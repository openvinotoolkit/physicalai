# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import signal
import subprocess  # noqa: S404  # nosec B404  # ruff: ignore[noqa-comments] - entrypoints use argv with shell disabled.
import tempfile
import weakref
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass

# nosemgrep: python.lang.compatibility.python37.python37-compatibility-importlib2 - Runtime requires Python 3.11+.
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from ._sources import _copy_tree

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping, Sequence

    # nosemgrep: python.lang.compatibility.python37.python37-compatibility-importlib2 - Runtime requires Python 3.11+.
    from importlib.resources.abc import Traversable
    from types import TracebackType
    from typing import IO, Any, Self

    from ._environment import PreparedEnvironment

# Host variables that redirect module lookup or interpreter identity away from the prepared venv.
_HOST_INTERPRETER_VARIABLES = frozenset({
    "PYTHONEXECUTABLE",
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONPLATLIBDIR",
    "PYTHONSAFEPATH",
    "PYTHONUSERBASE",
    "VIRTUAL_ENV",
    "VIRTUAL_ENV_PROMPT",
    "__PYVENV_LAUNCHER__",
})

# Runs the entrypoint without putting its folder or the working directory on sys.path.
# argv[1] is a lifeline pipe fd (or "-"). A detached supervisor in the worker's process group
# holds it and kills the group on EOF: when the host exits, or when the session releases it
# after the worker exits. Double-forking keeps the supervisor out of the worker's children,
# so it outlives the worker and cannot be reaped by the worker's own wait calls.
_BOOTSTRAP = """\
import sys
if sys.path[:1] == [""]:
    del sys.path[0]
import os
import runpy
import signal
lifeline = sys.argv[1]
sys.argv = sys.argv[2:]
def _supervise(fd):
    try:
        while os.read(fd, 1):
            pass
    finally:
        os.killpg(0, signal.SIGKILL)
if lifeline != "-":
    fd = int(lifeline)
    child = os.fork()
    if child == 0:
        try:
            # Detach before the second fork so waitpid() guarantees signals and stdio are set.
            if fd < 3:
                import fcntl
                new_fd = fcntl.fcntl(fd, fcntl.F_DUPFD, 3)
                os.close(fd)
                fd = new_fd
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                signal.signal(sig, signal.SIG_IGN)
            os.chdir("/")
            null = os.open(os.devnull, os.O_RDWR)
            for stream in (0, 1, 2):
                os.dup2(null, stream)
            if null > 2:
                os.close(null)
            # Close any extra descriptors passed via pass_fds so the supervisor does not pin them open.
            max_fd = os.sysconf("SC_OPEN_MAX") if hasattr(os, "sysconf") else 4096
            if max_fd < 0 or max_fd > 65536:
                try:
                    fds = [int(name) for name in os.listdir("/proc/self/fd")]
                except OSError:
                    fds = list(range(3, 65536))
                for extra in fds:
                    if extra >= 3 and extra != fd:
                        try:
                            os.close(extra)
                        except OSError:
                            pass
            else:
                if fd > 3:
                    os.closerange(3, fd)
                os.closerange(fd + 1, max(max_fd, fd + 1))
            if os.fork() == 0:
                _supervise(fd)
        finally:
            os._exit(0)
    os.close(fd)
    os.waitpid(child, 0)
runpy.run_path(sys.argv[0], run_name="__main__")
"""

_POSIX = os.name == "posix"
_GRACE_SECONDS = 5.0


@dataclass(frozen=True, init=False)
class ExternalInterpreter:
    """A Python interpreter managed outside isolate, such as a user's own virtual environment.

    Pass it to :func:`run` or :func:`start` in place of a prepared environment. The entrypoint
    runs the same way, in its own process group with the host lifeline and :meth:`Session.close`,
    but isolate does not choose its packages: the child gets the host environment unchanged,
    including ``PYTHONPATH`` and ``VIRTUAL_ENV``, and imports whatever the interpreter can.

    Attributes:
        python: Absolute path to the Python executable.
    """

    python: Path

    def __init__(self, python: str | os.PathLike[str]) -> None:
        """Store the interpreter as an absolute path.

        Args:
            python: Path to the Python executable, such as ``sys.executable`` or a command-line
                value. ``~`` is expanded, and a relative path is made absolute against the current
                directory, so a session's ``cwd`` does not change it.

        Raises:
            ValueError: If the path is empty.
        """
        if not os.fspath(python):
            msg = "External interpreter path cannot be empty."
            raise ValueError(msg)
        # absolute(), not resolve(): following a venv's bin/python symlink would run the base
        # interpreter without the venv.
        object.__setattr__(self, "python", Path(python).expanduser().absolute())


class Session:
    """A running process launched by :func:`start` or :func:`run`.

    On POSIX the worker runs in its own process group: :meth:`terminate`, :meth:`kill`, and
    :meth:`close` reach its subprocesses too. Subprocesses the worker leaves behind are killed
    once the session sees the worker exit, and the whole group is killed if the host exits
    without closing the session. On Windows only the direct child is signalled.

    A session that is garbage-collected without being closed releases its resources, which on
    POSIX also kills the worker's process group.
    """

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        command: Sequence[str],
        release_entrypoint: Callable[[], None],
    ) -> None:
        self._process = process
        self._command = tuple(command)
        # Runs at most once: on exit, close, garbage collection, or interpreter shutdown.
        self._release = weakref.finalize(self, release_entrypoint)
        self._group_killed = False
        self._closed = False

    @property
    def args(self) -> tuple[str, ...]:
        """Startup argv tuple, including the interpreter's ``-c`` bootstrap and lifeline argument."""
        return self._command

    @property
    def pid(self) -> int:
        """Child process ID."""
        return self._process.pid

    @property
    def returncode(self) -> int | None:
        """Child exit status, or ``None`` while it is running."""
        return self.poll()

    @property
    def stdin(self) -> IO[Any] | None:
        """Child stdin stream when one was opened."""
        return self._process.stdin

    @property
    def stdout(self) -> IO[Any] | None:
        """Child stdout stream when one was captured."""
        return self._process.stdout

    @property
    def stderr(self) -> IO[Any] | None:
        """Child stderr stream when one was captured."""
        return self._process.stderr

    def poll(self) -> int | None:
        """Check the process state.

        Returns:
            The exit status, or ``None`` if the child is still running.
        """
        result = self._process.poll()
        if result is not None:
            self._release_resources()
        return result

    def wait(self, timeout: float | None = None) -> int:
        """Wait for the child to exit.

        Args:
            timeout: Maximum seconds to wait, or ``None`` to wait indefinitely.

        Returns:
            The child exit status.
        """
        result = self._process.wait(timeout=timeout)
        self._release_resources()
        return result

    def communicate(
        self,
        input_data: bytes | None = None,
        timeout: float | None = None,
    ) -> tuple[bytes | None, bytes | None]:
        """Send optional input and collect captured output.

        Args:
            input_data: Bytes sent to the child's stdin.
            timeout: Maximum seconds to wait, or ``None`` to wait indefinitely.

        Returns:
            A tuple containing captured stdout and stderr.
        """
        result = self._process.communicate(input=input_data, timeout=timeout)
        if self._process.poll() is not None:
            self._release_resources()
        return result

    def send_signal(self, sig: int) -> None:
        """Send *sig* to the direct child process only."""
        self._process.send_signal(sig)

    def terminate(self) -> None:
        """Ask the child and its subprocesses to terminate."""
        if _POSIX:
            self._signal_group(signal.SIGTERM)
        else:
            self._process.terminate()

    def kill(self) -> None:
        """Force-kill the child and its subprocesses."""
        if _POSIX:
            self._signal_group(signal.SIGKILL)
        else:
            self._process.kill()

    def close(self, *, timeout: float = _GRACE_SECONDS) -> None:
        """Stop the child and its subprocesses, then release session resources.

        Closes the child's stdin pipe so it can exit on its own, then sends SIGTERM, then
        SIGKILL, waiting up to *timeout* seconds after each of the first two steps.

        Args:
            timeout: Maximum seconds to wait after closing stdin and after sending SIGTERM.
        """
        if self._closed:
            return
        self._closed = True
        try:
            running = self.poll() is None
            if running and self._process.stdin is not None:
                with suppress(OSError):
                    self._process.stdin.close()
                running = not self._exited_within(timeout)
            if running:
                self.terminate()
                running = not self._exited_within(timeout)
        finally:
            if self._process.poll() is None:
                self.kill()
                self._process.wait()
            if _POSIX:
                # Usually a no-op: poll() or wait() above already released the lifeline, and the
                # supervisor kills the group itself. This covers an interrupt that skipped them,
                # and signals before _release_resources() while the supervisor still pins the PGID.
                self._signal_group(signal.SIGKILL)
            self._release_resources()
            for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
                if stream is not None:
                    # Flushing buffered stdin fails once the worker has exited.
                    with suppress(OSError):
                        stream.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_value, traceback
        try:
            if _POSIX and exc_type is not None and issubclass(exc_type, KeyboardInterrupt) and self.poll() is None:
                # The worker's own process group does not receive the terminal's Ctrl+C.
                self._signal_group(signal.SIGINT)
                self._exited_within(_GRACE_SECONDS)
        finally:
            self.close()

    def _exited_within(self, timeout: float) -> bool:
        try:
            self.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        return True

    def _signal_group(self, sig: int) -> None:
        # Once the lifeline closes or SIGKILL hits the group, the supervisor exits and the OS may reuse this PGID.
        if self._group_killed or not self._release.alive:
            return
        if sig == signal.SIGKILL:
            self._group_killed = True
        with suppress(ProcessLookupError):
            os.killpg(self._process.pid, sig)

    def _release_resources(self) -> None:
        self._release()


def run(
    prepared: PreparedEnvironment | ExternalInterpreter,
    entrypoint: str | os.PathLike[str] | Traversable,
    args: Sequence[str] = (),
    *,
    check: bool = True,
    timeout: float | None = None,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    input_data: bytes | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[bytes]:
    """Run a Python entrypoint to completion in a prepared environment or external interpreter.

    From a prepared environment, only the entrypoint's own code and the environment's packages
    are importable: a single script, or a folder with ``__main__.py`` whose modules sit inside it.

    Args:
        prepared: Environment returned by :func:`physicalai.isolate.prepare`, or an
            :class:`ExternalInterpreter`.
        entrypoint: Script, folder with ``__main__.py``, or importlib resource executed for this run.
        args: Arguments forwarded as individual argv items after the entrypoint.
        check: Raise :class:`subprocess.CalledProcessError` for a non-zero status.
        timeout: Maximum seconds to wait for completion; on expiry the worker and its
            subprocesses are killed and :class:`subprocess.TimeoutExpired` is raised.
        cwd: Optional child working directory.
        env: Base child environment, defaulting to the host environment. For a prepared
            environment, Python path variables such as ``PYTHONPATH`` are removed even when set
            here, and the environment is activated; an external interpreter gets it unchanged.
        input_data: Bytes sent to the child's stdin.
        capture_output: Capture stdout and stderr in the completed-process result.

    Returns:
        The standard-library completed-process result.

    Raises:
        subprocess.CalledProcessError: If *check* is set and the worker exits with a non-zero status.
        subprocess.TimeoutExpired: If the worker does not finish within *timeout*.
    """
    session = start(
        prepared,
        entrypoint,
        args,
        cwd=cwd,
        env=env,
        stdin=None if input_data is None else subprocess.PIPE,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
    )
    with session:
        try:
            stdout, stderr = session.communicate(input_data, timeout=timeout)
        except subprocess.TimeoutExpired:
            session.kill()
            raise
        returncode = session.wait()
    command = list(session.args)
    if check and returncode:
        raise subprocess.CalledProcessError(returncode, command, stdout, stderr)
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def start(
    prepared: PreparedEnvironment | ExternalInterpreter,
    entrypoint: str | os.PathLike[str] | Traversable,
    args: Sequence[str] = (),
    *,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    stdin: int | IO[Any] | None = None,
    stdout: int | IO[Any] | None = None,
    stderr: int | IO[Any] | None = None,
    pass_fds: Sequence[int] = (),
    bufsize: int = -1,
) -> Session:
    """Start an entrypoint and return a controllable process session.

    From a prepared environment, only the entrypoint's own code and the environment's packages
    are importable: a single script, or a folder with ``__main__.py`` whose modules sit inside it.

    Args:
        prepared: Environment returned by :func:`physicalai.isolate.prepare`, or an
            :class:`ExternalInterpreter`.
        entrypoint: Script, folder with ``__main__.py``, or importlib resource executed for this session.
        args: Arguments forwarded as individual argv items after the entrypoint.
        cwd: Optional child working directory.
        env: Base child environment, defaulting to the host environment. For a prepared
            environment, Python path variables such as ``PYTHONPATH`` are removed even when set
            here, and the environment is activated; an external interpreter gets it unchanged.
        stdin: Optional child stdin stream or subprocess stream constant.
        stdout: Optional child stdout stream or subprocess stream constant.
        stderr: Optional child stderr stream or subprocess stream constant.
        pass_fds: Optional POSIX file descriptors inherited by the worker (for example, an
            out-of-band IPC pipe, socketpair, or shared-memory descriptor). Rejected with
            :class:`ValueError` on Windows.
        bufsize: Stream buffer size passed to :class:`subprocess.Popen` (``-1`` uses the
            system default; ``0`` selects unbuffered binary streams).

    Returns:
        A session exposing process status, streams, and lifecycle methods.
    """
    _check_launch(prepared, args, pass_fds)
    child_env = _child_environment(prepared, env)
    resources_stack = ExitStack()
    try:
        script = resources_stack.enter_context(_entrypoint_path(entrypoint))
        process, command = _launch(
            prepared,
            script,
            args,
            resources_stack,
            cwd=cwd,
            env=child_env,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            pass_fds=tuple(pass_fds),
            bufsize=bufsize,
        )
    except BaseException:
        resources_stack.close()
        raise
    return Session(process, command, resources_stack.close)


def _launch(
    prepared: PreparedEnvironment | ExternalInterpreter,
    script: Path,
    args: Sequence[str],
    resources_stack: ExitStack,
    *,
    cwd: str | os.PathLike[str] | None,
    env: dict[str, str],
    stdin: int | IO[Any] | None,
    stdout: int | IO[Any] | None,
    stderr: int | IO[Any] | None,
    pass_fds: tuple[int, ...] = (),
    bufsize: int = -1,
) -> tuple[subprocess.Popen[bytes], list[str]]:
    lifeline = None
    if _POSIX:
        lifeline, lifeline_write = os.pipe()
        resources_stack.callback(os.close, lifeline_write)
    try:
        command = _command(prepared, script, args, lifeline)
        forwarded_fds = pass_fds if lifeline is None else (lifeline, *pass_fds)
        process = subprocess.Popen(  # noqa: S603  # nosec B603  # ruff: ignore[noqa-comments] - explicit interpreter and entrypoint argv, no shell.
            command,
            bufsize=bufsize,
            cwd=cwd,
            env=env,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            shell=False,
            pass_fds=forwarded_fds,
            process_group=0 if _POSIX else None,
        )
    finally:
        if lifeline is not None:
            os.close(lifeline)
    return process, command


@contextmanager
def _entrypoint_path(
    entrypoint: str | os.PathLike[str] | Traversable,
) -> Generator[Path, None, None]:
    if isinstance(entrypoint, (str, os.PathLike)):
        path = Path(entrypoint).expanduser().resolve()
        _check_entrypoint(path)
        yield path
        return
    if entrypoint.is_dir():
        # resources.as_file() only extracts directories from Python 3.12 on.
        with tempfile.TemporaryDirectory(prefix="physicalai-isolate-entrypoint-") as temporary:
            path = Path(temporary).resolve() / "entrypoint"
            _copy_tree(entrypoint, path, label="Entrypoint")
            _check_entrypoint(path)
            yield path
        return
    with resources.as_file(entrypoint) as resource_path:
        path = Path(resource_path).resolve()
        _check_entrypoint(path)
        yield path


def _check_entrypoint(path: Path) -> None:
    if not (path.is_file() or (path / "__main__.py").is_file()):
        msg = f"Entrypoint must be a file or a folder containing __main__.py: {path}."
        raise FileNotFoundError(msg)


def _check_launch(
    prepared: PreparedEnvironment | ExternalInterpreter,
    args: Sequence[str],
    pass_fds: Sequence[int] = (),
) -> None:
    if isinstance(args, str) or not all(isinstance(argument, str) for argument in args):
        msg = "Entrypoint arguments must be a sequence of strings."
        raise TypeError(msg)
    if not all(isinstance(fd, int) and not isinstance(fd, bool) and fd >= 0 for fd in pass_fds):
        msg = "pass_fds must be a sequence of non-negative integer file descriptors."
        raise TypeError(msg)
    if pass_fds and not _POSIX:
        # subprocess only asserts this on Windows, and the assertion disappears under -O.
        msg = "pass_fds is only supported on POSIX."
        raise ValueError(msg)
    if not prepared.python.is_file():
        owner = "External interpreter" if isinstance(prepared, ExternalInterpreter) else "Prepared environment"
        msg = f"{owner} has no Python interpreter at {prepared.python}."
        raise FileNotFoundError(msg)


def _command(
    prepared: PreparedEnvironment | ExternalInterpreter,
    script: Path,
    args: Sequence[str],
    lifeline: int | None,
) -> list[str]:
    lifeline_argument = "-" if lifeline is None else str(lifeline)
    return [str(prepared.python), "-c", _BOOTSTRAP, lifeline_argument, str(script), *args]


def _child_environment(
    prepared: PreparedEnvironment | ExternalInterpreter,
    env: Mapping[str, str] | None,
) -> dict[str, str]:
    base = os.environ if env is None else env
    if isinstance(prepared, ExternalInterpreter):
        # The caller manages this interpreter, so its environment is theirs to set.
        return dict(base)
    child = {key: value for key, value in base.items() if _variable_name(key) not in _HOST_INTERPRETER_VARIABLES}
    path_key = next((key for key in child if _variable_name(key) == "PATH"), "PATH")
    search_path = child.get(path_key, os.defpath)
    child[path_key] = os.pathsep.join(part for part in (str(prepared.python.parent), search_path) if part)
    child["VIRTUAL_ENV"] = str(prepared.venv)
    return child


def _variable_name(key: str) -> str:
    # Windows environment variable names are case-insensitive.
    return key.upper() if os.name == "nt" else key


__all__ = ["ExternalInterpreter", "Session", "run", "start"]
