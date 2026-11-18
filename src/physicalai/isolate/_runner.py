# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import signal
import subprocess  # noqa: S404  # ruff: ignore[noqa-comments] - entrypoints use argv with shell disabled.
from contextlib import ExitStack, contextmanager, suppress
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping, Sequence
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
# argv[1] is a lifeline pipe fd (or "-"); EOF means the host died, so the worker group is killed.
_BOOTSTRAP = """\
import sys
if sys.path[:1] == [""]:
    del sys.path[0]
import os
import runpy
import signal
import threading
lifeline = sys.argv[1]
sys.argv = sys.argv[2:]
if lifeline != "-":
    fd = int(lifeline)
    os.set_inheritable(fd, False)
    def _watch_host():
        while os.read(fd, 1):
            pass
        os.killpg(0, signal.SIGKILL)
    threading.Thread(target=_watch_host, name="physicalai-isolate-lifeline", daemon=True).start()
runpy.run_path(sys.argv[0], run_name="__main__")
"""

_POSIX = os.name == "posix"
_GRACE_SECONDS = 5.0


class Session:
    """A running process launched from a prepared environment.

    On POSIX the worker runs in its own process group: :meth:`terminate`, :meth:`kill`, and
    :meth:`close` reach its subprocesses too, and the whole group is killed if the host exits
    without closing the session. On Windows only the direct child is signalled.
    """

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        command: Sequence[str],
        release_entrypoint: Callable[[], None],
    ) -> None:
        self._process = process
        self._command = tuple(command)
        self._release_entrypoint: Callable[[], None] | None = release_entrypoint
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
        """
        if self._closed:
            return
        self._closed = True
        running = self.poll() is None
        if running and self._process.stdin is not None:
            with suppress(OSError):
                self._process.stdin.close()
            running = not self._exited_within(timeout)
        if running:
            self.terminate()
            running = not self._exited_within(timeout)
        if running:
            self.kill()
            self.wait()
        if _POSIX:
            # Remove subprocesses the worker left behind; the group ID stays reserved while any live.
            self._signal_group(signal.SIGKILL)
        self._release_resources()
        for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
            if stream is not None:
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
        if _POSIX and exc_type is not None and issubclass(exc_type, KeyboardInterrupt) and self.poll() is None:
            # The worker's own process group does not receive the terminal's Ctrl+C.
            self._signal_group(signal.SIGINT)
            self._exited_within(_GRACE_SECONDS)
        self.close()

    def _exited_within(self, timeout: float) -> bool:
        try:
            self.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        return True

    def _signal_group(self, sig: int) -> None:
        with suppress(ProcessLookupError):
            os.killpg(self._process.pid, sig)

    def _release_resources(self) -> None:
        release = self._release_entrypoint
        if release is not None:
            self._release_entrypoint = None
            release()


def run(
    prepared: PreparedEnvironment,
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
    """Run a Python entrypoint to completion in a prepared environment.

    Only the entrypoint's own code is importable: a single script, or a folder with
    ``__main__.py`` whose modules sit inside that folder.

    Args:
        prepared: Environment returned by :func:`physicalai.isolate.prepare`.
        entrypoint: Script, folder with ``__main__.py``, or importlib resource executed for this run.
        args: Arguments forwarded as individual argv items after the entrypoint.
        check: Raise :class:`subprocess.CalledProcessError` for a non-zero status.
        timeout: Maximum seconds to wait for completion; on expiry the worker and its
            subprocesses are killed and :class:`subprocess.TimeoutExpired` is raised.
        cwd: Optional child working directory.
        env: Base child environment, defaulting to the host environment. Python path variables
            such as ``PYTHONPATH`` are removed even when set here, and the prepared environment
            is activated.
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
    prepared: PreparedEnvironment,
    entrypoint: str | os.PathLike[str] | Traversable,
    args: Sequence[str] = (),
    *,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    stdin: int | IO[Any] | None = None,
    stdout: int | IO[Any] | None = None,
    stderr: int | IO[Any] | None = None,
) -> Session:
    """Start an entrypoint and return a controllable process session.

    Only the entrypoint's own code is importable: a single script, or a folder with
    ``__main__.py`` whose modules sit inside that folder.

    Args:
        prepared: Environment returned by :func:`physicalai.isolate.prepare`.
        entrypoint: Script, folder with ``__main__.py``, or importlib resource executed for this session.
        args: Arguments forwarded as individual argv items after the entrypoint.
        cwd: Optional child working directory.
        env: Base child environment, defaulting to the host environment. Python path variables
            such as ``PYTHONPATH`` are removed even when set here, and the prepared environment
            is activated.
        stdin: Optional child stdin stream or subprocess stream constant.
        stdout: Optional child stdout stream or subprocess stream constant.
        stderr: Optional child stderr stream or subprocess stream constant.

    Returns:
        A session exposing process status, streams, and lifecycle methods.
    """
    resources_stack = ExitStack()
    try:
        script = resources_stack.enter_context(_entrypoint_path(entrypoint))
        lifeline = None
        if _POSIX:
            lifeline, lifeline_write = os.pipe()
            resources_stack.callback(os.close, lifeline_write)
        command = _command(prepared, script, args, lifeline)
        try:
            process = subprocess.Popen(  # noqa: S603  # ruff: ignore[noqa-comments] - explicit interpreter and entrypoint argv, no shell.
                command,
                cwd=cwd,
                env=_child_environment(prepared, env),
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                shell=False,
                pass_fds=() if lifeline is None else (lifeline,),
                process_group=0 if _POSIX else None,
            )
        finally:
            if lifeline is not None:
                os.close(lifeline)
    except BaseException:
        resources_stack.close()
        raise
    return Session(process, command, resources_stack.close)


@contextmanager
def _entrypoint_path(
    entrypoint: str | os.PathLike[str] | Traversable,
) -> Generator[Path, None, None]:
    if isinstance(entrypoint, (str, os.PathLike)):
        path = Path(entrypoint).expanduser().resolve()
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


def _command(
    prepared: PreparedEnvironment,
    script: Path,
    args: Sequence[str],
    lifeline: int | None,
) -> list[str]:
    if isinstance(args, str) or not all(isinstance(argument, str) for argument in args):
        msg = "Entrypoint arguments must be a sequence of strings."
        raise TypeError(msg)
    if not prepared.python.is_file():
        msg = f"Prepared environment has no Python interpreter at {prepared.python}."
        raise FileNotFoundError(msg)
    lifeline_argument = "-" if lifeline is None else str(lifeline)
    return [str(prepared.python), "-c", _BOOTSTRAP, lifeline_argument, str(script), *args]


def _child_environment(prepared: PreparedEnvironment, env: Mapping[str, str] | None) -> dict[str, str]:
    base = os.environ if env is None else env
    child = {key: value for key, value in base.items() if _variable_name(key) not in _HOST_INTERPRETER_VARIABLES}
    path_key = next((key for key in child if _variable_name(key) == "PATH"), "PATH")
    search_path = child.get(path_key, os.defpath)
    child[path_key] = os.pathsep.join(part for part in (str(prepared.python.parent), search_path) if part)
    child["VIRTUAL_ENV"] = str(prepared.venv)
    return child


def _variable_name(key: str) -> str:
    # Windows environment variable names are case-insensitive.
    return key.upper() if os.name == "nt" else key


__all__ = ["Session", "run", "start"]
