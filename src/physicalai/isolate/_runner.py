# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
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
_BOOTSTRAP = """\
import sys
if sys.path[:1] == [""]:
    del sys.path[0]
import runpy
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


class Session:
    """A running process launched from a prepared environment."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        command: Sequence[str],
        release_entrypoint: Callable[[], None],
    ) -> None:
        self._process = process
        self._command = tuple(command)
        self._release_entrypoint: Callable[[], None] | None = release_entrypoint

    @property
    def args(self) -> tuple[str, ...]:
        """Startup argv tuple, including the interpreter's ``-c`` bootstrap."""
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
        """Send *sig* to the child process."""
        self._process.send_signal(sig)

    def terminate(self) -> None:
        """Ask the child process to terminate."""
        self._process.terminate()

    def kill(self) -> None:
        """Force-kill the child process."""
        self._process.kill()

    def close(self, *, timeout: float = 5.0) -> None:
        """Stop a running child and release its process and entrypoint resources."""
        if self.poll() is None:
            with suppress(ProcessLookupError):
                self._process.terminate()
            try:
                self.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self.wait()
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
        del exc_type, exc_value, traceback
        self.close()

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
        timeout: Maximum seconds to wait for completion.
        cwd: Optional child working directory.
        env: Base child environment, defaulting to the host environment. Python path variables
            such as ``PYTHONPATH`` are removed even when set here, and the prepared environment
            is activated.
        input_data: Bytes sent to the child's stdin.
        capture_output: Capture stdout and stderr in the completed-process result.

    Returns:
        The standard-library completed-process result.
    """
    with _entrypoint_path(entrypoint) as script:
        command = _command(prepared, script, args)
        return subprocess.run(  # noqa: S603  # ruff: ignore[noqa-comments] - explicit interpreter and entrypoint argv, no shell.
            command,
            check=check,
            timeout=timeout,
            cwd=cwd,
            env=_child_environment(prepared, env),
            input=input_data,
            capture_output=capture_output,
            shell=False,
        )


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
        command = _command(prepared, script, args)
        process = subprocess.Popen(  # noqa: S603  # ruff: ignore[noqa-comments] - explicit interpreter and entrypoint argv, no shell.
            command,
            cwd=cwd,
            env=_child_environment(prepared, env),
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            shell=False,
        )
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


def _command(prepared: PreparedEnvironment, script: Path, args: Sequence[str]) -> list[str]:
    if isinstance(args, str) or not all(isinstance(argument, str) for argument in args):
        msg = "Entrypoint arguments must be a sequence of strings."
        raise TypeError(msg)
    if not prepared.python.is_file():
        msg = f"Prepared environment has no Python interpreter at {prepared.python}."
        raise FileNotFoundError(msg)
    return [str(prepared.python), "-c", _BOOTSTRAP, str(script), *args]


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
