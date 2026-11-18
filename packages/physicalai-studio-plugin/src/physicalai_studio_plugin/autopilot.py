# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The autopilot: how Physical AI Studio records what a simulation's scripted demonstrator does (protocol 1).

A simulation whose scenes have a demonstrator lets Physical AI Studio record its episodes. The simulation
keeps the task knowledge (resetting the scene, the demonstrator, deciding success); Physical AI Studio keeps
the dataset and the arm. In leader mode the simulation never moves the arm itself: the demonstrator only
publishes targets on the leader, and Physical AI Studio's follower writes them. So who drives the arm is
decided inside Physical AI Studio, by its control slot, as for any robot.

The protocol is a small episode API under :attr:`AutopilotSpec.path` (``/autopilot/v1``), at most one
episode at a time, ids chosen by the simulation:

- ``GET {path}`` returns :class:`AutopilotInfo`, with the current or last :class:`Episode`.
- ``POST {path}/episodes`` with :class:`EpisodeRequest` starts an episode (``preparing``, then ``ready``
  with its task); an episode in progress is ended first as ``aborted``.
- ``GET {path}/episodes/{id}``, ``POST {path}/episodes/{id}/begin`` (``ready`` → ``running``) and
  ``POST {path}/episodes/{id}/abort`` (→ ``ended``, idempotent).

An episode ends once, with its :data:`Outcome`. There is no finish: an ended episode stays readable until the
next ``POST``. Liveness: while an episode is in progress, any read of the simulation's leader or any request
under the path counts as a sign of life; after 1 s without one the simulation ends the episode as
``aborted``. Any other change to the scene (a route, the viewer, a scene switch) ends the episode as
``aborted`` first, then applies. Errors are ``{"detail": AutopilotError}``.

:class:`AutopilotClient` speaks it, from this machine only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Literal, Protocol, TypeVar, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError

if TYPE_CHECKING:
    import httpx

    from physicalai.robot.interface import Robot as PhysicalAIRobot

AUTOPILOT_PROTOCOL = 1
"""The protocol version this SDK describes."""
AUTOPILOT_PATH = "/autopilot/v1"
"""Where a simulation serves the protocol by default."""
LEADER_TIMEOUT_S = 1.0
"""How long the simulation lets an episode in progress go without a sign of life before it aborts it."""
MAX_ATTEMPTS = 5
MAX_RECOVERY_FRACTION = 0.5
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

_PayloadT = TypeVar("_PayloadT")
_PayloadT_contra = TypeVar("_PayloadT_contra", contravariant=True)

Phase = Literal["preparing", "ready", "running", "ended"]
"""An episode's phase: the scene is laid out (``preparing``), waits (``ready``), the demonstrator runs
(``running``), or it is over (``ended``, with its outcome)."""
Outcome = Literal["success", "failure", "stalled", "aborted"]
"""How an episode ended."""
AutopilotErrorCode = Literal["unsupported", "bad_option", "not_found", "wrong_phase"]
"""Why a request was refused: the scene has no autopilot or the native viewer is open (``409``); an option
the scene does not take (``422``); not the current or last episode (``404``); begin outside ``ready``
(``409``)."""


class _Strict(BaseModel):
    """A request: unknown fields are refused, so a typo never passes as a default."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class _Lenient(BaseModel):
    """An answer: fields a later version adds are ignored."""

    model_config = ConfigDict(extra="ignore", frozen=True)


class Episode(_Lenient):
    """An episode, in every episode answer."""

    id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._~-]+$")
    """URL-safe: used as a path segment in ``/episodes/{id}``."""
    """Chosen by the simulation, unique for its process's lifetime."""
    phase: Phase
    task: str | None = None
    """The instruction, known from ``ready``."""
    outcome: Outcome | None = None
    """Set when the phase is ``ended``."""
    reason: str | None = None
    """Why, for a failure, a stall or an abort."""
    clean: bool | None = None
    """For a success: without a retry."""


class AutopilotInfo(_Lenient):
    """``GET {path}``: the scene, whether it has an autopilot, and the current or last episode."""

    protocol: int = Field(ge=1)
    scene: str
    supported: bool
    reason: str | None = None
    """Why not, when ``supported`` is false."""
    options: list[str] = Field(default_factory=list)
    """The :class:`EpisodeOptions` fields this scene takes."""
    joint_names: list[str] = Field(default_factory=list)
    """The leader's joint names, in order: the simulated follower's."""
    episode: Episode | None = None


class EpisodeOptions(_Strict):
    """How to lay out an episode; ``None`` keeps the scene's default. Only :attr:`AutopilotInfo.options`."""

    max_attempts: int | None = Field(None, ge=1, le=MAX_ATTEMPTS)
    """The demonstrator's attempts per episode, retries included."""
    recovery_fraction: float | None = Field(None, ge=0.0, le=MAX_RECOVERY_FRACTION, allow_inf_nan=False)
    """The share of episodes that start from a stuck pose, to demonstrate a recovery."""

    def set_fields(self) -> set[str]:
        """Return the options that are set, which the scene must take."""
        return {name for name, value in self.model_dump().items() if value is not None}


class EpisodeRequest(_Strict):
    """``POST {path}/episodes``."""

    options: EpisodeOptions = EpisodeOptions()


class AutopilotError(_Lenient):
    """The ``detail`` of every ``4xx`` answer."""

    code: AutopilotErrorCode
    message: str


class LeaderFactory(Protocol[_PayloadT_contra]):
    """Builds the leader that reads a demonstrator's targets; see :class:`AutopilotSpec`."""

    def __call__(self, follower_payload: _PayloadT_contra, http_url: str) -> PhysicalAIRobot:
        """Return an unconnected leader for the follower configured by *follower_payload*.

        It reads the simulation at *http_url*; its class is exported with ``physicalai.config.export_config``.
        """
        ...


@dataclass(frozen=True)
class AutopilotSpec(Generic[_PayloadT]):
    """That a simulated robot type's simulation has an autopilot, and how to lead the follower from it.

    Set it on :attr:`SimulationLaunch.autopilot` of a simulated follower. Whether the
    running scene has one is :attr:`AutopilotInfo.supported`.
    """

    leader_for: LeaderFactory[_PayloadT]
    protocol: int = AUTOPILOT_PROTOCOL
    path: str = AUTOPILOT_PATH

    def __post_init__(self) -> None:
        """Validate the version and the path.

        Raises:
            ValueError: A version below 1, or a path that is not absolute, or has a query or a trailing slash.
        """
        if type(self.protocol) is not int or self.protocol < 1:
            msg = f"protocol must be a positive integer, got {self.protocol!r}"
            raise ValueError(msg)
        _check_path(self.path)


def _check_path(path: str) -> None:
    """Refuse a *path* that is not a plain absolute path on the simulation's own server.

    A URL or a network-path reference (``//host/...``) would make httpx send the request to another host, past
    :func:`loopback_root`.

    Raises:
        ValueError: The path is not absolute, starts with ``//``, ends with ``/`` or has a query, a fragment,
            a backslash or whitespace.
    """
    if (
        not path.startswith("/")
        or path.startswith("//")
        or path.endswith("/")
        or any(c in path for c in "?#\\")
        or any(c.isspace() for c in path)
    ):
        msg = f"path must be an absolute path without a query or a trailing slash, got {path!r}"
        raise ValueError(msg)


# ----------------------------------------------------------------------
# The client
# ----------------------------------------------------------------------


class AutopilotUnreachableError(Exception):
    """The simulation did not answer, or answered something that is not this protocol."""


class AutopilotRefusedError(Exception):
    """The simulation refused a request, with its :class:`AutopilotError`."""

    def __init__(self, status: int, code: str, message: str) -> None:
        """Keep the HTTP status, the error code and its message."""
        super().__init__(f"{code}: {message}")
        self.status = status
        self.code = code
        self.message = message


def loopback_root(http_url: str) -> str:
    """Return *http_url* as ``http://host:port`` when it is a root address on this machine.

    Returns:
        The root address.

    Raises:
        ValueError: Another scheme, a host that is not a loopback address, no port, credentials, a path, a
            query or a fragment.
    """
    parts = urlsplit(http_url)
    host = parts.hostname or ""
    if parts.scheme != "http" or host not in LOOPBACK_HOSTS:
        msg = f"The autopilot is only spoken to a simulation on this machine, not {http_url!r}"
        raise ValueError(msg)
    if parts.port is None or parts.username or parts.password or parts.path not in {"", "/"}:
        msg = f"The simulation's address {http_url!r} is not a root http://host:port address"
        raise ValueError(msg)
    if parts.query or parts.fragment:
        msg = f"The simulation's address {http_url!r} has a query or a fragment"
        raise ValueError(msg)
    netloc = f"[{host}]" if ":" in host else host
    return f"http://{netloc}:{parts.port}"


_MAX_ANSWER_BYTES = 64 * 1024
_FIRST_REDIRECT = 300
_M = TypeVar("_M", bound=BaseModel)


class AutopilotClient:
    """The protocol over HTTP, to a simulation on this machine: no proxy, no redirects, bounded answers.

    Synchronous methods and their ``a``-prefixed asynchronous twins, over ``httpx``.
    """

    def __init__(
        self,
        http_url: str,
        path: str = AUTOPILOT_PATH,
        *,
        timeout_s: float = 2.0,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Talk to the simulation at *http_url*, which :func:`loopback_root` checks (``ValueError``)."""
        import httpx  # noqa: PLC0415 - only the client needs it

        self.base_url = loopback_root(http_url)
        _check_path(path)
        self.path = path
        self._httpx = httpx
        self._sync = httpx.Client(
            base_url=self.base_url, timeout=timeout_s, follow_redirects=False, trust_env=False, transport=transport
        )
        self._async = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout_s,
            follow_redirects=False,
            trust_env=False,
            transport=async_transport,
        )

    def close(self) -> None:
        """Close the synchronous connections."""
        self._sync.close()

    async def aclose(self) -> None:
        """Close every connection."""
        self._sync.close()
        await self._async.aclose()

    def info(self) -> AutopilotInfo:
        """``GET {path}``.

        Returns:
            The scene and the current or last episode.
        """
        return self._call("GET", self.path, AutopilotInfo)

    def start_episode(self, options: EpisodeOptions | None = None) -> Episode:
        """``POST {path}/episodes``.

        Returns:
            The new episode, ``preparing``.
        """
        request = EpisodeRequest(options=options or EpisodeOptions())
        return self._call("POST", f"{self.path}/episodes", Episode, request)

    def episode(self, episode_id: str) -> Episode:
        """``GET {path}/episodes/{id}``.

        Returns:
            The episode.
        """
        return self._call("GET", f"{self.path}/episodes/{episode_id}", Episode)

    def begin(self, episode_id: str) -> Episode:
        """``POST {path}/episodes/{id}/begin``.

        Returns:
            The episode, ``running``.
        """
        return self._call("POST", f"{self.path}/episodes/{episode_id}/begin", Episode)

    def abort(self, episode_id: str) -> Episode:
        """``POST {path}/episodes/{id}/abort``.

        Returns:
            The episode, ``ended``.
        """
        return self._call("POST", f"{self.path}/episodes/{episode_id}/abort", Episode)

    async def ainfo(self) -> AutopilotInfo:
        """:meth:`info`, asynchronously.

        Returns:
            The scene and the current or last episode.
        """
        return await self._acall("GET", self.path, AutopilotInfo)

    async def astart_episode(self, options: EpisodeOptions | None = None) -> Episode:
        """:meth:`start_episode`, asynchronously.

        Returns:
            The new episode, ``preparing``.
        """
        request = EpisodeRequest(options=options or EpisodeOptions())
        return await self._acall("POST", f"{self.path}/episodes", Episode, request)

    async def aepisode(self, episode_id: str) -> Episode:
        """:meth:`episode`, asynchronously.

        Returns:
            The episode.
        """
        return await self._acall("GET", f"{self.path}/episodes/{episode_id}", Episode)

    async def abegin(self, episode_id: str) -> Episode:
        """:meth:`begin`, asynchronously.

        Returns:
            The episode, ``running``.
        """
        return await self._acall("POST", f"{self.path}/episodes/{episode_id}/begin", Episode)

    async def aabort(self, episode_id: str) -> Episode:
        """:meth:`abort`, asynchronously.

        Returns:
            The episode, ``ended``.
        """
        return await self._acall("POST", f"{self.path}/episodes/{episode_id}/abort", Episode)

    def _call(self, method: str, path: str, model: type[_M], body: BaseModel | None = None) -> _M:
        json_body = body.model_dump(mode="json") if body is not None else None
        try:
            with self._sync.stream(method, path, json=json_body) as response:
                content = bytearray()
                for chunk in response.iter_bytes():
                    content += chunk
                    _check_size(content, method, path)
                status = response.status_code
        except self._httpx.HTTPError as error:
            msg = f"The simulation at {self.base_url} does not answer: {error}"
            raise AutopilotUnreachableError(msg) from error
        return _parse(status, bytes(content), model, f"{method} {path}")

    async def _acall(self, method: str, path: str, model: type[_M], body: BaseModel | None = None) -> _M:
        json_body = body.model_dump(mode="json") if body is not None else None
        try:
            async with self._async.stream(method, path, json=json_body) as response:
                content = bytearray()
                async for chunk in response.aiter_bytes():
                    content += chunk
                    _check_size(content, method, path)
                status = response.status_code
        except self._httpx.HTTPError as error:
            msg = f"The simulation at {self.base_url} does not answer: {error}"
            raise AutopilotUnreachableError(msg) from error
        return _parse(status, bytes(content), model, f"{method} {path}")


def _check_size(content: bytearray, method: str, path: str) -> None:
    if len(content) > _MAX_ANSWER_BYTES:
        msg = f"The simulation answered {method} {path} with an oversized body"
        raise AutopilotUnreachableError(msg)


def _parse(status: int, content: bytes, model: type[_M], what: str) -> _M:
    if status >= _FIRST_REDIRECT:  # an error, or a redirect, which is never followed
        try:
            detail = AutopilotError.model_validate(json.loads(content)["detail"])
        except (ValueError, KeyError, TypeError, ValidationError):
            raise AutopilotRefusedError(status, f"http_{status}", content.decode(errors="replace")[:200]) from None
        raise AutopilotRefusedError(status, detail.code, detail.message)
    try:
        return cast("_M", cast("type[BaseModel]", model).model_validate_json(content))
    except ValidationError as error:
        msg = f"The simulation answered {what} with something that is not autopilot protocol {AUTOPILOT_PROTOCOL}"
        raise AutopilotUnreachableError(msg) from error


__all__ = [
    "AUTOPILOT_PATH",
    "AUTOPILOT_PROTOCOL",
    "LEADER_TIMEOUT_S",
    "MAX_ATTEMPTS",
    "MAX_RECOVERY_FRACTION",
    "AutopilotClient",
    "AutopilotError",
    "AutopilotErrorCode",
    "AutopilotInfo",
    "AutopilotRefusedError",
    "AutopilotSpec",
    "AutopilotUnreachableError",
    "Episode",
    "EpisodeOptions",
    "EpisodeRequest",
    "LeaderFactory",
    "Outcome",
    "Phase",
    "loopback_root",
]
