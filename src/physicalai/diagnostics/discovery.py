# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Interactive selection logic with injectable prompts and passive recipe output."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic_core import PydanticUndefined

from physicalai.config import Config
from physicalai.diagnostics.cameras import camera_config, supported_modes
from physicalai.diagnostics.export import component_bundle, runtime_document
from physicalai.diagnostics.identity import binding_paths, resolve_endpoint
from physicalai.diagnostics.render import safe_text

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from physicalai.diagnostics.models import Device, Endpoint, Inventory
    from physicalai.diagnostics.plugins import Catalog


class Prompts:
    """Small terminal-independent prompt interface, easily scripted in tests."""

    def __init__(self, *, read: Callable[[str], str] = input, write: Callable[[str], None] = print) -> None:
        """Inject prompt input/output independently of process-global streams."""
        self.read = read
        self.write = write

    def ask(self, label: str, default: str | None = None) -> str:
        """Read text; EOF is cancellation rather than an infinite prompt loop.

        Returns:
            Stripped input or the explicit default.
        """
        suffix = f" [{safe_text(default)}]" if default is not None else ""
        return self.read(safe_text(label) + suffix + ": ").strip() or default or ""

    def confirm(self, label: str, *, default: bool = False) -> bool:
        """Require an explicit yes/no response.

        Returns:
            Whether the user accepted the choice.
        """
        while True:
            answer = self.ask(label + " (yes/no)", "yes" if default else "no").lower()
            if answer in {"yes", "y", "no", "n"}:
                return answer in {"yes", "y"}
            self.write("Please answer yes or no.")

    def choose(self, label: str, options: Sequence[str]) -> int:
        """Select a numbered option, validating input instead of accepting guesses.

        Returns:
            The zero-based selection index.

        Raises:
            ValueError: If no options exist.
        """
        if not options:
            msg = f"No options available: {label}"
            raise ValueError(msg)
        self.write(label)
        for index, option in enumerate(options, start=1):
            self.write(f"  {index}. {safe_text(option)}")
        while True:
            answer = self.ask("Selection")
            if answer.isdecimal() and 1 <= int(answer) <= len(options):
                return int(answer) - 1
            self.write(f"Enter a number between 1 and {len(options)}.")

    def positive_number(self, label: str, default: str, *, integer: bool = False) -> float:
        """Reprompt for malformed or nonpositive numeric settings.

        Returns:
            A finite positive value, integral when requested.
        """
        while True:
            try:
                value = float(self.ask(label, default))
            except ValueError:
                self.write("Enter a positive number.")
                continue
            if math.isfinite(value) and value > 0 and (not integer or value.is_integer()):
                return value
            self.write("Enter a finite positive integer." if integer else "Enter a finite positive number.")


@dataclass(frozen=True)
class Selection:
    """Evidence to recheck immediately before writing an exported connection."""

    selector: str
    vendor_id: str
    product_id: str
    serial: str | None
    interface: str
    kind: str

    def validate(self, inventory: Inventory) -> None:
        """Refuse disappeared, ambiguous, or observably replaced selections.

        Raises:
            ValueError: If the selected endpoint no longer matches.
        """
        device, endpoint = resolve_endpoint(inventory, self.selector)
        if (device.vendor_id, device.product_id, device.serial, endpoint.interface, endpoint.kind) != (
            self.vendor_id,
            self.product_id,
            self.serial,
            self.interface,
            self.kind,
        ):
            msg = f"Hardware changed for {self.selector!r}; restart discovery and review the binding."
            raise ValueError(msg)


@dataclass
class RobotSelection:
    """Existing catalog builder/exporter descriptor, without a live driver."""

    type: str
    payload: BaseModel


class DiscoverySession:
    """Build recipes from a snapshot, never by constructing live driver objects."""

    def __init__(
        self,
        inventory: Inventory,
        catalog: Catalog,
        prompts: Prompts,
        *,
        refresh: Callable[[], Inventory] | None = None,
    ) -> None:
        """Keep the passive evidence and user selections for later revalidation.

        Raises:
            ValueError: If a redacted support inventory is supplied for export.
        """
        if inventory.redacted:
            msg = "Redacted support inventories cannot be used to export hardware configurations."
            raise ValueError(msg)
        self.inventory = inventory
        self.catalog = catalog
        self.prompts = prompts
        self.refresh = refresh
        self.selections: list[Selection] = []

    def _name(self, label: str, used: dict[str, Config]) -> str:
        while True:
            name = self.prompts.ask(label)
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", name) and name not in used:
                return name
            self.prompts.write("Use a unique name starting with a letter (letters, digits, _, -, .).")

    def _connection(self, device: Device, endpoint: Endpoint) -> str:
        options: list[tuple[str, str]] = []
        for binding in ("device", "port"):
            paths = binding_paths(self.inventory, endpoint, binding)
            if paths:
                options.append((paths[0], f"{binding}-bound: {paths[0]}"))
        options.append((endpoint.path, f"Ephemeral/unverified: {endpoint.path}"))
        index = self.prompts.choose("Choose a connection binding", [label for _, label in options])
        selector = options[index][0]
        for selection in self.selections:
            _, selected_endpoint = resolve_endpoint(self.inventory, selection.selector)
            if selected_endpoint.id == endpoint.id:
                msg = "This endpoint is already selected; do not export multiple owners through different aliases."
                raise ValueError(msg)
        if index == len(options) - 1 and not self.prompts.confirm(
            "This identifier can change or select another device after reconnect/reboot. Accept?"
        ):
            msg = "Unstable connection rejected. Choose a stable binding or configure the device manually."
            raise ValueError(msg)
        if "port-bound" in options[index][1]:
            self.prompts.write(
                "Port binding follows a slot, not a physical unit. Identical replacements cannot be verified."
            )
        self.selections.append(
            Selection(selector, device.vendor_id, device.product_id, device.serial, endpoint.interface, endpoint.kind)
        )
        return selector

    def _select_endpoint(self, kind: Literal["video", "serial"]) -> tuple[Device, Endpoint]:
        used = {selection.selector for selection in self.selections}
        if kind == "video":
            self.prompts.write(
                "Only supported UVC image streams are selectable; other functions remain in the inventory."
            )
        while True:
            options = [
                (device, endpoint)
                for device in self.inventory.devices
                for endpoint in device.endpoints
                if self._selectable(endpoint, kind)
                and endpoint.path not in used
                and not any(alias.path in used for alias in endpoint.aliases)
            ]
            labels = [f"{d.name} · {ep.path} · {d.location}" for d, ep in options]
            actions = ["select"] * len(options)
            if self.refresh is not None:
                if options:
                    labels.append("Locate a connection by safely unplugging/replugging one idle device")
                    actions.append("locate")
                labels.append("Refresh inventory (connect hardware first)")
                actions.append("refresh")
            index = self.prompts.choose(f"Select {kind} endpoint", labels)
            if actions[index] == "select":
                return options[index]
            if actions[index] == "locate":
                return self._locate_endpoint(kind)
            if self.refresh is not None:
                self.inventory = self.refresh()

    def _selectable(self, endpoint: Endpoint, kind: str) -> bool:
        if endpoint.kind != kind:
            return False
        if kind != "video":
            return True
        return endpoint.role == "capture" and (self.inventory.platform != "linux" or bool(supported_modes(endpoint)))

    def _locate_endpoint(self, kind: Literal["video", "serial"]) -> tuple[Device, Endpoint]:
        if self.refresh is None:
            msg = "No inventory refresh strategy is available."
            raise ValueError(msg)
        before = {d.id: d for d in self.inventory.devices if any(ep.kind == kind for ep in d.endpoints)}
        self.prompts.write(
            "Only disconnect hardware when it is safe and idle. This operation never opens a robot port."
        )
        self.prompts.ask("Unplug exactly one intended device, wait for the OS, then press Enter")
        unplugged = self.refresh()
        remaining = {d.id for d in unplugged.devices}
        removed = [device for key, device in before.items() if key not in remaining]
        if len(removed) != 1:
            msg = "Could not isolate one unplugged device; reconnect everything and restart discovery."
            raise ValueError(msg)
        original = removed[0]
        self.prompts.ask(
            f"Located {original.name} at {original.location}. Reconnect in the same slot, then press Enter"
        )
        returned = self.refresh()
        matches = [
            d
            for d in returned.devices
            if d.id == original.id
            and (d.vendor_id, d.product_id, d.serial) == (original.vendor_id, original.product_id, original.serial)
        ]
        if len(matches) != 1:
            msg = "The connection could not be matched reliably after replug; restart and inspect the new inventory."
            raise ValueError(msg)
        self.inventory = returned
        self.prompts.write("Located the connection; a port/index match alone does not prove physical device identity.")
        choices = [ep for ep in matches[0].endpoints if self._selectable(ep, kind)]
        index = self.prompts.choose("Choose the returned endpoint/stream", [ep.path for ep in choices])
        return matches[0], choices[index]

    def camera(self) -> Config:
        """Select a capture endpoint and an advertised compatible wire mode.

        Returns:
            A camera construction recipe with an explicit supported backend.

        Raises:
            ValueError: If no safe/exportable mode is available.
        """
        device, endpoint = self._select_endpoint("video")
        selector = self._connection(device, endpoint)
        modes = supported_modes(endpoint)
        if self.inventory.platform == "linux":
            if not modes:
                msg = "No advertised mode supported by the UVC Runtime backend. "
                msg += "Select another stream or use its vendor backend."
                raise ValueError(msg)
            mode = modes[
                self.prompts.choose(
                    "Choose wire mode (not measured throughput)",
                    [f"{m.width}x{m.height} @ {m.fps:g} FPS · {m.pixel_format}" for m in modes],
                )
            ]
            return camera_config(selector, mode)
        self.prompts.write("Native passive mode enumeration is unavailable; requested mode is not hardware-verified.")
        width = int(self.prompts.positive_number("Width", "640", integer=True))
        height = int(self.prompts.positive_number("Height", "480", integer=True))
        fps = int(self.prompts.positive_number("FPS", "30", integer=True))
        return Config(
            "physicalai.capture.UVCCamera",
            {
                "device": selector,
                "backend": "omnicamera",
                "width": width,
                "height": height,
                "fps": fps,
            },
        )

    def robot(self, *, role: str | None = None) -> tuple[Config, str]:
        """Select an installed model and use its explicit passive export hook.

        Returns:
            The robot recipe and its catalog role.

        Raises:
            ValueError: If compatible exporters or required bindings are missing.
            TypeError: If a plugin returns a non-Config recipe.
        """
        definitions = [
            definition
            for definition in self.catalog.definitions.values()
            if getattr(definition, "config_exporter", None) and (role is None or definition.role == role)
        ]
        if not definitions:
            msg = "No installed robot plugin supports passive recipe export for this role. "
            msg += "Install/update its Runtime plugin."
            raise ValueError(msg)
        definition = definitions[
            self.prompts.choose(
                "Choose the actual robot model (USB adapters do not prove model)",
                [f"{d.display_name} ({d.role})" for d in definitions],
            )
        ]
        model = definition.robot_payload
        if model is None or "connection_string" not in model.model_fields:
            msg = "This payload needs a specialized discovery adapter; use its manual Runtime configuration."
            raise ValueError(msg)
        device, endpoint = self._select_endpoint("serial")
        selector = self._connection(device, endpoint)
        values: dict[str, Any] = {"connection_string": selector, "serial_number": ""}
        for name, field in model.model_fields.items():
            if name in values:
                continue
            default = field.get_default(call_default_factory=True)
            default_text = None if default is PydanticUndefined else json.dumps(default, default=str)
            if field.description:
                self.prompts.write(safe_text(field.description))
            while True:
                raw = self.prompts.ask(name, default_text)
                try:
                    try:
                        value = json.loads(raw)
                    except ValueError:
                        value = raw
                    values[name] = TypeAdapter(field.rebuild_annotation()).validate_python(value)
                    break
                except ValidationError as exc:
                    self.prompts.write(safe_text(exc))
        payload = model.model_validate(values)
        self.prompts.write(
            "Review connection effects: "
            + safe_text(payload.model_dump(exclude={"connection_string", "serial_number"}))
        )
        if not self.prompts.confirm("Accept these settings for a future run? Export itself does not connect"):
            msg = "Driver settings were not approved."
            raise ValueError(msg)
        exporter = definition.config_exporter
        if exporter is None:
            msg = "Selected plugin has no passive config exporter."
            raise ValueError(msg)
        config = exporter(RobotSelection(type=definition.type, payload=payload))
        if not isinstance(config, Config):
            msg = "Plugin config_exporter did not return physicalai.config.Config."
            raise TypeError(msg)
        return config, definition.role

    def _policy_source(self, fps: float) -> Config:
        while True:
            policy = Path(self.prompts.ask("Existing exported policy directory")).expanduser().resolve()
            if (policy / "manifest.json").is_file():
                break
            self.prompts.write("Select an existing exported policy directory containing manifest.json.")
        selected = self.prompts.choose(
            "Policy execution mode",
            [
                "Synchronous (simplest; inference can slow the control loop)",
                "Asynchronous (background action-chunk inference)",
                "RTC (requires a reviewed RTC-compatible policy)",
            ],
        )
        execution = ("SyncExecution", "AsyncExecution", "RTCExecution")[selected]
        args: dict[str, object] = {
            "model": Config("physicalai.inference.InferenceModel", {"export_dir": str(policy)}).to_dict(),
            "execution": Config(
                f"physicalai.runtime.{execution}", {"fps": fps} if execution == "RTCExecution" else {}
            ).to_dict(),
        }
        if execution == "RTCExecution":
            if not self.prompts.confirm("Have you verified that this export supports RTC inputs and chunking?"):
                msg = "RTC compatibility was not confirmed."
                raise ValueError(msg)
            args["action_queue"] = Config("physicalai.runtime.RTCActionQueue").to_dict()
        task = self.prompts.ask("Policy task instruction (leave blank if unused)")
        if task:
            args["task"] = task
        return Config("physicalai.runtime.PolicySource", args)

    def build(self, *, components_only: bool) -> dict[str, object]:
        """Guide selection and return either reusable recipes or a Runtime document.

        Returns:
            A component bundle or complete CLI-shaped Runtime configuration.

        Raises:
            ValueError: If required selections or safety confirmations are absent.
        """
        cameras: dict[str, Config] = {}
        robots: dict[str, Config] = {}
        follower: Config | None = None
        if components_only:
            while self.prompts.confirm("Add a robot component?"):
                name = self._name("Robot name", robots)
                robots[name], _ = self.robot()
        else:
            follower, _ = self.robot(role="follower")
        while self.prompts.confirm("Add a camera?"):
            name = self._name("Camera name (must match the policy when using inference)", cameras)
            cameras[name] = self.camera()
        if components_only:
            if not robots and not cameras:
                msg = "Nothing selected; no empty component bundle will be written."
                raise ValueError(msg)
            return component_bundle(robots, cameras)
        source = self.prompts.choose(
            "Choose action source",
            [
                "Exported policy (no inference during setup)",
                "Leader teleoperation (same joint ordering/morphology required)",
            ],
        )
        fps = self.prompts.positive_number("Runtime control-loop FPS", "30")
        slower = [
            name
            for name, camera in cameras.items()
            if isinstance((rate := camera.init_args.get("fps")), (int, float)) and rate < fps
        ]
        if slower and not self.prompts.confirm(
            f"Cameras {slower} are configured below the control-loop rate and may slow it. Keep these rates?"
        ):
            msg = "Camera/control-loop rates were not approved; restart with matching rates."
            raise ValueError(msg)
        if source == 0:
            action_source = self._policy_source(fps)
        else:
            self.prompts.write(
                "This exports identity joint mapping; do not run mismatched arms without a reviewed mapping."
            )
            if not self.prompts.confirm("Have you verified leader/follower joint order, units, and calibration?"):
                msg = "Teleoperation compatibility was not confirmed."
                raise ValueError(msg)
            leader, _ = self.robot(role="leader")
            action_source = Config("physicalai.runtime.TeleopSource", {"leader": leader.to_dict()})
        if follower is None:
            msg = "A complete Runtime needs a selected follower robot."
            raise ValueError(msg)
        return runtime_document(follower, cameras, action_source, fps=fps)
