# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Notebook helpers for ``conveyor_sort_act.ipynb``: talk to the simulation's HTTP API and show a dashboard.

Nothing here is specific to Physical AI Runtime; the notebook keeps the Runtime code (loading the policy,
building and running the ``RobotRuntime``) in its own cells and uses this module for the user interface.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from typing import TYPE_CHECKING

import ipywidgets as widgets

if TYPE_CHECKING:
    import subprocess
    from collections.abc import Callable
    from typing import Any


def read_ready(sim: subprocess.Popen[str], log_path: str) -> dict[str, Any]:
    """Wait for the ``ready`` event of ``physicalai-mujoco start --status-json`` and return it."""
    for line in sim.stdout or ():
        event = json.loads(line)
        if event["event"] == "ready":
            return event
        if event["event"] == "error":
            msg = f"{event['message']} (see {log_path})"
            raise RuntimeError(msg)
    msg = f"the simulation exited before it was ready (see {log_path})"
    raise RuntimeError(msg)


class ConveyorDashboard:
    """The simulation's 3D viewer, its two cameras, the score and the controls, in one widget.

    Args:
        http_url: The simulation's HTTP address (``ready["http_url"]``).
        viewer_url: The simulation's 3D viewer address (``ready["viewer_url"]``).
        run: Runs the policy until ``stop`` is called. It is called in a background thread.
        stop: Asks ``run`` to return.
        shutdown: Stops the policy and the simulation for good.
    """

    def __init__(
        self,
        http_url: str,
        viewer_url: str,
        *,
        run: Callable[[], object],
        stop: Callable[[], object],
        shutdown: Callable[[], object],
    ) -> None:
        self._http_url = http_url
        self._run, self._stop, self._shutdown = run, stop, shutdown
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self._error: str | None = None
        self._done = threading.Event()

        self._viewer = widgets.HTML(f'<iframe src="{viewer_url}" style="width:100%;height:600px;border:1px solid #ccc"></iframe>')
        self._cameras = {name: widgets.Image(format="jpeg", width=360) for name in ("overview", "wrist")}
        self._status = widgets.HTML()
        self._run_button = widgets.Button(description="Run policy", icon="play", button_style="success")
        self._stop_button = widgets.Button(description="Stop", icon="stop", button_style="danger", disabled=True)
        self._reset_button = widgets.Button(description="Reset", icon="refresh", tooltip="Clear the belt and start a new episode")
        self._speed = widgets.FloatSlider(
            value=3.0, min=0.0, max=10.0, step=0.5, description="Belt cm/s", readout_format=".1f", continuous_update=False
        )
        self._shutdown_button = widgets.Button(description="Shut down", icon="power-off")

        self._run_button.on_click(self._on_run)
        self._stop_button.on_click(self._on_stop)
        self._reset_button.on_click(lambda _: self.reset())
        self._speed.observe(lambda change: self.post("/conveyor/belt-speed", {"speed": change["new"] / 100}), names="value")
        self._shutdown_button.on_click(self._on_shutdown)

        side = widgets.VBox(
            [widgets.HTML("<b>overview</b>"), self._cameras["overview"], widgets.HTML("<b>gripper</b>"), self._cameras["wrist"]]
        )
        panels = widgets.GridBox(
            [self._viewer, side],
            layout=widgets.Layout(grid_template_columns="minmax(480px, 1fr) 380px", grid_gap="12px", width="100%"),
        )
        controls = widgets.HBox([self._run_button, self._stop_button, self._reset_button, self._speed, self._shutdown_button])
        self.widget = widgets.VBox([controls, self._status, panels])
        threading.Thread(target=self._refresh, daemon=True).start()

    def _ipython_display_(self) -> None:
        from IPython.display import display  # noqa: PLC0415

        display(self.widget)

    # -- the simulation's HTTP API ----------------------------------------------------------------------

    def get(self, path: str) -> bytes:
        """Return the body of ``GET path``."""
        with urllib.request.urlopen(f"{self._http_url}{path}", timeout=5) as response:  # noqa: S310
            return response.read()

    def post(self, path: str, body: dict[str, Any] | None = None) -> None:
        """Send ``POST path`` with a JSON body."""
        request = urllib.request.Request(  # noqa: S310
            f"{self._http_url}{path}", data=json.dumps(body or {}).encode(), headers={"content-type": "application/json"}
        )
        urllib.request.urlopen(request, timeout=5).close()  # noqa: S310

    def pause_belt(self) -> None:
        """Hold the belt and the item feed."""
        self.post("/episode/auto-reset", {"enabled": False})

    def resume_belt(self) -> None:
        """Run the belt and feed items."""
        self.post("/episode/auto-reset", {"enabled": True})

    def reset(self) -> None:
        """Clear the belt and start a new episode. The belt stays paused until the policy runs."""
        self.pause_belt()
        self.post("/reset")

    # -- buttons ------------------------------------------------------------------------------------------

    @property
    def running(self) -> bool:
        """Whether the policy is running."""
        return self._thread is not None and self._thread.is_alive()

    def _work(self) -> None:
        try:
            self._run()
        except Exception as exc:  # noqa: BLE001 - show any failure in the dashboard
            self._error = repr(exc)

    def _on_run(self, _: widgets.Button) -> None:
        self._error, self._started = None, time.monotonic()
        self._run_button.disabled, self._stop_button.disabled = True, False
        self.resume_belt()  # the belt starts together with the policy, not before
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()

    def _on_stop(self, _: widgets.Button) -> None:
        self._stop_button.disabled = True
        self._stop()
        self.pause_belt()

    def _on_shutdown(self, _: widgets.Button) -> None:
        self._done.set()
        self._shutdown()
        for control in (self._run_button, self._stop_button, self._reset_button, self._shutdown_button, self._speed):
            control.disabled = True
        self._status.value = "<b>simulation stopped</b>"

    # -- live view ----------------------------------------------------------------------------------------

    def _render_status(self) -> str:
        episode = json.loads(self.get("/health"))["episode"]
        score = episode["score"]
        state = f"running {time.monotonic() - self._started:.0f} s" if self.running else "idle"
        if self._error:
            state = f"<span style='color:#b00'>stopped: {self._error}</span>"
        return (
            f"<b>{state}</b> &nbsp;|&nbsp; items fed {episode['spawned']} of {episode['items_per_episode']}"
            f" &nbsp;|&nbsp; <span style='color:#080'>correct {score['correct']}</span>"
            f" &nbsp; <span style='color:#b00'>wrong {score['wrong']}</span>"
            f" &nbsp; <span style='color:#a60'>missed {score['missed']}</span>"
            f" &nbsp;|&nbsp; episodes finished {episode['episode_count']}"
        )

    def _refresh(self) -> None:
        """Stream the cameras at about 10 frames per second and refresh the score twice a second."""
        tick = 0
        while not self._done.is_set():
            try:
                for name, image in self._cameras.items():
                    image.value = self.get(f"/cameras/{name}/frame.jpg")
                if tick % 5 == 0:
                    self._status.value = self._render_status()
                if not self.running and not self._shutdown_button.disabled:
                    self._run_button.disabled, self._stop_button.disabled = False, True
            except OSError:
                pass  # the simulation is not answering (yet, or any more)
            tick += 1
            self._done.wait(0.1)
