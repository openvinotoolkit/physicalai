from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("package", "urdf"),
    [
        ("plugin/src/physicalai_rebot_b601_plugin", "plugin/urdf"),  # source checkout / editable install
        ("site-packages/physicalai_rebot_b601_plugin", "site-packages/urdf"),  # wheel force-include
    ],
)
def test_bundled_urdfs_are_found_in_source_and_wheel_layouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    package: str,
    urdf: str,
) -> None:
    from physicalai_rebot_b601_plugin import _urdf

    (tmp_path / package).mkdir(parents=True)
    (tmp_path / urdf).mkdir(parents=True)
    monkeypatch.setattr(_urdf.ir, "files", lambda _name: tmp_path / package)

    assert _urdf.get_urdf_path() == tmp_path / urdf
