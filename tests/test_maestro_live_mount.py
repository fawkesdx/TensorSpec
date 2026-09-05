"""Optional live-mount smoke via ``TENSORSPEC_MAESTRO_SMOKE_DIR``.

Set the env var to a directory containing Maestro H5 files. Tests discover
files by basename glob (for example ``*_00742.h5``) and call
:func:`open_maestro` for plan/kind metadata only — no full tensor load.
Skipped when the env var is unset or no matching file is present.

Synthetic fixtures cover ``00567``-class two-motor line scans; live smoke
only verifies lazy open when a matching file exists on the mount.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tensorspec.core.io.loaders.maestro.lazy import open_maestro

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

_SMOKE_DIR = os.environ.get("TENSORSPEC_MAESTRO_SMOKE_DIR")

# (basename glob, expected kind, ndim, leading labels)
_SMOKE_CASES: list[tuple[str, str, int, list[str]]] = [
    ("*_00742.h5", "focus_xy_fine_5d", 5, ["Y", "X"]),
    ("*_00736.h5", "xy_fine_4d", 4, ["Y", "X"]),
    ("*_00737.h5", "xy_fine_4d", 4, ["Y", "X"]),
    ("*_00562.h5", "xy_fine_4d", 4, ["Y", "X"]),
    ("*_00563.h5", "xy_fine_4d", 4, ["Y", "X"]),
    ("*_00567.h5", "defl_x_line_4d", 4, ["X", "Slit Defl."]),
]

_SKIP_REASON = (
    "Set TENSORSPEC_MAESTRO_SMOKE_DIR to a folder with Maestro H5 files"
)


def _find_smoke_file(pattern: str) -> Path | None:
    if not _SMOKE_DIR:
        return None
    root = Path(_SMOKE_DIR)
    if not root.is_dir():
        return None
    matches = sorted(root.glob(pattern))
    return matches[0] if matches else None


pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("pattern", "expected_kind", "ndim", "label_prefix"),
    _SMOKE_CASES,
    ids=[pat.replace("*", "").replace(".h5", "") for pat, *_ in _SMOKE_CASES],
)
def test_live_open_maestro_smoke(pattern, expected_kind, ndim, label_prefix):
    path = _find_smoke_file(pattern)
    if path is None:
        pytest.skip(_SKIP_REASON)

    desc = open_maestro(str(path))
    try:
        assert desc.kind == expected_kind
        assert len(desc.shape) == ndim
        assert desc.labels[: len(label_prefix)] == label_prefix
        assert desc.shape[-2] > 0 and desc.shape[-1] > 0
        block = desc.read_block(0)
        assert block.ndim == 2
        assert block.shape[0] > 0 and block.shape[1] > 0
    finally:
        desc.close()
