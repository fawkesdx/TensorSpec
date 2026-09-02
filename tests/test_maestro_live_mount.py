"""Optional integration tests against live MAESTRO mount data.

Set env ``TENSORESPEC_MAESTRO_LIVE_DIR`` to a directory that contains the
sample H5 filenames below. Skipped when unset or files missing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tensorspec.core.io.arpes_loader import ARPESLoader

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

_LIVE_DIR = Path(os.environ["TENSORESPEC_MAESTRO_LIVE_DIR"]) if os.environ.get("TENSORESPEC_MAESTRO_LIVE_DIR") else None
_XY_FINE_4D = (_LIVE_DIR / "20260629_00736.h5") if _LIVE_DIR else None
_FOCUS_XY_FINE_5D = (_LIVE_DIR / "20260630_00742.h5") if _LIVE_DIR else None

_mount_available = bool(
    _LIVE_DIR
    and _LIVE_DIR.is_dir()
    and _XY_FINE_4D is not None
    and _FOCUS_XY_FINE_5D is not None
    and _XY_FINE_4D.is_file()
    and _FOCUS_XY_FINE_5D.is_file()
)
_skip_reason = "Set TENSORESPEC_MAESTRO_LIVE_DIR to a folder with sample Maestro H5 files"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _mount_available, reason=_skip_reason),
]


def test_live_xy_fine_4d():
    tensor = ARPESLoader.load(_XY_FINE_4D)

    assert tensor.ndim == 4
    assert tensor.labels[:2] == ["Y", "X"]
    assert tensor.value.shape[0] > 0 and tensor.value.shape[1] > 0


def test_live_focus_xy_fine_5d():
    tensor = ARPESLoader.load(_FOCUS_XY_FINE_5D)

    assert tensor.ndim == 5
    spatial_product = int(
        tensor.value.shape[0] * tensor.value.shape[1] * tensor.value.shape[2]
    )
    assert spatial_product == 17 * 81 * 81
    assert tensor.labels[:2] == ["Y", "X"]
