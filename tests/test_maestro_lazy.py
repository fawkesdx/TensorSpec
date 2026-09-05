import h5py
import numpy as np
import pytest

from tensorspec.core.io.loaders.maestro.lazy import open_maestro
from tests.test_maestro_defl_x_line_4d import _write_defl_x_h5
from tests.test_maestro_kinds import (
    _write_focus_xy_fine_h5,
    _write_xy_fine_h5,
)


def test_open_maestro_meta_without_loading_all(tmp_path, monkeypatch):
    path = tmp_path / "f5.h5"
    _write_focus_xy_fine_h5(
        path, nx=2, ny=2, n_defl=3, n_e=4, n_a=5
    )

    def fail_eager_read(*args, **kwargs):
        raise AssertionError("lazy open must not load the spectra buffer")

    monkeypatch.setattr(
        "tensorspec.core.io.loaders.maestro.reshape.load_spectra_buffer",
        fail_eager_read,
    )
    desc = open_maestro(str(path))
    try:
        assert desc.kind == "focus_xy_fine_5d"
        assert desc.shape == (2, 2, 3, 4, 5)
        assert desc.labels == ["Y", "X", "Slit Defl.", "Energy", "Angle"]
        block = desc.read_block(0)
        assert block.shape == (4, 5)
        assert block.flags.c_contiguous
    finally:
        desc.close()


def test_open_maestro_defl_x(tmp_path):
    path = tmp_path / "line.h5"
    _write_defl_x_h5(path, n_defl=3, n_x=4, n_e=5, n_a=6)
    desc = open_maestro(str(path))
    try:
        assert desc.kind == "defl_x_line_4d"
        assert desc.shape == (4, 3, 5, 6)
        assert desc.labels == ["X", "Slit Defl.", "Energy", "Angle"]
        assert desc.read_block(slice(1, 4)).shape == (3, 5, 6)
    finally:
        desc.close()


@pytest.mark.parametrize("points_last", [False, True])
def test_read_block_matches_detector_frame_orientation(tmp_path, points_last):
    path = tmp_path / f"xy-{points_last}.h5"
    _write_xy_fine_h5(
        path, points_last=points_last, nx=3, ny=2, n_e=4, n_a=5
    )
    desc = open_maestro(str(path))
    try:
        with h5py.File(path, "r") as f:
            dataset = f["2D_Data/Fixed_Spectra1"]
            expected = (
                dataset[:, :, 2].T
                if points_last
                else dataset[2, :, :].T
            )
        np.testing.assert_array_equal(desc.read_block(2), expected)
    finally:
        desc.close()


def test_open_maestro_partial_xy_shortens_full_cube(tmp_path):
    path = tmp_path / "partial.h5"
    _write_xy_fine_h5(
        path, points_last=False, nx=3, ny=2, n_e=4, n_a=3
    )
    with h5py.File(path, "r+") as f:
        dataset = f["2D_Data/Fixed_Spectra1"]
        raw = dataset[:5]
        attrs = dict(dataset.attrs)
        del f["2D_Data/Fixed_Spectra1"]
        dataset = f["2D_Data"].create_dataset("Fixed_Spectra1", data=raw)
        for name, value in attrs.items():
            dataset.attrs[name] = value

    desc = open_maestro(str(path))
    try:
        assert desc.shape == (1, 3, 4, 3)
        assert len(desc.axes[0]) == 1
        assert desc.metadata["partial_scan"] == {
            "expected": 6,
            "actual": 5,
            "kept_rows": 1,
        }
        with pytest.raises(IndexError):
            desc.read_block(3)
    finally:
        desc.close()


def test_read_block_rejects_noncontiguous_slice(tmp_path):
    path = tmp_path / "line.h5"
    _write_defl_x_h5(path)
    desc = open_maestro(str(path))
    try:
        with pytest.raises(ValueError, match="contiguous"):
            desc.read_block(slice(None, None, 2))
    finally:
        desc.close()
