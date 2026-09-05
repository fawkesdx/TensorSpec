import h5py
import numpy as np
import pytest

from tensorspec.core.io.loaders.maestro.kinds import defl_x_line_4d, fermi_defl_3d
from tensorspec.core.io.loaders.maestro.low_level_scan import parse_low_level_scan
from tensorspec.core.io.loaders.maestro.registry import match_kind, load_with_kind


def _row(tag, value, comment=""):
    return (tag, tag, value, comment)


def _defl_x_rows(*, n_defl=3, n_x=4):
    return [
        _row("lwlvnm", "'Two Motor'"),
        _row("scanpar", "F"),
        _row("lwlvlpn", "1"),
        _row("nmsbdv0", "2"),
        _row("nm_0_0", "'Slit Defl.'"),
        _row("un_0_0", "'Deg'"),
        _row("st_0_0", "-1.0"),
        _row("en_0_0", "1.0"),
        _row("n_0_0", str(n_defl)),
        _row("nm_0_1", "'Scan X'"),
        _row("un_0_1", "'um'"),
        _row("st_0_1", "0"),
        _row("en_0_1", str(n_x - 1)),
        _row("n_0_1", str(n_x)),
    ]


def _write_defl_x_h5(path, *, n_defl=3, n_x=4, n_e=5, n_a=6):
    n_points = n_defl * n_x
    # Fixed Spectra plane order is (angle, energy)
    shape = (n_a, n_e, n_points)
    raw = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    with h5py.File(path, "w") as f:
        f.create_group("0D_Data")
        headers = f.create_group("Headers")
        headers.create_dataset("DAQ_Fixed", data=np.array([]))
        headers.create_dataset(
            "Low_Level_Scan", data=np.array(_defl_x_rows(n_defl=n_defl, n_x=n_x), dtype=object)
        )
        g = f.create_group("2D_Data")
        ds = g.create_dataset("Fixed_Spectra1", data=raw)
        ds.attrs["unitNames"] = [b"eV", b"pixels"]
        ds.attrs["scaleOffset"] = [0.0, 0.0]
        ds.attrs["scaleDelta"] = [0.1, 1.0]


def test_two_motor_matches_defl_x_not_fermi(tmp_path):
    path = tmp_path / "line.h5"
    _write_defl_x_h5(path)
    with h5py.File(path, "r") as f:
        plan = parse_low_level_scan(f["Headers"]["Low_Level_Scan"][()])
    assert defl_x_line_4d.match(plan, True) is True
    assert fermi_defl_3d.match(plan, True) is False
    assert match_kind(plan, True) is defl_x_line_4d


def test_defl_x_load_shape_and_labels(tmp_path):
    path = tmp_path / "line.h5"
    _write_defl_x_h5(path, n_defl=3, n_x=4, n_e=5, n_a=6)
    with h5py.File(path, "r") as f:
        out = load_with_kind(f, str(path))
    assert out["metadata"]["kind"] == "defl_x_line_4d"
    assert out["labels"] == ["X", "Slit Defl.", "Energy", "Angle"]
    assert out["data"].shape == (4, 3, 5, 6)
