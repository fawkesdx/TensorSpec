import h5py
import numpy as np
import pytest

from tensorspec.core.io.loaders.maestro.kinds.fermi_defl_3d import (
    KIND_ID as FERMI_KIND_ID,
    match as fermi_match,
)
from tensorspec.core.io.loaders.maestro.kinds.focus_xy_fine_5d import (
    KIND_ID as FOCUS_KIND_ID,
    match as focus_match,
)
from tensorspec.core.io.loaders.maestro.kinds.xy_fine_4d import KIND_ID, match
from tensorspec.core.io.loaders.maestro.low_level_scan import parse_low_level_scan
from tensorspec.core.io.loaders.maestro.registry import load_with_kind, match_kind
from tensorspec.core.io.loaders.maestro.reshape import points_axis


def _row(tag, value, comment=""):
    return (tag, tag, value, comment)


def _xy_fine_rows(nx=3, ny=2):
    return [
        _row("lwlvnm", "'XY Scan Fine'"),
        _row("scanpar", "F"),
        _row("lwlvlpn", "1"),
        _row("nmsbdv0", "2"),
        _row("nm_0_0", "'Scan X'"),
        _row("un_0_0", "'um'"),
        _row("nm_0_1", "'Scan Y'"),
        _row("un_0_1", "'um'"),
        _row("st_0_0", "0"),
        _row("en_0_0", str(nx - 1)),
        _row("n_0_0", str(nx)),
        _row("st_0_1", "10"),
        _row("en_0_1", str(10 + ny - 1)),
        _row("n_0_1", str(ny)),
    ]


def _write_xy_fine_h5(
    path, *, points_last=True, nx=3, ny=2, n_e=4, n_a=5, num_cycles=None
):
    n_points = nx * ny
    # Fixed Spectra plane order is (angle, energy), not unitNames order.
    if points_last:
        shape = (n_a, n_e, n_points)
    else:
        shape = (n_points, n_a, n_e)

    raw = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    rows = _xy_fine_rows(nx=nx, ny=ny)

    with h5py.File(path, "w") as f:
        f.create_group("0D_Data")
        headers = f.create_group("Headers")
        headers.create_dataset("DAQ_Fixed", data=np.array([]))
        headers.create_dataset("Low_Level_Scan", data=np.array(rows, dtype=object))
        if num_cycles is not None:
            main = headers.create_group("Main")
            main.create_dataset("num_cycles", data=num_cycles)
        g = f.create_group("2D_Data")
        ds = g.create_dataset("Fixed_Spectra1", data=raw)
        ds.attrs["unitNames"] = [b"eV", b"pixels"]
        ds.attrs["scaleOffset"] = [0.0, 0.0]
        ds.attrs["scaleDelta"] = [0.25, 1.0]


def _focus_xy_fine_rows(*, nx=2, ny=2, n_defl=3):
    return [
        _row("lwlvnm", "'Focus XY Fine'"),
        _row("scanpar", "F"),
        _row("lwlvlpn", "2"),
        _row("nmsbdv0", "1"),
        _row("nm_0_0", "'Slit Defl.'"),
        _row("un_0_0", "'Deg'"),
        _row("st_0_0", "-1.0"),
        _row("en_0_0", "1.0"),
        _row("n_0_0", str(n_defl)),
        _row("nmsbdv1", "2"),
        _row("nm_1_0", "'Scan X'"),
        _row("un_1_0", "'um'"),
        _row("nm_1_1", "'Scan Y'"),
        _row("un_1_1", "'um'"),
        _row("st_1_0", "0"),
        _row("en_1_0", str(nx - 1)),
        _row("n_1_0", str(nx)),
        _row("st_1_1", "10"),
        _row("en_1_1", str(10 + ny - 1)),
        _row("n_1_1", str(ny)),
    ]


def _write_focus_xy_fine_h5(
    path,
    *,
    points_last=True,
    nx=2,
    ny=2,
    n_defl=3,
    n_e=4,
    n_a=5,
):
    n_points = nx * ny * n_defl
    if points_last:
        shape = (n_a, n_e, n_points)
    else:
        shape = (n_points, n_a, n_e)

    raw = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    rows = _focus_xy_fine_rows(nx=nx, ny=ny, n_defl=n_defl)

    with h5py.File(path, "w") as f:
        f.create_group("0D_Data")
        headers = f.create_group("Headers")
        headers.create_dataset("DAQ_Fixed", data=np.array([]))
        headers.create_dataset("Low_Level_Scan", data=np.array(rows, dtype=object))
        g = f.create_group("2D_Data")
        ds = g.create_dataset("Fixed_Spectra1", data=raw)
        ds.attrs["unitNames"] = [b"eV", b"pixels"]
        ds.attrs["scaleOffset"] = [0.0, 0.0]
        ds.attrs["scaleDelta"] = [0.25, 1.0]


def test_xy_fine_4d_match():
    plan = parse_low_level_scan(_xy_fine_rows())
    assert match(plan, is_fixed=True) is True
    assert match(plan, is_fixed=False) is False


def test_xy_fine_4d_load_points_last(tmp_path):
    path = tmp_path / "xy_fine.h5"
    nx, ny, n_e, n_a = 3, 2, 4, 5
    _write_xy_fine_h5(path, points_last=True, nx=nx, ny=ny, n_e=n_e, n_a=n_a)

    with h5py.File(path, "r") as f:
        result = load_with_kind(f, str(path))

    assert result["labels"] == ["Y", "X", "Energy", "Angle"]
    assert result["data"].shape == (ny, nx, n_e, n_a)
    assert len(result["axes"]) == 4
    assert result["units"] == ["um", "um", "eV", "deg"]
    np.testing.assert_allclose(result["axes"][0], np.linspace(10, 11, ny))
    np.testing.assert_allclose(result["axes"][1], np.linspace(0, 2, nx))
    assert len(result["axes"][2]) == n_e
    assert len(result["axes"][3]) == n_a
    assert result["is_fixed"] is True
    assert result["facility"] == "MAESTRO"
    assert result["mode"] == "XY Scan Fine"
    assert result["metadata"]["source_path"] == str(path)


def test_xy_fine_4d_load_points_first(tmp_path):
    path = tmp_path / "xy_fine_pf.h5"
    nx, ny, n_e, n_a = 3, 2, 4, 5
    _write_xy_fine_h5(path, points_last=False, nx=nx, ny=ny, n_e=n_e, n_a=n_a)

    with h5py.File(path, "r") as f:
        result = load_with_kind(f, str(path))

    assert result["data"].shape == (ny, nx, n_e, n_a)
    assert result["labels"] == ["Y", "X", "Energy", "Angle"]


def test_xy_fine_4d_aborted_scan_flattens_completed_points(tmp_path):
    path = tmp_path / "xy_fine_aborted.h5"
    nx, ny, actual, n_e, n_a = 3, 2, 5, 4, 3
    _write_xy_fine_h5(path, points_last=False, nx=nx, ny=ny, n_e=n_e, n_a=n_a)
    with h5py.File(path, "r+") as f:
        raw = f["2D_Data"]["Fixed_Spectra1"][:actual]
        del f["2D_Data"]["Fixed_Spectra1"]
        ds = f["2D_Data"].create_dataset("Fixed_Spectra1", data=raw)
        ds.attrs["unitNames"] = [b"eV", b"pixels"]
        ds.attrs["scaleOffset"] = [0.0, 0.0]
        ds.attrs["scaleDelta"] = [0.25, 1.0]

    with h5py.File(path, "r") as f:
        result = load_with_kind(f, str(path))

    assert result["data"].shape == (actual, n_e, n_a)
    assert result["labels"] == ["Point", "Energy", "Angle"]
    assert result["metadata"]["scan_plan"]["expected_cycles"] == nx * ny
    assert result["metadata"]["scan_plan"]["actual_cycles"] == actual
    assert "truncated from 6 to 5" in result["metadata"]["truncate_warning"]


def test_points_axis_rejects_two_truncated_candidates():
    with pytest.raises(ValueError, match="unambiguous points axis"):
        points_axis((5, 4, 5), 6, allow_truncated=True)


def test_header_num_cycles_identifies_aborted_scan_axis(tmp_path):
    path = tmp_path / "xy_fine_header_abort.h5"
    _write_xy_fine_h5(
        path, points_last=False, nx=3, ny=2, n_e=4, n_a=3, num_cycles=5
    )
    with h5py.File(path, "r+") as f:
        raw = f["2D_Data"]["Fixed_Spectra1"][:5]
        del f["2D_Data"]["Fixed_Spectra1"]
        ds = f["2D_Data"].create_dataset("Fixed_Spectra1", data=raw)
        ds.attrs["unitNames"] = [b"eV", b"pixels"]
        ds.attrs["scaleOffset"] = [0.0, 0.0]
        ds.attrs["scaleDelta"] = [0.25, 1.0]

    with h5py.File(path, "r") as f:
        result = load_with_kind(f, str(path))

    assert result["data"].shape == (5, 4, 3)
    assert result["metadata"]["scan_plan"]["expected_cycles"] == 6
    assert result["metadata"]["scan_plan"]["actual_cycles"] == 5
    assert result["metadata"]["scan_plan"]["num_cycles"] == 5
    assert "truncated from 6 to 5" in result["metadata"]["truncate_warning"]


def test_header_num_cycles_mismatch_rejects_unknown_dataset_length(tmp_path):
    path = tmp_path / "xy_fine_bad_cycles.h5"
    _write_xy_fine_h5(
        path, points_last=False, nx=3, ny=2, n_e=4, n_a=3, num_cycles=7
    )
    with h5py.File(path, "r+") as f:
        raw = f["2D_Data"]["Fixed_Spectra1"][:5]
        del f["2D_Data"]["Fixed_Spectra1"]
        ds = f["2D_Data"].create_dataset("Fixed_Spectra1", data=raw)
        ds.attrs["unitNames"] = [b"eV", b"pixels"]
        ds.attrs["scaleOffset"] = [0.0, 0.0]
        ds.attrs["scaleDelta"] = [0.25, 1.0]

    with h5py.File(path, "r") as f, pytest.raises(
        ValueError, match="neither scan-plan cycles 6 nor Headers/Main num_cycles 7"
    ):
        load_with_kind(f, str(path))


def test_match_kind_returns_xy_fine_4d():
    plan = parse_low_level_scan(_xy_fine_rows())
    mod = match_kind(plan, is_fixed=True)
    assert mod is not None
    assert mod.KIND_ID == KIND_ID


def test_no_kind_for_swept():
    plan = parse_low_level_scan(_xy_fine_rows())
    assert match_kind(plan, is_fixed=False) is None


def test_focus_xy_fine_5d_match():
    plan = parse_low_level_scan(_focus_xy_fine_rows())
    assert plan.expected_cycles == 12
    assert focus_match(plan, is_fixed=True) is True
    assert match(plan, is_fixed=True) is False


def test_focus_xy_fine_5d_load_points_last(tmp_path):
    path = tmp_path / "focus_xy_fine.h5"
    nx, ny, n_defl, n_e, n_a = 2, 2, 3, 4, 5
    _write_focus_xy_fine_h5(
        path, points_last=True, nx=nx, ny=ny, n_defl=n_defl, n_e=n_e, n_a=n_a
    )

    with h5py.File(path, "r") as f:
        result = load_with_kind(f, str(path))

    assert result["labels"] == ["Y", "X", "Slit Defl.", "Energy", "Angle"]
    assert result["data"].shape == (ny, nx, n_defl, n_e, n_a)
    assert len(result["axes"]) == 5
    assert result["units"] == ["um", "um", "Deg", "eV", "deg"]
    np.testing.assert_allclose(result["axes"][0], np.linspace(10, 11, ny))
    np.testing.assert_allclose(result["axes"][1], np.linspace(0, 1, nx))
    np.testing.assert_allclose(result["axes"][2], np.linspace(-1.0, 1.0, n_defl))
    assert result["metadata"]["kind"] == FOCUS_KIND_ID


def test_focus_xy_fine_5d_load_points_first(tmp_path):
    path = tmp_path / "focus_xy_fine_pf.h5"
    nx, ny, n_defl, n_e, n_a = 2, 2, 3, 4, 5
    _write_focus_xy_fine_h5(
        path, points_last=False, nx=nx, ny=ny, n_defl=n_defl, n_e=n_e, n_a=n_a
    )

    with h5py.File(path, "r") as f:
        result = load_with_kind(f, str(path))

    assert result["data"].shape == (ny, nx, n_defl, n_e, n_a)
    assert result["labels"] == ["Y", "X", "Slit Defl.", "Energy", "Angle"]


def test_match_kind_returns_focus_xy_fine_5d():
    plan = parse_low_level_scan(_focus_xy_fine_rows())
    mod = match_kind(plan, is_fixed=True)
    assert mod is not None
    assert mod.KIND_ID == FOCUS_KIND_ID


def _fermi_defl_rows(*, n_defl=3):
    return [
        _row("lwlvnm", "'Fermi Map'"),
        _row("scanpar", "F"),
        _row("lwlvlpn", "1"),
        _row("nmsbdv0", "1"),
        _row("nm_0_0", "'Slit Defl.'"),
        _row("un_0_0", "'Deg'"),
        _row("st_0_0", "-1.0"),
        _row("en_0_0", "1.0"),
        _row("n_0_0", str(n_defl)),
    ]


def _write_fermi_defl_h5(
    path,
    *,
    points_last=True,
    n_defl=3,
    n_e=4,
    n_a=5,
):
    n_points = n_defl
    if points_last:
        shape = (n_a, n_e, n_points)
    else:
        shape = (n_points, n_a, n_e)

    raw = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    rows = _fermi_defl_rows(n_defl=n_defl)

    with h5py.File(path, "w") as f:
        f.create_group("0D_Data")
        headers = f.create_group("Headers")
        headers.create_dataset("DAQ_Fixed", data=np.array([]))
        headers.create_dataset("Low_Level_Scan", data=np.array(rows, dtype=object))
        g = f.create_group("2D_Data")
        ds = g.create_dataset("Fixed_Spectra1", data=raw)
        ds.attrs["unitNames"] = [b"eV", b"pixels"]
        ds.attrs["scaleOffset"] = [0.0, 0.0]
        ds.attrs["scaleDelta"] = [0.25, 1.0]


def test_fermi_defl_3d_match():
    plan = parse_low_level_scan(_fermi_defl_rows())
    assert plan.expected_cycles == 3
    assert fermi_match(plan, is_fixed=True) is True
    assert focus_match(plan, is_fixed=True) is False
    assert match(plan, is_fixed=True) is False


def test_fermi_defl_3d_load_points_last(tmp_path):
    path = tmp_path / "fermi_defl.h5"
    n_defl, n_e, n_a = 3, 4, 5
    _write_fermi_defl_h5(path, points_last=True, n_defl=n_defl, n_e=n_e, n_a=n_a)

    with h5py.File(path, "r") as f:
        result = load_with_kind(f, str(path))

    assert result["labels"] == ["Slit Defl.", "Energy", "Angle"]
    assert result["data"].shape == (n_defl, n_e, n_a)
    assert len(result["axes"]) == 3
    assert result["units"] == ["Deg", "eV", "deg"]
    np.testing.assert_allclose(result["axes"][0], np.linspace(-1.0, 1.0, n_defl))
    assert len(result["axes"][1]) == n_e
    assert len(result["axes"][2]) == n_a
    assert result["is_fixed"] is True
    assert result["facility"] == "MAESTRO"
    assert result["mode"] == "Fermi Map"
    assert result["metadata"]["kind"] == FERMI_KIND_ID


def test_fermi_defl_3d_load_points_first(tmp_path):
    path = tmp_path / "fermi_defl_pf.h5"
    n_defl, n_e, n_a = 3, 4, 5
    _write_fermi_defl_h5(path, points_last=False, n_defl=n_defl, n_e=n_e, n_a=n_a)

    with h5py.File(path, "r") as f:
        result = load_with_kind(f, str(path))

    assert result["data"].shape == (n_defl, n_e, n_a)
    assert result["labels"] == ["Slit Defl.", "Energy", "Angle"]


def test_match_kind_returns_fermi_defl_3d():
    plan = parse_low_level_scan(_fermi_defl_rows())
    mod = match_kind(plan, is_fixed=True)
    assert mod is not None
    assert mod.KIND_ID == FERMI_KIND_ID
