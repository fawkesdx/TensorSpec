import h5py
import numpy as np

from tensorspec.core.io.loaders.maestro.registry import load_with_kind


def _row(tag, value, comment=""):
    return (tag, tag, value, comment)


def _write_partial_xy(path, *, nx=5, ny=4, actual_points=17, n_e=3, n_a=4):
    """expected=20, actual=17 -> complete rows = 17 // 5 = 3 -> keep 15 points."""
    expected = nx * ny
    assert actual_points < expected
    shape = (n_a, n_e, actual_points)
    raw = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    rows = [
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
        _row("st_0_1", "0"),
        _row("en_0_1", str(ny - 1)),
        _row("n_0_1", str(ny)),
    ]
    with h5py.File(path, "w") as f:
        f.create_group("0D_Data")
        headers = f.create_group("Headers")
        headers.create_dataset("DAQ_Fixed", data=np.array([]))
        headers.create_dataset("Low_Level_Scan", data=np.array(rows, dtype=object))
        main = headers.create_group("Main")
        main.create_dataset("num_cycles", data=actual_points)
        g = f.create_group("2D_Data")
        ds = g.create_dataset("Fixed_Spectra1", data=raw)
        ds.attrs["unitNames"] = [b"eV", b"pixels"]
        ds.attrs["scaleOffset"] = [0.0, 0.0]
        ds.attrs["scaleDelta"] = [0.1, 1.0]


def test_partial_xy_recovers_complete_rows(tmp_path):
    path = tmp_path / "partial.h5"
    _write_partial_xy(path, nx=5, ny=4, actual_points=17, n_e=3, n_a=4)
    with h5py.File(path, "r") as f:
        out = load_with_kind(f, str(path))
    # 3 complete Y rows × 5 X = 15 points as (Y=3, X=5, E, A)
    assert out["data"].shape == (3, 5, 3, 4)
    assert out["labels"] == ["Y", "X", "Energy", "Angle"]
    assert out["metadata"]["partial_scan"]["kept_rows"] == 3
    assert len(out["axes"][0]) == 3  # Y axis shortened
