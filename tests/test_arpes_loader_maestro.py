import h5py
import numpy as np
import pytest

from tensorspec.core.io.arpes_loader import ARPESLoader, MaestroLoader


def _row(tag, value, comment=""):
    return (tag, tag, value, comment)


def _write_xy_fine_h5(path, *, nx=3, ny=2, n_e=4, n_a=5):
    n_points = nx * ny
    shape = (n_e, n_a, n_points)
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
        _row("st_0_1", "10"),
        _row("en_0_1", str(10 + ny - 1)),
        _row("n_0_1", str(ny)),
    ]

    with h5py.File(path, "w") as f:
        f.create_group("0D_Data")
        headers = f.create_group("Headers")
        headers.create_dataset("DAQ_Fixed", data=np.array([]))
        headers.create_dataset("Low_Level_Scan", data=np.array(rows, dtype=object))
        ds = f.create_group("2D_Data").create_dataset("Fixed_Spectra1", data=raw)
        ds.attrs["unitNames"] = [b"eV", b"pixels"]
        ds.attrs["scaleOffset"] = [0.0, 0.0]
        ds.attrs["scaleDelta"] = [0.25, 1.0]


def test_arpes_loader_xy_fine_fixed(tmp_path):
    path = tmp_path / "xy_fine.h5"
    _write_xy_fine_h5(path)

    tensor = ARPESLoader.load(path)

    assert tensor.ndim == 4
    assert tensor.labels == ["Y", "X", "Energy", "Angle"]
    assert tensor.units == ["um", "um", "eV", "deg"]
    assert tensor.data_type == "XY Scan Fine"
    assert tensor.metadata["facility"] == "MAESTRO"
    assert tensor.metadata["is_fixed"] is True


def test_arpes_loader_legacy_axes_dict(tmp_path):
    path = tmp_path / "mock_legacy.h5"
    raw = np.arange(12, dtype=np.float32).reshape(3, 4)
    with h5py.File(path, "w") as f:
        headers = f.create_group("Headers")
        headers.create_dataset(
            "DAQ_Fixed",
            data=np.array([(b"", b"Mode", b"Fixed", b"")], dtype="S32"),
        )
        f.create_group("0D_Data")
        ds = f.create_group("2D_Data").create_dataset("Process_000", data=raw)
        ds.attrs["unitNames"] = [b"deg", b"eV"]
        ds.attrs["scaleOffset"] = [-1.0, 10.0]
        ds.attrs["scaleDelta"] = [1.0, 0.5]

    tensor = ARPESLoader.load(path)

    assert tensor.ndim == 2
    assert "Energy" in tensor.labels[0] or "Energy" in tensor.labels[1]
    assert tensor.metadata.get("is_fixed") is True


def test_arpes_loader_reraises_maestro_reshape_error(tmp_path, monkeypatch):
    path = tmp_path / "broken_maestro.h5"
    path.touch()
    error = ValueError("xy_fine_4d: reshape got (5, 4, 3), expected (2, 3, 4, 3).")

    monkeypatch.setattr(MaestroLoader, "load", lambda self: (_ for _ in ()).throw(error))

    with pytest.raises(ValueError, match="xy_fine_4d: reshape got"):
        ARPESLoader.load(path)
