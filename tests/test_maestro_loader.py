import os

import h5py
import numpy as np
import pytest

from tensorspec.core.io.loaders.maestro import MaestroLoader
from tensorspec.core.io.loaders.maestro.kinds import process000_generic


def _mode_rows(mode: str):
    return np.array(
        [(b"", b"Mode", mode.encode(), b"")],
        dtype="S32",
    )


def _write_process000(path, *, fixed=False, motors=True):
    raw = np.arange(6 * 3 * 4, dtype=np.float32).reshape(6, 3, 4)
    with h5py.File(path, "w") as f:
        zero_d = f.create_group("0D_Data")
        if motors:
            zero_d.create_dataset("Scan X", data=[0, 0, 1, 1, 2, 2])
            zero_d.create_dataset("Scan Y", data=[10, 11, 10, 11, 10, 11])
        headers = f.create_group("Headers")
        header_name = "DAQ_Fixed" if fixed else "DAQ_Swept"
        headers.create_dataset(header_name, data=_mode_rows("Generic Map"))
        dataset = f.create_group("2D_Data").create_dataset("Process_000", data=raw)
        dataset.attrs["unitNames"] = [b"eV", b"deg"]
        dataset.attrs["scaleOffset"] = [1.0, -2.0]
        dataset.attrs["scaleDelta"] = [0.5, 1.0]


def test_process000_generic_reshapes_unique_motor_grid(tmp_path):
    path = tmp_path / "generic.h5"
    _write_process000(path)

    with h5py.File(path, "r") as f:
        result = process000_generic.load(f, f["2D_Data"]["Process_000"], path=str(path))

    assert result["data"].shape == (3, 2, 3, 4)
    assert result["labels"] == ["Scan X", "Scan Y", "Energy", "Angle"]
    assert result["units"] == ["a.u.", "a.u.", "eV", "deg"]
    np.testing.assert_allclose(result["axes"][0], [0, 1, 2])
    np.testing.assert_allclose(result["axes"][1], [10, 11])
    assert result["metadata"]["kind"] == "process000_generic"


def test_process000_generic_transposes_fixed_detector(tmp_path):
    path = tmp_path / "fixed_2d.h5"
    with h5py.File(path, "w") as f:
        f.create_group("0D_Data")
        headers = f.create_group("Headers")
        headers.create_dataset("DAQ_Fixed", data=_mode_rows("Fixed"))
        dataset = f.create_group("2D_Data").create_dataset(
            "Process_000", data=np.arange(12).reshape(3, 4)
        )
        dataset.attrs["unitNames"] = [b"deg", b"eV"]
        dataset.attrs["scaleOffset"] = [-1.0, 10.0]
        dataset.attrs["scaleDelta"] = [1.0, 0.5]

    with h5py.File(path, "r") as f:
        result = process000_generic.load(f, f["2D_Data"]["Process_000"], path=str(path))

    assert result["data"].shape == (4, 3)
    assert result["labels"] == ["Energy", "Angle"]
    assert result["units"] == ["eV", "deg"]


def test_maestro_loader_process000_end_to_end(tmp_path):
    path = tmp_path / "generic.h5"
    _write_process000(path)

    result = MaestroLoader(path).load()

    assert os.environ["HDF5_USE_FILE_LOCKING"] == "FALSE"
    assert result["data"].shape == (3, 2, 3, 4)
    assert result["facility"] == "MAESTRO"


def test_maestro_loader_fixed_plan_miss_reports_cycles(tmp_path):
    path = tmp_path / "unknown_fixed.h5"
    rows = [
        ("lwlvnm", "lwlvnm", "'Unknown Fixed'", ""),
        ("scanpar", "scanpar", "F", ""),
        ("lwlvlpn", "lwlvlpn", "1", ""),
        ("nmsbdv0", "nmsbdv0", "1", ""),
        ("nm_0_0", "nm_0_0", "'Temperature'", ""),
        ("un_0_0", "un_0_0", "'K'", ""),
        ("st_0_0", "st_0_0", "1", ""),
        ("en_0_0", "en_0_0", "2", ""),
        ("n_0_0", "n_0_0", "2", ""),
    ]
    with h5py.File(path, "w") as f:
        f.create_group("0D_Data")
        headers = f.create_group("Headers")
        headers.create_dataset("DAQ_Fixed", data=np.array([]))
        headers.create_dataset("Low_Level_Scan", data=np.array(rows, dtype=object))
        dataset = f.create_group("2D_Data").create_dataset(
            "Process_000", data=np.zeros((2, 3, 4), dtype=np.float32)
        )
        dataset.attrs["unitNames"] = [b"eV", b"deg"]
        dataset.attrs["scaleOffset"] = [0.0, 0.0]
        dataset.attrs["scaleDelta"] = [1.0, 1.0]

    with pytest.raises(
        ValueError,
        match=r"expected_cycles=2.*actual_cycles=2",
    ):
        MaestroLoader(path).load()


def test_maestro_loader_swept_plan_miss_uses_process000(tmp_path):
    path = tmp_path / "unknown_swept.h5"
    rows = [
        ("lwlvnm", "lwlvnm", "'Unknown Swept'", ""),
        ("scanpar", "scanpar", "F", ""),
        ("lwlvlpn", "lwlvlpn", "1", ""),
        ("nmsbdv0", "nmsbdv0", "1", ""),
        ("nm_0_0", "nm_0_0", "'Temperature'", ""),
        ("un_0_0", "un_0_0", "'K'", ""),
        ("st_0_0", "st_0_0", "1", ""),
        ("en_0_0", "en_0_0", "2", ""),
        ("n_0_0", "n_0_0", "2", ""),
    ]
    with h5py.File(path, "w") as f:
        f.create_group("0D_Data").create_dataset("Temperature", data=[1.0, 2.0])
        headers = f.create_group("Headers")
        headers.create_dataset("DAQ_Swept", data=_mode_rows("Unknown Swept"))
        headers.create_dataset("Low_Level_Scan", data=np.array(rows, dtype=object))
        dataset = f.create_group("2D_Data").create_dataset(
            "Process_000", data=np.zeros((2, 3, 4), dtype=np.float32)
        )
        dataset.attrs["unitNames"] = [b"eV", b"deg"]
        dataset.attrs["scaleOffset"] = [0.0, 0.0]
        dataset.attrs["scaleDelta"] = [1.0, 1.0]

    result = MaestroLoader(path).load()

    assert result["metadata"]["kind"] == "process000_generic"
    assert result["data"].shape == (2, 3, 4)
