import h5py
import numpy as np
import pytest

from tensorspec.core.io.loaders.maestro.detect import (
    assert_maestro_signature,
    is_fixed_mode,
    select_spectra_dataset,
)
from tensorspec.core.io.loaders.maestro.detector import detector_axes


def test_select_fixed_spectra(tmp_path):
    p = tmp_path / "t.h5"
    with h5py.File(p, "w") as f:
        f.create_group("0D_Data")
        f.create_group("Headers")
        g = f.create_group("2D_Data")
        ds = g.create_dataset("Fixed_Spectra1", data=np.zeros((10, 20, 4), dtype=np.int32))
        ds.attrs["unitNames"] = [b"eV", b"pixels"]
        ds.attrs["scaleOffset"] = [0.0, 0.0]
        ds.attrs["scaleDelta"] = [0.1, 1.0]
    with h5py.File(p, "r") as f:
        assert_maestro_signature(f)
        ds = select_spectra_dataset(f)
        assert ds.name.endswith("Fixed_Spectra1")
        e, a, eu, au = detector_axes(ds, is_fixed=True)
        assert len(e) == 10 and len(a) == 20
        assert eu == "eV"
        np.testing.assert_allclose(e, np.arange(10) * 0.1)
        assert au in {"deg", "Angle", "pixels"}


def test_assert_maestro_signature_missing_group(tmp_path):
    p = tmp_path / "bad.h5"
    with h5py.File(p, "w") as f:
        f.create_group("0D_Data")
        f.create_group("2D_Data")
    with h5py.File(p, "r") as f:
        with pytest.raises(ValueError, match="Headers"):
            assert_maestro_signature(f)


def test_select_prefers_process_000(tmp_path):
    p = tmp_path / "t.h5"
    with h5py.File(p, "w") as f:
        f.create_group("0D_Data")
        f.create_group("Headers")
        g = f.create_group("2D_Data")
        g.create_dataset("Fixed_Spectra1", data=np.zeros((5, 5)))
        g.create_dataset("Process_000", data=np.zeros((8, 8)))
    with h5py.File(p, "r") as f:
        ds = select_spectra_dataset(f)
        assert ds.name.endswith("Process_000")


def test_is_fixed_mode(tmp_path):
    p = tmp_path / "t.h5"
    with h5py.File(p, "w") as f:
        f.create_group("0D_Data")
        f.create_group("2D_Data")
        headers = f.create_group("Headers")
        headers.create_dataset("DAQ_Fixed", data=np.array([]))
    with h5py.File(p, "r") as f:
        assert is_fixed_mode(f) is True

    p2 = tmp_path / "swept.h5"
    with h5py.File(p2, "w") as f:
        f.create_group("0D_Data")
        f.create_group("2D_Data")
        headers = f.create_group("Headers")
        headers.create_dataset("DAQ_Swept", data=np.array([]))
    with h5py.File(p2, "r") as f:
        assert is_fixed_mode(f) is False


def test_detector_axes_fallback_without_attrs(tmp_path):
    p = tmp_path / "t.h5"
    with h5py.File(p, "w") as f:
        g = f.create_group("2D_Data")
        g.create_dataset("Process_000", data=np.zeros((12, 16)))
    with h5py.File(p, "r") as f:
        ds = f["2D_Data"]["Process_000"]
        e, a, eu, au = detector_axes(ds, is_fixed=False)
    assert len(e) == 12 and len(a) == 16
    assert eu == "eV" and au == "deg"
