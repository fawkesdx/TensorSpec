"""End-to-end tests for streaming SSL preprocessing and its CLI."""

import json

import numpy as np

from tensorspec.core.io.loaders.maestro.lazy import open_maestro
from tensorspec.core.ml.ssl.calibrate import (
    DEG_PER_RAW_PX,
    resample_disp2d,
    slit_axis_degrees,
)
from tensorspec.core.ml.ssl.cli import main
from tensorspec.core.ml.ssl.preprocess import _estimate_stats, preprocess_file
from tensorspec.core.ml.ssl.shards import ShardDataset
from tensorspec.core.ml.ssl.spec import (
    NormSpec,
    PreprocessConfig,
    ResampleSpec,
    SampleSpec,
    TrimSpec,
    to_jsonable,
)
from tests.test_maestro_defl_x_line_4d import _write_defl_x_h5
from tests.test_maestro_kinds import _write_focus_xy_fine_h5


def _config(mode: str) -> PreprocessConfig:
    index_roles = ("y", "x", "defl") if mode == "disp2d" else ("y", "x")
    return PreprocessConfig(
        trim=TrimSpec(ranges={}, source_kind="focus_xy_fine_5d"),
        norm=NormSpec(subsample_points=4, dead_pixel_sigma=0.0),
        resample=ResampleSpec(energy_size=16, slit_size=16, defl_size=4),
        sample=SampleSpec(mode=mode, index_roles=index_roles),
    )


def test_preprocess_disp2d_end_to_end(tmp_path):
    src = tmp_path / "f5.h5"
    _write_focus_xy_fine_h5(
        src, nx=2, ny=2, n_defl=3, n_e=8, n_a=8
    )
    out = tmp_path / "ds"
    progress = []

    manifest = preprocess_file(
        str(src),
        str(out),
        _config("disp2d"),
        progress=lambda current, total: progress.append((current, total)),
    )

    dataset = ShardDataset(str(out))
    assert len(dataset) == 2 * 2 * 3
    sample, metadata = dataset[0]
    assert sample.shape == (16, 16)
    assert sample.dtype == np.float16
    assert metadata["index"] == {"y": 0, "x": 0, "defl": 0}
    assert dataset[-1][1]["index"] == {"y": 1, "x": 1, "defl": 2}
    assert progress[-1] == (12, 12)

    source = manifest["sources"][0]
    assert source["id"] == "f5.h5"
    assert "path" not in source
    assert source["kind"] == "focus_xy_fine_5d"
    assert source["shape"] == [2, 2, 3, 8, 8]
    assert source["size"] == src.stat().st_size
    assert len(source["sha256_head"]) == 64
    assert len(source["sha256_tail"]) == 64
    assert source["calibration"]["deg_per_raw_px"] > 0
    assert all("id" not in entry for entry in manifest["samples"])


def test_preprocess_fermi3d_stacks_each_spatial_point(tmp_path):
    src = tmp_path / "f5.h5"
    _write_focus_xy_fine_h5(
        src, nx=2, ny=2, n_defl=3, n_e=8, n_a=8
    )
    out = tmp_path / "ds"

    preprocess_file(str(src), str(out), _config("fermi3d"))

    dataset = ShardDataset(str(out))
    assert len(dataset) == 2 * 2
    sample, metadata = dataset[0]
    assert sample.shape == (4, 16, 16)
    assert sample.dtype == np.float16
    assert metadata["index"] == {"y": 0, "x": 0}
    assert dataset[-1][1]["index"] == {"y": 1, "x": 1}


def test_slit_trim_uses_same_calibrated_degree_axis_as_resampling(tmp_path):
    src = tmp_path / "f5.h5"
    _write_focus_xy_fine_h5(
        src, nx=1, ny=1, n_defl=2, n_e=8, n_a=9
    )
    out = tmp_path / "ds"
    config = _config("disp2d")
    config.trim.ranges["slit"] = (-0.06, 0.06)
    config.norm.clip_percentiles = (0.0, 100.0)
    config.resample.energy_size = 8
    config.resample.slit_size = 7

    preprocess_file(str(src), str(out), config)

    with open_maestro(str(src)) as descriptor:
        frame = descriptor.read_block(0)
        full_slit = slit_axis_degrees(
            9,
            scale_offset=0.0,
            scale_delta=1.0,
            deg_per_raw_px=DEG_PER_RAW_PX[("R4000", "Angular30")],
        )
        keep = np.flatnonzero((full_slit >= -0.06) & (full_slit <= 0.06))
        slit_slice = slice(int(keep[0]), int(keep[-1]) + 1)
        trimmed = frame[:, slit_slice].astype(np.float64)
        normalized = (trimmed - trimmed.min()) / (
            trimmed.max() - trimmed.min()
        )
        expected = resample_disp2d(
            normalized,
            descriptor.axes[-2],
            full_slit[slit_slice],
            config.resample,
        ).astype(np.float16)

    actual, _ = ShardDataset(str(out))[0]
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


def test_norm_subsample_is_seeded_and_respects_scan_axis_trims():
    class RecordingDescriptor:
        labels = ["Y", "X", "Slit Defl.", "Energy", "Angle"]
        shape = (2, 2, 3, 2, 2)

        def __init__(self):
            self.indices = []

        def read_block(self, index):
            self.indices.append(index)
            return np.full((2, 2), index + 1, dtype=np.float32)

    descriptor = RecordingDescriptor()
    config = _config("fermi3d")
    config.seed = 17
    config.norm.subsample_points = 3
    scan_slices = {
        "y": slice(None),
        "x": slice(1, 2),
        "defl": slice(1, 3),
        "energy": slice(None),
        "slit": slice(None),
    }
    eligible = np.array([4, 5, 10, 11])
    expected = np.random.default_rng(17).choice(
        eligible, size=3, replace=False
    )

    _estimate_stats(descriptor, scan_slices, config)

    assert descriptor.indices == expected.tolist()
    for index in descriptor.indices:
        y, x, defl = np.unravel_index(index, descriptor.shape[:-2])
        assert y in (0, 1)
        assert x == 1
        assert defl in (1, 2)


def test_cli_preprocess_loads_json_config_and_overrides_mode(tmp_path):
    src = tmp_path / "line.h5"
    _write_defl_x_h5(src, n_x=2, n_defl=3, n_e=8, n_a=8)
    out = tmp_path / "ds"
    config = _config("fermi3d")
    config.trim.source_kind = "defl_x_line_4d"
    config_path = tmp_path / "preprocess.json"
    config_path.write_text(json.dumps(to_jsonable(config)), encoding="utf-8")

    result = main(
        [
            "preprocess",
            "--input",
            str(src),
            "--out",
            str(out),
            "--mode",
            "disp2d",
            "--config",
            str(config_path),
        ]
    )

    assert result == 0
    dataset = ShardDataset(str(out))
    assert len(dataset) == 2 * 3
    assert dataset[0][0].shape == (16, 16)
    assert dataset.manifest["preprocess"]["sample"] == {
        "mode": "disp2d",
        "index_roles": ["x", "defl"],
    }
