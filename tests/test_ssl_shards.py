"""Tests for float16 SSL shards and their JSON manifest."""

import json

import numpy as np
import pytest

from tensorspec.core.ml.ssl.shards import (
    ShardDataset,
    ShardWriter,
    load_manifest,
    write_manifest,
)


def test_shard_roundtrip(tmp_path):
    """Catches broken shard rollover, float16 storage, or provenance lookup."""
    writer = ShardWriter(str(tmp_path), target_bytes=1024)
    for i in range(10):
        writer.add(
            np.ones((8, 8), dtype=np.float16) * (i + 1),
            {"source_id": "s", "index": {"x": i}},
        )

    partial = writer.close()
    write_manifest(
        tmp_path / "manifest.json",
        {**partial, "preprocess": {"sample": {"mode": "disp2d"}}, "sources": []},
    )

    dataset = ShardDataset(str(tmp_path))
    assert len(dataset) == 10
    sample, metadata = dataset[3]
    assert sample.shape == (8, 8)
    assert sample.dtype == np.float16
    np.testing.assert_array_equal(sample, np.full((8, 8), 4, dtype=np.float16))
    assert metadata["index"]["x"] == 3
    assert len(list(tmp_path.glob("shard_*.npy"))) == 2


def test_manifest_roundtrip_has_required_metadata(tmp_path):
    """Catches manifests that omit reproducibility keys or cannot reload."""
    writer = ShardWriter(str(tmp_path))
    writer.add(np.ones((2, 2)), {"source_id": "source", "index": {"y": 1}})
    partial = writer.close()
    manifest = {
        **partial,
        "preprocess": {"norm": {"scope": "per_sample"}},
        "sources": [],
    }

    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)

    loaded = load_manifest(path)
    assert loaded == json.loads(path.read_text(encoding="utf-8"))
    assert set(("preprocess", "sources", "samples", "created_utc", "versions")) <= set(loaded)
    assert set(("numpy", "h5py")) <= set(loaded["versions"])


def test_writer_drops_zero_and_nonfinite_samples(tmp_path):
    """Catches invalid normalized samples leaking into shard arrays."""
    writer = ShardWriter(str(tmp_path))
    for i in range(198):
        writer.add(np.ones((2, 2)), {"source_id": "s", "index": {"x": i}})
    writer.add(np.zeros((2, 2)), {"source_id": "s", "index": {"x": 198}})
    writer.add(
        np.array([[1.0, np.nan], [2.0, 3.0]]),
        {"source_id": "s", "index": {"x": 199}},
    )

    partial = writer.close()

    assert len(partial["samples"]) == 198
    assert partial["dropped_samples"] == 2


def test_writer_raises_when_drop_rate_exceeds_one_percent(tmp_path):
    """Catches silently accepting datasets with excessive invalid samples."""
    writer = ShardWriter(str(tmp_path))
    for i in range(98):
        writer.add(np.ones((2, 2)), {"source_id": "s", "index": {"x": i}})
    writer.add(np.zeros((2, 2)), {"source_id": "s", "index": {"x": 98}})
    writer.add(np.zeros((2, 2)), {"source_id": "s", "index": {"x": 99}})

    with pytest.raises(ValueError, match="drop rate"):
        writer.close()


def test_write_manifest_rejects_missing_required_keys(tmp_path):
    """Catches incomplete manifests being persisted as valid datasets."""
    with pytest.raises(ValueError, match="missing required keys"):
        write_manifest(tmp_path / "manifest.json", {"samples": []})


def test_write_manifest_rejects_incomplete_source_records(tmp_path):
    """Catches source records missing fields needed for reproducibility."""
    manifest = {
        "preprocess": {},
        "sources": [{"id": "source-only"}],
        "samples": [],
        "created_utc": "2026-09-04T00:00:00Z",
        "versions": {"numpy": "2", "h5py": "3"},
    }

    with pytest.raises(ValueError, match="source 0 missing required keys"):
        write_manifest(tmp_path / "manifest.json", manifest)
