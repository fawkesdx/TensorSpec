"""Float16 sample shards with JSON provenance manifests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
from typing import Any
import warnings

import numpy as np


_REQUIRED_MANIFEST_KEYS = {
    "preprocess",
    "sources",
    "samples",
    "created_utc",
    "versions",
}
_REQUIRED_SOURCE_KEYS = {
    "id",
    "size",
    "sha256_head",
    "sha256_tail",
    "kind",
    "shape",
    "detector",
    "dead_pixel",
    "calibration",
}
_REQUIRED_SAMPLE_KEYS = {"shard", "offset", "source_id", "index"}


@dataclass(frozen=True)
class Provenance:
    """Location and source coordinates for one stored sample."""

    source_id: str
    index: dict[str, int]
    shard_id: int
    offset: int


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _versions() -> dict[str, str]:
    versions = {
        "numpy": np.__version__,
        "h5py": _package_version("h5py"),
    }
    tensorspec_version = _package_version("tensorspec")
    if tensorspec_version != "unavailable":
        versions["tensorspec"] = tensorspec_version
    return versions


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _validate_manifest(manifest: dict) -> None:
    missing = sorted(_REQUIRED_MANIFEST_KEYS.difference(manifest))
    if missing:
        raise ValueError(f"manifest missing required keys: {', '.join(missing)}")

    if not isinstance(manifest["sources"], list):
        raise ValueError("manifest sources must be a list")
    for position, source in enumerate(manifest["sources"]):
        if not isinstance(source, dict):
            raise ValueError(f"source {position} must be an object")
        missing = sorted(_REQUIRED_SOURCE_KEYS.difference(source))
        if missing:
            raise ValueError(
                f"source {position} missing required keys: {', '.join(missing)}"
            )

    if not isinstance(manifest["samples"], list):
        raise ValueError("manifest samples must be a list")
    for position, sample in enumerate(manifest["samples"]):
        if not isinstance(sample, dict):
            raise ValueError(f"sample {position} must be an object")
        missing = sorted(_REQUIRED_SAMPLE_KEYS.difference(sample))
        if missing:
            raise ValueError(
                f"sample {position} missing required keys: {', '.join(missing)}"
            )

    versions = manifest["versions"]
    if not isinstance(versions, dict) or not {"numpy", "h5py"} <= versions.keys():
        raise ValueError("manifest versions requires numpy and h5py")


class ShardWriter:
    """Buffer normalized samples and write size-bounded ``.npy`` shards."""

    def __init__(self, out_dir: str, target_bytes: int = 512 * 1024 * 1024):
        if target_bytes <= 0:
            raise ValueError("target_bytes must be positive")

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.target_bytes = int(target_bytes)
        self._buffer: list[np.ndarray] = []
        self._buffer_bytes = 0
        self._shard_id = 0
        self._sample_shape: tuple[int, ...] | None = None
        self._samples: list[dict[str, Any]] = []
        self._total_samples = 0
        self._dropped_samples = 0
        self._closed = False

    def add(self, sample: np.ndarray, provenance: dict) -> None:
        """Append one valid sample, recording its source and index."""
        if self._closed:
            raise RuntimeError("cannot add samples after writer is closed")
        if "source_id" not in provenance or "index" not in provenance:
            raise ValueError("provenance requires source_id and index")

        self._total_samples += 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            stored = np.asarray(sample, dtype=np.float16)
        if not np.all(np.isfinite(stored)) or not np.any(stored):
            self._dropped_samples += 1
            return

        if self._sample_shape is None:
            self._sample_shape = stored.shape
        elif stored.shape != self._sample_shape:
            raise ValueError(
                f"sample shape {stored.shape} does not match {self._sample_shape}"
            )

        if self._buffer and self._buffer_bytes + stored.nbytes > self.target_bytes:
            self._flush()

        offset = len(self._buffer)
        self._buffer.append(np.ascontiguousarray(stored))
        self._buffer_bytes += stored.nbytes
        self._samples.append(
            {
                "shard": self._shard_id,
                "offset": offset,
                "source_id": str(provenance["source_id"]),
                "index": dict(provenance["index"]),
            }
        )

    def _flush(self) -> None:
        if not self._buffer:
            return
        shard = np.stack(self._buffer)
        np.save(self.out_dir / f"shard_{self._shard_id:05d}.npy", shard)
        self._shard_id += 1
        self._buffer.clear()
        self._buffer_bytes = 0

    def close(self) -> dict:
        """Finish writing and return manifest fields describing the samples."""
        if self._closed:
            raise RuntimeError("writer is already closed")
        self._closed = True

        drop_rate = (
            self._dropped_samples / self._total_samples
            if self._total_samples
            else 0.0
        )
        if drop_rate > 0.01:
            raise ValueError(
                f"sample drop rate {drop_rate:.2%} exceeds the 1% limit "
                f"({self._dropped_samples}/{self._total_samples})"
            )

        self._flush()
        return {
            "samples": self._samples,
            "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "versions": _versions(),
            "total_samples": self._total_samples,
            "dropped_samples": self._dropped_samples,
        }


def write_manifest(path: str, manifest: dict) -> None:
    """Validate and atomically write a dataset manifest."""
    _validate_manifest(manifest)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def load_manifest(path: str) -> dict:
    """Load and validate a dataset manifest."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_manifest(manifest)
    return manifest


class ShardDataset:
    """Torch-free indexable dataset backed by read-only numpy memmaps."""

    def __init__(self, out_dir: str):
        self.out_dir = Path(out_dir)
        self.manifest = load_manifest(self.out_dir / "manifest.json")
        self.samples = self.manifest["samples"]
        self._shards: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[np.ndarray, dict]:
        if i < 0:
            i += len(self)
        if i < 0 or i >= len(self):
            raise IndexError(i)

        provenance = self.samples[i]
        shard_id = int(provenance["shard"])
        if shard_id not in self._shards:
            path = self.out_dir / f"shard_{shard_id:05d}.npy"
            self._shards[shard_id] = np.load(path, mmap_mode="r", allow_pickle=False)
        sample = np.asarray(self._shards[shard_id][int(provenance["offset"])])
        return sample, dict(provenance)
