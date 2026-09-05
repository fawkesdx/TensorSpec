"""Streaming Maestro-to-shard preprocessing orchestration."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Callable

import numpy as np

from tensorspec.core.io.loaders.maestro.lazy import open_maestro
from tensorspec.core.ml.ssl.calibrate import (
    DEG_PER_RAW_PX,
    resample_disp2d,
    resample_fermi3d,
    slit_axis_degrees,
)
from tensorspec.core.ml.ssl.normalize import (
    FileNormStats,
    estimate_file_stats,
    normalize_sample,
)
from tensorspec.core.ml.ssl.resolve import enumerate_modes, role_for_label
from tensorspec.core.ml.ssl.shards import ShardWriter, write_manifest
from tensorspec.core.ml.ssl.spec import PreprocessConfig, to_jsonable
from tensorspec.core.ml.ssl.trim import apply_trim

_HASH_WINDOW_BYTES = 1024 * 1024
_DEFAULT_DETECTOR = ("R4000", "Angular30")


def _edge_hashes(path: Path) -> tuple[str, str]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        head = stream.read(_HASH_WINDOW_BYTES)
        stream.seek(max(0, size - _HASH_WINDOW_BYTES))
        tail = stream.read(_HASH_WINDOW_BYTES)
    return hashlib.sha256(head).hexdigest(), hashlib.sha256(tail).hexdigest()


def _decoded(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _detector_calibration(descriptor, config: PreprocessConfig) -> dict:
    """Read detector scale metadata, retaining safe defaults when absent."""
    metadata = descriptor.metadata.get("detector", {})
    attrs = descriptor._dataset.attrs
    units = [_decoded(value) for value in attrs.get("unitNames", [])]
    offsets = [float(value) for value in attrs.get("scaleOffset", [])]
    deltas = [float(value) for value in attrs.get("scaleDelta", [])]

    angle_index = next(
        (
            index
            for index, unit in enumerate(units)
            if any(token in unit.casefold() for token in ("pixel", "deg", "angle"))
        ),
        None,
    )
    scale_offset = (
        offsets[angle_index]
        if angle_index is not None and angle_index < len(offsets)
        else float(metadata.get("scale_offset", 0.0))
    )
    scale_delta = (
        deltas[angle_index]
        if angle_index is not None and angle_index < len(deltas)
        else float(metadata.get("scale_delta", 1.0))
    )
    angle_unit = units[angle_index] if angle_index is not None else ""
    deg_per_raw_px = config.resample.deg_per_raw_px
    if deg_per_raw_px is None:
        deg_per_raw_px = DEG_PER_RAW_PX[_DEFAULT_DETECTOR]

    return {
        "scale_offset": scale_offset,
        "scale_delta": scale_delta,
        "angle_unit": angle_unit,
        "deg_per_raw_px": float(deg_per_raw_px),
        "detector": _DEFAULT_DETECTOR[0],
        "lens_mode": _DEFAULT_DETECTOR[1],
    }


def _slit_axis(descriptor, calibration: dict) -> np.ndarray:
    slit_dimension = next(
        index
        for index, label in enumerate(descriptor.labels)
        if role_for_label(label) == "slit"
    )
    source_axis = np.asarray(descriptor.axes[slit_dimension])
    if "pixel" not in calibration["angle_unit"].casefold():
        return source_axis
    return slit_axis_degrees(
        source_axis.size,
        calibration["scale_offset"],
        calibration["scale_delta"],
        calibration["deg_per_raw_px"],
    )


def _trimmed_axes(descriptor, config: PreprocessConfig):
    trim = apply_trim(
        descriptor.labels,
        descriptor.axes,
        descriptor.shape,
        config.trim,
    )
    axes_by_role = {
        role_for_label(label): np.asarray(axis)
        for label, axis in zip(descriptor.labels, descriptor.axes)
    }
    return trim, axes_by_role


def _estimate_stats(
    descriptor,
    energy_slice: slice,
    slit_slice: slice,
    config: PreprocessConfig,
) -> FileNormStats:
    n_frames = int(np.prod(descriptor.shape[:-2]))
    count = min(n_frames, max(1, config.norm.subsample_points))
    indices = np.unique(np.linspace(0, n_frames - 1, count, dtype=int))
    frames = np.stack(
        [
            descriptor.read_block(int(index))[energy_slice, slit_slice]
            for index in indices
        ]
    )
    return estimate_file_stats(frames, config.norm)


def _source_record(
    path: Path,
    descriptor,
    source_id: str,
    stats: FileNormStats,
    calibration: dict,
) -> dict:
    sha256_head, sha256_tail = _edge_hashes(path)
    dead_mask = stats.dead_pixel_mask
    dead_count = int(np.count_nonzero(dead_mask)) if dead_mask is not None else 0
    detector_shape = list(descriptor.shape[-2:])
    return {
        "id": source_id,
        "size": path.stat().st_size,
        "sha256_head": sha256_head,
        "sha256_tail": sha256_tail,
        "kind": descriptor.kind,
        "shape": list(descriptor.shape),
        "detector": {
            "shape": detector_shape,
            "model": calibration["detector"],
            "lens_mode": calibration["lens_mode"],
        },
        "dead_pixel": {
            "count": dead_count,
            "fraction": dead_count / int(np.prod(detector_shape)),
        },
        "calibration": calibration,
    }


def _add_disp2d_samples(
    descriptor,
    writer: ShardWriter,
    mode,
    source_id: str,
    stats: FileNormStats,
    config: PreprocessConfig,
    energy_axis: np.ndarray,
    slit_axis: np.ndarray,
    energy_slice: slice,
    slit_slice: slice,
    progress: Callable[[int, int], None] | None,
) -> None:
    index_shape = tuple(
        descriptor.shape[
            next(
                index
                for index, label in enumerate(descriptor.labels)
                if role_for_label(label) == role
            )
        ]
        for role in mode.index_roles
    )
    for flat_index in range(mode.n_samples):
        frame = descriptor.read_block(flat_index)[energy_slice, slit_slice]
        normalized = normalize_sample(frame, stats, config.norm)
        sample = resample_disp2d(
            normalized,
            energy_axis,
            slit_axis,
            config.resample,
        )
        coordinates = np.unravel_index(flat_index, index_shape)
        writer.add(
            sample,
            {
                "source_id": source_id,
                "index": {
                    role: int(value)
                    for role, value in zip(mode.index_roles, coordinates)
                },
            },
        )
        if progress is not None:
            progress(flat_index + 1, mode.n_samples)


def _add_fermi3d_samples(
    descriptor,
    writer: ShardWriter,
    mode,
    source_id: str,
    stats: FileNormStats,
    config: PreprocessConfig,
    axes_by_role: dict,
    energy_axis: np.ndarray,
    slit_axis: np.ndarray,
    energy_slice: slice,
    slit_slice: slice,
    defl_slice: slice,
    progress: Callable[[int, int], None] | None,
) -> None:
    scan_roles = tuple(role_for_label(label) for label in descriptor.labels[:-2])
    scan_shape = descriptor.shape[:-2]
    defl_dimension = scan_roles.index("defl")
    defl_indices = range(*defl_slice.indices(scan_shape[defl_dimension]))
    defl_indices = tuple(defl_indices)
    defl_axis = axes_by_role["defl"][defl_slice]
    spatial_shape = tuple(
        scan_shape[scan_roles.index(role)] for role in mode.index_roles
    )

    for sample_index, spatial_coordinates in enumerate(np.ndindex(spatial_shape)):
        coordinate_by_role = dict(zip(mode.index_roles, spatial_coordinates))
        frames = []
        for defl_index in defl_indices:
            canonical_coordinates = tuple(
                defl_index if role == "defl" else coordinate_by_role[role]
                for role in scan_roles
            )
            flat_index = int(
                np.ravel_multi_index(canonical_coordinates, scan_shape)
            )
            frames.append(
                descriptor.read_block(flat_index)[energy_slice, slit_slice]
            )
        cube = np.stack(frames)
        normalized = normalize_sample(cube, stats, config.norm)
        sample = resample_fermi3d(
            normalized,
            defl_axis,
            energy_axis,
            slit_axis,
            config.resample,
        )
        writer.add(
            sample,
            {
                "source_id": source_id,
                "index": {
                    role: int(value)
                    for role, value in coordinate_by_role.items()
                },
            },
        )
        if progress is not None:
            progress(sample_index + 1, mode.n_samples)


def preprocess_file(
    path: str,
    out_dir: str,
    config: PreprocessConfig,
    *,
    source_id: str | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Stream one Maestro file through trim, normalize, resample, and shards."""
    source_path = Path(path)
    source_name = source_id or source_path.name
    if Path(source_name).is_absolute():
        raise ValueError("source_id must not be an absolute path")

    with open_maestro(str(source_path)) as descriptor:
        modes = {
            mode.name: mode
            for mode in enumerate_modes(descriptor.labels, descriptor.shape)
        }
        if config.sample.mode not in modes:
            raise ValueError(
                f"mode {config.sample.mode!r} unavailable for {descriptor.kind}"
            )
        mode = modes[config.sample.mode]
        effective_config = replace(
            config,
            sample=replace(config.sample, index_roles=mode.index_roles),
        )
        trim, axes_by_role = _trimmed_axes(descriptor, effective_config)
        energy_slice = trim.slices["energy"]
        slit_slice = trim.slices["slit"]
        defl_slice = trim.slices.get("defl", slice(None))
        calibration = _detector_calibration(descriptor, effective_config)
        energy_axis = axes_by_role["energy"][energy_slice]
        slit_axis = _slit_axis(descriptor, calibration)[slit_slice]
        stats = _estimate_stats(
            descriptor,
            energy_slice,
            slit_slice,
            effective_config,
        )

        writer = ShardWriter(out_dir)
        if mode.name == "disp2d":
            _add_disp2d_samples(
                descriptor,
                writer,
                mode,
                source_name,
                stats,
                effective_config,
                energy_axis,
                slit_axis,
                energy_slice,
                slit_slice,
                progress,
            )
        else:
            _add_fermi3d_samples(
                descriptor,
                writer,
                mode,
                source_name,
                stats,
                effective_config,
                axes_by_role,
                energy_axis,
                slit_axis,
                energy_slice,
                slit_slice,
                defl_slice,
                progress,
            )

        partial = writer.close()
        source = _source_record(
            source_path,
            descriptor,
            source_name,
            stats,
            calibration,
        )

    manifest = {
        **partial,
        "preprocess": {
            **to_jsonable(effective_config),
            "trim_warnings": trim.warnings,
        },
        "sources": [source],
    }
    write_manifest(str(Path(out_dir) / "manifest.json"), manifest)
    return manifest


__all__ = ["preprocess_file"]
