from __future__ import annotations

import h5py
import numpy as np

_PIXEL_ANGLE_STEP = 0.048


def detector_axes(
    dataset: h5py.Dataset,
    is_fixed: bool,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    shape = dataset.shape
    attrs = _read_scale_attrs(dataset)

    if attrs is None:
        n_energy, n_angle = _detector_shape(shape, 2, is_fixed)
        energy = np.linspace(0.0, 1.0, n_energy)
        angle = np.linspace(-1.0, 1.0, n_angle)
        return energy, angle, "eV", "deg"

    units, offsets, deltas = attrs
    energy_idx, angle_idx = _classify_detector_axes(units)
    n_energy = _axis_length(shape, energy_idx, len(units), is_fixed)
    n_angle = _axis_length(shape, angle_idx, len(units), is_fixed)

    energy_unit = units[energy_idx]
    angle_unit = units[angle_idx]

    energy = offsets[energy_idx] + np.arange(n_energy) * deltas[energy_idx]

    if is_fixed and _is_pixel_unit(angle_unit):
        angle = (np.arange(n_angle) - n_angle // 2) * _PIXEL_ANGLE_STEP
        angle_unit = "deg"
    else:
        angle = offsets[angle_idx] + np.arange(n_angle) * deltas[angle_idx]
        if _is_pixel_unit(angle_unit):
            angle_unit = "deg"

    return energy, angle, energy_unit, angle_unit


def _read_scale_attrs(
    dataset: h5py.Dataset,
) -> tuple[list[str], list[float], list[float]] | None:
    if not all(
        key in dataset.attrs
        for key in ("unitNames", "scaleOffset", "scaleDelta")
    ):
        return None
    units = [_decode_text(value) for value in dataset.attrs["unitNames"]]
    offsets = [float(value) for value in dataset.attrs["scaleOffset"]]
    deltas = [float(value) for value in dataset.attrs["scaleDelta"]]
    return units, offsets, deltas


def _classify_detector_axes(units: list[str]) -> tuple[int, int]:
    energy_idx = angle_idx = 0
    for idx, unit in enumerate(units):
        label = unit.casefold()
        if "ev" in label or "energy" in label:
            energy_idx = idx
        elif any(token in label for token in ("deg", "pixel", "angle")):
            angle_idx = idx
    return energy_idx, angle_idx


def _axis_length(shape: tuple[int, ...], axis_idx: int, n_named: int, is_fixed: bool) -> int:
    if len(shape) == n_named:
        return int(shape[axis_idx])
    if is_fixed and len(shape) > n_named:
        return int(shape[axis_idx])
    return int(shape[-(n_named - axis_idx)])


def _detector_shape(shape: tuple[int, ...], n_axes: int, is_fixed: bool) -> tuple[int, int]:
    if len(shape) == n_axes:
        return int(shape[0]), int(shape[1])
    if is_fixed and len(shape) > n_axes:
        return int(shape[0]), int(shape[1])
    return int(shape[-2]), int(shape[-1])


def _is_pixel_unit(unit: str) -> bool:
    return "pixel" in unit.casefold()


def _decode_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
