from __future__ import annotations

import h5py
import numpy as np

_PIXEL_ANGLE_STEP = 0.048


def fixed_detector_axes_for_plane(
    dataset: h5py.Dataset,
    *,
    n_angle: int,
    n_energy: int,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Build (energy, angle) axes for a Fixed Spectra detector plane.

    Plane storage order is ``(angle, energy)``. ``unitNames`` identify which
    scale is eV vs pixels; they do not define array axis order.
    """
    attrs = _read_scale_attrs(dataset)
    if attrs is None:
        energy = np.linspace(0.0, 1.0, n_energy)
        angle = (np.arange(n_angle) - n_angle // 2) * _PIXEL_ANGLE_STEP
        return energy, angle, "eV", "deg"

    units, offsets, deltas = attrs
    energy_idx, angle_idx = _classify_detector_axes(units)
    energy = offsets[energy_idx] + np.arange(n_energy) * deltas[energy_idx]
    energy_unit = units[energy_idx]
    angle_unit = units[angle_idx]
    if _is_pixel_unit(angle_unit):
        angle = (np.arange(n_angle) - n_angle // 2) * _PIXEL_ANGLE_STEP
        angle_unit = "deg"
    else:
        angle = offsets[angle_idx] + np.arange(n_angle) * deltas[angle_idx]
        if _is_pixel_unit(angle_unit):
            angle_unit = "deg"
    return energy, angle, energy_unit, angle_unit


def detector_axes(
    dataset: h5py.Dataset,
    is_fixed: bool,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Return (energy, angle) coordinate arrays for a spectra dataset.

    Maestro ``DAQ_Fixed`` / ``Fixed_Spectra*`` stores detector planes as
    ``(angle, energy)`` (SES Y then X). ``unitNames`` often lists ``eV``
    before ``pixels`` — that names the scales, not the array axis order.
    """
    shape = tuple(int(dim) for dim in dataset.shape)
    attrs = _read_scale_attrs(dataset)

    if is_fixed:
        n_angle, n_energy = _fixed_detector_lengths(shape)
        return fixed_detector_axes_for_plane(
            dataset, n_angle=n_angle, n_energy=n_energy
        )

    if attrs is None:
        n_energy, n_angle = _detector_shape(shape, 2, is_fixed=False)
        energy = np.linspace(0.0, 1.0, n_energy)
        angle = np.linspace(-1.0, 1.0, n_angle)
        return energy, angle, "eV", "deg"

    units, offsets, deltas = attrs
    energy_idx, angle_idx = _classify_detector_axes(units)
    n_energy = _axis_length(shape, energy_idx, len(units), is_fixed=False)
    n_angle = _axis_length(shape, angle_idx, len(units), is_fixed=False)

    energy_unit = units[energy_idx]
    angle_unit = units[angle_idx]
    energy = offsets[energy_idx] + np.arange(n_energy) * deltas[energy_idx]
    angle = offsets[angle_idx] + np.arange(n_angle) * deltas[angle_idx]
    if _is_pixel_unit(angle_unit):
        angle_unit = "deg"
    return energy, angle, energy_unit, angle_unit


def _fixed_detector_lengths(shape: tuple[int, ...]) -> tuple[int, int]:
    """Return (n_angle, n_energy) for Fixed spectra buffer shapes."""
    if len(shape) < 2:
        raise ValueError(f"Fixed spectra need >=2 dims, got {shape}.")
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    # Rank-3+: points axis is 0 or 2; detector plane is the other pair.
    # Prefer trailing detector (points-first) only when leading axis is the
    # clear odd-one-out vs two similar detector sizes — kinds pass paxis via
    # detector_dims_for_buffer; here use leading pair as Maestro default
    # (angle, energy, points) which matches live Fixed_Spectra1 files.
    return int(shape[0]), int(shape[1])


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
