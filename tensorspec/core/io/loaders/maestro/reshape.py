from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np


def points_axis(
    shape: tuple[int, ...],
    n_points: int,
    *,
    allow_truncated: bool = False,
) -> int:
    if len(shape) != 3:
        raise ValueError(f"Expected 3-D spectra buffer, got shape {shape}.")
    exact = [axis for axis in (0, 2) if shape[axis] == n_points]
    if exact:
        return exact[0]
    if allow_truncated:
        candidates = [axis for axis in (0, 2) if shape[axis] < n_points]
        if candidates:
            largest = max(int(shape[axis]) for axis in candidates)
            winners = [axis for axis in candidates if int(shape[axis]) == largest]
            if len(winners) == 1 and largest != int(shape[1]):
                return winners[0]
            raise ValueError(
                f"Cannot infer an unambiguous points axis for truncated "
                f"scan ({n_points} expected) in shape {shape}."
            )
    raise ValueError(
        f"No scan-points axis matches expected {n_points} in shape {shape}."
    )


def scan_cycles(
    f: h5py.File,
    shape: tuple[int, ...],
    expected: int,
) -> tuple[int, int, int | None]:
    """Resolve points axis while cross-checking Headers/Main/num_cycles."""
    num_cycles = header_num_cycles(f)
    if num_cycles is None or num_cycles == expected:
        paxis = points_axis(shape, expected, allow_truncated=True)
        return paxis, int(shape[paxis]), num_cycles

    plan_axes = [axis for axis in (0, 2) if int(shape[axis]) == expected]
    header_axes = [axis for axis in (0, 2) if int(shape[axis]) == num_cycles]
    matches = sorted(set(plan_axes + header_axes))
    if len(matches) != 1:
        if not matches:
            raise ValueError(
                f"Dataset shape {shape} matches neither scan-plan cycles "
                f"{expected} nor Headers/Main num_cycles {num_cycles}."
            )
        raise ValueError(
            f"Dataset shape {shape} does not identify an unambiguous points "
            f"axis between scan-plan cycles {expected} and "
            f"Headers/Main num_cycles {num_cycles}."
        )
    paxis = matches[0]
    return paxis, int(shape[paxis]), num_cycles


def header_num_cycles(f: h5py.File) -> int | None:
    main = f.get("Headers/Main")
    if main is None:
        return None
    if isinstance(main, h5py.Group):
        for key in main.keys():
            if key.casefold() == "num_cycles":
                return _coerce_int(main[key][()])
        for key, value in main.attrs.items():
            if key.casefold() == "num_cycles":
                return _coerce_int(value)
        return None

    values = main[()]
    if getattr(values.dtype, "names", None):
        for name in values.dtype.names:
            if name.casefold() == "num_cycles":
                return _coerce_int(values[name])

    flat = np.asarray(values).reshape(-1)
    for index, value in enumerate(flat[:-1]):
        text = _decode_text(value).strip().casefold()
        if text == "num_cycles":
            return _coerce_int(flat[index + 1])
    return None


def _coerce_int(value) -> int:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Headers/Main num_cycles is not scalar: {value!r}.")
    scalar = array.reshape(-1)[0]
    text = _decode_text(scalar).strip().strip("'\"")
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(
            f"Headers/Main num_cycles is not numeric: {text!r}."
        ) from exc
    if not number.is_integer():
        raise ValueError(f"Headers/Main num_cycles is not integral: {text!r}.")
    return int(number)


def load_spectra_buffer(dataset: h5py.Dataset) -> np.ndarray:
    buffer = np.empty(dataset.shape, dtype=np.float32)
    dataset.read_direct(buffer)
    return buffer


def abort_truncate_warning(expected: int, actual: int) -> str | None:
    if actual < expected:
        return (
            f"Aborted scan: truncated from {expected} to {actual} points."
        )
    return None


@dataclass(frozen=True)
class PartialGrid:
    kept_points: int
    kept_shape: tuple[int, ...]   # slow→fast scan dims after truncation
    kept_rows: int
    truncated_axis: int           # index in kept_shape that was shortened


def recover_partial_grid(actual: int, scan_shape: tuple[int, ...]) -> PartialGrid | None:
    if not scan_shape:
        return None
    fastest = scan_shape[-1]
    if fastest <= 0:
        return None
    complete = actual // fastest
    if complete < 1:
        return None
    kept_shape = list(scan_shape)
    # Outermost axis shortens when product of trailing dims == fastest
    # For standard XY: scan_shape=(n_y, n_x), fastest=n_x, kept=(complete, n_x)
    kept_shape[0] = complete
    if int(np.prod(kept_shape)) * 1 != complete * fastest:
        # general: only support shortening axis 0 when prod(scan_shape[1:]) == fastest
        if int(np.prod(scan_shape[1:])) != fastest:
            return None
    return PartialGrid(
        kept_points=complete * fastest,
        kept_shape=tuple(kept_shape),
        kept_rows=complete,
        truncated_axis=0,
    )


def flatten_aborted_buffer(
    buffer: np.ndarray,
    shape: tuple[int, ...],
    paxis: int,
    n_e: int,
    n_a: int,
    *,
    kind_id: str,
) -> np.ndarray:
    actual = shape[paxis]
    if paxis == 0:
        d1, d2 = shape[1], shape[2]
        block = buffer.reshape(actual, d1, d2)
        if (d1, d2) == (n_e, n_a):
            return block
        if (d1, d2) == (n_a, n_e):
            return np.transpose(block, (0, 2, 1))
    elif paxis == 2:
        d1, d2 = shape[0], shape[1]
        block = buffer.reshape(d1, d2, actual)
        if (d1, d2) == (n_e, n_a):
            return np.transpose(block, (2, 0, 1))
        if (d1, d2) == (n_a, n_e):
            return np.transpose(block, (2, 1, 0))
    raise ValueError(
        f"{kind_id}: detector dimensions do not match aborted buffer {shape}."
    )


def detector_dims_for_buffer(
    dataset: h5py.Dataset,
    shape: tuple[int, ...],
    paxis: int,
    *,
    is_fixed: bool,
) -> tuple[int, int, np.ndarray, np.ndarray, str, str]:
    """Map buffer detector axes to (n_e, n_a) and coordinate arrays."""
    from tensorspec.core.io.loaders.maestro.detector import (
        detector_axes,
        fixed_detector_axes_for_plane,
    )

    if paxis == 0:
        d1, d2 = int(shape[1]), int(shape[2])
    elif paxis == 2:
        d1, d2 = int(shape[0]), int(shape[1])
    else:
        raise ValueError(f"Unsupported points axis {paxis} for shape {shape}.")

    if is_fixed:
        # Fixed Spectra plane is always (angle, energy) regardless of unitNames order.
        energy, angle, energy_unit, angle_unit = fixed_detector_axes_for_plane(
            dataset, n_angle=d1, n_energy=d2
        )
        return len(energy), len(angle), energy, angle, energy_unit, angle_unit

    energy, angle, energy_unit, angle_unit = detector_axes(dataset, is_fixed=False)
    if (d1, d2) == (len(energy), len(angle)):
        return len(energy), len(angle), energy, angle, energy_unit, angle_unit
    if (d1, d2) == (len(angle), len(energy)):
        return len(energy), len(angle), energy, angle, energy_unit, angle_unit

    if paxis != 0:
        raise ValueError(
            f"Detector dims {(d1, d2)} do not match attrs "
            f"({len(energy)}, {len(angle)})."
        )

    return _detector_dims_points_first(
        dataset,
        d1,
        d2,
        is_fixed=False,
    )


def _detector_dims_points_first(
    dataset: h5py.Dataset,
    d1: int,
    d2: int,
    *,
    is_fixed: bool,
) -> tuple[int, int, np.ndarray, np.ndarray, str, str]:
    attrs = _read_scale_attrs(dataset)
    if attrs is None:
        energy = np.linspace(0.0, 1.0, d1)
        angle = np.linspace(-1.0, 1.0, d2)
        return d1, d2, energy, angle, "eV", "deg"

    units, offsets, deltas = attrs
    energy_idx = angle_idx = 0
    for idx, unit in enumerate(units):
        label = unit.casefold()
        if "ev" in label or "energy" in label:
            energy_idx = idx
        elif any(token in label for token in ("deg", "pixel", "angle")):
            angle_idx = idx

    if energy_idx == 0:
        n_e, n_a = d1, d2
    else:
        n_e, n_a = d2, d1

    energy = offsets[energy_idx] + np.arange(n_e) * deltas[energy_idx]
    angle_unit = units[angle_idx]
    if is_fixed and "pixel" in angle_unit.casefold():
        angle = (np.arange(n_a) - n_a // 2) * 0.048
        angle_unit = "deg"
    else:
        angle = offsets[angle_idx] + np.arange(n_a) * deltas[angle_idx]
        if "pixel" in angle_unit.casefold():
            angle_unit = "deg"

    return n_e, n_a, energy, angle, units[energy_idx], angle_unit


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


def _decode_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
