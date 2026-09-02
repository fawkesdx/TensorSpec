from __future__ import annotations

import os

import h5py
import numpy as np

from tensorspec.core.io.loaders.maestro.detect import is_fixed_mode
from tensorspec.core.io.loaders.maestro.reshape import abort_truncate_warning

KIND_ID = "process000_generic"


def load(
    f: h5py.File,
    dataset: h5py.Dataset,
    *,
    path: str | None = None,
) -> dict:
    raw = dataset[()]
    if raw.ndim < 2:
        raise ValueError(f"{KIND_ID}: expected at least 2-D Process_000 data.")

    motors = {
        name: np.asarray(f["0D_Data"][name][()])
        for name in f["0D_Data"].keys()
    }
    n_dim_1, n_dim_2 = int(raw.shape[-2]), int(raw.shape[-1])
    actual = int(raw.size // (n_dim_1 * n_dim_2))
    expected = len(next(iter(motors.values()))) if motors else actual
    warning = abort_truncate_warning(expected, actual)
    if warning:
        motors = {name: values[:actual] for name, values in motors.items()}

    units = dataset.attrs.get("unitNames", [b"eV", b"deg"])
    offsets = dataset.attrs.get("scaleOffset", [0.0, 0.0])
    deltas = dataset.attrs.get("scaleDelta", [0.01, 0.01])
    detector_units = [_decode(units[index]) for index in range(2)]
    detector_axes = [
        float(offsets[index]) + np.arange(size) * float(deltas[index])
        for index, size in enumerate((n_dim_1, n_dim_2))
    ]

    grid_shape = [
        len(np.unique(np.round(values, 3)))
        for values in motors.values()
    ]
    complete_grid = bool(motors) and not warning and np.prod(grid_shape) == actual
    if complete_grid:
        target_shape = tuple(grid_shape) + (n_dim_1, n_dim_2)
    elif motors:
        target_shape = (actual, n_dim_1, n_dim_2)
    else:
        target_shape = (n_dim_1, n_dim_2)
    data = np.reshape(raw, target_shape)

    fixed = is_fixed_mode(f)
    if fixed and not motors:
        data = np.transpose(data, (1, 0))
        detector_axes.reverse()
        detector_units.reverse()

    motor_labels = list(motors)
    if complete_grid:
        motor_axes = [
            np.unique(np.round(values, 3))
            for values in motors.values()
        ]
        labels = motor_labels
    elif motors:
        motor_axes = [np.arange(actual)]
        labels = ["Point"]
    else:
        motor_axes = []
        labels = []

    detector_labels = _detector_labels(detector_units)
    metadata = {
        "kind": KIND_ID,
        "source_path": path,
        "scan_plan": {
            "expected_cycles": expected,
            "actual_cycles": actual,
        },
    }
    if warning:
        metadata["truncate_warning"] = warning

    return {
        "name": os.path.basename(path or dataset.file.filename).removesuffix(".h5"),
        "data": np.ascontiguousarray(data),
        "labels": labels + detector_labels,
        "axes": motor_axes + detector_axes,
        "units": ["a.u."] * len(motor_axes) + detector_units,
        "mode": _mode_name(f, fixed) + (" (Aborted)" if warning else ""),
        "is_fixed": fixed,
        "facility": "MAESTRO",
        "metadata": metadata,
    }


def _detector_labels(units: list[str]) -> list[str]:
    labels = []
    for unit in units:
        folded = unit.casefold()
        if "ev" in folded or "energy" in folded:
            labels.append("Energy")
        elif any(token in folded for token in ("deg", "pixel", "angle")):
            labels.append("Angle")
        else:
            labels.append("Detector")
    if labels[0] == labels[1]:
        return ["Axis Y", "Axis X"]
    return labels


def _mode_name(f: h5py.File, fixed: bool) -> str:
    header_name = "DAQ_Fixed" if fixed else "DAQ_Swept"
    header = f["Headers"].get(header_name)
    if header is not None:
        for row in header[()]:
            if len(row) >= 3 and b"Mode" in row[1]:
                return _decode(row[2])
    return "Unknown Scan"


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
