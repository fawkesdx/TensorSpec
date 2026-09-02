from __future__ import annotations

import h5py
import numpy as np

from tensorspec.core.io.loaders.maestro.reshape import (
    abort_truncate_warning,
    detector_dims_for_buffer,
    flatten_aborted_buffer,
    load_spectra_buffer,
    scan_cycles,
)
from tensorspec.core.io.loaders.maestro.types import ScanPlan

KIND_ID = "fermi_defl_3d"


def match(plan: ScanPlan, is_fixed: bool) -> bool:
    if not is_fixed:
        return False
    if len(plan.loops) != 1:
        return False
    if plan.has_xy_mesh():
        return False
    if len(plan.angle_motors()) != 1:
        return False
    return True


def load(
    f: h5py.File,
    plan: ScanPlan,
    dataset: h5py.Dataset,
    *,
    path: str | None = None,
) -> dict:
    defl_motor = plan.angle_motors()[0]
    n_defl = defl_motor.n
    expected = plan.expected_cycles

    shape = tuple(int(dim) for dim in dataset.shape)
    paxis, actual, num_cycles = scan_cycles(f, shape, expected)
    warning = abort_truncate_warning(expected, actual)

    n_e, n_a, energy, angle, energy_unit, angle_unit = detector_dims_for_buffer(
        dataset,
        shape,
        paxis,
        is_fixed=True,
    )

    buffer = load_spectra_buffer(dataset)
    if warning:
        data = flatten_aborted_buffer(
            buffer, shape, paxis, n_e, n_a, kind_id=KIND_ID
        )
    elif paxis == 0:
        d1, d2 = shape[1], shape[2]
        block = buffer.reshape(n_defl, d1, d2)
        if (d1, d2) == (n_e, n_a):
            data = block
        elif (d1, d2) == (n_a, n_e):
            data = np.transpose(block, (0, 2, 1))
        else:
            raise ValueError(
                f"{KIND_ID}: detector dims {(d1, d2)} != "
                f"({n_e}, {n_a}) from dataset attrs."
            )
    elif paxis == 2:
        d1, d2 = shape[0], shape[1]
        block = buffer.reshape(d1, d2, n_defl)
        if (d1, d2) == (n_e, n_a):
            data = np.transpose(block, (2, 0, 1))
        elif (d1, d2) == (n_a, n_e):
            data = np.transpose(block, (2, 1, 0))
        else:
            raise ValueError(
                f"{KIND_ID}: detector dims {(d1, d2)} != "
                f"({n_e}, {n_a}) from dataset attrs."
            )
    else:
        raise ValueError(f"{KIND_ID}: unrecognized points axis {paxis} for {shape}.")

    if not warning and data.shape != (n_defl, n_e, n_a):
        raise ValueError(
            f"{KIND_ID}: reshape got {data.shape}, "
            f"expected ({n_defl}, {n_e}, {n_a})."
        )

    defl_axis = np.linspace(defl_motor.start, defl_motor.end, n_defl)

    metadata: dict = {
        "kind": KIND_ID,
        "source_path": path,
        "scan_plan": {
            "mode_name": plan.mode_name,
            "expected_cycles": expected,
            "actual_cycles": actual,
        },
    }
    if num_cycles is not None:
        metadata["scan_plan"]["num_cycles"] = num_cycles
    if warning:
        metadata["truncate_warning"] = warning

    if warning:
        labels = ["Point", "Energy", "Angle"]
        axes = [np.arange(actual), energy, angle]
        units = ["index", energy_unit, angle_unit]
    else:
        labels = [defl_motor.name, "Energy", "Angle"]
        axes = [defl_axis, energy, angle]
        units = [defl_motor.units, energy_unit, angle_unit]

    return {
        "data": np.ascontiguousarray(data),
        "labels": labels,
        "axes": axes,
        "units": units,
        "mode": plan.mode_name,
        "is_fixed": True,
        "facility": "MAESTRO",
        "metadata": metadata,
    }
