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
from tensorspec.core.io.loaders.maestro.types import ScanLoop, ScanPlan, _is_angle_motor

KIND_ID = "focus_xy_fine_5d"


def match(plan: ScanPlan, is_fixed: bool) -> bool:
    if not is_fixed:
        return False
    if len(plan.loops) != 2:
        return False
    if not plan.has_xy_mesh():
        return False
    if len(plan.angle_motors()) != 1:
        return False
    xy_loops = sum(1 for loop in plan.loops if _loop_is_xy(loop))
    angle_loops = sum(1 for loop in plan.loops if _loop_is_single_angle(loop))
    return xy_loops == 1 and angle_loops == 1


def load(
    f: h5py.File,
    plan: ScanPlan,
    dataset: h5py.Dataset,
    *,
    path: str | None = None,
) -> dict:
    xy = plan.xy_motors()
    if xy is None:
        raise ValueError(f"{KIND_ID}: scan plan has no XY mesh.")
    x_motor, y_motor = xy
    defl_motor = plan.angle_motors()[0]
    n_x, n_y, n_defl = x_motor.n, y_motor.n, defl_motor.n
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

    motors_in_order = [motor for loop in plan.loops for motor in loop.motors]
    scan_shape = tuple(motor.n for motor in motors_in_order)
    scan_roles = [_motor_role(motor) for motor in motors_in_order]
    role_axis = {role: idx for idx, role in enumerate(scan_roles)}
    out_scan = (role_axis["y"], role_axis["x"], role_axis["defl"])

    buffer = load_spectra_buffer(dataset)
    scan_ndim = len(scan_shape)

    if warning:
        data = flatten_aborted_buffer(
            buffer, shape, paxis, n_e, n_a, kind_id=KIND_ID
        )
    elif paxis == 0:
        d1, d2 = shape[1], shape[2]
        block = buffer.reshape(*scan_shape, d1, d2)
        if (d1, d2) == (n_e, n_a):
            e_axis, a_axis = scan_ndim, scan_ndim + 1
        elif (d1, d2) == (n_a, n_e):
            e_axis, a_axis = scan_ndim + 1, scan_ndim
        else:
            raise ValueError(
                f"{KIND_ID}: detector dims {(d1, d2)} != "
                f"({n_e}, {n_a}) from dataset attrs."
            )
        perm = out_scan + (e_axis, a_axis)
        data = np.transpose(block, perm)
    elif paxis == 2:
        d1, d2 = shape[0], shape[1]
        block = buffer.reshape(d1, d2, *scan_shape)
        if (d1, d2) == (n_e, n_a):
            e_axis, a_axis = 0, 1
        elif (d1, d2) == (n_a, n_e):
            e_axis, a_axis = 1, 0
        else:
            raise ValueError(
                f"{KIND_ID}: detector dims {(d1, d2)} != "
                f"({n_e}, {n_a}) from dataset attrs."
            )
        scan_offset = 2
        perm = (
            scan_offset + role_axis["y"],
            scan_offset + role_axis["x"],
            scan_offset + role_axis["defl"],
            e_axis,
            a_axis,
        )
        data = np.transpose(block, perm)
    else:
        raise ValueError(f"{KIND_ID}: unrecognized points axis {paxis} for {shape}.")

    if not warning and data.shape != (n_y, n_x, n_defl, n_e, n_a):
        raise ValueError(
            f"{KIND_ID}: reshape got {data.shape}, "
            f"expected ({n_y}, {n_x}, {n_defl}, {n_e}, {n_a})."
        )

    y_axis = np.linspace(y_motor.start, y_motor.end, n_y)
    x_axis = np.linspace(x_motor.start, x_motor.end, n_x)
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
        labels = ["Y", "X", defl_motor.name, "Energy", "Angle"]
        axes = [y_axis, x_axis, defl_axis, energy, angle]
        units = [
            y_motor.units,
            x_motor.units,
            defl_motor.units,
            energy_unit,
            angle_unit,
        ]

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


def _loop_is_xy(loop: ScanLoop) -> bool:
    if len(loop.motors) != 2:
        return False
    x_motor = y_motor = None
    for motor in loop.motors:
        label = motor.name.casefold()
        if label in {"scan x", "sample x"}:
            x_motor = motor
        elif label in {"scan y", "sample y"}:
            y_motor = motor
    return x_motor is not None and y_motor is not None


def _loop_is_single_angle(loop: ScanLoop) -> bool:
    return len(loop.motors) == 1 and _is_angle_motor(loop.motors[0].name)


def _motor_role(motor) -> str:
    label = motor.name.casefold()
    if label in {"scan x", "sample x"}:
        return "x"
    if label in {"scan y", "sample y"}:
        return "y"
    if _is_angle_motor(motor.name):
        return "defl"
    raise ValueError(f"{KIND_ID}: unrecognized scan motor {motor.name!r}.")
