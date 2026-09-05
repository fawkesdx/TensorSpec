from __future__ import annotations

from dataclasses import dataclass, field

import h5py
import numpy as np

from tensorspec.core.io.loaders.maestro.detect import (
    assert_maestro_signature,
    is_fixed_mode,
    select_spectra_dataset,
)
from tensorspec.core.io.loaders.maestro.low_level_scan import parse_low_level_scan
from tensorspec.core.io.loaders.maestro.registry import match_kind
from tensorspec.core.io.loaders.maestro.reshape import (
    detector_dims_for_buffer,
    recover_partial_grid,
    scan_cycles,
)
from tensorspec.core.io.loaders.maestro.types import ScanMotor, ScanPlan


@dataclass
class MaestroDescriptor:
    """Lazy Maestro metadata plus detector-frame hyperslab access.

    The descriptor owns one read-only HDF5 handle. Call :meth:`close` when
    finished, or use the descriptor as a context manager.
    """

    path: str
    kind: str
    labels: list[str]
    axes: list[np.ndarray]
    units: list[str]
    shape: tuple[int, ...]
    metadata: dict
    _file: h5py.File = field(repr=False)
    _dataset: h5py.Dataset = field(repr=False)
    _points_axis: int = field(repr=False)
    _n_energy: int = field(repr=False)
    _n_angle: int = field(repr=False)
    _readable_points: int = field(repr=False)

    def read_block(self, index: int | slice) -> np.ndarray:
        """Read one point or contiguous flat C-order span as ``(E, A)`` frames."""
        if not self._file.id.valid:
            raise ValueError("Maestro descriptor is closed.")

        selection = _normalize_selection(index, self._readable_points)
        if self._points_axis == 0:
            slab = self._dataset[selection, :, :]
        else:
            slab = self._dataset[:, :, selection]
        return detector_frame_from_slab(
            slab,
            self._n_energy,
            self._n_angle,
            self._points_axis,
        )

    def close(self) -> None:
        if self._file.id.valid:
            self._file.close()

    def __enter__(self) -> MaestroDescriptor:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def open_maestro(path: str) -> MaestroDescriptor:
    """Open supported Maestro scan metadata without loading its spectra buffer."""
    source_path = str(path)
    file = h5py.File(source_path, "r")
    try:
        assert_maestro_signature(file)
        headers = file.get("Headers")
        if headers is None or "Low_Level_Scan" not in headers:
            raise ValueError("Missing Headers/Low_Level_Scan.")

        fixed = is_fixed_mode(file)
        plan = parse_low_level_scan(headers["Low_Level_Scan"][()])
        module = match_kind(plan, fixed)
        if module is None:
            raise ValueError(
                f"No Maestro kind matched mode={plan.mode_name!r}, "
                f"is_fixed={fixed}."
            )

        dataset = select_spectra_dataset(file)
        dataset_shape = tuple(int(dim) for dim in dataset.shape)
        points_axis, actual, num_cycles = scan_cycles(
            file, dataset_shape, plan.expected_cycles
        )
        n_e, n_a, energy, angle, energy_unit, angle_unit = (
            detector_dims_for_buffer(
                dataset,
                dataset_shape,
                points_axis,
                is_fixed=fixed,
            )
        )

        labels, axes, units, scan_shape = _kind_layout(
            module.KIND_ID,
            plan,
            energy,
            angle,
            energy_unit,
            angle_unit,
        )
        metadata: dict = {
            "kind": module.KIND_ID,
            "source_path": source_path,
            "scan_plan": {
                "mode_name": plan.mode_name,
                "expected_cycles": plan.expected_cycles,
                "actual_cycles": actual,
            },
        }
        if num_cycles is not None:
            metadata["scan_plan"]["num_cycles"] = num_cycles

        readable_points = actual
        if actual < plan.expected_cycles:
            partial = recover_partial_grid(actual, scan_shape)
            if partial is not None:
                scan_shape = partial.kept_shape
                axes[0] = axes[0][: partial.kept_shape[0]]
                readable_points = partial.kept_points
                metadata["partial_scan"] = {
                    "expected": plan.expected_cycles,
                    "actual": actual,
                    "kept_rows": partial.kept_rows,
                }
            else:
                labels = ["Point", "Energy", "Angle"]
                axes = [np.arange(actual), energy, angle]
                units = ["index", energy_unit, angle_unit]
                scan_shape = (actual,)
                metadata["truncate_warning"] = (
                    f"Aborted scan: truncated from {plan.expected_cycles} "
                    f"to {actual} points."
                )

        return MaestroDescriptor(
            path=source_path,
            kind=module.KIND_ID,
            labels=labels,
            axes=axes,
            units=units,
            shape=(*scan_shape, n_e, n_a),
            metadata=metadata,
            _file=file,
            _dataset=dataset,
            _points_axis=points_axis,
            _n_energy=n_e,
            _n_angle=n_a,
            _readable_points=readable_points,
        )
    except BaseException:
        file.close()
        raise


def detector_frame_from_slab(
    slab: np.ndarray,
    n_e: int,
    n_a: int,
    points_axis: int,
) -> np.ndarray:
    """Convert a fixed-spectra ``(angle, energy)`` slab to ``(..., E, A)``."""
    array = np.asarray(slab)
    if points_axis == 0:
        detector_shape = array.shape[-2:]
        if detector_shape != (n_a, n_e):
            raise ValueError(
                f"Detector slab has shape {detector_shape}, "
                f"expected ({n_a}, {n_e})."
            )
        result = np.swapaxes(array, -2, -1)
    elif points_axis == 2:
        detector_shape = array.shape[:2]
        if detector_shape != (n_a, n_e):
            raise ValueError(
                f"Detector slab has shape {detector_shape}, "
                f"expected ({n_a}, {n_e})."
            )
        if array.ndim == 2:
            result = array.T
        else:
            result = np.transpose(array, (2, 1, 0))
    else:
        raise ValueError(f"Unsupported points axis {points_axis}.")
    return np.ascontiguousarray(result)


def _normalize_selection(
    index: int | slice,
    n_points: int,
) -> int | slice:
    if isinstance(index, bool):
        raise TypeError("Point index must be an integer or slice.")
    if isinstance(index, int):
        if index < 0 or index >= n_points:
            raise IndexError(
                f"Point index {index} outside valid range 0..{n_points - 1}."
            )
        return index
    if not isinstance(index, slice):
        raise TypeError("Point index must be an integer or slice.")
    if index.step not in (None, 1):
        raise ValueError("read_block only accepts contiguous slices.")
    start, stop, _ = index.indices(n_points)
    return slice(start, stop)


def _kind_layout(
    kind: str,
    plan: ScanPlan,
    energy: np.ndarray,
    angle: np.ndarray,
    energy_unit: str,
    angle_unit: str,
) -> tuple[list[str], list[np.ndarray], list[str], tuple[int, ...]]:
    defl_motors = plan.angle_motors()
    if kind == "xy_fine_4d":
        x_motor, y_motor = _require_xy(plan, kind)
        scan_motors = [y_motor, x_motor]
        labels = ["Y", "X"]
    elif kind == "focus_xy_fine_5d":
        x_motor, y_motor = _require_xy(plan, kind)
        scan_motors = [y_motor, x_motor, defl_motors[0]]
        labels = ["Y", "X", defl_motors[0].name]
    elif kind == "defl_x_line_4d":
        x_motor = next(
            (
                motor
                for motor in plan.loops[0].motors
                if motor.name.casefold() in {"scan x", "sample x"}
            ),
            None,
        )
        if x_motor is None:
            raise ValueError(f"{kind}: scan plan has no Scan X motor.")
        scan_motors = [x_motor, defl_motors[0]]
        labels = ["X", defl_motors[0].name]
    elif kind == "fermi_defl_3d":
        scan_motors = [defl_motors[0]]
        labels = [defl_motors[0].name]
    else:
        raise ValueError(f"Lazy Maestro access does not support kind {kind!r}.")

    axes = [_motor_axis(motor) for motor in scan_motors]
    units = [motor.units for motor in scan_motors]
    return (
        labels + ["Energy", "Angle"],
        axes + [energy, angle],
        units + [energy_unit, angle_unit],
        tuple(motor.n for motor in scan_motors),
    )


def _require_xy(plan: ScanPlan, kind: str) -> tuple[ScanMotor, ScanMotor]:
    motors = plan.xy_motors()
    if motors is None:
        raise ValueError(f"{kind}: scan plan has no XY mesh.")
    return motors


def _motor_axis(motor: ScanMotor) -> np.ndarray:
    return np.linspace(motor.start, motor.end, motor.n)


__all__ = ["MaestroDescriptor", "open_maestro"]
