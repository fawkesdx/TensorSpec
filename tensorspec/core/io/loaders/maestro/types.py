from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScanMotor:
    name: str
    units: str
    start: float
    end: float
    n: int


@dataclass(frozen=True)
class ScanLoop:
    motors: list[ScanMotor]


@dataclass(frozen=True)
class ScanPlan:
    mode_name: str
    loops: list[ScanLoop]
    parallel: bool

    @property
    def expected_cycles(self) -> int:
        total = 1
        for loop in self.loops:
            for motor in loop.motors:
                total *= motor.n
        return total

    def has_xy_mesh(self) -> bool:
        return self.xy_motors() is not None

    def angle_motors(self) -> list[ScanMotor]:
        motors: list[ScanMotor] = []
        for loop in self.loops:
            for motor in loop.motors:
                if _is_angle_motor(motor.name):
                    motors.append(motor)
        return motors

    def xy_motors(self) -> tuple[ScanMotor, ScanMotor] | None:
        for loop in self.loops:
            x_motor = y_motor = None
            for motor in loop.motors:
                label = motor.name.casefold()
                if label in {"scan x", "sample x"}:
                    x_motor = motor
                elif label in {"scan y", "sample y"}:
                    y_motor = motor
            if x_motor is not None and y_motor is not None:
                return x_motor, y_motor
        return None


def _is_angle_motor(name: str) -> bool:
    label = name.casefold()
    if label.endswith(" x") or label.endswith(" y"):
        return False
    if " x" in label or " y" in label:
        return False
    keywords = ("defl", "deflection", "theta", "tilt", "phi")
    return any(keyword in label for keyword in keywords)
