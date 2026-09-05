"""Map Maestro axis labels to SSL roles and enumerate extractable sample modes."""

from __future__ import annotations

import math
from dataclasses import dataclass

from tensorspec.core.io.loaders.maestro.types import _is_angle_motor
from tensorspec.core.ml.ssl.spec import AxisRole, SampleModeName

_SPATIAL_INDEX_ORDER: tuple[AxisRole, ...] = ("y", "x")
_DISP2D_INDEX_ORDER: tuple[AxisRole, ...] = ("y", "x", "defl")
_FERMI3D_PAYLOAD: tuple[AxisRole, ...] = ("defl", "energy", "slit")
_DISP2D_PAYLOAD: tuple[AxisRole, ...] = ("energy", "slit")


@dataclass(frozen=True)
class SampleMode:
    name: SampleModeName
    n_samples: int
    sample_shape: tuple[int, ...]
    index_roles: tuple[AxisRole, ...]
    payload_roles: tuple[AxisRole, ...]


def role_for_label(label: str) -> AxisRole:
    normalized = label.casefold()

    if normalized == "energy":
        return "energy"
    if normalized == "angle" or "slit angle" in normalized:
        return "slit"
    if normalized in {"scan x", "sample x", "x"}:
        return "x"
    if normalized in {"scan y", "sample y", "y"}:
        return "y"
    if _is_angle_motor(label):
        return "defl"
    return "other"


def roles_for_labels(labels: list[str]) -> list[AxisRole]:
    return [role_for_label(label) for label in labels]


def _role_sizes(labels: list[str], shape: tuple[int, ...]) -> dict[AxisRole, int]:
    if len(labels) != len(shape):
        msg = f"labels length {len(labels)} != shape length {len(shape)}"
        raise ValueError(msg)
    sizes: dict[AxisRole, int] = {}
    for label, size in zip(labels, shape):
        role = role_for_label(label)
        if role in sizes and sizes[role] != size:
            msg = f"duplicate role {role!r} with mismatched sizes for label {label!r}"
            raise ValueError(msg)
        sizes[role] = size
    return sizes


def _index_product(sizes: dict[AxisRole, int], roles: tuple[AxisRole, ...]) -> int:
    present = tuple(role for role in roles if role in sizes)
    if not present:
        return 1
    return math.prod(sizes[role] for role in present)


def _payload_shape(sizes: dict[AxisRole, int], roles: tuple[AxisRole, ...]) -> tuple[int, ...]:
    return tuple(sizes[role] for role in roles)


def enumerate_modes(labels: list[str], shape: tuple[int, ...]) -> list[SampleMode]:
    """Return extractable sample modes per spec section 5.2."""
    sizes = _role_sizes(labels, shape)
    if "energy" not in sizes or "slit" not in sizes:
        return []

    modes: list[SampleMode] = []

    if "defl" in sizes:
        fermi_index = tuple(role for role in _SPATIAL_INDEX_ORDER if role in sizes)
        modes.append(
            SampleMode(
                name="fermi3d",
                n_samples=_index_product(sizes, fermi_index),
                sample_shape=_payload_shape(sizes, _FERMI3D_PAYLOAD),
                index_roles=fermi_index,
                payload_roles=_FERMI3D_PAYLOAD,
            )
        )

        disp_index = tuple(role for role in _DISP2D_INDEX_ORDER if role in sizes)
        modes.append(
            SampleMode(
                name="disp2d",
                n_samples=_index_product(sizes, disp_index),
                sample_shape=_payload_shape(sizes, _DISP2D_PAYLOAD),
                index_roles=disp_index,
                payload_roles=_DISP2D_PAYLOAD,
            )
        )
        return modes

    if any(role in sizes for role in _SPATIAL_INDEX_ORDER):
        disp_index = tuple(role for role in _SPATIAL_INDEX_ORDER if role in sizes)
        modes.append(
            SampleMode(
                name="disp2d",
                n_samples=_index_product(sizes, disp_index),
                sample_shape=_payload_shape(sizes, _DISP2D_PAYLOAD),
                index_roles=disp_index,
                payload_roles=_DISP2D_PAYLOAD,
            )
        )

    return modes
