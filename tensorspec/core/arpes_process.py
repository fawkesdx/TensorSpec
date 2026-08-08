"""Experimental ARPES coordinate transforms (angle → k∥, later hv → kz).

Uses ``ARPESKinematics`` for the vacuum projection. Angle axes are recentered
on a user-chosen Γ (center value), then mapped with a degrees-per-unit scale
so Detector Pixel and true-degree axes share one path.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tensorspec.core.data_models import TensorData
from tensorspec.core.kinematics import ARPESKinematics


ANGLE_HINTS = (
    "pixel",
    "angle",
    "slit",
    "defl",
    "theta",
    "beta",
    "phi",
    "detector",
)
ENERGY_HINTS = ("energy", "ev", "binding", "kinetic")
PHOTON_HINTS = ("photon", "mono_ev", "mono energy", "hv")
AVOID_ANGLE = ("sample x", "sample y", "sample z", "photon", "time", "index", "mono")


def _label_matches(label: str, hints: Sequence[str]) -> bool:
    lower = label.lower()
    return any(token in lower for token in hints)


def _is_photon_label(label: str) -> bool:
    lower = label.lower()
    return any(token in lower for token in PHOTON_HINTS)


def infer_axis_roles(tensor: TensorData) -> Dict[str, Optional[int]]:
    """Guess energy / photon / primary angle / secondary angle indices from labels."""
    energy_idx = None
    photon_idx = None
    angle_idxs: List[int] = []
    for index, label in enumerate(tensor.labels):
        lower = label.lower()
        if _is_photon_label(label):
            photon_idx = index
            continue
        if any(tok in lower for tok in AVOID_ANGLE):
            continue
        if _label_matches(label, ENERGY_HINTS) and "pixel" not in lower:
            energy_idx = index
        elif _label_matches(label, ANGLE_HINTS):
            angle_idxs.append(index)

    primary = angle_idxs[0] if angle_idxs else (1 if tensor.ndim > 1 else 0)
    secondary = angle_idxs[1] if len(angle_idxs) > 1 else None
    if energy_idx is None:
        for index, label in enumerate(tensor.labels):
            if index in (photon_idx, primary, secondary):
                continue
            if _label_matches(label, ENERGY_HINTS) and not _is_photon_label(label):
                energy_idx = index
                break
    return {
        "energy_axis": energy_idx if energy_idx is not None else 0,
        "photon_axis": photon_idx,
        "angle_axis": primary,
        "beta_axis": secondary,
    }


def _kinetic_energy(
    energy_axis: np.ndarray,
    *,
    photon_energy: float,
    work_function: float,
    energy_mode: str,
) -> Tuple[np.ndarray, str]:
    energy = np.asarray(energy_axis, dtype=float)
    mode = energy_mode
    if mode == "auto":
        median = float(np.nanmedian(energy))
        span = float(np.nanmax(energy) - np.nanmin(energy))
        if median > 5.0 and span > 5.0:
            mode = "kinetic"
        else:
            mode = "binding_negative" if median <= 0 else "binding_positive"

    if mode == "kinetic":
        e_kin = np.maximum(energy, 1e-6)
    elif mode == "binding_negative":
        # Occupied states stored as E <= 0 relative to EF.
        e_kin = np.maximum(photon_energy - work_function + energy, 1e-6)
    else:
        # Occupied states as positive binding energy.
        e_kin = np.maximum(photon_energy - work_function - energy, 1e-6)
    return e_kin, mode


def _angle_to_k_axis(
    coords: np.ndarray,
    *,
    center: float,
    deg_per_unit: float,
    e_kin_ref: float,
) -> np.ndarray:
    theta = (np.asarray(coords, dtype=float) - float(center)) * float(deg_per_unit)
    kx, _ = ARPESKinematics.angle_to_k_parallel(e_kin_ref, theta, beta=0.0)
    return np.asarray(kx, dtype=float)


def suggest_center(
    tensor: TensorData,
    *,
    angle_axis: int,
    energy_axis: Optional[int] = None,
    fixed: Optional[Dict[int, int]] = None,
) -> Dict[str, Any]:
    """Brightness-based Γ hint. User should always fine-tune."""
    fixed = dict(fixed or {})
    axis = np.asarray(tensor.axes[angle_axis], dtype=float)
    data = np.asarray(tensor.value, dtype=float)

    work = data
    for dim in range(tensor.ndim - 1, -1, -1):
        if dim == angle_axis:
            continue
        if energy_axis is not None and dim == energy_axis:
            continue
        held = fixed.get(dim, work.shape[dim] // 2)
        work = np.take(work, int(np.clip(held, 0, work.shape[dim] - 1)), axis=dim)

    if energy_axis is not None and work.ndim == 2:
        e_ax_in_work = 1 if angle_axis < energy_axis else 0
        n_e = work.shape[e_ax_in_work]
        start = max(0, n_e - max(3, n_e // 10))
        if e_ax_in_work == 0:
            profile = work[start:, :].sum(axis=0)
        else:
            profile = work[:, start:].sum(axis=1)
    else:
        profile = np.asarray(work, dtype=float).ravel()

    profile = np.asarray(profile, dtype=float).ravel()
    if profile.size != axis.size or not np.any(np.isfinite(profile)):
        index = int(axis.size // 2)
    else:
        index = int(np.nanargmax(profile))
    return {
        "index": index,
        "value": float(axis[index]),
        "method": "brightness_near_ef",
    }


def convert_inplane_to_k(
    tensor: TensorData,
    *,
    angle_axis: int,
    energy_axis: Optional[int] = None,
    beta_axis: Optional[int] = None,
    center: float,
    deg_per_unit: float = 1.0,
    beta_center: float = 0.0,
    beta_deg_per_unit: float = 1.0,
    photon_energy: float,
    work_function: float = 4.5,
    energy_mode: str = "auto",
    e_kin_ref: Optional[float] = None,
) -> TensorData:
    """
    Relabel angle axes as k∥ using an EF-referenced kinetic energy.

    Intensity is not resampled: this is a coordinate transform for interactive
    preview and first-order analysis. Full warped E-dependent resampling can
    follow later.
    """
    if not 0 <= angle_axis < tensor.ndim:
        raise ValueError(f"angle_axis {angle_axis} out of range for {tensor.ndim}D data.")
    if beta_axis is not None and not 0 <= beta_axis < tensor.ndim:
        raise ValueError(f"beta_axis {beta_axis} out of range.")
    if energy_axis is not None and not 0 <= energy_axis < tensor.ndim:
        raise ValueError(f"energy_axis {energy_axis} out of range.")

    if e_kin_ref is None:
        if energy_axis is not None:
            e_kin_axis, resolved_mode = _kinetic_energy(
                tensor.axes[energy_axis],
                photon_energy=photon_energy,
                work_function=work_function,
                energy_mode=energy_mode,
            )
            e_kin_ref = float(np.nanmax(e_kin_axis))
        else:
            resolved_mode = energy_mode
            e_kin_ref = max(photon_energy - work_function, 1e-6)
    else:
        resolved_mode = energy_mode
        e_kin_ref = max(float(e_kin_ref), 1e-6)

    new_axes = [np.asarray(ax, dtype=float).copy() for ax in tensor.axes]
    new_labels = list(tensor.labels)
    new_units = list(tensor.units)

    new_axes[angle_axis] = _angle_to_k_axis(
        tensor.axes[angle_axis],
        center=center,
        deg_per_unit=deg_per_unit,
        e_kin_ref=e_kin_ref,
    )
    new_labels[angle_axis] = "kx"
    new_units[angle_axis] = "1/A"

    if beta_axis is not None:
        new_axes[beta_axis] = _angle_to_k_axis(
            tensor.axes[beta_axis],
            center=beta_center,
            deg_per_unit=beta_deg_per_unit,
            e_kin_ref=e_kin_ref,
        )
        new_labels[beta_axis] = "ky"
        new_units[beta_axis] = "1/A"

    metadata = dict(tensor.metadata or {})
    metadata.update(
        {
            "Processed": "inplane_k",
            "Gamma_Center": float(center),
            "Gamma_Center_Beta": float(beta_center) if beta_axis is not None else None,
            "Deg_Per_Unit": float(deg_per_unit),
            "Beta_Deg_Per_Unit": float(beta_deg_per_unit) if beta_axis is not None else None,
            "Photon_Energy_eV": float(photon_energy),
            "Work_Function_eV": float(work_function),
            "Energy_Mode": resolved_mode,
            "E_kin_Ref_eV": float(e_kin_ref),
            "Angle_Axis": int(angle_axis),
            "Beta_Axis": int(beta_axis) if beta_axis is not None else None,
            "Energy_Axis": int(energy_axis) if energy_axis is not None else None,
            "Source_DataType": tensor.data_type,
        }
    )

    return TensorData(
        value=np.asarray(tensor.value, dtype=float).copy(),
        axes=new_axes,
        labels=new_labels,
        units=new_units,
        data_type=f"{tensor.data_type} (k∥)",
        metadata=metadata,
    )


def surface_bz_polygon_2d(
    structure,
    h: int = 0,
    k: int = 0,
    l: int = 1,
) -> Optional[Dict[str, Any]]:
    """Projected surface BZ as a closed (kx, ky) polygon in Å⁻¹."""
    from tensorspec.core.crystallography import CrystalEngine

    bz_data = CrystalEngine.calculate_brillouin_zone(structure)
    if not bz_data:
        return None

    recip_matrix = structure.lattice.reciprocal_lattice.matrix
    hkl = (int(h), int(k), int(l))
    z_surf, y_surf = CrystalEngine.get_hkl_surface_frame(hkl, recip_matrix, azimuthal_ref=None)
    x_surf = np.cross(y_surf, z_surf)
    r_hkl_to_bulk = np.column_stack((x_surf, y_surf, z_surf))
    r_bulk_to_hkl = np.linalg.inv(r_hkl_to_bulk)

    surf = CrystalEngine.calculate_surface_projection(
        np.asarray(bz_data["points"]), structure, hkl[0], hkl[1], hkl[2]
    )
    if not surf or surf.get("projected_bounds") is None:
        return None

    bounds_bulk = np.asarray(surf["projected_bounds"], dtype=float)
    bounds_sample = bounds_bulk @ r_bulk_to_hkl.T
    kx = bounds_sample[:, 0]
    ky = bounds_sample[:, 1]
    # Close the loop
    if len(kx) and (kx[0] != kx[-1] or ky[0] != ky[-1]):
        kx = np.append(kx, kx[0])
        ky = np.append(ky, ky[0])

    return {
        "kx": [float(v) for v in kx],
        "ky": [float(v) for v in ky],
        "hkl": list(hkl),
        "n_vertices": int(len(kx) - 1),
    }


def convert_hv_to_kz(
    tensor: TensorData,
    *,
    photon_axis: int,
    work_function: float = 4.5,
    inner_potential: float = 15.0,
    theta_deg: float = 0.0,
    binding_ref: float = 0.0,
    include_photon_momentum: bool = False,
    photon_incidence_angle: float = 45.0,
) -> TensorData:
    """
    Relabel a photon-energy axis as kz (free-electron final state).

    Uses E_kin = hv − Φ − E_B(ref) at each hv point (default E_B=0 at EF).
    Intensity is not resampled — coordinate transform for Vo tuning.
    """
    if photon_axis is None or not 0 <= photon_axis < tensor.ndim:
        raise ValueError("photon_axis is required for hv → kz conversion.")

    hv = np.asarray(tensor.axes[photon_axis], dtype=float)
    e_kin = np.maximum(hv - float(work_function) - float(binding_ref), 1e-6)
    kz = np.asarray(
        ARPESKinematics.calculate_kz(
            e_kin,
            float(theta_deg),
            float(inner_potential),
            include_photon_momentum=include_photon_momentum,
            photon_energy=hv if include_photon_momentum else None,
            photon_incidence_angle=float(photon_incidence_angle),
        ),
        dtype=float,
    )

    new_axes = [np.asarray(ax, dtype=float).copy() for ax in tensor.axes]
    new_labels = list(tensor.labels)
    new_units = list(tensor.units)
    new_axes[photon_axis] = kz
    new_labels[photon_axis] = "kz"
    new_units[photon_axis] = "1/A"

    metadata = dict(tensor.metadata or {})
    metadata.update(
        {
            "Processed": "hv_kz",
            "Inner_Potential_eV": float(inner_potential),
            "Work_Function_eV": float(work_function),
            "Theta_Deg": float(theta_deg),
            "Binding_Ref_eV": float(binding_ref),
            "Photon_Momentum_Correction": bool(include_photon_momentum),
            "Photon_Incidence_Deg": float(photon_incidence_angle),
            "Photon_Axis": int(photon_axis),
            "kz_min": float(np.nanmin(kz)),
            "kz_max": float(np.nanmax(kz)),
            "Source_DataType": tensor.data_type,
        }
    )

    return TensorData(
        value=np.asarray(tensor.value, dtype=float).copy(),
        axes=new_axes,
        labels=new_labels,
        units=new_units,
        data_type=f"{tensor.data_type} (kz)",
        metadata=metadata,
    )


def perpendicular_bz_guides(
    structure,
    h: int = 0,
    k: int = 0,
    l: int = 1,
    n_zones: int = 4,
) -> Dict[str, Any]:
    """
    kz zone-boundary guides along the surface normal.

    Edges sit at n · |G_hkl| / 2 (Å⁻¹), i.e. ±π/d, ±2π/d, … for the chosen HKL.
    """
    hkl = (int(h), int(k), int(l))
    if hkl == (0, 0, 0):
        raise ValueError("HKL cannot be (0,0,0).")
    d_hkl = float(structure.lattice.d_hkl(hkl))
    if d_hkl <= 0:
        raise ValueError("Invalid d_hkl for the requested plane.")
    g_mag = 2.0 * np.pi / d_hkl
    half = g_mag / 2.0
    lines = [float(n * half) for n in range(-int(n_zones), int(n_zones) + 1) if n != 0]
    return {
        "kz_lines": lines,
        "half_g": float(half),
        "g_mag": float(g_mag),
        "d_hkl": d_hkl,
        "hkl": list(hkl),
    }
