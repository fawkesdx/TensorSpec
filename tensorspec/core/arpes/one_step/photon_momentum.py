"""Photon wavevector in the Chinook/Grizzly lab frame (Approach C soft X-ray).

Lab axes match ``compute_A_lab`` / Fresnel:

- Incidence plane: lab ``x``–``y``.
- Surface normal toward vacuum: lab ``+y``.
- LH p-pol: ``A = [cos α, -sin α, 0]``.

Photon propagation toward the sample (into −y) and ⊥ ``A``:

    â_beam = [-sin α, -cos α, 0]

so ``q = (hv / ħc) * â_beam`` with ``ħc = ARPESKinematics.HBAR_C``.

Sign for later mesh wiring (Task 3): crystal / bulk photoelectron momentum
conserves photon-in as ``K_crystal = K_pe - R(q_lab)`` (rotate ``q_lab`` with
the same maps as ``K`` / ``A``).
"""

from __future__ import annotations

import numpy as np

try:
    from tensorspec.core.kinematics import ARPESKinematics

    _HBAR_C = float(ARPESKinematics.HBAR_C)
except ImportError:
    # Keep synced with tensorspec.core.kinematics.ARPESKinematics.HBAR_C
    # (remote runner uploads this file without the tensorspec package).
    _HBAR_C = 1973.269804

__all__ = ["photon_q_lab"]


def photon_q_lab(hv_eV: float, incidence_deg: float) -> np.ndarray:
    """Return photon wavevector ``q`` in lab coordinates, shape ``(3,)``.

    Magnitude ``|q| = hv / HBAR_C`` (Å⁻¹). Direction = beam toward sample in
    the ``x``–``y`` incidence plane (see module docstring).
    """
    alpha = np.radians(float(incidence_deg))
    q_mag = float(hv_eV) / _HBAR_C
    return q_mag * np.array([-np.sin(alpha), -np.cos(alpha), 0.0], dtype=float)
