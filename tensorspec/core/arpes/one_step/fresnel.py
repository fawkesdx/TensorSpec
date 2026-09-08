"""Fresnel transmission amplitudes for ARPES local-field correction (Approach C).

Vacuum → isotropic solid with complex refractive index ``n``. Lab frame matches
``compute_A_lab`` in ``chinook_arpes_kmesh``:

- Incidence plane: lab ``x``–``y`` (p / LH lives here).
- s / LV: lab ``z``.
- Surface-normal direction in that plane: lab ``y`` (LH uses
  ``A = [cos α, -sin α, 0]``).

Amplitude transmission (field E, not intensity), ``n_i = 1``, ``n_t = n``:

    t_s = 2 n_i cos θ_i / (n_i cos θ_i + n_t cos θ_t)
    t_p = 2 n_i cos θ_i / (n_t cos θ_i + n_i cos θ_t)

with Snell ``sin θ_t = sin θ_i / n``. Cosines use complex square-root continuation
(finite for TIR / absorbing ``n``; no hard zero-out).

Vector rebuild (v1): **vacuum-angle basis** — scale the vacuum s and p
cartesian pieces by ``t_s`` / ``t_p`` without remapping onto the refracted
propagation angle. Then ``n = 1`` is an exact identity. Phase of complex ``t``
is kept.
"""

from __future__ import annotations

import numpy as np

__all__ = ["fresnel_ts_tp", "apply_fresnel_to_A_lab"]


def fresnel_ts_tp(incidence_deg: float, n: complex) -> tuple[complex, complex]:
    """Return ``(t_s, t_p)`` for vacuum → solid with refractive index ``n``."""
    n_i = 1.0 + 0.0j
    n_t = complex(n)
    theta_i = np.radians(float(incidence_deg))
    sin_i = np.sin(theta_i)
    cos_i = np.cos(theta_i)
    # Complex continuation of Snell's law (handles TIR / complex n).
    sin_t = sin_i / n_t
    cos_t = np.sqrt(1.0 - sin_t * sin_t)

    t_s = (2.0 * n_i * cos_i) / (n_i * cos_i + n_t * cos_t)
    t_p = (2.0 * n_i * cos_i) / (n_t * cos_i + n_i * cos_t)
    return complex(t_s), complex(t_p)


def apply_fresnel_to_A_lab(
    A_vac: np.ndarray,
    incidence_deg: float,
    n: complex,
) -> np.ndarray:
    """Scale vacuum lab ``A`` by Fresnel ``t_s`` / ``t_p`` (vacuum-angle basis).

    Decomposition for this codebase's LH/LV convention:
    - p: in-plane components ``(A_x, A_y, 0)``
    - s: ``(0, 0, A_z)``
    """
    A = np.asarray(A_vac)
    t_s, t_p = fresnel_ts_tp(incidence_deg, n)

    out = np.array(
        [t_p * A[0], t_p * A[1], t_s * A[2]],
        dtype=np.result_type(A.dtype, complex),
    )
    # Preserve real dtype when inputs and n yield a real result.
    if np.isrealobj(A) and np.isreal(n) and np.isreal(out).all():
        return np.real(out).astype(A.dtype, copy=False)
    return out
