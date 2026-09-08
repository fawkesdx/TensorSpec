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

Vector rebuild (v2): **refracted-angle basis** — project vacuum in-plane A onto
``Ê_p(θ_i)``, then rebuild ``t_p A_p Ê_p(θ_t)`` with Snell ``θ_t``. Then
``n = 1`` is an exact identity. Phase of complex ``t`` is kept.
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
    """Scale + remap vacuum lab A onto refracted p-basis (Approach C v2)."""
    A = np.asarray(A_vac)
    n_c = complex(n)
    t_s, t_p = fresnel_ts_tp(incidence_deg, n_c)

    theta_i = np.radians(float(incidence_deg))
    sin_i = np.sin(theta_i)
    cos_i = np.cos(theta_i)
    sin_t = sin_i / n_c
    cos_t = np.sqrt(1.0 - sin_t * sin_t)

    # Complex-safe projection of in-plane A onto vacuum Ê_p(θ_i).
    e_px_i = cos_i
    e_py_i = -sin_i
    A_p = A[0] * e_px_i + A[1] * e_py_i

    e_px_t = cos_t
    e_py_t = -sin_t

    out = np.array(
        [t_p * A_p * e_px_t, t_p * A_p * e_py_t, t_s * A[2]],
        dtype=np.result_type(A.dtype, complex),
    )
    if np.isrealobj(A) and np.isreal(n) and np.isreal(out).all():
        return np.real(out).astype(A.dtype, copy=False)
    return out
