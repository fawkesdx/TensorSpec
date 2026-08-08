"""ARPES analyzer ΔE and deflector Δk helpers (no Chinook)."""

from __future__ import annotations

import math

K_FACTOR = 0.5123  # Å^-1 / sqrt(eV)


def analyzer_delta_e(slit_mm: float, pass_energy: float) -> float:
    """Scienta-like estimate: (slit_mm / 400) * pass_energy_eV."""
    return (float(slit_mm) / 400.0) * float(pass_energy)


def total_delta_e(ana: float, beam: float = 0.0, extra: float = 0.0) -> float:
    """Combine analyzer, beamline, and extra ΔE in quadrature."""
    return math.sqrt(max(ana, 0.0) ** 2 + max(beam, 0.0) ** 2 + max(extra, 0.0) ** 2)


def deflector_dk(hv: float, work_function: float, deflector_deg: float) -> float:
    """Δk_y from deflector angle: K * sqrt(Ek) * sin(θ)."""
    ek = max(float(hv) - float(work_function), 0.0)
    return K_FACTOR * math.sqrt(ek) * math.sin(math.radians(float(deflector_deg)))
