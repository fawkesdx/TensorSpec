"""SC / CDW gap fitting on ARPES EDCs (Dynes DOS × FD × resolution).

Results go to ``/analysis/gap_fit`` for a single curve or a momentum stack.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

from tensorspec.core.arpes_peakfit import (
    K_B,
    extract_plane_line,
    fwhm_to_gaussian_sigma,
    fermi_dirac,
)

ML_SCHEMA_VERSION = "gap_fit_v1"


def dynes_dos(energy: np.ndarray, delta: float, gamma: float) -> np.ndarray:
    """Dynes density of states (coherence peaks + in-gap states)."""
    d = max(float(delta), 0.0)
    g = max(float(gamma), 1e-6)
    z = np.asarray(energy, dtype=complex) - 1j * g
    dos = np.real(z / np.sqrt(z * z - d * d))
    return np.maximum(dos, 0.0)


def gap_model(
    energy: np.ndarray,
    *,
    amplitude: float,
    delta: float,
    gamma: float,
    temperature: float,
    mu: float,
    bg0: float = 0.0,
    bg1: float = 0.0,
    analyzer_fwhm: float = 0.0,
) -> np.ndarray:
    """I(E) = A · Dynes(E) · FD(E) (+ linear bg), optionally Gauss-broadened."""
    e = np.asarray(energy, dtype=float)
    spectrum = (
        float(amplitude) * dynes_dos(e - float(mu), delta, gamma) * fermi_dirac(e, mu, temperature)
        + float(bg0)
        + float(bg1) * (e - float(mu))
    )
    if analyzer_fwhm and analyzer_fwhm > 0:
        de = float(np.median(np.diff(e))) if e.size > 1 else 0.0
        if de != 0:
            sigma_eV = fwhm_to_gaussian_sigma(analyzer_fwhm)
            sigma_bins = abs(sigma_eV / de)
            if sigma_bins > 0.05:
                spectrum = gaussian_filter1d(spectrum, sigma_bins, mode="nearest")
    return spectrum


def suggest_gap_seeds(energy: np.ndarray, values: np.ndarray) -> Dict[str, float]:
    """Rough Δ from coherence-peak half-separation; Γ from peak width."""
    e = np.asarray(energy, dtype=float)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(e) & np.isfinite(y)
    e, y = e[mask], y[mask]
    if e.size < 5:
        return {"amplitude": 1.0, "delta": 0.01, "gamma": 0.005, "bg0": 0.0, "bg1": 0.0}
    # Prefer peaks below EF (occupied)
    occupied = e <= 0
    if occupied.sum() >= 5:
        e_use, y_use = e[occupied], y[occupied]
    else:
        e_use, y_use = e, y
    i_peak = int(np.argmax(y_use))
    delta0 = abs(float(e_use[i_peak]))
    if delta0 < 1e-4:
        delta0 = 0.01
    # Local width estimate
    half = 0.5 * (y_use[i_peak] + float(np.nanmin(y_use)))
    above = np.where(y_use >= half)[0]
    if above.size >= 2:
        gamma0 = 0.5 * abs(float(e_use[above[-1]] - e_use[above[0]]))
    else:
        gamma0 = 0.3 * delta0
    return {
        "amplitude": float(max(y_use[i_peak] - np.nanmin(y_use), 1e-6)),
        "delta": float(delta0),
        "gamma": float(max(gamma0, 1e-4)),
        "bg0": float(np.nanmin(y)),
        "bg1": 0.0,
    }


def fit_gap_curve(
    energy: np.ndarray,
    values: np.ndarray,
    *,
    gap_type: str = "sc",
    temperature: float = 10.0,
    mu: float = 0.0,
    analyzer_fwhm: float = 0.0,
    seeds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Fit one EDC with Dynes × FD (± resolution)."""
    e = np.asarray(energy, dtype=float)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(e) & np.isfinite(y)
    e, y = e[mask], y[mask]
    if e.size < 8:
        raise ValueError("Need at least 8 finite samples for a gap fit.")
    gap_type = gap_type.lower().strip()
    if gap_type not in ("sc", "cdw", "dynes"):
        raise ValueError("gap_type must be 'sc' or 'cdw'.")

    seeds = dict(seeds or suggest_gap_seeds(e, y))
    # params: amp, delta, gamma, bg0, bg1
    p0 = np.array(
        [
            max(seeds.get("amplitude", 1.0), 1e-9),
            max(seeds.get("delta", 0.01), 1e-6),
            max(seeds.get("gamma", 0.005), 1e-6),
            float(seeds.get("bg0", 0.0)),
            float(seeds.get("bg1", 0.0)),
        ],
        dtype=float,
    )
    e_span = float(np.ptp(e)) or 1.0
    lo = np.array([1e-12, 1e-6, 1e-6, -np.inf, -np.inf])
    hi = np.array([np.inf, max(e_span, 0.5), max(e_span, 0.5), np.inf, np.inf])

    def residual(p):
        return (
            gap_model(
                e,
                amplitude=p[0],
                delta=p[1],
                gamma=p[2],
                temperature=temperature,
                mu=mu,
                bg0=p[3],
                bg1=p[4],
                analyzer_fwhm=analyzer_fwhm,
            )
            - y
        )

    result = least_squares(residual, p0, bounds=(lo, hi), max_nfev=4000)
    y_fit = gap_model(
        e,
        amplitude=result.x[0],
        delta=result.x[1],
        gamma=result.x[2],
        temperature=temperature,
        mu=mu,
        bg0=result.x[3],
        bg1=result.x[4],
        analyzer_fwhm=analyzer_fwhm,
    )
    resid = y - y_fit
    chi2 = float(np.sum(resid**2) / max(len(y) - len(result.x), 1))
    return {
        "x": e.tolist(),
        "y": y.tolist(),
        "y_fit": y_fit.tolist(),
        "residual": resid.tolist(),
        "gap_type": "cdw" if gap_type == "cdw" else "sc",
        "delta": float(result.x[1]),
        "gamma": float(result.x[2]),
        "amplitude": float(result.x[0]),
        "background": {"offset": float(result.x[3]), "slope": float(result.x[4])},
        "temperature": float(temperature),
        "mu": float(mu),
        "analyzer_fwhm": float(analyzer_fwhm),
        "chi2": chi2,
        "success": bool(result.success),
        "message": str(result.message),
        "ml_schema_version": ML_SCHEMA_VERSION,
        "seeds_used": seeds,
    }


def fit_gap_stack(
    plane_yx: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    *,
    gap_type: str = "sc",
    temperature: float = 10.0,
    mu: float = 0.0,
    analyzer_fwhm: float = 0.0,
    half_width: int = 0,
    scan_indices: Optional[Sequence[int]] = None,
    propagate_seeds: bool = True,
    seeds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Fit Dynes gap on every EDC (vs momentum) of a cut."""
    plane = np.asarray(plane_yx, dtype=float)
    x_axis = np.asarray(x_axis, dtype=float)
    y_axis = np.asarray(y_axis, dtype=float)
    n_scan = plane.shape[1]
    if scan_indices is None:
        scan_indices = list(range(n_scan))
    else:
        scan_indices = [int(i) for i in scan_indices]

    n = len(scan_indices)
    scan_vals = np.full(n, np.nan)
    delta = np.full(n, np.nan)
    gamma = np.full(n, np.nan)
    amp = np.full(n, np.nan)
    chi2 = np.full(n, np.nan)
    ok = np.zeros(n, dtype=np.int8)
    current = dict(seeds) if seeds else None

    for row, idx in enumerate(scan_indices):
        line = extract_plane_line(
            plane, x_axis, y_axis, mode="edc", index=idx, half_width=half_width
        )
        try:
            fit = fit_gap_curve(
                line["axis"],
                line["values"],
                gap_type=gap_type,
                temperature=temperature,
                mu=mu,
                analyzer_fwhm=analyzer_fwhm,
                seeds=current,
            )
        except Exception:
            scan_vals[row] = line["scan_value"]
            continue
        scan_vals[row] = line["scan_value"]
        delta[row] = fit["delta"]
        gamma[row] = fit["gamma"]
        amp[row] = fit["amplitude"]
        chi2[row] = fit["chi2"]
        ok[row] = 1 if fit["success"] else 0
        if propagate_seeds:
            current = {
                "amplitude": fit["amplitude"],
                "delta": fit["delta"],
                "gamma": fit["gamma"],
                "bg0": fit["background"]["offset"],
                "bg1": fit["background"]["slope"],
            }

    return {
        "mode": "edc",
        "gap_type": "cdw" if gap_type.lower().strip() == "cdw" else "sc",
        "scan_coord_name": "momentum",
        "scan": scan_vals,
        "delta": delta,
        "gamma": gamma,
        "amplitude": amp,
        "chi2": chi2,
        "success": ok,
        "temperature": float(temperature),
        "mu": float(mu),
        "analyzer_fwhm": float(analyzer_fwhm),
        "half_width": int(half_width),
        "ml_schema_version": ML_SCHEMA_VERSION,
        "n_points": n,
    }


def gap_stack_to_xarray(stack: Dict[str, Any]):
    import xarray as xr

    scan_name = stack["scan_coord_name"]
    return xr.Dataset(
        data_vars={
            "delta": ((scan_name,), stack["delta"]),
            "gamma": ((scan_name,), stack["gamma"]),
            "amplitude": ((scan_name,), stack["amplitude"]),
            "chi2": ((scan_name,), stack["chi2"]),
            "success": ((scan_name,), stack["success"].astype(np.int8)),
        },
        coords={scan_name: stack["scan"]},
        attrs={
            "mode": stack["mode"],
            "gap_type": stack["gap_type"],
            "temperature": stack["temperature"],
            "mu": stack["mu"],
            "analyzer_fwhm": stack["analyzer_fwhm"],
            "half_width": stack["half_width"],
            "ml_schema_version": stack["ml_schema_version"],
            "usable_for_ml": True,
            "usable_for_tb_feedback": True,
        },
    )
