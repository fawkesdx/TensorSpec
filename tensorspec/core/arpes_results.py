"""Quasiparticle result curves from EDC/MDC peak tables.

Builds δE(E), dispersion E(k), k_F, m*/v_F, and integrated-intensity traces
from ``/analysis/{mdc,edc}_peakfit``, then fits FL / MFL self-energy models
to the linewidth vs energy. Outputs are shaped for ``/analysis/qp_results``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy.optimize import curve_fit

ML_SCHEMA_VERSION = "qp_results_v1"
# ħ² / (2 m_e) in eV·Å² → convert (k in 1/Å, E in eV) to m*/m_e
HBAR2_OVER_2ME = 3.81


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(arrays[0].shape[0], dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def extract_peak_series(
    scan: np.ndarray,
    center: np.ndarray,
    width: np.ndarray,
    integrated: np.ndarray,
    success: Optional[np.ndarray] = None,
    *,
    peak: int = 0,
) -> Dict[str, np.ndarray]:
    """Pull one peak column from a stack table into 1D series."""
    scan = np.asarray(scan, dtype=float)
    if center.ndim == 1:
        c = np.asarray(center, dtype=float)
        w = np.asarray(width, dtype=float)
        integ = np.asarray(integrated, dtype=float)
    else:
        p = int(peak)
        c = np.asarray(center[:, p], dtype=float)
        w = np.asarray(width[:, p], dtype=float)
        integ = np.asarray(integrated[:, p], dtype=float)
    if success is None:
        ok = np.ones(scan.shape[0], dtype=bool)
    else:
        ok = np.asarray(success).astype(bool)
    mask = ok & _finite_mask(scan, c, w, integ)
    return {
        "scan": scan[mask],
        "center": c[mask],
        "width": w[mask],
        "integrated": integ[mask],
        "n_points": int(mask.sum()),
    }


def mdc_dispersion(series: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """MDC stack: scan=energy, center=k → dispersion points (k, E)."""
    return {
        "k": np.asarray(series["center"], dtype=float),
        "energy": np.asarray(series["scan"], dtype=float),
        "width": np.asarray(series["width"], dtype=float),
        "integrated": np.asarray(series["integrated"], dtype=float),
    }


def edc_dispersion(series: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """EDC stack: scan=momentum, center=E → dispersion points (k, E)."""
    return {
        "k": np.asarray(series["scan"], dtype=float),
        "energy": np.asarray(series["center"], dtype=float),
        "width": np.asarray(series["width"], dtype=float),
        "integrated": np.asarray(series["integrated"], dtype=float),
    }


def find_k_fermi(
    k: np.ndarray,
    energy: np.ndarray,
    *,
    e_fermi: float = 0.0,
) -> Dict[str, Any]:
    """Interpolate k where the dispersion crosses ``e_fermi`` (linear segments)."""
    k = np.asarray(k, dtype=float)
    energy = np.asarray(energy, dtype=float)
    order = np.argsort(energy)
    e = energy[order]
    kk = k[order]
    crossings = []
    for i in range(len(e) - 1):
        e0, e1 = e[i], e[i + 1]
        if (e0 - e_fermi) * (e1 - e_fermi) > 0:
            continue
        if e1 == e0:
            continue
        t = (e_fermi - e0) / (e1 - e0)
        crossings.append(float(kk[i] + t * (kk[i + 1] - kk[i])))
    if not crossings:
        # Nearest-energy fallback
        j = int(np.argmin(np.abs(e - e_fermi)))
        return {
            "k_fermi": [float(kk[j])],
            "e_fermi": float(e_fermi),
            "method": "nearest",
            "n_crossings": 0,
        }
    return {
        "k_fermi": crossings,
        "e_fermi": float(e_fermi),
        "method": "interpolate",
        "n_crossings": len(crossings),
    }


def fit_parabolic_mass(
    k: np.ndarray,
    energy: np.ndarray,
    *,
    k0: Optional[float] = None,
    e_window: Optional[tuple[float, float]] = None,
) -> Dict[str, Any]:
    """Fit E = E0 + (ħ²/2m*) (k - k0)². Returns m*/m_e."""
    k = np.asarray(k, dtype=float)
    energy = np.asarray(energy, dtype=float)
    mask = _finite_mask(k, energy)
    if e_window is not None:
        lo, hi = e_window
        mask &= (energy >= lo) & (energy <= hi)
    k, energy = k[mask], energy[mask]
    if k.size < 4:
        raise ValueError("Need at least 4 points for parabolic m* fit.")
    if k0 is None:
        # Band extremum: energy farthest below EF (binding) or min |E|
        k0 = float(k[int(np.argmin(energy))])

    def model(kk, e0, inv_mass):
        return e0 + HBAR2_OVER_2ME * inv_mass * (kk - k0) ** 2

    p0 = [float(np.min(energy)), 1.0]
    popt, pcov = curve_fit(model, k, energy, p0=p0, maxfev=5000)
    e0, inv_mass = float(popt[0]), float(popt[1])
    if abs(inv_mass) < 1e-12:
        raise ValueError("Degenerate parabolic curvature.")
    mstar = 1.0 / inv_mass
    yfit = model(k, e0, inv_mass)
    resid = energy - yfit
    chi2 = float(np.sum(resid**2) / max(len(energy) - 2, 1))
    return {
        "model": "parabolic",
        "k0": float(k0),
        "E0": e0,
        "m_star_over_m_e": mstar,
        "chi2": chi2,
        "k": k.tolist(),
        "energy": energy.tolist(),
        "energy_fit": yfit.tolist(),
        "cov_diag": np.diag(pcov).tolist() if pcov is not None else None,
    }


def fit_fermi_velocity(
    k: np.ndarray,
    energy: np.ndarray,
    *,
    e_fermi: float = 0.0,
    e_window: float = 0.05,
    k_fermi: Optional[float] = None,
) -> Dict[str, Any]:
    """Linear fit E ≈ E_F + ħ v_F (k - k_F) near the Fermi level.

    Returns ``v_F`` in eV·Å (multiply by 1.519e6 for m/s if needed).
    """
    k = np.asarray(k, dtype=float)
    energy = np.asarray(energy, dtype=float)
    mask = _finite_mask(k, energy) & (np.abs(energy - e_fermi) <= e_window)
    k, energy = k[mask], energy[mask]
    if k.size < 3:
        raise ValueError("Need at least 3 points near E_F for v_F fit.")
    if k_fermi is None:
        kf = find_k_fermi(k, energy, e_fermi=e_fermi)
        k_fermi = float(kf["k_fermi"][0])

    def model(kk, vf):
        return e_fermi + vf * (kk - k_fermi)

    popt, pcov = curve_fit(model, k, energy, p0=[1.0], maxfev=2000)
    vf = float(popt[0])
    yfit = model(k, vf)
    resid = energy - yfit
    chi2 = float(np.sum(resid**2) / max(len(energy) - 1, 1))
    return {
        "model": "linear_fermi",
        "k_fermi": float(k_fermi),
        "e_fermi": float(e_fermi),
        "v_F_eV_A": vf,
        "chi2": chi2,
        "k": k.tolist(),
        "energy": energy.tolist(),
        "energy_fit": yfit.tolist(),
    }


def _fl_model(omega: np.ndarray, gamma0: float, alpha: float) -> np.ndarray:
    return gamma0 + alpha * np.asarray(omega, dtype=float) ** 2


def _mfl_model(omega: np.ndarray, gamma0: float, alpha: float) -> np.ndarray:
    return gamma0 + alpha * np.abs(np.asarray(omega, dtype=float))


def fit_self_energy(
    energy: np.ndarray,
    width: np.ndarray,
    *,
    model: str = "fl",
    e_fermi: float = 0.0,
    e_min: Optional[float] = None,
    e_max: Optional[float] = None,
) -> Dict[str, Any]:
    """Fit linewidth Γ(ω) with FL (Γ0 + α ω²) or MFL (Γ0 + α |ω|).

    ``energy`` is the scan energy (binding or kinetic); ω = energy - e_fermi.
    ``width`` is the Lorentzian γ (HWHM) from peakfit.
    """
    model = model.lower().strip()
    if model not in ("fl", "mfl", "fermi_liquid", "marginal"):
        raise ValueError("model must be 'fl' or 'mfl'.")
    is_fl = model in ("fl", "fermi_liquid")
    energy = np.asarray(energy, dtype=float)
    width = np.asarray(width, dtype=float)
    omega = energy - float(e_fermi)
    mask = _finite_mask(omega, width) & (width > 0)
    if e_min is not None:
        mask &= energy >= e_min
    if e_max is not None:
        mask &= energy <= e_max
    omega, width, energy = omega[mask], width[mask], energy[mask]
    if omega.size < 3:
        raise ValueError("Need at least 3 finite linewidth points for FL/MFL fit.")

    fn = _fl_model if is_fl else _mfl_model
    p0 = [float(np.min(width)), 1.0]
    bounds = ([0.0, 0.0], [np.inf, np.inf])
    popt, pcov = curve_fit(fn, omega, width, p0=p0, bounds=bounds, maxfev=5000)
    gamma0, alpha = float(popt[0]), float(popt[1])
    yfit = fn(omega, gamma0, alpha)
    resid = width - yfit
    chi2 = float(np.sum(resid**2) / max(len(width) - 2, 1))
    return {
        "model": "fl" if is_fl else "mfl",
        "formula": "Γ = Γ0 + α ω²" if is_fl else "Γ = Γ0 + α |ω|",
        "gamma0": gamma0,
        "alpha": alpha,
        "e_fermi": float(e_fermi),
        "chi2": chi2,
        "energy": energy.tolist(),
        "omega": omega.tolist(),
        "width": width.tolist(),
        "width_fit": yfit.tolist(),
        "n_points": int(omega.size),
    }


def build_qp_results(
    *,
    mode: str,
    scan: Sequence[float],
    center: np.ndarray,
    width: np.ndarray,
    integrated: np.ndarray,
    success: Optional[Sequence[bool]] = None,
    peak: int = 0,
    e_fermi: float = 0.0,
    fit_mass: bool = True,
    fit_vf: bool = True,
    se_model: Optional[str] = "fl",
    se_e_min: Optional[float] = None,
    se_e_max: Optional[float] = None,
    mass_e_window: Optional[tuple[float, float]] = None,
    vf_e_window: float = 0.08,
) -> Dict[str, Any]:
    """Full QP summary from one peak column of a peakfit stack."""
    mode = mode.lower().strip()
    series = extract_peak_series(
        np.asarray(scan, dtype=float),
        np.asarray(center, dtype=float),
        np.asarray(width, dtype=float),
        np.asarray(integrated, dtype=float),
        np.asarray(success) if success is not None else None,
        peak=peak,
    )
    if series["n_points"] < 3:
        raise ValueError("Too few successful peakfit points for QP results.")

    disp = mdc_dispersion(series) if mode == "mdc" else edc_dispersion(series)
    k = disp["k"]
    energy = disp["energy"]
    gamma = disp["width"]
    integ = disp["integrated"]

    # Sort by energy for plotting δE–E and intensity–E
    order = np.argsort(energy)
    e_sorted = energy[order]
    gamma_sorted = gamma[order]
    integ_sorted = integ[order]
    k_sorted = k[order]

    kf = find_k_fermi(k, energy, e_fermi=e_fermi)
    out: Dict[str, Any] = {
        "mode": mode,
        "peak": int(peak),
        "ml_schema_version": ML_SCHEMA_VERSION,
        "e_fermi": float(e_fermi),
        "dispersion": {
            "k": k_sorted.tolist(),
            "energy": e_sorted.tolist(),
        },
        "delta_e": {
            "energy": e_sorted.tolist(),
            "width": gamma_sorted.tolist(),
            "note": "Lorentzian γ (HWHM) vs energy; proxy for |Im Σ|",
        },
        "integrated_intensity": {
            "energy": e_sorted.tolist(),
            "integrated": integ_sorted.tolist(),
            "note": "Peak-component integrated intensity vs energy",
        },
        "k_fermi": kf,
    }

    if fit_mass:
        try:
            out["effective_mass"] = fit_parabolic_mass(
                k, energy, e_window=mass_e_window
            )
        except Exception as exc:
            out["effective_mass"] = {"error": str(exc)}

    if fit_vf:
        try:
            k_ref = kf["k_fermi"][0] if kf["k_fermi"] else None
            out["fermi_velocity"] = fit_fermi_velocity(
                k, energy, e_fermi=e_fermi, e_window=vf_e_window, k_fermi=k_ref
            )
        except Exception as exc:
            out["fermi_velocity"] = {"error": str(exc)}

    if se_model:
        try:
            # Self-energy vs binding uses the scan energy for MDC (natural);
            # for EDC use peak center energy already in `energy`.
            out["self_energy"] = fit_self_energy(
                e_sorted,
                gamma_sorted,
                model=se_model,
                e_fermi=e_fermi,
                e_min=se_e_min,
                e_max=se_e_max,
            )
        except Exception as exc:
            out["self_energy"] = {"error": str(exc)}

    return out


def qp_results_to_xarray(results: Dict[str, Any]):
    """Persist curves + scalar fits under ``/analysis/qp_results``."""
    import xarray as xr

    disp = results["dispersion"]
    n = len(disp["energy"])
    data_vars = {
        "k": (("point",), np.asarray(disp["k"], dtype=float)),
        "energy": (("point",), np.asarray(disp["energy"], dtype=float)),
        "width": (("point",), np.asarray(results["delta_e"]["width"], dtype=float)),
        "integrated": (
            ("point",),
            np.asarray(results["integrated_intensity"]["integrated"], dtype=float),
        ),
    }
    attrs: Dict[str, Any] = {
        "mode": results["mode"],
        "peak": results["peak"],
        "e_fermi": results["e_fermi"],
        "ml_schema_version": results["ml_schema_version"],
        "k_fermi": results.get("k_fermi"),
        "usable_for_ml": True,
        "usable_for_tb_feedback": True,
    }
    for key in ("effective_mass", "fermi_velocity", "self_energy"):
        block = results.get(key)
        if isinstance(block, dict) and "error" not in block:
            # Store scalars in attrs; drop long arrays from attrs
            slim = {
                k: v
                for k, v in block.items()
                if not isinstance(v, list)
            }
            attrs[key] = slim
            if key == "self_energy" and "width_fit" in block:
                data_vars["width_fit"] = (
                    ("point",),
                    np.asarray(block["width_fit"], dtype=float),
                )
        elif isinstance(block, dict) and "error" in block:
            attrs[f"{key}_error"] = block["error"]

    return xr.Dataset(
        data_vars=data_vars,
        coords={"point": np.arange(n, dtype=int)},
        attrs=attrs,
    )
