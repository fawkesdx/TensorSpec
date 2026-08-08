"""EDC / MDC peak fitting for ARPES cuts (Lorentzian or Voigt + optional FD).

Results are shaped for ``/analysis/mdc_peakfit`` and ``/analysis/edc_peakfit``
so a later ML / TB feedback tab can consume peak tables without re-parsing UI state.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.special import voigt_profile

# Boltzmann constant in eV/K
K_B = 8.617333262145e-5
ML_SCHEMA_VERSION = "peakfit_v1"


def fermi_dirac(energy: np.ndarray, mu: float, temperature: float) -> np.ndarray:
    t = max(float(temperature), 1e-6)
    x = (np.asarray(energy, dtype=float) - float(mu)) / (K_B * t)
    # Stable evaluation
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[~pos] = 1.0 / (np.exp(x[~pos]) + 1.0)
    ex = np.exp(-x[pos])
    out[pos] = ex / (ex + 1.0)
    return out


def lorentzian(x: np.ndarray, center: float, amplitude: float, gamma: float) -> np.ndarray:
    g = max(float(gamma), 1e-9)
    return float(amplitude) * (g ** 2) / ((np.asarray(x, dtype=float) - float(center)) ** 2 + g ** 2)


def voigt(x: np.ndarray, center: float, amplitude: float, gamma: float, sigma: float) -> np.ndarray:
    g = max(float(gamma), 1e-12)
    s = max(float(sigma), 1e-12)
    return float(amplitude) * voigt_profile(np.asarray(x, dtype=float) - float(center), s, g)


def fwhm_to_gaussian_sigma(fwhm: float) -> float:
    return max(float(fwhm), 0.0) / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def parse_resolution_eV(metadata: Optional[Dict[str, Any]]) -> Optional[float]:
    if not metadata:
        return None
    for key in ("Log_Energy_Resolution", "Energy_Resolution", "Energy Resolution"):
        val = metadata.get(key)
        if val is None:
            continue
        if isinstance(val, (int, float)):
            return float(val)
        text = str(val)
        import re

        match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
        if match:
            return float(match.group(1))
    return None


def parse_temperature_K(metadata: Optional[Dict[str, Any]]) -> Optional[float]:
    if not metadata:
        return None
    for key in ("Temperature", "Log_Temperature", "Cryostat_A", "temperature"):
        val = metadata.get(key)
        if isinstance(val, (int, float)) and np.isfinite(val):
            return float(val)
        if isinstance(val, str):
            import re

            match = re.search(r"([0-9]+(?:\.[0-9]+)?)", val)
            if match:
                return float(match.group(1))
    log = metadata.get("Measurement_Log") or {}
    if isinstance(log, dict):
        for key in ("Temperature (K)", "Cryostat A (K)", "Cryostat B (K)"):
            val = log.get(key)
            if val and str(val).upper() != "N/A":
                import re

                match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(val))
                if match:
                    return float(match.group(1))
    return None


def _model_curve(
    x: np.ndarray,
    params: np.ndarray,
    *,
    n_peaks: int,
    lineshape: str,
    sigma_fixed: Optional[float],
    include_fd: bool,
    temperature: float,
    mu: float,
) -> np.ndarray:
    # params: [bg0, bg1, (center, amp, gamma[, sigma])*n_peaks]
    bg0, bg1 = params[0], params[1]
    y = bg0 + bg1 * x
    cursor = 2
    for _ in range(n_peaks):
        center = params[cursor]
        amp = params[cursor + 1]
        gamma = abs(params[cursor + 2])
        if lineshape == "voigt":
            if sigma_fixed is not None:
                sigma = sigma_fixed
                cursor += 3
            else:
                sigma = abs(params[cursor + 3])
                cursor += 4
            y = y + voigt(x, center, amp, gamma, sigma)
        else:
            y = y + lorentzian(x, center, amp, gamma)
            cursor += 3
    if include_fd:
        y = y * fermi_dirac(x, mu, temperature)
    return y


def _pack_params(
    seeds: Sequence[Dict[str, float]],
    *,
    lineshape: str,
    sigma_fixed: Optional[float],
    bg0: float,
    bg1: float,
) -> np.ndarray:
    vals = [bg0, bg1]
    for seed in seeds:
        vals.extend([seed["center"], seed["amplitude"], abs(seed["width"])])
        if lineshape == "voigt" and sigma_fixed is None:
            vals.append(abs(seed.get("sigma", seed["width"] * 0.5)))
    return np.asarray(vals, dtype=float)


def _bounds(
    seeds: Sequence[Dict[str, float]],
    x: np.ndarray,
    *,
    lineshape: str,
    sigma_fixed: Optional[float],
) -> tuple[np.ndarray, np.ndarray]:
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    span = max(x_max - x_min, 1e-6)
    y_span = 1.0
    lo = [-np.inf, -np.inf]
    hi = [np.inf, np.inf]
    for seed in seeds:
        lo.extend([x_min - 0.1 * span, 0.0, span * 1e-5])
        hi.extend([x_max + 0.1 * span, np.inf, span])
        if lineshape == "voigt" and sigma_fixed is None:
            lo.append(span * 1e-5)
            hi.append(span)
    return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)


def suggest_seeds(
    x: np.ndarray,
    y: np.ndarray,
    n_peaks: int,
    *,
    default_width: Optional[float] = None,
) -> List[Dict[str, float]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    span = float(np.nanmax(x) - np.nanmin(x)) or 1.0
    width = default_width if default_width and default_width > 0 else span * 0.03
    prominence = max(float(np.nanmax(y) - np.nanmin(y)) * 0.05, 1e-12)
    peaks, props = find_peaks(y, prominence=prominence, distance=max(3, len(x) // 40))
    if len(peaks) == 0:
        peaks = np.array([int(np.nanargmax(y))], dtype=int)
    # Strongest first
    order = np.argsort(y[peaks])[::-1]
    peaks = peaks[order][: max(1, n_peaks)]
    seeds = []
    for idx in peaks:
        seeds.append(
            {
                "center": float(x[idx]),
                "amplitude": float(max(y[idx], 1e-12)),
                "width": float(width),
            }
        )
    while len(seeds) < n_peaks:
        seeds.append(
            {
                "center": float(np.nanmean(x)),
                "amplitude": float(max(np.nanmax(y) * 0.2, 1e-12)),
                "width": float(width),
            }
        )
    return seeds[:n_peaks]


def fit_curve(
    x: np.ndarray,
    y: np.ndarray,
    seeds: Sequence[Dict[str, float]],
    *,
    lineshape: str = "lorentzian",
    analyzer_fwhm: float = 0.0,
    include_fd: bool = False,
    temperature: float = 10.0,
    mu: float = 0.0,
) -> Dict[str, Any]:
    """Fit one EDC or MDC curve."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 5:
        raise ValueError("Need at least 5 finite samples to fit.")
    if not seeds:
        raise ValueError("Provide at least one peak seed.")

    lineshape = lineshape.lower().strip()
    if lineshape not in ("lorentzian", "voigt"):
        raise ValueError("lineshape must be 'lorentzian' or 'voigt'.")

    sigma_fixed = fwhm_to_gaussian_sigma(analyzer_fwhm) if lineshape == "voigt" and analyzer_fwhm > 0 else None
    # For Voigt with no analyzer width, leave sigma free per peak.
    if lineshape == "voigt" and analyzer_fwhm <= 0:
        sigma_fixed = None

    bg0 = float(np.nanmin(y))
    bg1 = 0.0
    p0 = _pack_params(seeds, lineshape=lineshape, sigma_fixed=sigma_fixed, bg0=bg0, bg1=bg1)
    lo, hi = _bounds(seeds, x, lineshape=lineshape, sigma_fixed=sigma_fixed)

    def residual(p):
        return _model_curve(
            x,
            p,
            n_peaks=len(seeds),
            lineshape=lineshape,
            sigma_fixed=sigma_fixed,
            include_fd=include_fd and True,
            temperature=temperature,
            mu=mu,
        ) - y

    result = least_squares(residual, p0, bounds=(lo, hi), max_nfev=4000)
    y_fit = _model_curve(
        x,
        result.x,
        n_peaks=len(seeds),
        lineshape=lineshape,
        sigma_fixed=sigma_fixed,
        include_fd=include_fd,
        temperature=temperature,
        mu=mu,
    )
    resid = y - y_fit
    chi2 = float(np.sum(resid ** 2) / max(len(y) - len(result.x), 1))

    peaks_out = []
    cursor = 2
    for i in range(len(seeds)):
        center = float(result.x[cursor])
        amp = float(result.x[cursor + 1])
        gamma = float(abs(result.x[cursor + 2]))
        if lineshape == "voigt":
            if sigma_fixed is not None:
                sigma = float(sigma_fixed)
                cursor += 3
            else:
                sigma = float(abs(result.x[cursor + 3]))
                cursor += 4
            component = voigt(x, center, amp, gamma, sigma)
            width = gamma
        else:
            sigma = 0.0
            cursor += 3
            component = lorentzian(x, center, amp, gamma)
            width = gamma
        # Trapezoidal integrated intensity of the peak component (pre-FD).
        integrated = float(np.trapezoid(component, x) if hasattr(np, "trapezoid") else np.trapz(component, x))
        peaks_out.append(
            {
                "peak": i,
                "center": center,
                "amplitude": amp,
                "width": width,
                "sigma": sigma,
                "integrated": integrated,
            }
        )

    return {
        "x": x.tolist(),
        "y": y.tolist(),
        "y_fit": y_fit.tolist(),
        "residual": resid.tolist(),
        "peaks": peaks_out,
        "background": {"offset": float(result.x[0]), "slope": float(result.x[1])},
        "chi2": chi2,
        "success": bool(result.success),
        "message": str(result.message),
        "lineshape": lineshape,
        "include_fd": bool(include_fd),
        "temperature": float(temperature),
        "mu": float(mu),
        "analyzer_fwhm": float(analyzer_fwhm),
        "ml_schema_version": ML_SCHEMA_VERSION,
    }


def extract_plane_line(
    plane_yx: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    *,
    mode: str,
    index: int,
    half_width: int = 0,
) -> Dict[str, np.ndarray]:
    """
    From a display plane ``values[y, x]``, extract one EDC or MDC.

    MDC: intensity vs X at fixed Y (energy) index.
    EDC: intensity vs Y at fixed X (momentum) index.
    """
    plane = np.asarray(plane_yx, dtype=float)
    x_axis = np.asarray(x_axis, dtype=float)
    y_axis = np.asarray(y_axis, dtype=float)
    mode = mode.lower()
    if mode == "mdc":
        i0 = int(np.clip(index - half_width, 0, plane.shape[0] - 1))
        i1 = int(np.clip(index + half_width + 1, 1, plane.shape[0]))
        curve = plane[i0:i1, :].sum(axis=0)
        return {"axis": x_axis, "values": curve, "scan_value": float(y_axis[int(np.clip(index, 0, len(y_axis) - 1))])}
    if mode == "edc":
        i0 = int(np.clip(index - half_width, 0, plane.shape[1] - 1))
        i1 = int(np.clip(index + half_width + 1, 1, plane.shape[1]))
        curve = plane[:, i0:i1].sum(axis=1)
        return {"axis": y_axis, "values": curve, "scan_value": float(x_axis[int(np.clip(index, 0, len(x_axis) - 1))])}
    raise ValueError("mode must be 'mdc' or 'edc'.")


def fit_stack(
    plane_yx: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    seeds: Sequence[Dict[str, float]],
    *,
    mode: str = "mdc",
    lineshape: str = "lorentzian",
    analyzer_fwhm: float = 0.0,
    include_fd: bool = False,
    temperature: float = 10.0,
    mu: float = 0.0,
    half_width: int = 0,
    scan_indices: Optional[Sequence[int]] = None,
    propagate_seeds: bool = True,
) -> Dict[str, Any]:
    """
    Fit every MDC (along energy) or EDC (along momentum) on a 2D cut.

    Returns an ML-ready peak table keyed by scan coordinate.
    """
    mode = mode.lower()
    plane = np.asarray(plane_yx, dtype=float)
    x_axis = np.asarray(x_axis, dtype=float)
    y_axis = np.asarray(y_axis, dtype=float)
    n_scan = plane.shape[0] if mode == "mdc" else plane.shape[1]
    if scan_indices is None:
        scan_indices = list(range(n_scan))
    else:
        scan_indices = [int(i) for i in scan_indices]

    n_peaks = len(seeds)
    scan_vals = []
    centers = np.full((len(scan_indices), n_peaks), np.nan)
    amps = np.full_like(centers, np.nan)
    widths = np.full_like(centers, np.nan)
    sigmas = np.full_like(centers, np.nan)
    integrated = np.full_like(centers, np.nan)
    chi2 = np.full(len(scan_indices), np.nan)
    ok = np.zeros(len(scan_indices), dtype=bool)

    current_seeds = [dict(s) for s in seeds]
    for row, scan_index in enumerate(scan_indices):
        line = extract_plane_line(
            plane, x_axis, y_axis, mode=mode, index=scan_index, half_width=half_width
        )
        try:
            fit = fit_curve(
                line["axis"],
                line["values"],
                current_seeds,
                lineshape=lineshape,
                analyzer_fwhm=analyzer_fwhm,
                include_fd=include_fd and mode == "edc",
                temperature=temperature,
                mu=mu,
            )
        except Exception:
            scan_vals.append(line["scan_value"])
            continue
        scan_vals.append(line["scan_value"])
        chi2[row] = fit["chi2"]
        ok[row] = fit["success"]
        for peak in fit["peaks"]:
            p = peak["peak"]
            centers[row, p] = peak["center"]
            amps[row, p] = peak["amplitude"]
            widths[row, p] = peak["width"]
            sigmas[row, p] = peak["sigma"]
            integrated[row, p] = peak["integrated"]
        if propagate_seeds:
            current_seeds = [
                {
                    "center": float(centers[row, p]),
                    "amplitude": float(max(amps[row, p], 1e-12)),
                    "width": float(max(widths[row, p], 1e-9)),
                    "sigma": float(max(sigmas[row, p], 1e-9)),
                }
                for p in range(n_peaks)
                if np.isfinite(centers[row, p])
            ]
            # Keep length stable
            while len(current_seeds) < n_peaks:
                current_seeds.append(dict(seeds[len(current_seeds)]))

    scan_coord_name = "energy" if mode == "mdc" else "momentum"
    fit_axis_name = "momentum" if mode == "mdc" else "energy"
    return {
        "mode": mode,
        "lineshape": lineshape,
        "scan_coord_name": scan_coord_name,
        "fit_axis_name": fit_axis_name,
        "scan": np.asarray(scan_vals, dtype=float),
        "peak": np.arange(n_peaks, dtype=int),
        "center": centers,
        "amplitude": amps,
        "width": widths,
        "sigma": sigmas,
        "integrated": integrated,
        "chi2": chi2,
        "success": ok,
        "analyzer_fwhm": float(analyzer_fwhm),
        "temperature": float(temperature),
        "mu": float(mu),
        "include_fd": bool(include_fd and mode == "edc"),
        "half_width": int(half_width),
        "n_peaks": n_peaks,
        "ml_schema_version": ML_SCHEMA_VERSION,
        "seeds": [dict(s) for s in seeds],
    }


def stack_to_xarray(stack: Dict[str, Any]):
    """Build an xarray.Dataset for DataTree ``/analysis/{mode}_peakfit``."""
    import xarray as xr

    scan_name = stack["scan_coord_name"]
    ds = xr.Dataset(
        data_vars={
            "center": ((scan_name, "peak"), stack["center"]),
            "amplitude": ((scan_name, "peak"), stack["amplitude"]),
            "width": ((scan_name, "peak"), stack["width"]),
            "sigma": ((scan_name, "peak"), stack["sigma"]),
            "integrated": ((scan_name, "peak"), stack["integrated"]),
            "chi2": ((scan_name,), stack["chi2"]),
            "success": ((scan_name,), stack["success"].astype(np.int8)),
        },
        coords={
            scan_name: stack["scan"],
            "peak": stack["peak"],
        },
        attrs={
            "mode": stack["mode"],
            "lineshape": stack["lineshape"],
            "fit_axis_name": stack["fit_axis_name"],
            "analyzer_fwhm": stack["analyzer_fwhm"],
            "temperature": stack["temperature"],
            "mu": stack["mu"],
            "include_fd": stack["include_fd"],
            "half_width": stack["half_width"],
            "n_peaks": stack["n_peaks"],
            "ml_schema_version": stack["ml_schema_version"],
            "seeds": stack["seeds"],
            "usable_for_ml": True,
            "usable_for_tb_feedback": True,
        },
    )
    return ds
