from __future__ import annotations

from typing import Any

import numpy as np

ENERGY_ALIASES = ("energy", "E", "hv", "photon_energy", "PhotonEnergy", "eV")


def resolve_energy(n_frames: int, metadata: dict) -> tuple[np.ndarray, str]:
    """Return (energy, source) where source is 'csv' or 'index'."""
    table = metadata.get("beamline_table") or {}
    series = table.get("series") or {}
    lower_map = {str(k).lower(): k for k in series}

    for alias in ENERGY_ALIASES:
        key = lower_map.get(alias.lower())
        if key is None:
            continue
        arr = np.asarray(series[key], dtype=float)
        if arr.shape == (n_frames,):
            return arr, "csv"

    return np.arange(n_frames, dtype=float), "index"


def extract_spectrum(stack: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """stack (n,y,x) → spectrum (n,). mask None = all pixels."""
    stack = np.asarray(stack, dtype=float)
    if stack.ndim != 3:
        raise ValueError("stack must have shape (n_frames, y, x)")

    if mask is None:
        return stack.mean(axis=(1, 2))

    mask = np.asarray(mask, dtype=bool)
    if mask.shape != stack.shape[1:]:
        raise ValueError("mask shape must match stack spatial dimensions (y, x)")
    if not mask.any():
        raise ValueError("empty ROI mask")

    return stack[:, mask].mean(axis=1)


def fit_linear_preedge(
    energy: np.ndarray, spectrum: np.ndarray, e0: float, e1: float
) -> dict:
    """Return slope, intercept, bg (full axis). ValueError if <2 points in window."""
    energy = np.asarray(energy, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    lo, hi = (e0, e1) if e0 <= e1 else (e1, e0)
    window = (energy >= lo) & (energy <= hi)
    if int(window.sum()) < 2:
        raise ValueError("pre-edge window must contain at least 2 points")

    slope, intercept = np.polyfit(energy[window], spectrum[window], 1)
    bg = slope * energy + intercept
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "bg": bg,
    }


def fit_two_step_pre_post(
    energy: np.ndarray,
    spectrum: np.ndarray,
    pre_e0: float,
    pre_e1: float,
    post_e0: float,
    post_e1: float,
) -> dict:
    """Return pre/post slopes, intercepts, and bg on full axis."""
    energy = np.asarray(energy, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    pre_lo, pre_hi = (pre_e0, pre_e1) if pre_e0 <= pre_e1 else (pre_e1, pre_e0)
    post_lo, post_hi = (
        (post_e0, post_e1) if post_e0 <= post_e1 else (post_e1, post_e0)
    )
    if pre_hi >= post_lo:
        raise ValueError("pre-edge end must be before post-edge start")

    pre_window = (energy >= pre_lo) & (energy <= pre_hi)
    post_window = (energy >= post_lo) & (energy <= post_hi)
    if int(pre_window.sum()) < 2:
        raise ValueError("pre-edge window must contain at least 2 points")
    if int(post_window.sum()) < 2:
        raise ValueError("post-edge window must contain at least 2 points")

    pre_slope, pre_intercept = np.polyfit(energy[pre_window], spectrum[pre_window], 1)
    post_slope, post_intercept = np.polyfit(
        energy[post_window], spectrum[post_window], 1
    )

    y_pre = pre_slope * energy + pre_intercept
    y_post = post_slope * energy + post_intercept
    y_at_pre_end = pre_slope * pre_hi + pre_intercept
    y_at_post_start = post_slope * post_lo + post_intercept

    t = (energy - pre_hi) / (post_lo - pre_hi)
    bg_connect = (1.0 - t) * y_at_pre_end + t * y_at_post_start
    bg = np.where(energy <= pre_hi, y_pre, np.where(energy >= post_lo, y_post, bg_connect))

    return {
        "pre_slope": float(pre_slope),
        "pre_intercept": float(pre_intercept),
        "post_slope": float(post_slope),
        "post_intercept": float(post_intercept),
        "bg": bg,
    }


def fit_background(
    method: str,
    energy: np.ndarray,
    spectrum: np.ndarray,
    *,
    e0: float,
    e1: float,
    post_e0: float | None = None,
    post_e1: float | None = None,
) -> dict:
    """Fit background using linear or two_step method."""
    method_norm = str(method).strip().lower()
    if method_norm == "linear":
        fit = fit_linear_preedge(energy, spectrum, e0, e1)
        return {**fit, "method": "linear"}
    if method_norm == "two_step":
        if post_e0 is None or post_e1 is None:
            raise ValueError("two_step requires post_e0 and post_e1")
        fit = fit_two_step_pre_post(energy, spectrum, e0, e1, post_e0, post_e1)
        return {**fit, "method": "two_step"}
    raise ValueError(f"unknown background method: {method!r}")


def ensemble_background(
    method: str,
    energy: np.ndarray,
    spectrum: np.ndarray,
    *,
    e0: float,
    e1: float,
    post_e0: float | None = None,
    post_e1: float | None = None,
    delta: float,
    n: int,
    seed: int = 0,
) -> dict:
    """bg_mean, bg_std, subtracted_mean, subtracted_std, n_valid."""
    if n < 1:
        raise ValueError("ensemble n must be at least 1")
    if delta < 0:
        raise ValueError("ensemble delta must be non-negative")

    energy = np.asarray(energy, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    e_min, e_max = float(energy.min()), float(energy.max())
    rng = np.random.default_rng(seed)
    method_norm = str(method).strip().lower()
    two_step = method_norm == "two_step"
    if two_step and (post_e0 is None or post_e1 is None):
        raise ValueError("two_step requires post_e0 and post_e1")

    bg_samples: list[np.ndarray] = []
    sub_samples: list[np.ndarray] = []
    for _ in range(n):
        e0_j = float(np.clip(rng.uniform(e0 - delta, e0 + delta), e_min, e_max))
        e1_j = float(np.clip(rng.uniform(e1 - delta, e1 + delta), e_min, e_max))
        fit_kwargs: dict[str, Any] = {"e0": e0_j, "e1": e1_j}
        if two_step:
            fit_kwargs["post_e0"] = float(
                np.clip(rng.uniform(post_e0 - delta, post_e0 + delta), e_min, e_max)
            )
            fit_kwargs["post_e1"] = float(
                np.clip(rng.uniform(post_e1 - delta, post_e1 + delta), e_min, e_max)
            )
        try:
            fit = fit_background(method_norm, energy, spectrum, **fit_kwargs)
        except ValueError:
            continue
        bg_samples.append(fit["bg"])
        sub_samples.append(spectrum - fit["bg"])

    if not bg_samples:
        raise ValueError("ensemble produced no valid samples")

    bg_arr = np.stack(bg_samples)
    sub_arr = np.stack(sub_samples)
    return {
        "bg_mean": bg_arr.mean(axis=0),
        "bg_std": bg_arr.std(axis=0, ddof=0),
        "subtracted_mean": sub_arr.mean(axis=0),
        "subtracted_std": sub_arr.std(axis=0, ddof=0),
        "n_valid": len(bg_samples),
    }


def ensemble_preedge(
    energy: np.ndarray,
    spectrum: np.ndarray,
    e0: float,
    e1: float,
    *,
    delta: float,
    n: int,
    seed: int = 0,
) -> dict:
    """bg_mean, bg_std, subtracted_mean, subtracted_std, n_valid."""
    return ensemble_background(
        "linear",
        energy,
        spectrum,
        e0=e0,
        e1=e1,
        delta=delta,
        n=n,
        seed=seed,
    )


def apply_bg_to_stack(stack: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """I'[..., i] = I[..., i] - bg[i]."""
    stack = np.asarray(stack, dtype=float)
    if stack.ndim != 3:
        raise ValueError("stack must have shape (n_frames, y, x)")
    bg = np.asarray(bg, dtype=float)
    if bg.shape != (stack.shape[0],):
        raise ValueError("bg length must match n_frames")
    return stack - bg.reshape(-1, 1, 1)


def is_bg_output_node(node: str) -> bool:
    """True if node is a background-subtracted processed child, not a fit source."""
    node = node.strip("/")
    if node == "processed/bg":
        return True
    if node.startswith("processed/"):
        tag = node.split("/", 1)[1]
        return bool(tag) and tag.endswith("_bg")
    return False


def bg_child_name(source_node: str) -> str:
    """Map source viewer node to processed child name."""
    node = source_node.strip("/")
    if is_bg_output_node(node):
        raise ValueError(
            f"cannot use background output as source: {source_node!r}"
        )
    if node in {"raw", "processed"}:
        return "bg"
    if node.startswith("processed/"):
        tag = node.split("/", 1)[1]
        if not tag or "/" in tag:
            raise ValueError(f"invalid processed source node: {source_node!r}")
        return f"{tag}_bg"
    raise ValueError(f"unsupported source node: {source_node!r}")


def analysis_dataset(
    energy: np.ndarray,
    spectrum: np.ndarray,
    fit: dict[str, Any],
    ensemble: dict[str, Any],
    *,
    e0: float,
    e1: float,
    post_e0: float | None = None,
    post_e1: float | None = None,
    energy_source: str,
    source_node: str,
    channel: int = 0,
    use_roi: bool = False,
    roi: dict | None = None,
    ensemble_delta: float | None = None,
    ensemble_n: int | None = None,
    seed: int = 0,
):
    """Build xarray Dataset for /analysis/background."""
    import xarray as xr

    method = str(fit.get("method", "linear"))
    attrs: dict[str, Any] = {
        "method": method,
        "e0": float(e0),
        "e1": float(e1),
        "energy_source": energy_source,
        "source_node": source_node,
        "channel": int(channel),
        "use_roi": bool(use_roi),
        "ensemble_n_valid": int(ensemble["n_valid"]),
        "seed": int(seed),
    }
    if method == "two_step":
        attrs["post_e0"] = float(post_e0 if post_e0 is not None else fit["post_e0"])
        attrs["post_e1"] = float(post_e1 if post_e1 is not None else fit["post_e1"])
        attrs["pre_slope"] = float(fit["pre_slope"])
        attrs["pre_intercept"] = float(fit["pre_intercept"])
        attrs["post_slope"] = float(fit["post_slope"])
        attrs["post_intercept"] = float(fit["post_intercept"])
    else:
        attrs["slope"] = float(fit["slope"])
        attrs["intercept"] = float(fit["intercept"])
    if ensemble_delta is not None:
        attrs["ensemble_delta"] = float(ensemble_delta)
    if ensemble_n is not None:
        attrs["ensemble_n"] = int(ensemble_n)
    if roi is not None:
        attrs["roi"] = roi

    return xr.Dataset(
        data_vars={
            "raw_spectrum": (("energy",), np.asarray(spectrum, dtype=float)),
            "bg": (("energy",), np.asarray(ensemble["bg_mean"], dtype=float)),
            "bg_std": (("energy",), np.asarray(ensemble["bg_std"], dtype=float)),
            "subtracted": (
                ("energy",),
                np.asarray(ensemble["subtracted_mean"], dtype=float),
            ),
            "subtracted_std": (
                ("energy",),
                np.asarray(ensemble["subtracted_std"], dtype=float),
            ),
        },
        coords={"energy": np.asarray(energy, dtype=float)},
        attrs=attrs,
    )
