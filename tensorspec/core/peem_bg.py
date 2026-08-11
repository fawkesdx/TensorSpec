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
    if n < 1:
        raise ValueError("ensemble n must be at least 1")
    if delta < 0:
        raise ValueError("ensemble delta must be non-negative")

    energy = np.asarray(energy, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    e_min, e_max = float(energy.min()), float(energy.max())
    rng = np.random.default_rng(seed)

    bg_samples: list[np.ndarray] = []
    sub_samples: list[np.ndarray] = []
    for _ in range(n):
        e0_j = float(np.clip(rng.uniform(e0 - delta, e0 + delta), e_min, e_max))
        e1_j = float(np.clip(rng.uniform(e1 - delta, e1 + delta), e_min, e_max))
        try:
            fit = fit_linear_preedge(energy, spectrum, e0_j, e1_j)
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

    attrs: dict[str, Any] = {
        "slope": float(fit["slope"]),
        "intercept": float(fit["intercept"]),
        "e0": float(e0),
        "e1": float(e1),
        "energy_source": energy_source,
        "source_node": source_node,
        "channel": int(channel),
        "use_roi": bool(use_roi),
        "ensemble_n_valid": int(ensemble["n_valid"]),
        "seed": int(seed),
    }
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
