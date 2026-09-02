"""1D XAS / XMCD operations via shared PEEM BG and sum-rule core."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from tensorspec.core.data_models import TensorData
from tensorspec.core.io.xas_loaders import load_xas_pair, load_xas_spectrum
from tensorspec.core.peem_bg import (
    analysis_dataset as bg_analysis_dataset,
    ensemble_background,
    fit_background,
)
from tensorspec.core.peem_sumrule import (
    analysis_dataset as sumrule_analysis_dataset,
    apply_i0,
    ensemble_sumrule,
    integrate_windows,
    moments,
    pick_source_kind,
)
from tensorspec.core.workspace import global_workspace

SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SUMRULE_PAIRS = (("CP", "CM"), ("LH", "LV"))


def _safe_label(name: str, fallback: str) -> str:
    cleaned = (name.strip() or fallback)[:64]
    if not SAFE_NAME.match(cleaned):
        raise ValueError(
            "Name must start with a letter or digit and contain only "
            "letters, digits, '.', '_' or '-'."
        )
    return cleaned


def _require_xas(name: str) -> TensorData:
    tensor = global_workspace.pull_tensor_data(name, "raw")
    if tensor is None:
        raise ValueError(f"'{name}' is not spectroscopy data in workspace.")
    if not str(tensor.data_type).startswith("Experimental XAS"):
        raise ValueError(f"'{name}' is not XAS data.")
    return tensor


def _processed_tensor(name: str) -> TensorData | None:
    tensor = global_workspace.pull_tensor_data(name, "processed")
    if tensor is None:
        return None
    ok_single = (
        tensor.labels == ["energy"]
        and tensor.value.ndim == 1
        and tensor.value.size > 0
    )
    ok_pair = (
        tensor.labels == ["channel", "energy"]
        and tensor.value.ndim == 2
        and tensor.value.shape[0] == 2
        and tensor.value.shape[1] > 0
    )
    if ok_single or ok_pair:
        return tensor
    raise ValueError("Processed XAS must be 1D (energy) or paired (channel, energy).")


def _separated_tensor(name: str, node: str) -> TensorData:
    rel = node.strip("/")
    if not rel.startswith("processed/") or rel.count("/") != 1:
        raise ValueError("Invalid separated node path.")
    tensor = global_workspace.pull_tensor_data(name, rel)
    if tensor is None:
        raise ValueError(f"No data at '{rel}'.")
    if tensor.labels != ["energy"] or tensor.value.ndim != 1:
        raise ValueError(f"Invalid 1D channel at '{rel}'.")
    return tensor


def _default_ensemble_delta(energy: np.ndarray) -> float:
    span = float(np.max(energy) - np.min(energy))
    return 0.05 * span if span > 0 else 1.0


def _energy_and_spectrum(tensor: TensorData, channel: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if tensor.labels == ["energy"]:
        return np.asarray(tensor.axes[0], dtype=float), np.asarray(tensor.value, dtype=float)
    if tensor.labels == ["channel", "energy"]:
        if not 0 <= channel < tensor.value.shape[0]:
            raise ValueError("channel index out of range")
        return (
            np.asarray(tensor.axes[1], dtype=float),
            np.asarray(tensor.value[channel], dtype=float),
        )
    raise ValueError("Expected 1D XAS tensor.")


def _pull_bg_source(name: str, node: str, channel: int) -> tuple[np.ndarray, TensorData, TensorData]:
    raw = _require_xas(name)
    node = node.strip("/")
    if node == "raw":
        energy, spectrum = _energy_and_spectrum(raw, channel)
        return spectrum, raw, raw  # stack placeholder: 1D spectrum as "stack"
    if node == "processed":
        processed = _processed_tensor(name)
        if processed is None:
            raise ValueError(f"'{name}' has no processed data.")
        energy, spectrum = _energy_and_spectrum(processed, channel)
        return spectrum, processed, raw
    if node.startswith("processed/"):
        tensor = _separated_tensor(name, node)
        energy = np.asarray(tensor.axes[0], dtype=float)
        spectrum = np.asarray(tensor.value, dtype=float)
        return spectrum, tensor, raw
    raise ValueError("node must be 'raw', 'processed', or 'processed/<tag>'.")


def _run_bg_fit(
    name: str,
    *,
    node: str = "raw",
    channel: int = 0,
    method: str = "linear",
    e0: float = 0.0,
    e1: float = 1.0,
    post_e0: float | None = None,
    post_e1: float | None = None,
    ensemble_delta: float | None = None,
    ensemble_n: int = 21,
    seed: int | None = None,
) -> dict[str, Any]:
    spectrum, source_tensor, raw = _pull_bg_source(name, node, channel)
    energy, _ = _energy_and_spectrum(source_tensor, channel)
    energy_source = "csv"
    fit = fit_background(
        method,
        energy,
        spectrum,
        e0=e0,
        e1=e1,
        post_e0=post_e0,
        post_e1=post_e1,
    )
    delta = ensemble_delta if ensemble_delta is not None else _default_ensemble_delta(energy)
    ensemble = ensemble_background(
        method,
        energy,
        spectrum,
        e0=e0,
        e1=e1,
        post_e0=post_e0,
        post_e1=post_e1,
        delta=delta,
        n=ensemble_n,
        seed=seed,
    )
    return {
        "energy": energy,
        "spectrum": spectrum,
        "fit": fit,
        "ensemble": ensemble,
        "delta": delta,
        "energy_source": energy_source,
        "source_tensor": source_tensor,
        "raw": raw,
        "node": node,
        "channel": channel,
    }


def _bg_fit_to_dict(result: dict[str, Any], *, e0: float, e1: float, post_e0, post_e1) -> dict[str, Any]:
    fit = result["fit"]
    ensemble = result["ensemble"]
    method = str(fit.get("method", "linear"))
    out: dict[str, Any] = {
        "energy": np.asarray(result["energy"], dtype=float),
        "spectrum": np.asarray(result["spectrum"], dtype=float),
        "bg": np.asarray(ensemble["bg_mean"], dtype=float),
        "subtracted": np.asarray(ensemble["subtracted_mean"], dtype=float),
        "method": method,
        "energy_source": result["energy_source"],
        "e0": float(e0),
        "e1": float(e1),
        "ensemble_n_valid": int(ensemble["n_valid"]),
    }
    if method == "two_step":
        out.update(
            post_e0=float(post_e0 if post_e0 is not None else fit["post_e0"]),
            post_e1=float(post_e1 if post_e1 is not None else fit["post_e1"]),
        )
    else:
        out.update(slope=float(fit["slope"]), intercept=float(fit["intercept"]))
    return out


def _bg_subtracted_tensor(
    source_tensor: TensorData,
    subtracted: np.ndarray,
    *,
    child_name: str,
    source_node: str,
    channel: int,
) -> TensorData:
    energy, _ = _energy_and_spectrum(source_tensor, channel)
    meta = dict(source_tensor.metadata or {})
    meta.update(
        {
            "bg_subtracted": True,
            "bg_source_node": source_node,
            "bg_channel": int(channel),
            "analysis_node": "background",
            "bg_child": child_name,
        }
    )
    return TensorData(
        value=np.asarray(subtracted, dtype=float),
        axes=[energy],
        labels=["energy"],
        units=["eV"],
        data_type="Experimental XAS (bg subtracted)",
        metadata=meta,
    )


def _bg_child_name(source_node: str) -> str:
    node = source_node.strip("/")
    if node in {"raw", "processed"}:
        return "bg"
    if node.startswith("processed/"):
        tag = node.split("/", 1)[1]
        if tag.endswith("_bg"):
            raise ValueError(f"cannot use background output as source: {source_node!r}")
        return f"{tag}_bg"
    raise ValueError(f"unsupported source node: {source_node!r}")


def _sumrule_tags(name: str) -> tuple[str, str]:
    processed = _processed_tensor(name)
    if processed is None:
        raw = _require_xas(name)
        if raw.labels == ["channel", "energy"]:
            tags = list(raw.metadata.get("channel_tags") or ["CP", "CM"])
            return (str(tags[0]), str(tags[1]))
        raise ValueError("Sum rule requires paired CP/CM or LH/LV spectra.")
    tags = list(processed.metadata.get("channel_tags") or [])
    if len(tags) >= 2:
        return (str(tags[0]), str(tags[1]))
    children = set(global_workspace.list_processed_children(name))
    for pair in _SUMRULE_PAIRS:
        if pair[0] in children and pair[1] in children:
            return pair
    if processed.labels == ["channel", "energy"]:
        return ("CP", "CM")
    raise ValueError("Cannot resolve sum-rule channel pair.")


def _sumrule_available_nodes(name: str) -> list[str]:
    nodes = [f"processed/{t}" for t in global_workspace.list_processed_children(name)]
    processed = _processed_tensor(name)
    if processed is not None and processed.labels == ["channel", "energy"]:
        nodes.append("processed")
    raw = _require_xas(name)
    if raw.labels == ["channel", "energy"]:
        nodes.append("raw")
    return nodes


def _resolve_sumrule_spectra(
    name: str,
    tags: tuple[str, str],
    source_kind: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, str], str]:
    t0, t1 = tags
    if source_kind == "bg":
        plus = _separated_tensor(name, f"processed/{t0}_bg")
        minus = _separated_tensor(name, f"processed/{t1}_bg")
    elif source_kind == "separated":
        plus = _separated_tensor(name, f"processed/{t0}")
        minus = _separated_tensor(name, f"processed/{t1}")
    else:
        node = "processed"
        tensor = global_workspace.pull_tensor_data(name, node)
        if tensor is None or tensor.labels != ["channel", "energy"]:
            tensor = _require_xas(name)
        if tensor.labels != ["channel", "energy"]:
            raise ValueError("Paired sum rule needs (channel, energy) data.")
        energy = np.asarray(tensor.axes[1], dtype=float)
        return (
            energy,
            np.asarray(tensor.value[0], dtype=float),
            np.asarray(tensor.value[1], dtype=float),
            tags,
            source_kind,
        )
    energy = np.asarray(plus.axes[0], dtype=float)
    return energy, np.asarray(plus.value, dtype=float), np.asarray(minus.value, dtype=float), tags, source_kind


def _sumrule_bg_params(name: str, energy: np.ndarray) -> tuple[str, float | None, float | None, float | None, float | None, float, int]:
    analysis = global_workspace.pull_analysis_data(name, "background")
    if analysis is None:
        return "linear", None, None, None, None, 0.0, 1
    attrs = analysis.attrs or {}
    e0, e1 = attrs.get("e0"), attrs.get("e1")
    if e0 is None or e1 is None:
        return "linear", None, None, None, None, 0.0, 1
    bg_delta = attrs.get("ensemble_delta")
    bg_delta = float(bg_delta) if bg_delta is not None else _default_ensemble_delta(energy)
    bg_n = int(attrs.get("ensemble_n", 21))
    method = str(attrs.get("method", "linear"))
    post_e0, post_e1 = attrs.get("post_e0"), attrs.get("post_e1")
    return (
        method,
        float(e0),
        float(e1),
        float(post_e0) if post_e0 is not None else None,
        float(post_e1) if post_e1 is not None else None,
        bg_delta,
        bg_n,
    )


def _run_sumrule(
    name: str,
    *,
    nh: float = 1.0,
    l3_lo: float = 0.0,
    l3_hi: float = 1.0,
    l2_lo: float = 2.0,
    l2_hi: float = 3.0,
    r_lo: float = 0.0,
    r_hi: float = 4.0,
    window_delta: float | None = None,
    window_n: int = 21,
    bg_delta: float | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    raw = _require_xas(name)
    tags = _sumrule_tags(name)
    source_kind = pick_source_kind(_sumrule_available_nodes(name), tags)
    energy, mu_plus, mu_minus, tags, source_kind = _resolve_sumrule_spectra(
        name, tags, source_kind
    )

    n_pts = energy.size
    mu_plus, i0p = apply_i0(mu_plus, (raw.metadata or {}).get("I0"))
    mu_minus, i0m = apply_i0(mu_minus, (raw.metadata or {}).get("I0"))
    i0_applied = i0p and i0m

    l3 = (l3_lo, l3_hi)
    l2 = (l2_lo, l2_hi)
    r_win = (r_lo, r_hi)
    if window_delta is None:
        window_delta = _default_ensemble_delta(energy)

    bg_method, bg_e0, bg_e1, bg_post_e0, bg_post_e1, bg_delta_default, bg_n = (
        _sumrule_bg_params(name, energy)
    )
    if source_kind == "bg":
        bg_e0, bg_e1, bg_post_e0, bg_post_e1 = None, None, None, None
    bg_delta_val = bg_delta if bg_delta is not None else bg_delta_default

    integrals = integrate_windows(energy, mu_plus, mu_minus, l3=l3, l2=l2, r_win=r_win)
    moment_vals = moments(integrals["p"], integrals["q"], integrals["r"], nh)
    ensemble = ensemble_sumrule(
        energy,
        mu_plus,
        mu_minus,
        l3=l3,
        l2=l2,
        r_win=r_win,
        nh=nh,
        window_delta=window_delta,
        window_n=window_n,
        bg_e0=bg_e0,
        bg_e1=bg_e1,
        bg_method=bg_method,
        bg_post_e0=bg_post_e0,
        bg_post_e1=bg_post_e1,
        bg_delta=bg_delta_val,
        bg_n=bg_n,
        seed=seed,
    )

    return {
        "energy": energy,
        "mu_plus": mu_plus,
        "mu_minus": mu_minus,
        "i0_applied": i0_applied,
        "source_kind": source_kind,
        "tags": tags,
        "integrals": integrals,
        "moment_vals": moment_vals,
        "ensemble": ensemble,
        "window_delta": window_delta,
        "bg_e0": bg_e0,
        "bg_e1": bg_e1,
        "bg_method": bg_method,
        "bg_post_e0": bg_post_e0,
        "bg_post_e1": bg_post_e1,
        "bg_delta": bg_delta_val,
        "bg_n": bg_n,
        "nh": nh,
        "l3_lo": l3_lo,
        "l3_hi": l3_hi,
        "l2_lo": l2_lo,
        "l2_hi": l2_hi,
        "r_lo": r_lo,
        "r_hi": r_hi,
        "n_points": n_pts,
    }


def _sumrule_to_dict(result: dict[str, Any]) -> dict[str, Any]:
    ens = result["ensemble"]
    d_mu = result["mu_plus"] - result["mu_minus"]
    return {
        "energy": np.asarray(result["energy"], dtype=float),
        "mu_plus": np.asarray(result["mu_plus"], dtype=float),
        "mu_minus": np.asarray(result["mu_minus"], dtype=float),
        "dichroism": np.asarray(d_mu, dtype=float),
        "nh": float(result["nh"]),
        "l3_lo": float(result["l3_lo"]),
        "l3_hi": float(result["l3_hi"]),
        "l2_lo": float(result["l2_lo"]),
        "l2_hi": float(result["l2_hi"]),
        "r_lo": float(result["r_lo"]),
        "r_hi": float(result["r_hi"]),
        "p": float(ens["p_mean"]),
        "q": float(ens["q_mean"]),
        "r": float(ens["r_mean"]),
        "p_std": float(ens["p_std"]),
        "q_std": float(ens["q_std"]),
        "r_std": float(ens["r_std"]),
        "m_orb": float(ens["m_orb_mean"]),
        "m_orb_std": float(ens["m_orb_std"]),
        "m_spin_plus_dipole": float(ens["m_spin_plus_dipole_mean"]),
        "m_spin_plus_dipole_std": float(ens["m_spin_plus_dipole_std"]),
        "i0_applied": result["i0_applied"],
        "source_kind": result["source_kind"],
        "tag_plus": result["tags"][0],
        "tag_minus": result["tags"][1],
        "energy_source": "csv",
        "ensemble_n_valid": int(ens["n_valid"]),
        "ensemble_n_valid_bg": int(ens.get("n_valid_bg", 0)),
    }


class XasService:
    """Qt-facing 1D XAS using shared PEEM BG / sum-rule engines."""

    def load_path(self, path: str | Path, name: str | None = None) -> dict[str, Any]:
        path = Path(path)
        tensor = load_xas_spectrum(path)
        label = _safe_label(name or "", path.stem)
        global_workspace.push_spectroscopy_data(label, tensor)
        if tensor.labels == ["channel", "energy"]:
            global_workspace.write_processed_data(label, tensor)
            tags = tensor.metadata.get("channel_tags", [])
            for ch, tag in enumerate(tags):
                child = TensorData(
                    value=np.asarray(tensor.value[ch], dtype=float),
                    axes=[np.asarray(tensor.axes[1], dtype=float)],
                    labels=["energy"],
                    units=["eV"],
                    data_type=f"Experimental XAS ({tag})",
                    metadata=dict(tensor.metadata or {}, channel_tag=tag),
                )
                global_workspace.write_processed_child_data(label, str(tag), child)
        return self.get_meta(label)

    def load_pair(
        self,
        plus_path: str | Path,
        minus_path: str | Path,
        *,
        tag_plus: str = "CP",
        tag_minus: str = "CM",
        name: str | None = None,
    ) -> dict[str, Any]:
        tensor = load_xas_pair(plus_path, minus_path, tag_plus=tag_plus, tag_minus=tag_minus)
        fallback = Path(plus_path).stem
        label = _safe_label(name or "", fallback)
        global_workspace.push_spectroscopy_data(label, tensor)
        global_workspace.write_processed_data(label, tensor)
        for ch, tag in enumerate([tag_plus, tag_minus]):
            child = TensorData(
                value=np.asarray(tensor.value[ch], dtype=float),
                axes=[np.asarray(tensor.axes[1], dtype=float)],
                labels=["energy"],
                units=["eV"],
                data_type=f"Experimental XAS ({tag})",
                metadata=dict(tensor.metadata or {}, channel_tag=tag),
            )
            global_workspace.write_processed_child_data(label, tag, child)
        return self.get_meta(label)

    def get_meta(self, name: str) -> dict[str, Any]:
        raw = _require_xas(name)
        processed = global_workspace.pull_tensor_data(name, "processed")
        children = global_workspace.list_processed_children(name)
        n_points = int(raw.value.shape[-1])
        paired = raw.labels == ["channel", "energy"]
        tags = list(raw.metadata.get("channel_tags") or [])
        has_bg = global_workspace.pull_analysis_data(name, "background") is not None
        has_sumrule = global_workspace.pull_analysis_data(name, "sumrule") is not None
        return {
            "name": name,
            "n_points": n_points,
            "paired": paired,
            "channel_tags": tags,
            "has_processed": processed is not None,
            "processed_children": children,
            "has_background": has_bg,
            "has_sumrule": has_sumrule,
            "I0_present": (raw.metadata or {}).get("I0") is not None,
            "source": str((raw.metadata or {}).get("source", "")),
        }

    def bg_preview(
        self,
        name: str,
        node: str = "raw",
        channel: int = 0,
        method: str = "linear",
        e0: float = 0.0,
        e1: float = 1.0,
        post_e0: float | None = None,
        post_e1: float | None = None,
        ensemble_delta: float | None = None,
        ensemble_n: int = 21,
    ) -> dict[str, Any]:
        _require_xas(name)
        result = _run_bg_fit(
            name,
            node=node,
            channel=channel,
            method=method,
            e0=e0,
            e1=e1,
            post_e0=post_e0,
            post_e1=post_e1,
            ensemble_delta=ensemble_delta,
            ensemble_n=ensemble_n,
        )
        return _bg_fit_to_dict(result, e0=e0, e1=e1, post_e0=post_e0, post_e1=post_e1)

    def bg_apply(
        self,
        name: str,
        node: str = "raw",
        channel: int = 0,
        method: str = "linear",
        e0: float = 0.0,
        e1: float = 1.0,
        post_e0: float | None = None,
        post_e1: float | None = None,
        ensemble_delta: float | None = None,
        ensemble_n: int = 21,
    ) -> dict[str, Any]:
        _require_xas(name)
        result = _run_bg_fit(
            name,
            node=node,
            channel=channel,
            method=method,
            e0=e0,
            e1=e1,
            post_e0=post_e0,
            post_e1=post_e1,
            ensemble_delta=ensemble_delta,
            ensemble_n=ensemble_n,
        )
        fit = result["fit"]
        ensemble = result["ensemble"]
        ds = bg_analysis_dataset(
            result["energy"],
            result["spectrum"],
            fit,
            ensemble,
            e0=e0,
            e1=e1,
            post_e0=post_e0,
            post_e1=post_e1,
            energy_source="csv",
            source_node=node,
            channel=channel,
            use_roi=False,
            roi=None,
            ensemble_delta=result["delta"],
            ensemble_n=ensemble_n,
        )
        if not global_workspace.write_analysis_data(name, "background", ds):
            raise ValueError(f"XAS data '{name}' not found.")

        child = _bg_child_name(node)
        bg_tensor = _bg_subtracted_tensor(
            result["source_tensor"],
            ensemble["subtracted_mean"],
            child_name=child,
            source_node=node,
            channel=channel,
        )
        if not global_workspace.write_processed_child_data(name, child, bg_tensor):
            raise ValueError(f"XAS data '{name}' not found.")
        return {"name": name, "processed_bg_node": f"processed/{child}", "n_points": int(bg_tensor.value.size)}

    def sumrule_preview(
        self,
        name: str,
        nh: float = 1.0,
        l3_lo: float = 0.0,
        l3_hi: float = 1.0,
        l2_lo: float = 2.0,
        l2_hi: float = 3.0,
        r_lo: float = 0.0,
        r_hi: float = 4.0,
        window_delta: float | None = None,
        window_n: int = 21,
        bg_delta: float | None = None,
    ) -> dict[str, Any]:
        _require_xas(name)
        return _sumrule_to_dict(
            _run_sumrule(
                name,
                nh=nh,
                l3_lo=l3_lo,
                l3_hi=l3_hi,
                l2_lo=l2_lo,
                l2_hi=l2_hi,
                r_lo=r_lo,
                r_hi=r_hi,
                window_delta=window_delta,
                window_n=window_n,
                bg_delta=bg_delta,
            )
        )

    def sumrule_apply(
        self,
        name: str,
        nh: float = 1.0,
        l3_lo: float = 0.0,
        l3_hi: float = 1.0,
        l2_lo: float = 2.0,
        l2_hi: float = 3.0,
        r_lo: float = 0.0,
        r_hi: float = 4.0,
        window_delta: float | None = None,
        window_n: int = 21,
        bg_delta: float | None = None,
    ) -> dict[str, Any]:
        _require_xas(name)
        result = _run_sumrule(
            name,
            nh=nh,
            l3_lo=l3_lo,
            l3_hi=l3_hi,
            l2_lo=l2_lo,
            l2_hi=l2_hi,
            r_lo=r_lo,
            r_hi=r_hi,
            window_delta=window_delta,
            window_n=window_n,
            bg_delta=bg_delta,
        )
        ens = result["ensemble"]
        ds = sumrule_analysis_dataset(
            result["energy"],
            result["mu_plus"],
            result["mu_minus"],
            result["integrals"],
            {
                "p": ens["p_std"],
                "q": ens["q_std"],
                "r": ens["r_std"],
            },
            result["moment_vals"],
            {
                "m_orb": ens["m_orb_std"],
                "m_spin_plus_dipole": ens["m_spin_plus_dipole_std"],
            },
            ens,
            nh=nh,
            l3=(l3_lo, l3_hi),
            l2=(l2_lo, l2_hi),
            r_win=(r_lo, r_hi),
            i0_applied=result["i0_applied"],
            source_kind=result["source_kind"],
            tags=result["tags"],
            window_delta=result["window_delta"],
            window_n=window_n,
            bg_e0=result["bg_e0"],
            bg_e1=result["bg_e1"],
            bg_delta=result["bg_delta"],
            bg_n=result["bg_n"],
            use_roi=False,
            roi=None,
        )
        if not global_workspace.write_analysis_data(name, "sumrule", ds):
            raise ValueError(f"XAS data '{name}' not found.")
        preview = _sumrule_to_dict(result)
        return {
            "name": name,
            "tag_plus": preview["tag_plus"],
            "tag_minus": preview["tag_minus"],
            "source_kind": preview["source_kind"],
        }

    def get_plot_spectrum(self, name: str, node: str = "raw", channel: int = 0) -> dict[str, Any]:
        raw = _require_xas(name)
        node = node.strip("/")
        if node == "raw":
            tensor = raw
        elif node == "processed":
            tensor = _processed_tensor(name)
            if tensor is None:
                raise ValueError("No processed spectrum.")
        elif node.startswith("processed/"):
            tensor = _separated_tensor(name, node)
        else:
            raise ValueError("Invalid node.")
        energy, spectrum = _energy_and_spectrum(tensor, channel)
        return {"energy": energy, "spectrum": spectrum, "label": node}
