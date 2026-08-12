"""PEEM TIF load, metadata, frame, and deferred beamline-CSV endpoints."""
from __future__ import annotations

import math
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import NamedTuple
from uuid import uuid4

import numpy as np
import tifffile
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from tensorspec.core.io.peem_loaders import (
    find_beamline_csv,
    load_beamline_csv,
    load_tif_sequence,
    load_tif_stack,
)
from tensorspec.core.data_models import TensorData
from tensorspec.core.peem_bg import (
    analysis_dataset as bg_analysis_dataset,
    apply_bg_to_stack,
    bg_child_name,
    ensemble_preedge,
    extract_spectrum,
    fit_linear_preedge,
    is_bg_output_node,
    resolve_energy,
)
from tensorspec.core.peem_engine import drift_correct, pair_stack, separate_pairs
from tensorspec.core.peem_roi import roi_to_mask
from tensorspec.core.peem_sumrule import (
    analysis_dataset as sumrule_analysis_dataset,
    apply_i0,
    ensemble_sumrule,
    integrate_windows,
    moments,
    pick_source_kind,
)
from tensorspec.web.server.schemas import (
    PeemBgApplySummary,
    PeemBgPreviewResponse,
    PeemBgRequest,
    PeemDriftRequest,
    PeemDriftSummary,
    PeemFrame,
    PeemLoadSummary,
    PeemMeta,
    PeemPairRequest,
    PeemPairSummary,
    PeemSeparateSummary,
    PeemSumruleApplySummary,
    PeemSumrulePreviewResponse,
    PeemSumruleRequest,
)
from tensorspec.web.server.session import Session, current_session

router = APIRouter(prefix="/api/peem", tags=["peem"])

MAX_PEEM_BYTES = 512 * 1024 * 1024
MAX_PEEM_FRAMES = 10_000
MAX_CSV_BYTES = 8 * 1024 * 1024
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _resolve_allowed(path: Path, session: Session) -> Path:
    resolved = path.expanduser().resolve()
    roots = [
        Path(session.workspace.project_dir).resolve(),
        Path.home().resolve(),
    ]
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise HTTPException(status_code=403, detail="Path outside allowed roots.")
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="Path not found.")
    return resolved


def _safe_label(name: str, fallback: str) -> str:
    cleaned = (name.strip() or fallback)[:64]
    if not SAFE_NAME.match(cleaned):
        raise HTTPException(
            status_code=422,
            detail="Name must start with a letter or digit and contain only "
            "letters, digits, '.', '_' or '-'.",
        )
    return cleaned


def _safe_upload_name(filename: str, fallback: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", Path(filename).name)[:120] or fallback


def _read_upload(upload: UploadFile, limit: int, kind: str) -> bytes:
    payload = upload.file.read(limit + 1)
    if len(payload) > limit:
        raise HTTPException(status_code=413, detail=f"{kind} exceeds the size limit.")
    if not payload:
        raise HTTPException(status_code=400, detail=f"Uploaded {kind.lower()} is empty.")
    return payload


def _tif_paths(source_path: Path) -> list[Path]:
    if source_path.is_dir():
        candidates = sorted(source_path.glob("*.tif")) + sorted(
            source_path.glob("*.tiff")
        )
        seen: set[str] = set()
        paths: list[Path] = []
        for candidate in candidates:
            key = candidate.name.casefold()
            if key not in seen:
                seen.add(key)
                paths.append(candidate)
        return paths
    if source_path.suffix.lower() in (".tif", ".tiff"):
        return [source_path]
    raise HTTPException(
        status_code=400, detail="Server path must be a TIF file or directory."
    )


def _float64_bytes_for_shape(shape: tuple[int, ...] | list[int]) -> int:
    """Byte count for float64 expansion; use math.prod to avoid int64 overflow."""
    dims = [int(size) for size in shape]
    if any(size < 0 for size in dims):
        raise ValueError(f"Invalid TIF shape {shape}")
    return math.prod(dims) * 8


def _enforce_peem_size(source_path: Path) -> None:
    paths = _tif_paths(source_path)
    if sum(path.stat().st_size for path in paths) > MAX_PEEM_BYTES:
        raise HTTPException(status_code=413, detail="PEEM TIF files exceed 512 MB.")

    n_frames = 0
    float64_bytes = 0
    try:
        for path in paths:
            with tifffile.TiffFile(path) as tif:
                if not tif.pages:
                    continue
                if len(tif.pages) == 1:
                    shape = tuple(int(size) for size in tif.pages[0].shape)
                    n_frames += shape[0] if len(shape) == 3 else 1
                    float64_bytes += _float64_bytes_for_shape(shape)
                else:
                    n_frames += len(tif.pages)
                    float64_bytes += sum(
                        _float64_bytes_for_shape(page.shape) for page in tif.pages
                    )
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Could not inspect PEEM TIF data: {exc}"
        ) from exc

    if n_frames > MAX_PEEM_FRAMES:
        raise HTTPException(
            status_code=413,
            detail=f"PEEM data exceeds the {MAX_PEEM_FRAMES}-frame limit.",
        )
    if float64_bytes > MAX_PEEM_BYTES:
        raise HTTPException(
            status_code=413,
            detail="PEEM float64 expansion exceeds the 512 MB limit.",
        )


def _extract_zip(payload: bytes, destination: Path) -> Path:
    archive_path = destination / "_upload.zip"
    archive_path.write_bytes(payload)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            if sum(member.file_size for member in members) > MAX_PEEM_BYTES:
                raise HTTPException(
                    status_code=413, detail="Extracted ZIP exceeds the 512 MB limit."
                )
            root = destination.resolve()
            for member in members:
                target = (destination / member.filename).resolve()
                if not (target == root or root in target.parents):
                    raise HTTPException(status_code=400, detail="Unsafe path in ZIP archive.")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as output:
                    output.write(source.read())
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=422, detail="Could not parse ZIP archive.") from exc
    finally:
        archive_path.unlink(missing_ok=True)

    tif_paths = sorted(destination.rglob("*.tif")) + sorted(destination.rglob("*.tiff"))
    if not tif_paths:
        raise HTTPException(status_code=422, detail="ZIP contains no TIF files.")
    parents = {path.parent for path in tif_paths}
    if len(parents) != 1:
        raise HTTPException(
            status_code=422, detail="ZIP TIF files must be in one directory."
        )
    return parents.pop()


def _csv_path_from_inputs(
    *,
    csv: UploadFile | None,
    csv_path: str | None,
    session: Session,
    upload_dir: Path,
) -> Path | None:
    if csv is not None and csv_path:
        raise HTTPException(
            status_code=400, detail="Provide either csv or csv_path, not both."
        )
    if csv is not None:
        filename = _safe_upload_name(csv.filename or "", "beamline.csv")
        if not filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Expected a .csv file.")
        destination = upload_dir / filename
        destination.write_bytes(_read_upload(csv, MAX_CSV_BYTES, "CSV"))
        return destination.resolve()
    if csv_path:
        resolved = _resolve_allowed(Path(csv_path), session)
        if not resolved.is_file():
            raise HTTPException(status_code=400, detail="CSV path must be a file.")
        if resolved.suffix.lower() != ".csv":
            raise HTTPException(status_code=400, detail="Expected a .csv file.")
        if resolved.stat().st_size > MAX_CSV_BYTES:
            raise HTTPException(status_code=413, detail="CSV exceeds the 8 MB limit.")
        return resolved
    return None


def _auto_csv_choice(
    directory: Path, preferred_stem: str
) -> tuple[Path | None, list[Path]]:
    """Mirror loader auto-selection so size is checked before CSV parsing."""
    candidates = find_beamline_csv(directory, preferred_stem)
    if len(candidates) == 1:
        return candidates[0], candidates
    preferred = preferred_stem.casefold()
    matches = [
        path
        for path in candidates
        if path.stem.casefold() == preferred or preferred in path.stem.casefold()
    ]
    return (matches[0] if len(matches) == 1 else None), candidates


def _require_tensor(session: Session, name: str):
    tensor = session.workspace.pull_tensor_data(name, "raw")
    if tensor is None:
        raise HTTPException(
            status_code=404,
            detail=f"'{name}' is not spectroscopy data in this session.",
        )
    if tensor.data_type != "Experimental PEEM":
        raise HTTPException(status_code=404, detail=f"'{name}' is not PEEM data.")
    return tensor


def _processed_tensor(session: Session, name: str):
    tensor = session.workspace.pull_tensor_data(name, "processed")
    if tensor is None:
        return None
    shape = tuple(int(size) for size in tensor.value.shape)
    valid_raw = (
        tensor.data_type == "Experimental PEEM"
        and tensor.labels == ["frame", "y", "x"]
        and tensor.value.ndim == 3
        and len(shape) == 3
        and all(size > 0 for size in shape)
    )
    valid_pair = (
        tensor.data_type == "Experimental PEEM (paired)"
        and tensor.labels == ["pair", "channel", "y", "x"]
        and tensor.value.ndim == 4
        and len(shape) == 4
        and shape[1] == 2
        and all(size > 0 for size in shape)
    )
    if not (valid_raw or valid_pair):
        raise HTTPException(
            status_code=422,
            detail="Processed PEEM data must be a non-empty (frame, y, x) "
            "cube or (pair, channel=2, y, x) paired cube.",
        )
    return tensor


def _processed_pair_tensor(session: Session, name: str):
    tensor = _processed_tensor(session, name)
    if tensor is None:
        return None
    if tensor.value.ndim != 4:
        raise HTTPException(
            status_code=422,
            detail="Processed PEEM data must be a non-empty "
            "(pair, channel=2, y, x) paired cube.",
        )
    return tensor


def _separated_tensor(session: Session, name: str, node: str):
    """Pull processed/<tag> channel stack; validate 3D PEEM channel cube."""
    rel = node.strip("/")
    if not rel.startswith("processed/") or rel.count("/") != 1:
        raise HTTPException(status_code=422, detail="Invalid separated node path.")
    tensor = session.workspace.pull_tensor_data(name, rel)
    if tensor is None:
        raise HTTPException(status_code=404, detail=f"No data at '{rel}'.")
    tag = rel.split("/", 1)[1]
    ok = (
        tensor.value.ndim == 3
        and list(tensor.labels) == ["frame", "y", "x"]
        and tensor.data_type.startswith("Experimental PEEM")
    )
    if not ok:
        raise HTTPException(
            status_code=422, detail=f"Invalid channel stack at '{rel}'."
        )
    return tensor, tag


def _default_ensemble_delta(energy: np.ndarray, energy_source: str) -> float:
    if energy_source == "index":
        return 1.0
    span = float(np.max(energy) - np.min(energy))
    return 0.05 * span if span > 0 else 1.0


def _pull_bg_source(
    session: Session, name: str, node: str, channel: int
) -> tuple[np.ndarray, TensorData, TensorData]:
    """Return (3D stack, source tensor, raw tensor for energy metadata)."""
    raw = _require_tensor(session, name)
    node = node.strip("/")
    if is_bg_output_node(node):
        raise HTTPException(
            status_code=422,
            detail=(
                f"node '{node}' is a background output; "
                "use raw, processed, or a non-bg separated channel."
            ),
        )
    if node == "raw":
        stack = np.asarray(raw.value, dtype=float)
        if stack.ndim != 3:
            raise HTTPException(status_code=422, detail="Raw PEEM data must be 3D.")
        return stack, raw, raw
    if node == "processed":
        processed = _processed_tensor(session, name)
        if processed is None:
            raise HTTPException(
                status_code=404, detail=f"PEEM data '{name}' has no processed data."
            )
        if processed.value.ndim == 4:
            if not 0 <= channel < processed.value.shape[1]:
                raise HTTPException(
                    status_code=404, detail=f"Channel index {channel} is out of range."
                )
            stack = np.asarray(processed.value[:, channel], dtype=float)
        elif processed.value.ndim == 3:
            stack = np.asarray(processed.value, dtype=float)
        else:
            raise HTTPException(status_code=422, detail="Invalid processed PEEM shape.")
        return stack, processed, raw
    if node.startswith("processed/"):
        tensor, _tag = _separated_tensor(session, name, node)
        stack = np.asarray(tensor.value, dtype=float)
        return stack, tensor, raw
    raise HTTPException(
        status_code=422,
        detail="node must be 'raw', 'processed', or 'processed/<tag>'.",
    )


def _bg_roi_mask(request: PeemBgRequest, ny: int, nx: int) -> np.ndarray | None:
    if not request.use_roi:
        return None
    if request.roi is None:
        raise HTTPException(status_code=422, detail="roi required when use_roi is true.")
    try:
        return roi_to_mask(ny, nx, request.roi.model_dump(exclude_none=True))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class _BgFitResult(NamedTuple):
    stack: np.ndarray
    source_tensor: TensorData
    energy: np.ndarray
    energy_source: str
    spectrum: np.ndarray
    fit: dict
    ensemble: dict
    delta: float
    roi_dict: dict | None


def _run_bg_fit(session: Session, name: str, request: PeemBgRequest) -> _BgFitResult:
    stack, source_tensor, raw = _pull_bg_source(
        session, name, request.node, request.channel
    )
    energy, energy_source = resolve_energy(stack.shape[0], raw.metadata or {})
    mask = _bg_roi_mask(request, stack.shape[1], stack.shape[2])
    try:
        spectrum = extract_spectrum(stack, mask)
        fit = fit_linear_preedge(energy, spectrum, request.e0, request.e1)
        delta = request.ensemble_delta
        if delta is None:
            delta = _default_ensemble_delta(energy, energy_source)
        ensemble = ensemble_preedge(
            energy,
            spectrum,
            request.e0,
            request.e1,
            delta=delta,
            n=request.ensemble_n,
            seed=request.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    roi_dict = (
        request.roi.model_dump(exclude_none=True)
        if request.use_roi and request.roi
        else None
    )
    return _BgFitResult(
        stack=stack,
        source_tensor=source_tensor,
        energy=energy,
        energy_source=energy_source,
        spectrum=spectrum,
        fit=fit,
        ensemble=ensemble,
        delta=delta,
        roi_dict=roi_dict,
    )


def _compute_bg_preview(
    session: Session, name: str, request: PeemBgRequest
) -> PeemBgPreviewResponse:
    result = _run_bg_fit(session, name, request)
    return PeemBgPreviewResponse(
        energy=[float(v) for v in result.energy],
        spectrum=[float(v) for v in result.spectrum],
        bg=[float(v) for v in result.ensemble["bg_mean"]],
        bg_std=[float(v) for v in result.ensemble["bg_std"]],
        subtracted=[float(v) for v in result.ensemble["subtracted_mean"]],
        subtracted_std=[float(v) for v in result.ensemble["subtracted_std"]],
        slope=float(result.fit["slope"]),
        intercept=float(result.fit["intercept"]),
        energy_source=result.energy_source,
        e0=float(request.e0),
        e1=float(request.e1),
        ensemble_n_valid=int(result.ensemble["n_valid"]),
    )


def _bg_subtracted_tensor(
    source_tensor: TensorData,
    subtracted: np.ndarray,
    *,
    child_name: str,
    source_node: str,
    channel: int,
) -> TensorData:
    n, _y, _x = subtracted.shape
    if source_tensor.value.ndim == 4:
        y_axis = np.asarray(source_tensor.axes[2])
        x_axis = np.asarray(source_tensor.axes[3])
        y_unit = source_tensor.units[2]
        x_unit = source_tensor.units[3]
    else:
        y_axis = np.asarray(source_tensor.axes[1])
        x_axis = np.asarray(source_tensor.axes[2])
        y_unit = source_tensor.units[1]
        x_unit = source_tensor.units[2]

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
        value=subtracted,
        axes=[np.arange(n), y_axis, x_axis],
        labels=["frame", "y", "x"],
        units=["", y_unit, x_unit],
        data_type="Experimental PEEM (bg subtracted)",
        metadata=meta,
    )


def _analysis_to_preview(ds) -> PeemBgPreviewResponse:
    attrs = ds.attrs or {}
    return PeemBgPreviewResponse(
        energy=[float(v) for v in ds.coords["energy"].values],
        spectrum=[float(v) for v in ds["raw_spectrum"].values],
        bg=[float(v) for v in ds["bg"].values],
        bg_std=[float(v) for v in ds["bg_std"].values],
        subtracted=[float(v) for v in ds["subtracted"].values],
        subtracted_std=[float(v) for v in ds["subtracted_std"].values],
        slope=float(attrs["slope"]),
        intercept=float(attrs["intercept"]),
        energy_source=str(attrs["energy_source"]),
        e0=float(attrs["e0"]),
        e1=float(attrs["e1"]),
        ensemble_n_valid=int(attrs["ensemble_n_valid"]),
    )


def _bg_meta_fields(session: Session, name: str) -> dict:
    analysis = session.workspace.pull_analysis_data(name, "background")
    if analysis is None:
        return {
            "has_background": False,
            "has_processed_bg": False,
            "energy_source": None,
            "processed_bg_node": None,
            "n_bg_frames": None,
        }
    attrs = analysis.attrs or {}
    energy_source = attrs.get("energy_source")
    source_node = str(attrs.get("source_node", "raw"))
    try:
        child = bg_child_name(source_node)
    except ValueError:
        child = "bg"
    processed_bg_node = f"processed/{child}"
    children = session.workspace.list_processed_children(name)
    has_processed_bg = child in children
    n_bg_frames: int | None = None
    if has_processed_bg:
        bg_tensor = session.workspace.pull_tensor_data(name, processed_bg_node)
        if bg_tensor is not None and bg_tensor.value.ndim >= 1:
            n_bg_frames = int(bg_tensor.value.shape[0])
    return {
        "has_background": True,
        "has_processed_bg": has_processed_bg,
        "energy_source": str(energy_source) if energy_source is not None else None,
        "processed_bg_node": processed_bg_node,
        "n_bg_frames": n_bg_frames,
    }


_SUMRULE_PAIRS = (("CP", "CM"), ("LH", "LV"))


def _sumrule_tags(session: Session, name: str) -> tuple[str, str]:
    processed = _processed_tensor(session, name)
    if processed is None:
        raise HTTPException(
            status_code=422,
            detail="Sum rule requires paired or separated CP/CM (or LH/LV) stacks.",
        )
    channel_tags = processed.metadata.get("channel_tags", [])
    if len(channel_tags) >= 2:
        return (str(channel_tags[0]), str(channel_tags[1]))
    children = set(session.workspace.list_processed_children(name))
    for pair in _SUMRULE_PAIRS:
        if pair[0] in children and pair[1] in children:
            return pair
    if processed.value.ndim == 4:
        return ("CP", "CM")
    raise HTTPException(
        status_code=422,
        detail="Cannot resolve sum-rule channel pair (CP/CM or LH/LV).",
    )


def _sumrule_available_nodes(session: Session, name: str) -> list[str]:
    nodes = [f"processed/{tag}" for tag in session.workspace.list_processed_children(name)]
    processed = _processed_tensor(session, name)
    if processed is not None and processed.value.ndim == 4:
        nodes.append("processed")
    return nodes


def _resolve_sumrule_stacks(
    session: Session,
    name: str,
    tags: tuple[str, str],
    source_kind: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, str], str]:
    t0, t1 = tags
    if source_kind == "bg":
        plus_tensor, _ = _separated_tensor(session, name, f"processed/{t0}_bg")
        minus_tensor, _ = _separated_tensor(session, name, f"processed/{t1}_bg")
        plus_stack = np.asarray(plus_tensor.value, dtype=float)
        minus_stack = np.asarray(minus_tensor.value, dtype=float)
    elif source_kind == "separated":
        plus_tensor, _ = _separated_tensor(session, name, f"processed/{t0}")
        minus_tensor, _ = _separated_tensor(session, name, f"processed/{t1}")
        plus_stack = np.asarray(plus_tensor.value, dtype=float)
        minus_stack = np.asarray(minus_tensor.value, dtype=float)
    else:
        paired = _processed_pair_tensor(session, name)
        plus_stack = np.asarray(paired.value[:, 0], dtype=float)
        minus_stack = np.asarray(paired.value[:, 1], dtype=float)
    return plus_stack, minus_stack, tags, source_kind


def _sumrule_roi_mask(request: PeemSumruleRequest, ny: int, nx: int) -> np.ndarray | None:
    if not request.use_roi:
        return None
    if request.roi is None:
        raise HTTPException(status_code=422, detail="roi required when use_roi is true.")
    try:
        return roi_to_mask(ny, nx, request.roi.model_dump(exclude_none=True))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _sumrule_i0(
    raw_metadata: dict, n_frames: int, spectrum: np.ndarray
) -> tuple[np.ndarray, bool]:
    i0 = (raw_metadata or {}).get("I0")
    if i0 is None:
        return apply_i0(spectrum, None)
    if isinstance(i0, list):
        if len(i0) != n_frames:
            return apply_i0(spectrum, None)
        return apply_i0(spectrum, np.asarray(i0, dtype=float))
    return apply_i0(spectrum, i0)


def _sumrule_bg_params(
    session: Session, name: str, energy: np.ndarray, energy_source: str
) -> tuple[float | None, float | None, float, int]:
    analysis = session.workspace.pull_analysis_data(name, "background")
    if analysis is None:
        return None, None, 0.0, 1
    attrs = analysis.attrs or {}
    e0 = attrs.get("e0")
    e1 = attrs.get("e1")
    if e0 is None or e1 is None:
        return None, None, 0.0, 1
    bg_delta = attrs.get("ensemble_delta")
    if bg_delta is None:
        bg_delta = _default_ensemble_delta(energy, energy_source)
    else:
        bg_delta = float(bg_delta)
    bg_n = int(attrs.get("ensemble_n", 21))
    return float(e0), float(e1), bg_delta, bg_n


class _SumruleResult(NamedTuple):
    energy: np.ndarray
    energy_source: str
    mu_plus: np.ndarray
    mu_minus: np.ndarray
    i0_applied: bool
    source_kind: str
    tags: tuple[str, str]
    integrals: dict[str, float]
    moment_vals: dict[str, float]
    ensemble: dict
    window_delta: float
    bg_e0: float | None
    bg_e1: float | None
    bg_delta: float
    bg_n: int
    roi_dict: dict | None


def _run_sumrule(
    session: Session, name: str, request: PeemSumruleRequest
) -> _SumruleResult:
    raw = _require_tensor(session, name)
    tags = _sumrule_tags(session, name)
    try:
        source_kind = pick_source_kind(_sumrule_available_nodes(session, name), tags)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    plus_stack, minus_stack, tags, source_kind = _resolve_sumrule_stacks(
        session, name, tags, source_kind
    )
    if plus_stack.shape != minus_stack.shape:
        raise HTTPException(status_code=422, detail="Plus/minus stacks have mismatched shape.")
    if plus_stack.ndim != 3:
        raise HTTPException(status_code=422, detail="Sum-rule stacks must be 3D.")

    energy, energy_source = resolve_energy(plus_stack.shape[0], raw.metadata or {})
    mask = _sumrule_roi_mask(request, plus_stack.shape[1], plus_stack.shape[2])
    try:
        mu_plus = extract_spectrum(plus_stack, mask)
        mu_minus = extract_spectrum(minus_stack, mask)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    mu_plus, i0_plus = _sumrule_i0(raw.metadata or {}, plus_stack.shape[0], mu_plus)
    mu_minus, i0_minus = _sumrule_i0(raw.metadata or {}, minus_stack.shape[0], mu_minus)
    i0_applied = i0_plus and i0_minus

    l3 = (request.l3_lo, request.l3_hi)
    l2 = (request.l2_lo, request.l2_hi)
    r_win = (request.r_lo, request.r_hi)
    window_delta = request.window_delta
    if window_delta is None:
        window_delta = _default_ensemble_delta(energy, energy_source)

    bg_e0, bg_e1, bg_delta_default, bg_n_default = _sumrule_bg_params(
        session, name, energy, energy_source
    )
    if source_kind == "bg":
        bg_e0, bg_e1 = None, None
    bg_delta = request.bg_delta if request.bg_delta is not None else bg_delta_default
    bg_n = request.bg_n if bg_e0 is not None else 1

    try:
        integrals = integrate_windows(energy, mu_plus, mu_minus, l3=l3, l2=l2, r_win=r_win)
        moment_vals = moments(integrals["p"], integrals["q"], integrals["r"], request.nh)
        ensemble = ensemble_sumrule(
            energy,
            mu_plus,
            mu_minus,
            l3=l3,
            l2=l2,
            r_win=r_win,
            nh=request.nh,
            window_delta=window_delta,
            window_n=request.window_n,
            bg_e0=bg_e0,
            bg_e1=bg_e1,
            bg_delta=bg_delta,
            bg_n=bg_n,
            seed=request.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    roi_dict = (
        request.roi.model_dump(exclude_none=True)
        if request.use_roi and request.roi
        else None
    )
    return _SumruleResult(
        energy=energy,
        energy_source=energy_source,
        mu_plus=mu_plus,
        mu_minus=mu_minus,
        i0_applied=i0_applied,
        source_kind=source_kind,
        tags=tags,
        integrals=integrals,
        moment_vals=moment_vals,
        ensemble=ensemble,
        window_delta=window_delta,
        bg_e0=bg_e0,
        bg_e1=bg_e1,
        bg_delta=bg_delta,
        bg_n=bg_n,
        roi_dict=roi_dict,
    )


def _sumrule_to_preview(result: _SumruleResult) -> PeemSumrulePreviewResponse:
    ens = result.ensemble
    d_mu = result.mu_plus - result.mu_minus
    return PeemSumrulePreviewResponse(
        energy=[float(v) for v in result.energy],
        mu_plus=[float(v) for v in result.mu_plus],
        mu_minus=[float(v) for v in result.mu_minus],
        dichroism=[float(v) for v in d_mu],
        p=float(ens["p_mean"]),
        q=float(ens["q_mean"]),
        r=float(ens["r_mean"]),
        p_std=float(ens["p_std"]),
        q_std=float(ens["q_std"]),
        r_std=float(ens["r_std"]),
        m_orb=float(ens["m_orb_mean"]),
        m_orb_std=float(ens["m_orb_std"]),
        m_spin_plus_dipole=float(ens["m_spin_plus_dipole_mean"]),
        m_spin_plus_dipole_std=float(ens["m_spin_plus_dipole_std"]),
        i0_applied=result.i0_applied,
        source_kind=result.source_kind,
        tag_plus=result.tags[0],
        tag_minus=result.tags[1],
        energy_source=result.energy_source,
        ensemble_n_valid=int(ens["n_valid"]),
        ensemble_n_valid_bg=int(ens.get("n_valid_bg", 0)),
    )


def _analysis_to_sumrule_preview(ds) -> PeemSumrulePreviewResponse:
    attrs = ds.attrs or {}
    return PeemSumrulePreviewResponse(
        energy=[float(v) for v in ds.coords["energy"].values],
        mu_plus=[float(v) for v in ds["mu_plus"].values],
        mu_minus=[float(v) for v in ds["mu_minus"].values],
        dichroism=[float(v) for v in ds["dichroism"].values],
        p=float(attrs["p"]),
        q=float(attrs["q"]),
        r=float(attrs["r"]),
        p_std=float(attrs["p_std"]),
        q_std=float(attrs["q_std"]),
        r_std=float(attrs["r_std"]),
        m_orb=float(attrs["m_orb"]),
        m_orb_std=float(attrs["m_orb_std"]),
        m_spin_plus_dipole=float(attrs["m_spin_plus_dipole"]),
        m_spin_plus_dipole_std=float(attrs["m_spin_plus_dipole_std"]),
        i0_applied=bool(attrs.get("i0_applied", False)),
        source_kind=str(attrs.get("source_kind", "")),
        tag_plus=str(attrs.get("tag_plus", "")),
        tag_minus=str(attrs.get("tag_minus", "")),
        energy_source=str(attrs.get("energy_source", "index")),
        ensemble_n_valid=int(attrs.get("ensemble_n_valid", 0)),
        ensemble_n_valid_bg=int(attrs.get("ensemble_n_valid_bg", 0)),
    )


def _sumrule_meta_fields(session: Session, name: str) -> dict:
    analysis = session.workspace.pull_analysis_data(name, "sumrule")
    if analysis is None:
        return {
            "has_sumrule": False,
            "sumrule_i0_applied": None,
            "sumrule_tags": [],
        }
    attrs = analysis.attrs or {}
    tags: list[str] = []
    tag_plus = attrs.get("tag_plus")
    tag_minus = attrs.get("tag_minus")
    if tag_plus is not None and tag_minus is not None:
        tags = [str(tag_plus), str(tag_minus)]
    return {
        "has_sumrule": True,
        "sumrule_i0_applied": bool(attrs.get("i0_applied", False)),
        "sumrule_tags": tags,
    }


def _summary(
    name: str,
    tensor,
    *,
    csv_prompt: bool = False,
    csv_candidates: list[str] | None = None,
) -> PeemLoadSummary:
    metadata = tensor.metadata or {}
    pol = [str(value) for value in metadata.get("pol", [])]
    return PeemLoadSummary(
        name=name,
        shape=[int(size) for size in tensor.value.shape],
        n_frames=int(tensor.value.shape[0]),
        data_type=tensor.data_type,
        pol_summary=dict(Counter(pol)),
        source=str(metadata.get("source", "")),
        loader=str(metadata.get("loader", "")),
        csv_attached=bool(metadata.get("csv_attached", False)),
        I0_present=metadata.get("I0") is not None,
        csv_prompt=csv_prompt,
        csv_candidates=csv_candidates or [],
    )


@router.post("/load", response_model=PeemLoadSummary)
def load_peem(
    file: UploadFile | None = File(default=None),
    server_path: str | None = Form(default=None),
    csv: UploadFile | None = File(default=None),
    csv_path: str | None = Form(default=None),
    name: str | None = Form(default=None),
    session: Session = Depends(current_session),
) -> PeemLoadSummary:
    """Load one TIF stack, a TIF directory, or an uploaded ZIP sequence."""
    if (file is None) == (not server_path):
        raise HTTPException(
            status_code=400, detail="Provide exactly one of file or server_path."
        )
    label = _safe_label(name, "peem") if name is not None else None

    upload_root = Path(session.workspace.project_dir) / "uploads" / "peem"
    upload_root.mkdir(parents=True, exist_ok=True)
    upload_dir = upload_root
    if file is not None or csv is not None:
        upload_dir = upload_root / uuid4().hex
        upload_dir.mkdir()
    source_path: Path
    load_directory: Path
    fallback: str

    if file is not None:
        filename = _safe_upload_name(file.filename or "", "peem.tif")
        payload = _read_upload(file, MAX_PEEM_BYTES, "PEEM file")
        fallback = Path(filename).stem or "peem"
        if filename.lower().endswith(".zip"):
            source_path = _extract_zip(payload, upload_dir)
            load_directory = source_path
        elif filename.lower().endswith((".tif", ".tiff")):
            source_path = upload_dir / filename
            source_path.write_bytes(payload)
            load_directory = source_path.parent
        else:
            raise HTTPException(
                status_code=400, detail="Expected a .tif, .tiff, or .zip file."
            )
    else:
        source_path = _resolve_allowed(Path(server_path or ""), session)
        fallback = source_path.stem if source_path.is_file() else source_path.name
        load_directory = source_path if source_path.is_dir() else source_path.parent

    _enforce_peem_size(source_path)

    explicit_csv = _csv_path_from_inputs(
        csv=csv,
        csv_path=csv_path,
        session=session,
        upload_dir=upload_dir,
    )
    auto_candidates: list[Path] = []
    loader_csv = explicit_csv
    if explicit_csv is None:
        preferred_stem = source_path.name if source_path.is_dir() else source_path.stem
        loader_csv, auto_candidates = _auto_csv_choice(
            load_directory, preferred_stem
        )
        if loader_csv is not None and loader_csv.stat().st_size > MAX_CSV_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Auto-discovered CSV exceeds the 8 MB limit.",
            )

    try:
        if source_path.is_dir():
            tensor = load_tif_sequence(source_path, csv_path=loader_csv)
        elif source_path.suffix.lower() in (".tif", ".tiff"):
            tensor = load_tif_stack(source_path, csv_path=loader_csv)
        else:
            raise HTTPException(
                status_code=400, detail="Server path must be a TIF file or directory."
            )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Could not load PEEM data: {exc}"
        ) from exc

    if label is None:
        label = _safe_label(_safe_upload_name(fallback, "peem"), "peem")
    session.workspace.push_spectroscopy_data(label, tensor)
    attached = bool((tensor.metadata or {}).get("csv_attached", False))
    candidates = (
        []
        if attached or explicit_csv is not None
        else [str(path) for path in auto_candidates]
    )
    return _summary(
        label,
        tensor,
        csv_prompt=not attached and explicit_csv is None,
        csv_candidates=candidates,
    )


@router.post("/{name}/attach-csv", response_model=PeemLoadSummary)
def attach_csv(
    name: str,
    csv: UploadFile | None = File(default=None),
    csv_path: str | None = Form(default=None),
    session: Session = Depends(current_session),
) -> PeemLoadSummary:
    """Attach or replace beamline metadata without rebuilding image data."""
    tensor = _require_tensor(session, name)
    upload_root = Path(session.workspace.project_dir) / "uploads" / "peem"
    upload_root.mkdir(parents=True, exist_ok=True)
    upload_dir = upload_root
    if csv is not None:
        upload_dir = upload_root / uuid4().hex
        upload_dir.mkdir()
    selected = _csv_path_from_inputs(
        csv=csv,
        csv_path=csv_path,
        session=session,
        upload_dir=upload_dir,
    )
    if selected is None:
        raise HTTPException(status_code=400, detail="Provide csv or csv_path.")

    try:
        csv_meta = load_beamline_csv(selected)
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Could not parse beamline CSV: {exc}"
        ) from exc
    attrs = {
        "csv_attached": True,
        "beamline_csv": str(selected),
        "I0": csv_meta.get("I0"),
        "beamline_table": {
            "columns": csv_meta.get("columns"),
            "series": csv_meta.get("series"),
        },
        "beam_current": csv_meta.get("beam_current"),
    }
    if not session.workspace.merge_spectroscopy_raw_attrs(name, attrs):
        raise HTTPException(status_code=404, detail=f"PEEM data '{name}' not found.")
    return _summary(name, _require_tensor(session, name))


@router.post("/{name}/pair", response_model=PeemPairSummary)
def pair_peem(
    name: str,
    request: PeemPairRequest,
    session: Session = Depends(current_session),
) -> PeemPairSummary:
    """Pair tagged raw frames and replace the workspace /processed cube."""
    tensor = _require_tensor(session, name)
    try:
        paired = pair_stack(tensor, request.mode)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not session.workspace.write_processed_data(name, paired):
        raise HTTPException(status_code=404, detail=f"PEEM data '{name}' not found.")

    metadata = paired.metadata or {}
    return PeemPairSummary(
        name=name,
        n_pairs=int(paired.value.shape[0]),
        channel_tags=[str(value) for value in metadata.get("channel_tags", [])],
        unpaired_count=len(metadata.get("unpaired", [])),
        mode=str(metadata.get("pair_mode", request.mode)),
        shape=[int(size) for size in paired.value.shape],
    )


@router.post("/{name}/separate", response_model=PeemSeparateSummary)
def separate_peem(
    name: str,
    session: Session = Depends(current_session),
) -> PeemSeparateSummary:
    _require_tensor(session, name)
    paired = _processed_pair_tensor(session, name)
    if paired is None:
        raise HTTPException(
            status_code=422,
            detail="Separate requires a paired /processed cube. Run Stack Pairs first.",
        )
    try:
        channels = separate_pairs(paired)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    for tag, td in channels.items():
        if not session.workspace.write_processed_child_data(name, tag, td):
            raise HTTPException(status_code=404, detail=f"PEEM data '{name}' not found.")
    tags = sorted(channels)
    sample = channels[tags[0]]
    return PeemSeparateSummary(
        name=name,
        channels=tags,
        n_frames=int(sample.value.shape[0]),
        shape=[int(s) for s in sample.value.shape],
    )


@router.post("/{name}/drift", response_model=PeemDriftSummary)
def drift_peem(
    name: str,
    request: PeemDriftRequest,
    session: Session = Depends(current_session),
) -> PeemDriftSummary:
    """Drift-correct raw or processed PEEM data into /processed."""
    if request.source == "raw":
        tensor = _require_tensor(session, name)
    else:
        tensor = _processed_tensor(session, name)
        if tensor is None:
            raise HTTPException(
                status_code=404, detail=f"PEEM data '{name}' has no processed data."
            )
    try:
        corrected = drift_correct(
            tensor,
            ref_index=request.ref_index,
            roi=request.roi.model_dump(exclude_none=True),
            search_radius=request.search_radius,
            track_channel=request.track_channel,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not session.workspace.write_processed_data(name, corrected):
        raise HTTPException(status_code=404, detail=f"PEEM data '{name}' not found.")

    shifts = corrected.metadata.get("drift_shifts", [])
    return PeemDriftSummary(
        name=name,
        source=request.source,
        n_planes=int(corrected.value.shape[0]),
        ref_index=request.ref_index,
        search_radius=request.search_radius,
        max_abs_dx=max((abs(int(item["dx"])) for item in shifts), default=0),
        max_abs_dy=max((abs(int(item["dy"])) for item in shifts), default=0),
        shape=[int(size) for size in corrected.value.shape],
    )


@router.post("/{name}/bg/preview", response_model=PeemBgPreviewResponse)
def bg_preview_peem(
    name: str,
    request: PeemBgRequest,
    session: Session = Depends(current_session),
) -> PeemBgPreviewResponse:
    """Fit linear pre-edge background; return curves without writing the tree."""
    _require_tensor(session, name)
    return _compute_bg_preview(session, name, request)


@router.post("/{name}/bg/apply", response_model=PeemBgApplySummary)
def bg_apply_peem(
    name: str,
    request: PeemBgRequest,
    session: Session = Depends(current_session),
) -> PeemBgApplySummary:
    """Write /analysis/background and BG-subtracted /processed child."""
    _require_tensor(session, name)
    result = _run_bg_fit(session, name, request)
    subtracted = apply_bg_to_stack(result.stack, result.ensemble["bg_mean"])

    ds = bg_analysis_dataset(
        result.energy,
        result.spectrum,
        result.fit,
        result.ensemble,
        e0=request.e0,
        e1=request.e1,
        energy_source=result.energy_source,
        source_node=request.node,
        channel=request.channel,
        use_roi=request.use_roi,
        roi=result.roi_dict,
        ensemble_delta=result.delta,
        ensemble_n=request.ensemble_n,
        seed=request.seed,
    )
    if not session.workspace.write_analysis_data(name, "background", ds):
        raise HTTPException(status_code=404, detail=f"PEEM data '{name}' not found.")

    child = bg_child_name(request.node)
    bg_tensor = _bg_subtracted_tensor(
        result.source_tensor,
        subtracted,
        child_name=child,
        source_node=request.node,
        channel=request.channel,
    )
    if not session.workspace.write_processed_child_data(name, child, bg_tensor):
        raise HTTPException(status_code=404, detail=f"PEEM data '{name}' not found.")

    shape = [int(s) for s in subtracted.shape]
    return PeemBgApplySummary(
        name=name,
        processed_bg_node=f"processed/{child}",
        n_frames=shape[0],
        shape=shape,
        energy_source=result.energy_source,
    )


@router.get("/{name}/bg/spectrum", response_model=PeemBgPreviewResponse)
def bg_spectrum_peem(
    name: str,
    session: Session = Depends(current_session),
) -> PeemBgPreviewResponse:
    """Return stored /analysis/background curves."""
    _require_tensor(session, name)
    ds = session.workspace.pull_analysis_data(name, "background")
    if ds is None:
        raise HTTPException(
            status_code=404, detail=f"PEEM data '{name}' has no background analysis."
        )
    return _analysis_to_preview(ds)


@router.post("/{name}/sumrule/preview", response_model=PeemSumrulePreviewResponse)
def sumrule_preview_peem(
    name: str,
    request: PeemSumruleRequest,
    session: Session = Depends(current_session),
) -> PeemSumrulePreviewResponse:
    """Compute XMCD sum rule; return curves and moments without writing the tree."""
    _require_tensor(session, name)
    return _sumrule_to_preview(_run_sumrule(session, name, request))


@router.post("/{name}/sumrule/apply", response_model=PeemSumruleApplySummary)
def sumrule_apply_peem(
    name: str,
    request: PeemSumruleRequest,
    session: Session = Depends(current_session),
) -> PeemSumruleApplySummary:
    """Write /analysis/sumrule only."""
    _require_tensor(session, name)
    result = _run_sumrule(session, name, request)
    ens = result.ensemble
    ds = sumrule_analysis_dataset(
        result.energy,
        result.mu_plus,
        result.mu_minus,
        integrals=result.integrals,
        integral_stds={
            "p": ens["p_std"],
            "q": ens["q_std"],
            "r": ens["r_std"],
        },
        moment_vals=result.moment_vals,
        moment_stds={
            "m_orb": ens["m_orb_std"],
            "m_spin_plus_dipole": ens["m_spin_plus_dipole_std"],
        },
        ensemble=ens,
        nh=request.nh,
        l3=(request.l3_lo, request.l3_hi),
        l2=(request.l2_lo, request.l2_hi),
        r_win=(request.r_lo, request.r_hi),
        i0_applied=result.i0_applied,
        source_kind=result.source_kind,
        tags=result.tags,
        window_delta=result.window_delta,
        window_n=request.window_n,
        bg_e0=result.bg_e0,
        bg_e1=result.bg_e1,
        bg_delta=result.bg_delta if result.bg_e0 is not None else None,
        bg_n=result.bg_n if result.bg_e0 is not None else None,
        seed=request.seed,
        use_roi=request.use_roi,
        roi=result.roi_dict,
    )
    ds.attrs["energy_source"] = result.energy_source
    if not session.workspace.write_analysis_data(name, "sumrule", ds):
        raise HTTPException(status_code=404, detail=f"PEEM data '{name}' not found.")

    return PeemSumruleApplySummary(
        name=name,
        i0_applied=result.i0_applied,
        source_kind=result.source_kind,
        tag_plus=result.tags[0],
        tag_minus=result.tags[1],
        energy_source=result.energy_source,
    )


@router.get("/{name}/sumrule", response_model=PeemSumrulePreviewResponse)
def sumrule_get_peem(
    name: str,
    session: Session = Depends(current_session),
) -> PeemSumrulePreviewResponse:
    """Return stored /analysis/sumrule curves and moments."""
    _require_tensor(session, name)
    ds = session.workspace.pull_analysis_data(name, "sumrule")
    if ds is None:
        raise HTTPException(
            status_code=404, detail=f"PEEM data '{name}' has no sum-rule analysis."
        )
    return _analysis_to_sumrule_preview(ds)


@router.get("/{name}/meta", response_model=PeemMeta)
def get_meta(
    name: str,
    session: Session = Depends(current_session),
) -> PeemMeta:
    tensor = _require_tensor(session, name)
    metadata = tensor.metadata or {}
    processed = _processed_tensor(session, name)
    processed_metadata = (processed.metadata or {}) if processed is not None else {}
    processed_is_pair = processed is not None and processed.value.ndim == 4
    processed_is_frame = processed is not None and processed.value.ndim == 3
    bg_meta = _bg_meta_fields(session, name)
    sumrule_meta = _sumrule_meta_fields(session, name)
    return PeemMeta(
        name=name,
        shape=[int(size) for size in tensor.value.shape],
        labels=list(tensor.labels),
        n_frames=int(tensor.value.shape[0]),
        frame_names=[str(value) for value in metadata.get("frame_names", [])],
        pol=[str(value) for value in metadata.get("pol", [])],
        csv_attached=bool(metadata.get("csv_attached", False)),
        I0_present=metadata.get("I0") is not None,
        I0=metadata.get("I0"),
        has_processed=processed is not None,
        processed_shape=(
            [int(size) for size in processed.value.shape] if processed is not None else None
        ),
        processed_is_paired=processed_is_pair,
        n_processed_frames=(
            int(processed.value.shape[0]) if processed_is_frame else None
        ),
        pair_mode=(
            str(processed_metadata.get("pair_mode"))
            if processed_metadata.get("pair_mode") is not None
            else None
        ),
        n_pairs=int(processed.value.shape[0]) if processed_is_pair else None,
        channel_tags=[
            str(value) for value in processed_metadata.get("channel_tags", [])
        ],
        unpaired_count=len(processed_metadata.get("unpaired", [])),
        has_drift=processed_metadata.get("drift_method") is not None,
        drift_method=(
            str(processed_metadata["drift_method"])
            if processed_metadata.get("drift_method") is not None
            else None
        ),
        separated_channels=session.workspace.list_processed_children(name),
        has_background=bg_meta["has_background"],
        has_processed_bg=bg_meta["has_processed_bg"],
        energy_source=bg_meta["energy_source"],
        processed_bg_node=bg_meta["processed_bg_node"],
        n_bg_frames=bg_meta["n_bg_frames"],
        has_sumrule=sumrule_meta["has_sumrule"],
        sumrule_i0_applied=sumrule_meta["sumrule_i0_applied"],
        sumrule_tags=sumrule_meta["sumrule_tags"],
    )


@router.get("/{name}/frame/{i}", response_model=PeemFrame)
def get_frame(
    name: str,
    i: int,
    node: str = "raw",
    channel: int = 0,
    session: Session = Depends(current_session),
) -> PeemFrame:
    raw_tensor = _require_tensor(session, name)
    tag: str | None = None
    if node == "raw":
        tensor = raw_tensor
        if not 0 <= i < tensor.value.shape[0]:
            raise HTTPException(status_code=404, detail=f"Frame index {i} is out of range.")
        frame = np.asarray(tensor.value[i], dtype=float)
    elif node == "processed":
        tensor = _processed_tensor(session, name)
        if tensor is None:
            raise HTTPException(
                status_code=404, detail=f"PEEM data '{name}' has no processed data."
            )
        if not 0 <= i < tensor.value.shape[0]:
            kind = "Pair" if tensor.value.ndim == 4 else "Frame"
            raise HTTPException(
                status_code=404, detail=f"{kind} index {i} is out of range."
            )
        if tensor.value.ndim == 4:
            if not 0 <= channel < tensor.value.shape[1]:
                raise HTTPException(
                    status_code=404, detail=f"Channel index {channel} is out of range."
                )
            frame = np.asarray(tensor.value[i, channel], dtype=float)
        else:
            frame = np.asarray(tensor.value[i], dtype=float)
    elif node.startswith("processed/"):
        tensor, tag = _separated_tensor(session, name, node)
        if not 0 <= i < tensor.value.shape[0]:
            raise HTTPException(
                status_code=404, detail=f"Frame index {i} is out of range."
            )
        frame = np.asarray(tensor.value[i], dtype=float)
    else:
        raise HTTPException(
            status_code=422,
            detail="node must be 'raw', 'processed', or 'processed/<tag>'.",
        )

    finite = frame[np.isfinite(frame)]
    if finite.size:
        vmin, vmax = np.percentile(finite, [1, 99])
        if vmin == vmax:
            vmin, vmax = finite.min(), finite.max()
    else:
        vmin = vmax = 0.0
    metadata = tensor.metadata or {}
    pol = metadata.get("pol", [])
    frame_names = metadata.get("frame_names", [])
    channel_tags = metadata.get("channel_tags", [])
    is_pair = node == "processed" and tensor.value.ndim == 4
    is_separated = node.startswith("processed/")
    return PeemFrame(
        index=i,
        shape=[int(size) for size in frame.shape],
        intensity=frame.tolist(),
        vmin=float(vmin),
        vmax=float(vmax),
        pol=(
            str(tag)
            if is_separated
            else str(pol[i]) if not is_pair and i < len(pol) else None
        ),
        frame_name=(
            str(frame_names[i]) if not is_pair and i < len(frame_names) else None
        ),
        node=node,
        pair=i if is_pair else None,
        channel=channel if is_pair else None,
        channel_tag=(
            str(tag)
            if is_separated
            else (
                str(channel_tags[channel])
                if is_pair and channel < len(channel_tags)
                else None
            )
        ),
    )
