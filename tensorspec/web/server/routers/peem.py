"""PEEM TIF load, metadata, frame, and deferred beamline-CSV endpoints."""
from __future__ import annotations

import re
import zipfile
from collections import Counter
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from tensorspec.core.io.peem_loaders import (
    find_beamline_csv,
    load_beamline_csv,
    load_tif_sequence,
    load_tif_stack,
)
from tensorspec.web.server.schemas import PeemFrame, PeemLoadSummary, PeemMeta
from tensorspec.web.server.session import Session, current_session

router = APIRouter(prefix="/api/peem", tags=["peem"])

MAX_PEEM_BYTES = 512 * 1024 * 1024
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
        raise HTTPException(status_code=404, detail=f"Not found: {resolved}")
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
    tensor = session.workspace.pull_tensor_data(name)
    if tensor is None:
        raise HTTPException(
            status_code=404,
            detail=f"'{name}' is not spectroscopy data in this session.",
        )
    if tensor.data_type != "Experimental PEEM":
        raise HTTPException(status_code=404, detail=f"'{name}' is not PEEM data.")
    return tensor


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

    explicit_csv = _csv_path_from_inputs(
        csv=csv,
        csv_path=csv_path,
        session=session,
        upload_dir=upload_dir,
    )
    auto_candidates: list[Path] = []
    loader_csv = explicit_csv
    if explicit_csv is None:
        loader_csv, auto_candidates = _auto_csv_choice(load_directory, fallback)
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

    label = _safe_label(name or fallback, "peem")
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


@router.get("/{name}/meta", response_model=PeemMeta)
def get_meta(
    name: str,
    session: Session = Depends(current_session),
) -> PeemMeta:
    tensor = _require_tensor(session, name)
    metadata = tensor.metadata or {}
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
    )


@router.get("/{name}/frame/{i}", response_model=PeemFrame)
def get_frame(
    name: str,
    i: int,
    session: Session = Depends(current_session),
) -> PeemFrame:
    tensor = _require_tensor(session, name)
    if not 0 <= i < tensor.value.shape[0]:
        raise HTTPException(status_code=404, detail=f"Frame index {i} is out of range.")
    frame = np.asarray(tensor.value[i], dtype=float)
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
    return PeemFrame(
        index=i,
        shape=[int(size) for size in frame.shape],
        intensity=frame.tolist(),
        vmin=float(vmin),
        vmax=float(vmax),
        pol=str(pol[i]) if i < len(pol) else None,
        frame_name=str(frame_names[i]) if i < len(frame_names) else None,
    )
