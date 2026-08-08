"""ARPES viewer and matrix-element simulation endpoints.

Slice / profile arithmetic is delegated to `core.tensor_ops`. Simulations go
through `ARPESEngineRouter` on the shared job queue (callable worker). Figure
export uses the headless matplotlib backend. No GUI toolkit imports here.
"""
from __future__ import annotations

import asyncio
import json
import re
import struct
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, WebSocket, WebSocketDisconnect

from tensorspec.core import tensor_ops as ops
from tensorspec.core.arpes_engine import ARPESEngineRouter
from tensorspec.core.data_models import TensorData
from tensorspec.core.dft import band_service
from tensorspec.core.dft_engine import DFTEngineRouter
from tensorspec.core.io.arpes_loader import ARPESLoader
from tensorspec.plotting.backends.arpes_figure import export_slice_figure
from tensorspec.web.server.jobs import Job, JobStatus, get_job_queue
from tensorspec.web.server.schemas import (
    ArpesLoadSummary,
    ArpesSimPushRequest,
    ArpesSimRequest,
    AxisInfo,
    Curve,
    FigureExportRequest,
    JobInfo,
    ProfileRequest,
    ProfileResponse,
    SliceRequest,
    TensorAxes,
)
from tensorspec.web.server.session import SESSION_COOKIE, Session, current_session, session_store

router = APIRouter(prefix="/api/arpes", tags=["arpes"])

MAX_SIM_VOXELS = 80 * 80 * 80
MAX_MESH_POINTS = 40 * 40
MAX_ARPES_BYTES = 512 * 1024 * 1024
MAX_LOG_BYTES = 8 * 1024 * 1024
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _safe_label(name: str, fallback: str) -> str:
    cleaned = (name.strip() or fallback)[:64]
    if not SAFE_NAME.match(cleaned):
        raise HTTPException(
            status_code=422,
            detail="Name must start with a letter or digit and contain only "
            "letters, digits, '.', '_' or '-'.",
        )
    return cleaned


@router.post("/load", response_model=ArpesLoadSummary)
async def load_arpes_file(
    file: UploadFile = File(...),
    log: UploadFile | None = File(default=None),
    name: str = Form(default=""),
    session: Session = Depends(current_session),
) -> ArpesLoadSummary:
    """Upload a MAESTRO (or other) HDF5 and push it into the session workspace."""
    filename = file.filename or ""
    if not filename.lower().endswith((".h5", ".hdf5")):
        raise HTTPException(status_code=400, detail="Expected a .h5 / .hdf5 file.")

    payload = await file.read()
    if len(payload) > MAX_ARPES_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 512 MB limit.")
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    upload_dir = Path(session.workspace.project_dir) / "uploads" / "arpes"
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_file = re.sub(r"[^\w.\-]+", "_", Path(filename).name)[:120] or "upload.h5"
    dest = upload_dir / safe_file
    dest.write_bytes(payload)

    log_path = None
    if log is not None and log.filename:
        log_bytes = await log.read()
        if len(log_bytes) > MAX_LOG_BYTES:
            raise HTTPException(status_code=413, detail="Measurement log exceeds the 8 MB limit.")
        log_name = re.sub(r"[^\w.\-]+", "_", Path(log.filename).name)[:120] or "measurement_log.csv"
        log_path = upload_dir / log_name
        log_path.write_bytes(log_bytes)

    try:
        tensor = ARPESLoader.load(dest, measurement_log_path=log_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse the file: {exc}") from exc

    label = _safe_label(name or Path(filename).stem, "arpes_dataset")
    session.workspace.push_spectroscopy_data(label, tensor)
    meta = tensor.metadata or {}
    return ArpesLoadSummary(
        name=label,
        shape=[int(n) for n in tensor.value.shape],
        labels=list(tensor.labels),
        units=list(tensor.units),
        data_type=tensor.data_type,
        facility=str(meta.get("Facility", "Unknown")),
        measurement_type=meta.get("Measurement_Type"),
        source_file=safe_file,
    )


def _require_tensor(session: Session, name: str):
    tensor = session.workspace.pull_tensor_data(name)
    if tensor is None:
        raise HTTPException(
            status_code=404,
            detail=f"'{name}' is not spectroscopy data in this session.",
        )
    return tensor


def _require_structure(session: Session, name: str):
    structure = session.workspace.pull_structure_object(name)
    if structure is None:
        raise HTTPException(
            status_code=404,
            detail=f"'{name}' is not a crystal with stored atoms in this session.",
        )
    return structure


def _validate_axes(tensor, x_idx: int, y_idx: int) -> None:
    if x_idx == y_idx:
        raise HTTPException(status_code=422, detail="The X and Y axes must differ.")
    for index in (x_idx, y_idx):
        if not 0 <= index < tensor.ndim:
            raise HTTPException(
                status_code=422,
                detail=f"Axis {index} is outside this {tensor.ndim}D tensor.",
            )


def _pack_plane(header: dict, plane: np.ndarray) -> bytes:
    """
    Frames a plane as: uint32 header length, header JSON, then float32 values.

    The header is padded so the numeric payload starts on a 4-byte boundary,
    which lets the browser wrap it in a Float32Array with no copy.
    """
    raw = json.dumps(header).encode("utf-8")
    padding = (-len(raw)) % 4
    raw += b" " * padding
    body = np.ascontiguousarray(plane, dtype="<f4").tobytes()
    return struct.pack("<I", len(raw)) + raw + body


def _require_job(job_id: str, session: Session) -> Job:
    job = get_job_queue().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job.")
    if job.session_id != session.session_id:
        raise HTTPException(status_code=403, detail="That job belongs to another session.")
    return job


def _experiment_kwargs(request: ArpesSimRequest) -> dict:
    return {
        "photon_energy": request.photon_energy,
        "work_function": request.work_function,
        "inner_potential": request.inner_potential,
        "temperature": request.temperature,
        "incidence_angle": request.incidence_angle,
        "polarization": request.polarization,
        "lin_pol_angle": request.lin_pol_angle,
        "matrix_element_mode": request.matrix_element_mode,
        "manip_theta": request.manip_theta,
        "manip_azimuth": request.manip_azimuth,
        "manip_tilt": request.manip_tilt,
        "hkl": (request.h, request.k, request.l),
        "k_bounds": {
            "X": [request.kx.min, request.kx.max, request.kx.steps],
            "Y": [request.ky.min, request.ky.max, request.ky.steps],
            "E": [request.energy.min, request.energy.max, request.energy.steps],
        },
        "se_width": request.se_width,
        "res_E": request.res_E,
        "res_k": request.res_k,
        "slit_angle": request.slit_angle,
    }


def _build_sim_worker(session_id: str, request: ArpesSimRequest):
    """Closure that owns the sim inputs; the queue only passes the Job."""

    def worker(job: Job) -> None:
        session = session_store._sessions.get(session_id)
        if session is None:
            raise RuntimeError("Session expired before the simulation started.")

        voxels = request.kx.steps * request.ky.steps * request.energy.steps
        if voxels > MAX_SIM_VOXELS:
            raise ValueError(
                f"Detector grid is {voxels} voxels; the shared-server cap is {MAX_SIM_VOXELS}."
            )
        mesh_pts = request.mesh_resolution ** 2
        if mesh_pts > MAX_MESH_POINTS:
            raise ValueError(
                f"TB mesh is {mesh_pts} k-points; the shared-server cap is {MAX_MESH_POINTS}."
            )

        structure = session.workspace.pull_structure_object(request.crystal_name)
        if structure is None:
            raise ValueError(f"Crystal '{request.crystal_name}' is missing from the workspace.")

        job.append_log(f"[arpes] loading {request.crystal_name} ({len(structure)} sites)")
        engine_router = DFTEngineRouter()
        engine_router.load_structure(structure)
        chinook = engine_router.chinook

        shells = chinook.get_default_hopping(structure.composition.reduced_formula)
        shell_keys = list(shells.keys())
        hoppings = list(request.hoppings[: len(shell_keys)])
        while len(hoppings) < len(shell_keys):
            hoppings.append(0.0)

        job.append_log(
            f"[arpes] building {request.mesh_resolution}×{request.mesh_resolution} TB mesh"
        )
        if job._cancel.is_set():
            return

        band_data = band_service.calculate_2d_mesh(
            chinook,
            kx_min=request.kx.min,
            kx_max=request.kx.max,
            ky_min=request.ky.min,
            ky_max=request.ky.max,
            resolution=request.mesh_resolution,
            shell_keys=shell_keys,
            hoppings=hoppings,
            cutoffs=request.cutoffs,
            onsite_e=request.onsite_e,
            tb_mode=request.tb_mode,
        )
        job.append_log(f"[arpes] mesh ready ({band_data['n_bands']} bands)")

        if job._cancel.is_set():
            return

        job.append_log(f"[arpes] running Option {request.model}")
        arpes = ARPESEngineRouter()
        results = arpes.run_simulation(request.model, band_data, _experiment_kwargs(request))
        intensity = np.asarray(results["intensity_broadened"], dtype=float)
        kx_ax, ky_ax = results.get("k_axes", (None, None))
        e_ax = results.get("e_axis")
        if kx_ax is None:
            kx_ax = np.linspace(request.kx.min, request.kx.max, intensity.shape[0])
            ky_ax = np.linspace(request.ky.min, request.ky.max, intensity.shape[1])
            e_ax = np.linspace(request.energy.min, request.energy.max, intensity.shape[2])

        # Viewer convention: (E, kx, ky)
        cube = np.transpose(intensity, (2, 0, 1))
        job.result = {
            "store_as": request.store_as,
            "crystal_name": request.crystal_name,
            "model": request.model,
            "intensity": cube,
            "axes": {
                "E": np.asarray(e_ax, dtype=float),
                "kx": np.asarray(kx_ax, dtype=float),
                "ky": np.asarray(ky_ax, dtype=float),
            },
            "shape": list(cube.shape),
        }
        job.append_log(f"[arpes] cube shape {list(cube.shape)} (E, kx, ky)")

    return worker


@router.post("/simulate", response_model=JobInfo)
def queue_simulation(
    request: ArpesSimRequest,
    session: Session = Depends(current_session),
) -> JobInfo:
    """Queue Option A or B1; rebuilds a 2D TB mesh from the chosen crystal."""
    _require_structure(session, request.crystal_name)
    voxels = request.kx.steps * request.ky.steps * request.energy.steps
    if voxels > MAX_SIM_VOXELS:
        raise HTTPException(
            status_code=422,
            detail=f"Detector grid is {voxels} voxels; reduce steps (cap {MAX_SIM_VOXELS}).",
        )

    run_dir = Path(session.workspace.project_dir) / "arpes_jobs" / request.store_as
    run_dir.mkdir(parents=True, exist_ok=True)
    queue = get_job_queue()
    try:
        job = queue.submit_callable(
            session_id=session.session_id,
            run_name=request.store_as,
            run_dir=run_dir,
            worker=_build_sim_worker(session.session_id, request),
            total_steps=2,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return JobInfo(**job.to_dict())


@router.get("/jobs/{job_id}", response_model=JobInfo)
def get_sim_job(job_id: str, session: Session = Depends(current_session)) -> JobInfo:
    return JobInfo(**_require_job(job_id, session).to_dict())


@router.post("/jobs/{job_id}/cancel", response_model=JobInfo)
def cancel_sim_job(job_id: str, session: Session = Depends(current_session)) -> JobInfo:
    try:
        job = get_job_queue().cancel(job_id, session.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown job.")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    return JobInfo(**job.to_dict())


@router.post("/jobs/{job_id}/push")
def push_sim_result(
    job_id: str,
    request: ArpesSimPushRequest,
    session: Session = Depends(current_session),
):
    """Promote a finished simulation cube into the spectroscopy workspace."""
    job = _require_job(job_id, session)
    if job.status != JobStatus.SUCCEEDED or not isinstance(job.result, dict):
        raise HTTPException(status_code=409, detail="Simulation is not ready to push.")

    name = request.name or job.result.get("store_as") or "simulated_arpes"
    axes = job.result["axes"]
    tensor = TensorData(
        value=np.asarray(job.result["intensity"], dtype=float),
        axes=[axes["E"], axes["kx"], axes["ky"]],
        labels=["Energy", "kx (Slit)", "ky (Deflection)"],
        units=["eV", "1/A", "1/A"],
        data_type="Simulated ARPES",
        metadata={
            "Source": "ARPESEngineRouter",
            "Model": job.result.get("model"),
            "Crystal": job.result.get("crystal_name"),
            "Job": job_id,
        },
    )
    session.workspace.push_spectroscopy_data(name, tensor)
    return {"name": name, "shape": list(tensor.value.shape)}


@router.get("/jobs/{job_id}/preview")
def preview_sim_slice(
    job_id: str,
    session: Session = Depends(current_session),
    e_index: int = 0,
):
    """Constant-energy contour from a finished sim, for the simulator canvas."""
    job = _require_job(job_id, session)
    if job.status != JobStatus.SUCCEEDED or not isinstance(job.result, dict):
        raise HTTPException(status_code=409, detail="Simulation is not ready.")

    cube = np.asarray(job.result["intensity"], dtype=float)
    e_index = int(np.clip(e_index, 0, cube.shape[0] - 1))
    plane = cube[e_index].T
    kx = job.result["axes"]["kx"]
    ky = job.result["axes"]["ky"]
    e_ax = job.result["axes"]["E"]
    header = {
        "shape": [int(plane.shape[0]), int(plane.shape[1])],
        "x_axis": [float(v) for v in kx],
        "y_axis": [float(v) for v in ky],
        "extent": [float(kx[0]), float(kx[-1]), float(ky[0]), float(ky[-1])],
        "vmin": float(np.nanmin(plane)),
        "vmax": float(np.nanmax(plane) or 1.0),
        "stride": [1, 1],
        "x_label": "kx (Slit)",
        "y_label": "ky (Deflection)",
        "x_unit": "1/A",
        "y_unit": "1/A",
        "full_shape": [int(plane.shape[0]), int(plane.shape[1])],
        "energy": float(e_ax[e_index]),
        "e_index": e_index,
        "n_energy": int(cube.shape[0]),
    }
    return Response(content=_pack_plane(header, plane), media_type="application/octet-stream")


@router.websocket("/jobs/{job_id}/logs")
async def sim_job_logs(websocket: WebSocket, job_id: str):
    await websocket.accept()
    session_id = websocket.cookies.get(SESSION_COOKIE)
    if not session_id:
        await websocket.send_json({"type": "error", "detail": "Missing session cookie."})
        await websocket.close()
        return

    session = session_store.get_or_create(session_id)
    job = get_job_queue().get(job_id)
    if job is None or job.session_id != session.session_id:
        await websocket.send_json({"type": "error", "detail": "Unknown job."})
        await websocket.close()
        return

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str] = asyncio.Queue()

    def on_line(line: str) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, line)

    unsubscribe = job.subscribe(on_line)
    try:
        await websocket.send_json({"type": "status", **job.to_dict()})
        while True:
            if job.status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED):
                while not queue.empty():
                    await websocket.send_json({"type": "log", "line": queue.get_nowait()})
                await websocket.send_json({"type": "status", **job.to_dict()})
                break
            try:
                line = await asyncio.wait_for(queue.get(), timeout=0.5)
                await websocket.send_json({"type": "log", "line": line})
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "status", **job.to_dict()})
    except WebSocketDisconnect:
        pass
    finally:
        unsubscribe()


@router.get("/{name}/axes", response_model=TensorAxes)
def get_axes(name: str, session: Session = Depends(current_session)) -> TensorAxes:
    """Dimensions of a tensor, plus a sensible opening view."""
    tensor = _require_tensor(session, name)
    described = ops.describe_axes(tensor)

    default_y = 0 if tensor.ndim > 1 else 0
    default_x = 1 if tensor.ndim > 1 else 0
    fixed = {
        axis["index"]: axis["size"] // 2
        for axis in described
        if axis["index"] not in (default_x, default_y)
    }

    return TensorAxes(
        name=name,
        data_type=tensor.data_type,
        ndim=tensor.ndim,
        axes=[AxisInfo(**axis) for axis in described],
        default_x=default_x,
        default_y=default_y,
        default_fixed=fixed,
    )


@router.post("/{name}/slice")
def get_slice(
    name: str,
    request: SliceRequest,
    session: Session = Depends(current_session),
) -> Response:
    """A 2D plane, decimated to fit the viewport, as a binary payload."""
    tensor = _require_tensor(session, name)
    _validate_axes(tensor, request.x_idx, request.y_idx)

    result = ops.extract_slice(tensor, request.x_idx, request.y_idx, request.fixed)
    plane, x_axis, y_axis, steps = ops.downsample_plane(
        result["values"], result["x_axis"], result["y_axis"], request.max_points
    )

    header = {
        "shape": [int(plane.shape[0]), int(plane.shape[1])],
        "x_axis": [float(v) for v in x_axis],
        "y_axis": [float(v) for v in y_axis],
        "extent": [float(v) for v in result["extent"]],
        "vmin": result["vmin"],
        "vmax": result["vmax"],
        "stride": [int(steps[0]), int(steps[1])],
        "x_label": tensor.labels[request.x_idx],
        "y_label": tensor.labels[request.y_idx],
        "x_unit": tensor.units[request.x_idx],
        "y_unit": tensor.units[request.y_idx],
        "full_shape": [int(result["values"].shape[0]), int(result["values"].shape[1])],
    }
    return Response(content=_pack_plane(header, plane), media_type="application/octet-stream")


@router.post("/{name}/profiles", response_model=ProfileResponse)
def get_profiles(
    name: str,
    request: ProfileRequest,
    session: Session = Depends(current_session),
) -> ProfileResponse:
    """Curves through the crosshair (EDC/MDC when energy is on Y)."""
    tensor = _require_tensor(session, name)
    _validate_axes(tensor, request.x_idx, request.y_idx)

    if request.ortho_idx is not None:
        if not 0 <= request.ortho_idx < tensor.ndim:
            raise HTTPException(status_code=422, detail="Orthogonal axis is out of range.")
        if request.ortho_idx in (request.x_idx, request.y_idx):
            raise HTTPException(
                status_code=422,
                detail="The orthogonal axis must differ from the displayed axes.",
            )

    result = ops.extract_slice(tensor, request.x_idx, request.y_idx, request.fixed)
    plane = result["values"]

    x_center = min(request.x_center, plane.shape[1] - 1)
    y_center = min(request.y_center, plane.shape[0] - 1)
    x_bounds = ops.integration_bounds(x_center, request.dx, plane.shape[1])
    y_bounds = ops.integration_bounds(y_center, request.dy, plane.shape[0])

    profiles = ops.extract_profiles(plane, x_bounds, y_bounds, request.mode)

    ortho_curve = None
    if request.ortho_idx is not None:
        ortho = ops.extract_orthogonal_profile(
            tensor, request.ortho_idx, request.x_idx, request.y_idx,
            x_bounds, y_bounds, request.fixed, request.mode,
        )
        ortho_curve = Curve(
            axis=[float(v) for v in ortho["axis"]],
            values=[float(v) for v in ortho["values"]],
            label=tensor.labels[request.ortho_idx],
            unit=tensor.units[request.ortho_idx],
        )

    return ProfileResponse(
        x=Curve(
            axis=[float(v) for v in result["x_axis"]],
            values=[float(v) for v in profiles["x"]],
            label=tensor.labels[request.x_idx],
            unit=tensor.units[request.x_idx],
        ),
        y=Curve(
            axis=[float(v) for v in result["y_axis"]],
            values=[float(v) for v in profiles["y"]],
            label=tensor.labels[request.y_idx],
            unit=tensor.units[request.y_idx],
        ),
        ortho=ortho_curve,
        window={
            "x1": x_bounds[0], "x2": x_bounds[1],
            "y1": y_bounds[0], "y2": y_bounds[1],
        },
        mode=request.mode,
    )


@router.post("/{name}/export/figure")
def export_figure(
    name: str,
    request: FigureExportRequest,
    session: Session = Depends(current_session),
):
    """Matplotlib PDF/SVG of the current slice (Qt vector-export parity)."""
    tensor = _require_tensor(session, name)
    _validate_axes(tensor, request.x_idx, request.y_idx)

    result = ops.extract_slice(tensor, request.x_idx, request.y_idx, request.fixed)
    plane = result["values"]
    x_center = min(request.x_center, plane.shape[1] - 1)
    y_center = min(request.y_center, plane.shape[0] - 1)
    x_bounds = ops.integration_bounds(x_center, request.dx, plane.shape[1])
    y_bounds = ops.integration_bounds(y_center, request.dy, plane.shape[0])

    x_profile = y_profile = None
    if request.include_profiles:
        profiles = ops.extract_profiles(plane, x_bounds, y_bounds, request.mode)
        x_profile = profiles["x"]
        y_profile = profiles["y"]

    cross = (
        float(result["x_axis"][x_center]),
        float(result["y_axis"][y_center]),
    )
    payload = export_slice_figure(
        plane,
        result["x_axis"],
        result["y_axis"],
        x_profile=x_profile,
        y_profile=y_profile,
        x_label=tensor.labels[request.x_idx],
        y_label=tensor.labels[request.y_idx],
        x_unit=tensor.units[request.x_idx],
        y_unit=tensor.units[request.y_idx],
        crosshair=cross,
        title=request.title or name,
        fmt=request.fmt,
    )
    media = "application/pdf" if request.fmt == "pdf" else "image/svg+xml"
    filename = f"{name}_figure.{request.fmt}"
    return Response(
        content=payload,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


