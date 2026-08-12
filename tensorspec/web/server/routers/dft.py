"""DFT endpoints: tight-binding bands and the Quantum ESPRESSO pipeline.

Band structures still run synchronously. QE runs go through the job queue:
inputs are generated into the session workspace, commands are built from the
server allowlist, and stdout is streamed over a WebSocket. No user-supplied
shell text is ever executed.
"""
from __future__ import annotations

import io
import os
import time
import zipfile

import numpy as np
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from tensorspec.core.dft import band_service
from tensorspec.core.dft import qe_pipeline
from tensorspec.core.dft import qe_slab
from tensorspec.core.dft.nbnd_suggest import suggest_nbnd_base
from tensorspec.core.dft.qe_pipeline import PipelineParams, SolverPaths
from tensorspec.core.dft_engine import DFTEngineRouter
from tensorspec.core import mlip_engine
from tensorspec.web.server.config import REPO_ROOT, load_solver_config
from tensorspec.web.server.jobs import get_job_queue
from tensorspec.web.server.schemas import (
    BandRequest,
    BandResult,
    FatBandRequest,
    FatBandResult,
    GapPredictRequest,
    IsoenergyRequest,
    IsoenergyResult,
    JobInfo,
    QEGenerateResponse,
    QERequest,
    SlabPrepareRequest,
    SlabPrepareResult,
    SolverStatus,
    StructureOption,
)
from tensorspec.web.server.session import Session, current_session, session_store

router = APIRouter(prefix="/api/dft", tags=["dft"])

DIAGONALISATION_BUDGET = 5e8


def _require_structure(session: Session, name: str):
    structure = session.workspace.pull_structure_object(name)
    if structure is None:
        raise HTTPException(
            status_code=404,
            detail=f"'{name}' is not a crystal with stored atoms in this session.",
        )
    return structure


def _engine_for(structure) -> DFTEngineRouter:
    engine = DFTEngineRouter()
    engine.load_structure(structure)
    return engine


def _orbital_count(engine, structure, use_soc: bool) -> int:
    total = 0
    for site in structure:
        total += len(engine._get_orbital_basis(site.specie.symbol))
    return total * 2 if use_soc else total


def _display_label(label: str) -> str:
    return (
        label.replace("$\\Gamma$", "\u0393")
        .replace("\\Gamma", "\u0393")
        .replace("$", "")
    )


def _wannier_dir(session: Session, name: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:64] or "structure"
    path = Path(session.workspace.project_dir) / "wannier" / safe
    path.mkdir(parents=True, exist_ok=True)
    return path


def _wannier_hr_path(session: Session, name: str) -> Path | None:
    hr = _wannier_dir(session, name) / "wannier90_hr.dat"
    return hr if hr.is_file() else None


def _solver_status() -> SolverStatus:
    try:
        cfg = load_solver_config(require_binaries=False)
        cfg.require_exists()
        return SolverStatus(
            available=True,
            pw=str(cfg.pw),
            wannier90=str(cfg.wannier90),
            pw2wannier90=str(cfg.pw2wannier90),
            mpirun=str(cfg.mpirun) if cfg.mpirun else None,
            pseudo_dir=str(cfg.pseudo_dir),
            max_mpi_ranks=cfg.max_mpi_ranks,
        )
    except Exception as exc:
        return SolverStatus(
            available=False,
            max_mpi_ranks=8,
            detail=str(exc),
        )


def _remote_qe_script_path() -> Path:
    return REPO_ROOT / "scripts" / "remote_qe.sh"


def _einstein_ssh_commands(
    run_dir: Path, mpi_ranks: int, host: str
) -> list[list[str]]:
    script = _remote_qe_script_path()
    return [
        [
            "bash",
            str(script),
            str(run_dir),
            "--np",
            str(mpi_ranks),
            "--host",
            host,
        ]
    ]


def _params_from_request(
    request: QERequest, max_mpi_ranks: int, *, structure=None
) -> PipelineParams:
    nbnd = int(request.nbnd)
    if structure is not None:
        base = suggest_nbnd_base(structure)
        suggested = min(2000, max(1, base * (2 if request.use_soc else 1)))
        # HTML default is 12; if the UI never applied the suggestion, prefer
        # the structure-based count. Keep any intentional non-default override.
        if nbnd == 12 and suggested != 12:
            nbnd = suggested
    return PipelineParams(
        ecutwfc=request.ecutwfc,
        nbnd=nbnd,
        kx=request.kx,
        ky=request.ky,
        kz=request.kz,
        use_soc=request.use_soc,
        mlwf_mode=request.mlwf_mode,
        use_mpi=request.use_mpi,
        mpi_ranks=min(request.mpi_ranks, max_mpi_ranks),
        slab_mode=bool(request.slab_mode),
        functional=request.functional,
    )


def _safe_crystal_label(name: str, fallback: str) -> str:
    import re

    cleaned = (name.strip() or fallback)[:64]
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$", cleaned):
        raise HTTPException(
            status_code=422,
            detail="Name must start with a letter or digit and contain only "
            "letters, digits, '.', '_' or '-'.",
        )
    return cleaned


def _prepare_run(
    session: Session,
    crystal_name: str,
    request: QERequest,
    *,
    relative_outdir: bool,
):
    structure = _require_structure(session, crystal_name)
    try:
        cfg = load_solver_config(require_binaries=False)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    try:
        run_dir = qe_pipeline.resolve_run_dir(session.workspace.project_dir, request.run_name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    params = _params_from_request(request, cfg.max_mpi_ranks, structure=structure)
    try:
        files = qe_pipeline.generate_inputs(
            structure,
            run_dir,
            params,
            pseudo_dir=cfg.pseudo_dir,
            relative_outdir=relative_outdir,
        )
        script = qe_pipeline.write_hpc_script(
            run_dir, params, max_mpi_ranks=cfg.max_mpi_ranks
        )
        files.append(script.name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return cfg, run_dir, params, files


@router.get("/structures", response_model=list[StructureOption])
def list_structures(session: Session = Depends(current_session)) -> list[StructureOption]:
    options = []
    for name, item in session.workspace._data.items():
        if item.get("type") != "crystal_structure":
            continue
        structure = item.get("structure")
        if structure is None:
            continue

        engine = _engine_for(structure)
        formula = structure.composition.reduced_formula
        shells = engine.get_default_hopping(formula)

        options.append(StructureOption(
            name=name,
            formula=formula,
            n_sites=len(structure),
            shell_keys=list(shells.keys()),
            default_hoppings=[float(v) for v in shells.values()],
            suggest_slab_qe=qe_slab.suggest_slab_qe(structure),
            lattice_c=float(structure.lattice.c),
            suggest_nbnd=suggest_nbnd_base(structure),
        ))
    return options


@router.get("/slab-presets")
def slab_presets():
    """Educational slab cleave presets for DFT Suite Prepare slab."""
    return {"presets": qe_slab.list_slab_presets()}


@router.post("/{name}/prepare-slab", response_model=SlabPrepareResult)
def prepare_slab(
    name: str,
    request: SlabPrepareRequest,
    session: Session = Depends(current_session),
) -> SlabPrepareResult:
    """Cleave bulk CIF / Tab-1 structure into a vacuum slab; store for QE."""
    structure = _require_structure(session, name)
    try:
        result = qe_slab.prepare_slab(
            structure,
            preset=request.preset,
            h=request.h,
            k=request.k,
            l=request.l,
            num_layers=request.num_layers,
            vacuum=request.vacuum,
            bond_threshold=request.bond_threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    store_as = request.store_as.strip() or f"{name}_slab"
    label = _safe_crystal_label(store_as, f"{name}_slab")
    slab = result["structure"]
    session.workspace.push_crystal_structure(
        label, slab.lattice.matrix, structure=slab
    )
    return SlabPrepareResult(
        stored_as=label,
        formula=result["formula"],
        n_sites=result["n_sites"],
        hkl=result["hkl"],
        num_layers=result["num_layers"],
        vacuum=result["vacuum"],
        preset=result["preset"],
        suggest_slab_qe=True,
        lattice_c=float(slab.lattice.c),
    )


@router.get("/solvers", response_model=SolverStatus)
def solver_status() -> SolverStatus:
    """Whether this server can queue QE jobs, and what the MPI rank cap is."""
    return _solver_status()


@router.post("/{name}/qe/generate", response_model=QEGenerateResponse)
def generate_qe_inputs(
    name: str,
    request: QERequest,
    session: Session = Depends(current_session),
) -> QEGenerateResponse:
    """Write SCF/NSCF/Wannier inputs into the session's qe_runs directory."""
    cfg, run_dir, params, files = _prepare_run(
        session, name, request, relative_outdir=False
    )
    status = _solver_status()
    return QEGenerateResponse(
        run_name=request.run_name,
        run_dir=str(run_dir.relative_to(session.workspace.project_dir)),
        files=files,
        mpi_ranks_capped=params.mpi_ranks,
        max_mpi_ranks=cfg.max_mpi_ranks,
        solvers_available=status.available,
    )


@router.post("/{name}/qe/bundle")
def download_qe_bundle(
    name: str,
    request: QERequest,
    session: Session = Depends(current_session),
):
    """
    Zip the generated inputs for an HPC submission.

    Uses relative ``outdir`` paths and bare solver names in ``run_pipeline.sh``
    so the archive is portable. The server does not execute that script.
    """
    cfg, run_dir, params, files = _prepare_run(
        session, name, request, relative_outdir=True
    )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in run_dir.rglob("*"):
            if path.is_file():
                archive.write(path, arcname=str(path.relative_to(run_dir.parent)))
    buffer.seek(0)

    filename = f"{qe_pipeline.sanitize_run_name(request.run_name)}.zip"
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/{name}/qe/queue", response_model=JobInfo)
def queue_qe_run(
    name: str,
    request: QERequest,
    session: Session = Depends(current_session),
) -> JobInfo:
    """Generate inputs if needed and enqueue the allowlisted pipeline."""
    cfg, run_dir, params, _files = _prepare_run(
        session, name, request, relative_outdir=False
    )
    if request.backend == "einstein_ssh":
        script = _remote_qe_script_path()
        if not script.is_file():
            raise HTTPException(503, detail=f"remote_qe.sh not found at {script}")
        host = os.environ.get("TENSORSPEC_QE_SSH_HOST", "einstein").strip() or "einstein"
        np_ranks = min(request.mpi_ranks, cfg.max_mpi_ranks)
        commands = _einstein_ssh_commands(run_dir.resolve(), np_ranks, host)
    else:
        try:
            cfg.require_exists()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

        solvers = SolverPaths(
            pw=cfg.pw,
            wannier90=cfg.wannier90,
            pw2wannier90=cfg.pw2wannier90,
            mpirun=cfg.mpirun,
        )
        commands = qe_pipeline.build_pipeline_commands(
            solvers, params, max_mpi_ranks=cfg.max_mpi_ranks
        )

    queue = get_job_queue(cfg.max_global_jobs, cfg.max_jobs_per_session)
    try:
        job = queue.submit(
            session_id=session.session_id,
            run_name=request.run_name,
            run_dir=run_dir,
            commands=commands,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return JobInfo(**job.to_dict())


@router.get("/jobs/{job_id}", response_model=JobInfo)
def get_job(job_id: str, session: Session = Depends(current_session)) -> JobInfo:
    queue = get_job_queue()
    job = queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job.")
    if job.session_id != session.session_id:
        raise HTTPException(status_code=403, detail="That job belongs to another session.")
    return JobInfo(**job.to_dict())


@router.post("/jobs/{job_id}/cancel", response_model=JobInfo)
def cancel_job(job_id: str, session: Session = Depends(current_session)) -> JobInfo:
    queue = get_job_queue()
    try:
        job = queue.cancel(job_id, session.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown job.")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    return JobInfo(**job.to_dict())


@router.websocket("/jobs/{job_id}/logs")
async def job_logs(websocket: WebSocket, job_id: str):
    """Stream a job's stdout lines to the browser."""
    await websocket.accept()
    cookie = websocket.cookies.get("ts_session")
    session = session_store.get_or_create(cookie)

    queue = get_job_queue()
    job = queue.get(job_id)
    if job is None or job.session_id != session.session_id:
        await websocket.send_json({"type": "error", "message": "Unknown job."})
        await websocket.close()
        return

    import asyncio

    loop = asyncio.get_running_loop()
    outbound: asyncio.Queue[str] = asyncio.Queue()

    def on_line(line: str) -> None:
        loop.call_soon_threadsafe(outbound.put_nowait, line)

    unsubscribe = job.subscribe(on_line)
    try:
        while True:
            try:
                line = await asyncio.wait_for(outbound.get(), timeout=0.5)
                await websocket.send_json({"type": "log", "line": line})
            except asyncio.TimeoutError:
                pass

            if job.status.value in ("succeeded", "failed", "cancelled"):
                # Drain anything still buffered, then send the final status.
                while not outbound.empty():
                    await websocket.send_json({"type": "log", "line": outbound.get_nowait()})
                await websocket.send_json({"type": "status", "job": job.to_dict()})
                break
    except WebSocketDisconnect:
        pass
    finally:
        unsubscribe()


@router.get("/{name}/bz-context")
def bz_context(name: str, session: Session = Depends(current_session)):
    """Educational note: folded supercell BZ vs standard high-symmetry path."""
    structure = _require_structure(session, name)
    return band_service.describe_bz_context(structure)


@router.post("/{name}/gap-predict")
def predict_gap(
    name: str,
    request: GapPredictRequest,
    session: Session = Depends(current_session),
):
    """MEGNet scalar band-gap prediction (surrogate when full bands are heavy)."""
    structure = _require_structure(session, name)
    try:
        return mlip_engine.predict_band_gap(structure, fidelity=request.fidelity)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/{name}/wannier")
async def upload_wannier_hr(
    name: str,
    file: UploadFile = File(...),
    scf_out: UploadFile | None = File(default=None),
    session: Session = Depends(current_session),
):
    """Store wannier90_hr.dat (and optional scf.out) for later band solves."""
    _require_structure(session, name)
    fname = (file.filename or "").lower()
    if "hr" not in fname and not fname.endswith(".dat"):
        raise HTTPException(
            status_code=400,
            detail="Expected a wannier90_hr.dat (or *.dat) Hamiltonian file.",
        )
    payload = await file.read()
    if len(payload) > 64 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File exceeds the 64 MB limit.")

    dest = _wannier_dir(session, name)
    hr_path = dest / "wannier90_hr.dat"
    hr_path.write_bytes(payload)

    scf_saved = False
    if scf_out is not None and scf_out.filename:
        scf_bytes = await scf_out.read()
        if scf_bytes:
            (dest / "scf.out").write_bytes(scf_bytes)
            scf_saved = True

    return {
        "name": name,
        "hr_path": str(hr_path),
        "bytes": len(payload),
        "scf_out_saved": scf_saved,
    }


@router.post("/{name}/bands", response_model=BandResult)
def compute_bands(
    name: str,
    request: BandRequest,
    session: Session = Depends(current_session),
) -> BandResult:
    """Solves a 1D high-symmetry band structure and stores it in the session."""
    structure = _require_structure(session, name)
    engine = _engine_for(structure)

    orbitals = _orbital_count(engine, structure, request.use_soc)
    segments = max(1, len(request.custom_labels.split(";")) - 1) if request.path_mode == "custom" else 5
    estimated_k = segments * request.points_per_segment

    shells = engine.get_default_hopping(structure.composition.reduced_formula)

    w90_filepath = None
    if request.use_wannier:
        hr = _wannier_hr_path(session, name)
        if hr is None:
            raise HTTPException(
                status_code=422,
                detail="No uploaded wannier90_hr.dat for this crystal. Load one first.",
            )
        w90_filepath = str(hr)

    needs_overlay = bool(request.overlay_wannier)
    hr_overlay = None
    if needs_overlay:
        hr_overlay = _wannier_hr_path(session, name)
        if hr_overlay is None:
            raise HTTPException(
                status_code=422,
                detail="No uploaded wannier90_hr.dat for this crystal. Load one first.",
            )
    solve_factor = 2 if (needs_overlay and not request.use_wannier) else 1
    if estimated_k * orbitals ** 3 * solve_factor > DIAGONALISATION_BUDGET:
        raise HTTPException(
            status_code=422,
            detail=(
                f"About {estimated_k} k-points on {orbitals} orbitals is too large to solve "
                "in one request. Reduce points per segment, or use a smaller cell."
            ),
        )

    band_kwargs = dict(
        path_mode=request.path_mode,
        custom_coords=request.custom_coords,
        custom_labels=request.custom_labels,
        points_per_segment=request.points_per_segment,
        shell_keys=list(shells.keys()),
        hoppings=request.hoppings,
        cutoffs=request.cutoffs,
        onsite_e=request.onsite_e,
        orbital_shifts={
            "0": request.shift_s,
            "1": request.shift_p,
            "2": request.shift_d,
        },
        use_soc=request.use_soc,
        soc_strength=request.soc_strength,
        tb_mode=request.tb_mode,
    )

    started = time.perf_counter()
    try:
        result = band_service.calculate_bands(
            engine,
            w90_filepath=w90_filepath,
            **band_kwargs,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    eigenvalues = np.asarray(result["eigenvalues"])
    k_dist = np.asarray(result["k_dist"], dtype=float)
    node_idx = result["node_idx"] or []

    overlay_bands = None
    if needs_overlay:
        if request.use_wannier:
            overlay_evals = eigenvalues
        else:
            try:
                overlay_result = band_service.calculate_bands(
                    engine,
                    w90_filepath=str(hr_overlay),
                    **band_kwargs,
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc))
            overlay_evals = np.asarray(overlay_result["eigenvalues"])
        overlay_bands = [
            [float(v) for v in overlay_evals[:, b]] for b in range(overlay_evals.shape[1])
        ]

    elapsed = time.perf_counter() - started

    session.workspace.push_band_structure(
        f"{name}_bands",
        k_dist,
        eigenvalues,
        result["eigenvectors"],
        result["k_vecs"],
        node_idx,
        result["labels"],
        orbital_positions=[site.coords.tolist() for site in structure],
        orbital_labels=[str(l) for l in (result["orbital_labels"] or [])],
    )

    return BandResult(
        name=f"{name}_bands",
        k_dist=[float(v) for v in k_dist],
        bands=[[float(v) for v in eigenvalues[:, b]] for b in range(eigenvalues.shape[1])],
        node_positions=[float(k_dist[i]) for i in node_idx],
        node_labels=[_display_label(str(l)) for l in (result["labels"] or [])],
        n_bands=int(eigenvalues.shape[1]),
        n_kpoints=int(eigenvalues.shape[0]),
        fermi_energy=float(result["fermi_energy"]),
        energy_min=float(eigenvalues.min()),
        energy_max=float(eigenvalues.max()),
        orbital_labels=[str(l) for l in (result["orbital_labels"] or [])],
        elapsed_seconds=round(elapsed, 3),
        path_kind=str(result.get("path_kind") or "standard"),
        path_title=str(result.get("path_title") or ""),
        path_note=str(result.get("path_note") or ""),
        likely_folded=bool(result.get("likely_folded")),
        weights=(
            [[float(v) for v in result["weights"][:, b]] for b in range(eigenvalues.shape[1])]
            if result.get("weights") is not None
            else None
        ),
        weight_min=float(np.min(result["weights"])) if result.get("weights") is not None else 0.0,
        weight_max=float(np.max(result["weights"])) if result.get("weights") is not None else 1.0,
        unfolded=result.get("weights") is not None,
        overlay_bands=overlay_bands,
    )


@router.post("/{name}/isoenergy", response_model=IsoenergyResult)
def compute_isoenergy(
    name: str,
    request: IsoenergyRequest,
    session: Session = Depends(current_session),
) -> IsoenergyResult:
    """Diagonalises a kx–ky mesh and returns Gaussian isoenergy density."""
    structure = _require_structure(session, name)
    engine = _engine_for(structure)

    orbitals = _orbital_count(engine, structure, request.use_soc)
    res = int(request.resolution)
    estimated_k = res * res
    if estimated_k * orbitals ** 3 > DIAGONALISATION_BUDGET:
        raise HTTPException(
            status_code=422,
            detail=(
                f"About {estimated_k} k-points on {orbitals} orbitals is too large to solve "
                "in one request. Reduce resolution, or use a smaller cell."
            ),
        )

    if request.use_wannier:
        raise HTTPException(
            status_code=422,
            detail="Isoenergy mesh does not support Wannier90 yet; use manual TB hoppings.",
        )

    shells = engine.get_default_hopping(structure.composition.reduced_formula)

    started = time.perf_counter()
    try:
        mesh = band_service.calculate_2d_mesh(
            engine,
            kx_min=request.kx_min,
            kx_max=request.kx_max,
            ky_min=request.ky_min,
            ky_max=request.ky_max,
            resolution=res,
            shell_keys=list(shells.keys()),
            hoppings=request.hoppings,
            cutoffs=request.cutoffs,
            onsite_e=request.onsite_e,
            orbital_shifts={
                "0": request.shift_s,
                "1": request.shift_p,
                "2": request.shift_d,
            },
            use_soc=request.use_soc,
            soc_strength=request.soc_strength,
            tb_mode=request.tb_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    eigenvalues = np.asarray(mesh["eigenvalues"])
    intensity = band_service.isoenergy_density(
        eigenvalues,
        energy=request.energy,
        smear=request.smear,
        grid_shape=mesh["grid_shape"],
    )
    elapsed = time.perf_counter() - started
    kx = np.asarray(mesh["kx"], dtype=float)
    ky = np.asarray(mesh["ky"], dtype=float)

    return IsoenergyResult(
        name=f"{name}_isoenergy",
        kx=[float(v) for v in kx],
        ky=[float(v) for v in ky],
        intensity=[[float(v) for v in row] for row in intensity],
        energy=float(request.energy),
        smear=float(request.smear),
        n_bands=int(mesh["n_bands"]),
        resolution=res,
        elapsed_seconds=round(elapsed, 3),
        fermi_energy=float(mesh.get("fermi_energy") or 0.0),
    )


@router.post("/{name}/bands/fat", response_model=FatBandResult)
def compute_fat_bands(
    name: str,
    request: FatBandRequest,
    session: Session = Depends(current_session),
) -> FatBandResult:
    """Re-project cached eigenvectors onto a fat-band target (no re-diagonalize)."""
    bands_name = name if name.endswith("_bands") else f"{name}_bands"
    stored = session.workspace.pull_band_structure(bands_name)
    if stored is None:
        # Allow callers that already pass the bands node name without suffix logic
        stored = session.workspace.pull_band_structure(name)
        bands_name = name if stored is not None else bands_name
    if stored is None:
        raise HTTPException(
            status_code=404,
            detail=f"No band structure '{bands_name}'. Calculate bands first.",
        )

    evecs = np.asarray(stored.get("eigenvectors"))
    labels = [str(l) for l in (stored.get("orbital_labels") or [])]
    if evecs.ndim != 3:
        raise HTTPException(
            status_code=422,
            detail="Cached band structure has no usable eigenvectors for fat projection.",
        )
    if not labels:
        # Reconstruct placeholder labels if an older cache entry lacks them
        labels = [f"orb_{i}" for i in range(evecs.shape[1])]

    try:
        idxs = band_service.resolve_fat_indices(labels, request.fat_target)
        weights = band_service.fat_band_weights(evecs, idxs)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    n_bands = int(weights.shape[1])
    target = (request.fat_target or "none").strip() or "none"
    if target.lower() in ("none", "none (standard lines)") or not idxs:
        return FatBandResult(
            name=bands_name,
            fat_target="none",
            fat_weights=None,
            fat_n_orbitals=0,
            orbital_labels=labels,
            n_bands=n_bands,
            n_kpoints=int(weights.shape[0]),
        )

    return FatBandResult(
        name=bands_name,
        fat_target=target,
        fat_weights=[[float(v) for v in weights[:, b]] for b in range(n_bands)],
        fat_n_orbitals=len(idxs),
        orbital_labels=labels,
        n_bands=n_bands,
        n_kpoints=int(weights.shape[0]),
    )
