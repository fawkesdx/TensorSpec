"""DFT endpoints: tight-binding band structures from session crystals.

Delegates to `core.dft.band_service`; no physics lives here. Quantum ESPRESSO
execution is deliberately absent from this module -- it belongs behind a job
queue with a solver allowlist, not on a synchronous request.
"""
from __future__ import annotations

import time

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from tensorspec.core.dft import band_service
from tensorspec.core.dft_engine import DFTEngineRouter
from tensorspec.web.server.schemas import BandRequest, BandResult, StructureOption
from tensorspec.web.server.session import Session, current_session

router = APIRouter(prefix="/api/dft", tags=["dft"])

# Rough cost of dense diagonalisation: one k-point costs about n_orbitals^3.
# This budget keeps a synchronous request to a few seconds; heavier runs are
# what the job queue is for.
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
    """PyMatgen returns matplotlib mathtext; the browser wants plain text."""
    return (
        label.replace("$\\Gamma$", "\u0393")
        .replace("\\Gamma", "\u0393")
        .replace("$", "")
    )


@router.get("/structures", response_model=list[StructureOption])
def list_structures(session: Session = Depends(current_session)) -> list[StructureOption]:
    """Crystals in this session that carry atoms, with their hopping shells."""
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
        ))
    return options


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
    if estimated_k * orbitals ** 3 > DIAGONALISATION_BUDGET:
        raise HTTPException(
            status_code=422,
            detail=(
                f"About {estimated_k} k-points on {orbitals} orbitals is too large to solve "
                "in one request. Reduce points per segment, or use a smaller cell."
            ),
        )

    shells = engine.get_default_hopping(structure.composition.reduced_formula)

    started = time.perf_counter()
    try:
        result = band_service.calculate_bands(
            engine,
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
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    elapsed = time.perf_counter() - started

    eigenvalues = np.asarray(result["eigenvalues"])
    k_dist = np.asarray(result["k_dist"], dtype=float)
    node_idx = result["node_idx"] or []

    # Store alongside the crystal so other suites can pull the dispersion.
    session.workspace.push_band_structure(
        f"{name}_bands",
        k_dist,
        eigenvalues,
        result["eigenvectors"],
        result["k_vecs"],
        node_idx,
        result["labels"],
        orbital_positions=[site.coords.tolist() for site in structure],
    )

    return BandResult(
        name=f"{name}_bands",
        k_dist=[float(v) for v in k_dist],
        # Transposed to band-major so the browser draws one polyline per band.
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
    )
