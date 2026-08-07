"""Crystal endpoints: load a CIF and hand its geometry to the browser.

Routes requests to `core.crystallography`; it performs no geometry of its own.
The response is renderer-agnostic (positions, radii, bond pairs) so three.js,
PyVista, or an exporter can all consume the same payload.
"""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pymatgen.core import Structure

from tensorspec.core.crystallography import CrystalEngine
from tensorspec.web.server.schemas import (
    Atom,
    Bond,
    CrystalGeometry,
    CrystalSummary,
    GeometryRequest,
)
from tensorspec.web.server.session import Session, current_session

router = APIRouter(prefix="/api/crystal", tags=["crystal"])

# A shared server should not parse an arbitrarily large upload into memory.
MAX_CIF_BYTES = 8 * 1024 * 1024

# Beyond this the browser cannot draw smoothly, so refuse rather than hang the tab.
MAX_RENDER_ATOMS = 20000

DEFAULT_RADIUS = 1.2


def _summarize(name: str, structure: Structure) -> CrystalSummary:
    lattice = structure.lattice
    symmetry = CrystalEngine.get_symmetry_info(structure)
    return CrystalSummary(
        name=name,
        formula=structure.composition.reduced_formula,
        spacegroup=symmetry["spacegroup"],
        n_sites=len(structure),
        lattice={
            "a": lattice.a, "b": lattice.b, "c": lattice.c,
            "alpha": lattice.alpha, "beta": lattice.beta, "gamma": lattice.gamma,
            "volume": lattice.volume,
        },
    )


def _require_structure(session: Session, name: str) -> Structure:
    structure = session.workspace.pull_structure_object(name)
    if structure is None:
        raise HTTPException(
            status_code=404,
            detail=f"'{name}' is not a crystal with stored atoms in this session.",
        )
    return structure


@router.post("/load", response_model=CrystalSummary)
async def load_cif(
    file: UploadFile = File(...),
    name: str = Form(default=""),
    session: Session = Depends(current_session),
) -> CrystalSummary:
    """Parse an uploaded CIF and store it in the caller's workspace."""
    if not file.filename or not file.filename.lower().endswith((".cif", ".vasp", ".poscar")):
        raise HTTPException(status_code=400, detail="Expected a .cif file.")

    payload = await file.read()
    if len(payload) > MAX_CIF_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 8 MB limit.")

    try:
        structure = Structure.from_str(payload.decode("utf-8", errors="replace"), fmt="cif")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse the CIF: {exc}")

    # Never trust the client's name for a dictionary key it will later address.
    label = (name.strip() or file.filename.rsplit(".", 1)[0])[:64]
    session.workspace.push_crystal_structure(
        label, structure.lattice.matrix, structure=structure
    )
    return _summarize(label, structure)


@router.get("/{name}/summary", response_model=CrystalSummary)
def get_summary(name: str, session: Session = Depends(current_session)) -> CrystalSummary:
    return _summarize(name, _require_structure(session, name))


@router.post("/{name}/geometry", response_model=CrystalGeometry)
def get_geometry(
    name: str,
    request: GeometryRequest,
    session: Session = Depends(current_session),
) -> CrystalGeometry:
    """Expand to a supercell and return atoms, bonds, and the cell frame."""
    structure = _require_structure(session, name)

    projected = len(structure) * request.cell_count
    if projected > MAX_RENDER_ATOMS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{projected} atoms exceeds the {MAX_RENDER_ATOMS} the browser can draw. "
                "Reduce the supercell."
            ),
        )

    supercell = structure.copy()
    if request.cell_count > 1:
        supercell.make_supercell([request.nx, request.ny, request.nz])

    coords = supercell.cart_coords
    atoms = [
        Atom(
            element=site.specie.symbol,
            position=[float(v) for v in coords[idx]],
            radius=float(site.specie.atomic_radius or DEFAULT_RADIUS),
        )
        for idx, site in enumerate(supercell)
    ]

    bonds: list[Bond] = []
    if request.show_bonds:
        found = CrystalEngine.compute_bonds(
            supercell, thresh_multiplier=request.bond_threshold
        )
        bonds = [Bond(i=int(a), j=int(b)) for a, b in zip(found["i"], found["j"])]

    center = coords.mean(axis=0) if len(coords) else np.zeros(3)

    return CrystalGeometry(
        name=name,
        atoms=atoms,
        bonds=bonds,
        cell=[[float(v) for v in row] for row in supercell.lattice.matrix],
        center=[float(v) for v in center],
        elements=sorted({site.specie.symbol for site in supercell}),
        n_atoms=len(atoms),
    )
