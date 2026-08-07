"""Crystal endpoints: CIF load, geometry, CDW, stack/twist, and Brillouin zone.

Routes requests to `core.crystallography`; it performs no geometry of its own.
The response is renderer-agnostic so three.js, PyVista, or an exporter can all
consume the same payload.
"""
from __future__ import annotations

import re

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pymatgen.core import Structure

from tensorspec.core.crystallography import CrystalEngine
from tensorspec.web.server.schemas import (
    Atom,
    Bond,
    BZGeometry,
    BZRequest,
    CrystalGeometry,
    CrystalSummary,
    GeometryRequest,
    MoireRequest,
    MoireResult,
    StackRequest,
    TemplateRequest,
)
from tensorspec.web.server.session import Session, current_session

router = APIRouter(prefix="/api/crystal", tags=["crystal"])

MAX_CIF_BYTES = 8 * 1024 * 1024
MAX_RENDER_ATOMS = 20000
DEFAULT_RADIUS = 1.2
DUMMY_CELL_A = 499.0

BUILTIN_TEMPLATES = [
    "Graphene (Monolayer)",
    "Graphene (AB Bilayer)",
    "Graphene (AA Bilayer)",
    "TaIrTe4 (1T')",
    "MoS2",
    "WSe2",
    "h-BN",
]

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


def _store(session: Session, name: str, structure: Structure) -> CrystalSummary:
    session.workspace.push_crystal_structure(
        name, structure.lattice.matrix, structure=structure
    )
    return _summarize(name, structure)


def _geometry_from_structure(
    name: str,
    structure: Structure,
    *,
    show_bonds: bool = True,
    bond_threshold: float = 1.15,
) -> CrystalGeometry:
    if len(structure) > MAX_RENDER_ATOMS:
        raise HTTPException(
            status_code=422,
            detail=f"{len(structure)} atoms exceeds the {MAX_RENDER_ATOMS} the browser can draw.",
        )

    coords = structure.cart_coords
    atoms = [
        Atom(
            element=site.specie.symbol,
            position=[float(v) for v in coords[idx]],
            radius=float(site.specie.atomic_radius or DEFAULT_RADIUS),
        )
        for idx, site in enumerate(structure)
    ]

    bonds: list[Bond] = []
    if show_bonds:
        found = CrystalEngine.compute_bonds(structure, thresh_multiplier=bond_threshold)
        bonds = [Bond(i=int(a), j=int(b)) for a, b in zip(found["i"], found["j"])]

    center = coords.mean(axis=0) if len(coords) else np.zeros(3)
    show_cell = float(structure.lattice.a) < DUMMY_CELL_A

    return CrystalGeometry(
        name=name,
        atoms=atoms,
        bonds=bonds,
        cell=[[float(v) for v in row] for row in structure.lattice.matrix],
        center=[float(v) for v in center],
        elements=sorted({site.specie.symbol for site in structure}),
        n_atoms=len(atoms),
        show_cell=show_cell,
    )


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

    label = _safe_label(name or file.filename.rsplit(".", 1)[0], "structure")
    return _store(session, label, structure)


@router.get("/templates")
def list_templates() -> dict:
    return {"templates": BUILTIN_TEMPLATES}


@router.post("/templates", response_model=CrystalSummary)
def add_template(
    request: TemplateRequest,
    session: Session = Depends(current_session),
) -> CrystalSummary:
    """Materialise a built-in 2D template into the session workspace."""
    structure = CrystalEngine.generate_template_structure(request.template_name)
    if structure is None:
        raise HTTPException(status_code=404, detail=f"Unknown template '{request.template_name}'.")
    label = _safe_label(
        request.store_as or request.template_name.replace(" ", "_").replace("(", "").replace(")", "").replace("'", ""),
        "template",
    )
    return _store(session, label, structure)


@router.post("/stack", response_model=CrystalGeometry)
def build_stack(
    request: StackRequest,
    session: Session = Depends(current_session),
) -> CrystalGeometry:
    """Rotate, shift and combine layers into one heterostructure."""
    layers_data = []
    projected = 0
    for layer in request.layers:
        structure = _require_structure(session, layer.name)
        projected += len(structure) * layer.sc_x * layer.sc_y
        layers_data.append({
            "struct": structure,
            "sc_x": layer.sc_x,
            "sc_y": layer.sc_y,
            "z_shift": layer.z_shift,
            "twist": layer.twist,
        })

    if projected > MAX_RENDER_ATOMS:
        raise HTTPException(
            status_code=422,
            detail=f"{projected} atoms exceeds the {MAX_RENDER_ATOMS} the browser can draw.",
        )

    stacked = CrystalEngine.build_heterostructure_stack(layers_data)
    label = _safe_label(request.store_as, "heterostructure")
    _store(session, label, stacked)
    return _geometry_from_structure(
        label, stacked, show_bonds=request.show_bonds, bond_threshold=request.bond_threshold
    )


@router.post("/moire", response_model=MoireResult)
def compute_moire(
    request: MoireRequest,
    session: Session = Depends(current_session),
) -> MoireResult:
    """Moiré periodicity for exactly two layers, plus a gold envelope polygon."""
    layer1 = _require_structure(session, request.layer1)
    layer2 = _require_structure(session, request.layer2)
    result = CrystalEngine.calculate_moire_superlattice(
        layer1, layer2, request.twist1, request.twist2
    )

    envelope = None
    matrix = result.get("matrix")
    if matrix is not None:
        m = np.asarray(matrix, dtype=float)
        if m.shape == (2, 2):
            a = m[0]
            b = m[1]
            z0, z1 = request.z_min, request.z_max
            # Closed polygon at mid-height for a simple LineLoop.
            z_mid = 0.5 * (z0 + z1)
            corners = [
                [0.0, 0.0, z_mid],
                [float(a[0]), float(a[1]), z_mid],
                [float(a[0] + b[0]), float(a[1] + b[1]), z_mid],
                [float(b[0]), float(b[1]), z_mid],
                [0.0, 0.0, z_mid],
            ]
            envelope = corners

    return MoireResult(
        status=result.get("status", "unknown"),
        message=result.get("message"),
        periodicity=result.get("periodicity"),
        n_cells=result.get("n_cells"),
        matrix=[[float(v) for v in row] for row in matrix] if matrix is not None else None,
        envelope=envelope,
    )


@router.get("/{name}/summary", response_model=CrystalSummary)
def get_summary(name: str, session: Session = Depends(current_session)) -> CrystalSummary:
    return _summarize(name, _require_structure(session, name))


@router.post("/{name}/geometry", response_model=CrystalGeometry)
def get_geometry(
    name: str,
    request: GeometryRequest,
    session: Session = Depends(current_session),
) -> CrystalGeometry:
    """Expand to a supercell, optionally apply CDW, and return atoms/bonds/cell."""
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

    if request.cdw_enabled:
        supercell = CrystalEngine.apply_cdw_distortion(
            supercell,
            request.cdw_target,
            (request.cdw_qx, request.cdw_qy, request.cdw_qz),
            (request.cdw_ax, request.cdw_ay, request.cdw_az),
            request.cdw_phase,
        )

    return _geometry_from_structure(
        name,
        supercell,
        show_bonds=request.show_bonds,
        bond_threshold=request.bond_threshold,
    )


@router.post("/{name}/bz", response_model=BZGeometry)
def get_brillouin_zone(
    name: str,
    request: BZRequest,
    session: Session = Depends(current_session),
) -> BZGeometry:
    """Wigner–Seitz cell of the reciprocal lattice, JSON-safe for three.js."""
    structure = _require_structure(session, name)
    raw = CrystalEngine.calculate_brillouin_zone(structure)
    if not raw:
        raise HTTPException(status_code=422, detail="Could not build a Brillouin zone for this cell.")

    scale = request.scale
    hull = np.asarray(raw["hull_points"], dtype=float) * scale
    simplices = [[int(i) for i in tri] for tri in raw["simplices"]]
    edges = [
        [[float(v) * scale for v in p] for p in edge]
        for edge in raw["edges"]
    ]

    surface_vertices = None
    surface_simplices = None
    projection_lines = None
    if request.surface and not (request.h == 0 and request.k == 0 and request.l == 0):
        surface = CrystalEngine.calculate_surface_projection(
            raw["hull_points"], structure, request.h, request.k, request.l
        )
        if surface is not None:
            offset = 1.5 * float(np.max(np.abs(raw["hull_points"]))) * scale
            normal = np.asarray(surface.get("normal", [0, 0, 1]), dtype=float)
            if np.linalg.norm(normal) > 0:
                normal = normal / np.linalg.norm(normal)
            else:
                normal = np.array([0.0, 0.0, 1.0])

            # projected_bounds are the silhouette vertices in the Miller plane;
            # origin_plane adds a centre point used by the fan triangulation.
            base = np.asarray(surface["projected_bounds"], dtype=float) * scale
            hover_ring = base + normal * offset
            centre = hover_ring.mean(axis=0)
            hover = np.vstack([hover_ring, centre])
            surface_vertices = [[float(v) for v in row] for row in hover]
            surface_simplices = [[int(i) for i in tri] for tri in surface["simplices"]]
            silhouette = np.asarray(surface["silhouette_3d"], dtype=float) * scale
            projection_lines = [
                [[float(v) for v in silhouette[i]], [float(v) for v in hover_ring[i]]]
                for i in range(min(len(silhouette), len(hover_ring)))
            ]

    return BZGeometry(
        name=name,
        hull_points=[[float(v) for v in row] for row in hull],
        simplices=simplices,
        edges=edges,
        scale=scale,
        style=request.style,
        surface_vertices=surface_vertices,
        surface_simplices=surface_simplices,
        projection_lines=projection_lines,
    )
