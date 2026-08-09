"""Crystal endpoints: CIF load, geometry, CDW, stack/twist, and Brillouin zone.

Routes requests to `core.crystallography`; it performs no geometry of its own.
The response is renderer-agnostic so three.js, PyVista, or an exporter can all
consume the same payload.
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from tensorspec.core.crystallography import CrystalEngine
from tensorspec.core.io.exporters import SceneExporter
from tensorspec.plotting.backends.crystal_figure import export_crystal_figure
from tensorspec.web.server.geometry_filter import (
    filter_geometry_atoms_bonds,
    filter_structure_by_omit,
    normalize_omit_indices,
)
from tensorspec.web.server.schemas import (
    Atom,
    Bond,
    BZGeometry,
    Polyhedron,
    BZRequest,
    CrystalCifRequest,
    CrystalFigureExportRequest,
    CrystalGeometry,
    CrystalSummary,
    ExfoliateRequest,
    ExfoliateResult,
    GapPredictRequest,
    GeometryRequest,
    MoireRequest,
    MoireResult,
    PushCrystalRequest,
    RelaxRequest,
    SceneExportRequest,
    StackRequest,
    TemplateRequest,
)
from tensorspec.web.server.session import Session, current_session
from tensorspec.core import mlip_engine
from fastapi.responses import Response

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


def _detect_structure_fmt(filename: str) -> str:
    lower = (filename or "").lower()
    if lower.endswith(".cif"):
        return "cif"
    if lower.endswith(".vasp") or lower.endswith(".poscar") or lower.endswith("poscar"):
        return "poscar"
    return "cif"


def _site_element_and_label(site) -> tuple[str, str]:
    """Return (element_symbol, display_label) for one pymatgen site."""
    try:
        element = site.specie.symbol
    except Exception:
        # Disordered: take the majority species
        if getattr(site, "species", None) is not None:
            element = sorted(site.species.items(), key=lambda kv: -kv[1])[0][0].symbol
        else:
            raise
    label = getattr(site, "label", None) or element
    return element, str(label)


def _apply_basis(structure: Structure, basis: str) -> Structure:
    if basis == "conventional":
        try:
            return SpacegroupAnalyzer(structure).get_conventional_standard_structure()
        except Exception:
            return structure.copy()
    if basis == "primitive":
        try:
            return SpacegroupAnalyzer(structure).get_primitive_standard_structure()
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Could not convert to primitive basis: {exc}",
            ) from exc
    return structure.copy()


def _is_default_knobs(
    *,
    omit_atom_indices: list[int],
    nx: int,
    ny: int,
    nz: int,
    basis: str,
) -> bool:
    return (
        not omit_atom_indices
        and nx == 1
        and ny == 1
        and nz == 1
        and basis == "conventional"
    )


def _build_knobbed_structure(
    structure: Structure,
    *,
    basis: str,
    nx: int,
    ny: int,
    nz: int,
    omit_atom_indices: list[int] | None = None,
) -> Structure:
    """Apply basis, optional supercell, and omit filter — same path as Draw/export."""
    structure = _apply_basis(structure, basis)
    cell_count = nx * ny * nz
    projected = len(structure) * cell_count
    if projected > MAX_RENDER_ATOMS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{projected} atoms exceeds the {MAX_RENDER_ATOMS} the browser can draw. "
                "Reduce the supercell."
            ),
        )
    out = structure.copy()
    if cell_count > 1:
        out.make_supercell([nx, ny, nz])
    omit = normalize_omit_indices(omit_atom_indices, len(out))
    if omit:
        out = filter_structure_by_omit(out, omit)
    return out


def _filter_geometry_by_omit(geo: CrystalGeometry, omit: set[int]) -> CrystalGeometry:
    if not omit:
        return geo
    filtered_atoms, filtered_bonds = filter_geometry_atoms_bonds(geo.atoms, geo.bonds, omit)
    return geo.model_copy(
        update={
            "atoms": filtered_atoms,
            "bonds": filtered_bonds,
            "n_atoms": len(filtered_atoms),
            "elements": sorted({a.element for a in filtered_atoms}),
        }
    )


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
    show_polyhedra: bool = False,
    bond_threshold: float = 1.15,
) -> CrystalGeometry:
    if len(structure) > MAX_RENDER_ATOMS:
        raise HTTPException(
            status_code=422,
            detail=f"{len(structure)} atoms exceeds the {MAX_RENDER_ATOMS} the browser can draw.",
        )

    coords = structure.cart_coords
    atoms = []
    for idx, site in enumerate(structure):
        try:
            element, label = _site_element_and_label(site)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Could not resolve species for site {idx}: {exc}",
            ) from exc
        radius = DEFAULT_RADIUS
        try:
            radius = float(site.specie.atomic_radius or DEFAULT_RADIUS)
        except Exception:
            radius = DEFAULT_RADIUS
        atoms.append(
            Atom(
                element=element,
                label=label,
                position=[float(v) for v in coords[idx]],
                radius=radius,
            )
        )

    bonds: list[Bond] = []
    bond_i = np.array([], dtype=int)
    bond_j = np.array([], dtype=int)
    if show_bonds or show_polyhedra:
        found = CrystalEngine.compute_bonds(structure, thresh_multiplier=bond_threshold)
        bond_i, bond_j = found["i"], found["j"]
        if show_bonds:
            bonds = [Bond(i=int(a), j=int(b)) for a, b in zip(bond_i, bond_j)]

    polyhedra: list[Polyhedron] = []
    if show_polyhedra:
        raw = CrystalEngine.compute_coordination_polyhedra(coords, bond_i, bond_j)
        polyhedra = [
            Polyhedron(
                center=p["center"],
                vertices=p["vertices"],
                simplices=p["simplices"],
                vertex_atom_indices=p["vertex_atom_indices"],
            )
            for p in raw
        ]

    center = coords.mean(axis=0) if len(coords) else np.zeros(3)
    show_cell = float(structure.lattice.a) < DUMMY_CELL_A

    return CrystalGeometry(
        name=name,
        atoms=atoms,
        bonds=bonds,
        polyhedra=polyhedra,
        cell=[[float(v) for v in row] for row in structure.lattice.matrix],
        center=[float(v) for v in center],
        elements=sorted({a.element for a in atoms}),
        n_atoms=len(atoms),
        show_cell=show_cell,
    )


# Jmol/CPK subset for DCC export; unknown symbols → FALLBACK_CPK.
_CPK_COLORS: dict[str, str] = {
    "H": "#FFFFFF", "He": "#D9FFFF", "Li": "#CC80FF", "Be": "#C2FF00", "B": "#FFB5B5",
    "C": "#909090", "N": "#3050F8", "O": "#FF0D0D", "F": "#90E050", "Ne": "#B3E3F5",
    "Na": "#AB5CF2", "Mg": "#8AFF00", "Al": "#BFA6A6", "Si": "#F0C8A0", "P": "#FF8000",
    "S": "#FFFF30", "Cl": "#1FF01F", "Ar": "#80D1E3", "K": "#8F40D4", "Ca": "#3DFF00",
    "Sc": "#E6E6E6", "Ti": "#BFC2C7", "V": "#A6A6AB", "Cr": "#8A99C7", "Mn": "#9C7AC7",
    "Fe": "#E06633", "Co": "#F090A0", "Ni": "#50D050", "Cu": "#C88033", "Zn": "#7D80B0",
    "Ga": "#C28F8F", "Ge": "#668F8F", "As": "#BD80E3", "Se": "#FFA100", "Br": "#A62929",
    "Kr": "#5CB8D1", "Rb": "#702EB0", "Sr": "#00FF00", "Y": "#94FFFF", "Zr": "#94E0E0",
    "Nb": "#73C2C9", "Mo": "#54B5B5", "Tc": "#3B9E9E", "Ru": "#248F8F", "Rh": "#0A7D8C",
    "Pd": "#006985", "Ag": "#C0C0C0", "Cd": "#FFD98F", "In": "#A67573", "Sn": "#668080",
    "Sb": "#9E63B5", "Te": "#D47A00", "I": "#940094", "Xe": "#429EB0", "Cs": "#57178F",
    "Ba": "#00C900", "La": "#70D4FF", "Hf": "#4DC2FF", "Ta": "#4DA6FF", "W": "#2194D6",
    "Re": "#267DAB", "Os": "#266696", "Ir": "#175487", "Pt": "#D0D0E0", "Au": "#FFD123",
    "Hg": "#B8B8D0", "Tl": "#A6544D", "Pb": "#575961", "Bi": "#9E4FB5", "U": "#008FFF",
}
FALLBACK_CPK = "#808080"
LATTICE_EDGE_COLOR = "#6f7891"
BOND_EXPORT_COLOR = "#888888"
BOND_EXPORT_RADIUS = 0.1


def _element_cpk(symbol: str) -> str:
    return _CPK_COLORS.get(symbol, FALLBACK_CPK)


def _lattice_edge_tuples(cell: list[list[float]], center: list[float]) -> list[tuple]:
    """12 parallelepiped edges from origin along a,b,c — same frame as viewer_3d._drawCell."""
    a = np.asarray(cell[0], dtype=float)
    b = np.asarray(cell[1], dtype=float)
    c = np.asarray(cell[2], dtype=float)
    origin = np.zeros(3)
    corners = [
        origin,
        a,
        b,
        c,
        a + b,
        a + c,
        b + c,
        a + b + c,
    ]
    edges = [
        (0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4),
        (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7),
    ]
    ctr = np.asarray(center, dtype=float)
    out: list[tuple] = []
    for s, e in edges:
        p1 = corners[s] - ctr
        p2 = corners[e] - ctr
        out.append(
            (float(p1[0]), float(p1[1]), float(p1[2]),
             float(p2[0]), float(p2[1]), float(p2[2]),
             LATTICE_EDGE_COLOR)
        )
    return out


def _scene_export_parts(
    geo: CrystalGeometry,
    *,
    include_atoms: bool,
    include_cell: bool,
    include_bz: bool,
    bz_geometry: BZGeometry | None,
) -> tuple[list, list, list, dict | None]:
    """Build SceneExporter tuples; positions are center-relative like the three.js viewer."""
    center = np.asarray(geo.center, dtype=float)
    atoms: list = []
    bonds: list = []
    lattice: list = []
    bz: dict | None = None

    if include_atoms:
        for atom in geo.atoms:
            pos = np.asarray(atom.position, dtype=float) - center
            atoms.append(
                (float(pos[0]), float(pos[1]), float(pos[2]),
                 float(atom.radius), _element_cpk(atom.element))
            )
        for bond in geo.bonds:
            p1 = np.asarray(geo.atoms[bond.i].position, dtype=float) - center
            p2 = np.asarray(geo.atoms[bond.j].position, dtype=float) - center
            bonds.append(
                (float(p1[0]), float(p1[1]), float(p1[2]),
                 float(p2[0]), float(p2[1]), float(p2[2]),
                 BOND_EXPORT_RADIUS, BOND_EXPORT_COLOR)
            )

    if include_cell:
        lattice = _lattice_edge_tuples(geo.cell, geo.center)

    if include_bz and bz_geometry is not None:
        hull = getattr(bz_geometry, "hull_points", None)
        simplices = getattr(bz_geometry, "simplices", None)
        if hull is not None and simplices is not None:
            # Center-relative BZ to match viewer overlay (viewer subtracts crystal center).
            verts = [
                [float(p[0]) - float(center[0]),
                 float(p[1]) - float(center[1]),
                 float(p[2]) - float(center[2])]
                for p in hull
            ]
            faces = [[int(i) for i in tri] for tri in simplices]
            bz = {"verts": verts, "faces": faces}

    return atoms, bonds, lattice, bz


@router.post("/load", response_model=CrystalSummary)
async def load_cif(
    file: UploadFile = File(...),
    name: str = Form(default=""),
    session: Session = Depends(current_session),
) -> CrystalSummary:
    """Parse an uploaded CIF and store it in the caller's workspace."""
    fname = file.filename or ""
    if not fname.lower().endswith((".cif", ".vasp", ".poscar")) and not fname.upper().endswith("POSCAR"):
        raise HTTPException(status_code=400, detail="Expected a .cif, .vasp, or POSCAR file.")

    payload = await file.read()
    if len(payload) > MAX_CIF_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 8 MB limit.")

    fmt = _detect_structure_fmt(fname)
    try:
        structure = Structure.from_str(payload.decode("utf-8", errors="replace"), fmt=fmt)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse the CIF: {exc}")

    label = _safe_label(name or fname.rsplit(".", 1)[0], "structure")
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

    stacked = CrystalEngine.build_heterostructure_stack(
        layers_data, vacuum=request.vacuum
    )
    label = _safe_label(request.store_as, "heterostructure")
    _store(session, label, stacked)
    return _geometry_from_structure(
        label,
        stacked,
        show_bonds=request.show_bonds,
        show_polyhedra=request.show_polyhedra,
        bond_threshold=request.bond_threshold,
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


@router.post("/exfoliate", response_model=ExfoliateResult)
def exfoliate_bulk(
    request: ExfoliateRequest,
    session: Session = Depends(current_session),
) -> ExfoliateResult:
    """Cleave a bulk crystal into a monolayer / N-layer sheet for stacking."""
    bulk = _require_structure(session, request.source_name)
    gap = None
    hkl = None

    try:
        if request.mode == "vdw":
            mono, gap = CrystalEngine.extract_monolayer_vdw(bulk)
            default_name = f"{request.source_name}_vdW_mono"
        else:
            hkl = (request.h, request.k, request.l)
            mono = CrystalEngine.extract_monolayer_miller(
                bulk,
                hkl,
                num_layers=request.num_layers,
                vacuum=request.vacuum,
            )
            default_name = (
                f"{request.source_name}_{request.num_layers}L_"
                f"{request.h}{request.k}{request.l}"
            )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Exfoliation failed: {exc}")

    if len(mono) > MAX_RENDER_ATOMS:
        raise HTTPException(
            status_code=422,
            detail=f"{len(mono)} atoms exceeds the {MAX_RENDER_ATOMS} the browser can draw.",
        )

    label = _safe_label(request.store_as or default_name, "monolayer")
    summary = _store(session, label, mono)
    return ExfoliateResult(
        summary=summary,
        mode=request.mode,
        gap_angstrom=float(gap) if gap is not None else None,
        hkl=[int(v) for v in hkl] if hkl is not None else None,
        num_layers=request.num_layers if request.mode == "miller" else 1,
        vacuum=request.vacuum if request.mode == "miller" else None,
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
    structure = _apply_basis(structure, request.basis)

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
        show_polyhedra=request.show_polyhedra,
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


@router.get("/mlip/models")
def mlip_models():
    """List pretrained MLIP options for Tab 3 relaxation."""
    return {"models": mlip_engine.list_models(), "installed": mlip_engine.mlip_available()}


@router.get("/{name}/cif")
def download_cif(name: str, session: Session = Depends(current_session)):
    """Download the workspace structure as a CIF (for DFT / archiving)."""
    structure = _require_structure(session, name)
    try:
        text = structure.to(fmt="cif")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"CIF export failed: {exc}") from exc
    filename = f"{name}.cif"
    return Response(
        content=text.encode("utf-8"),
        media_type="chemical/x-cif",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/{name}/cif")
def download_cif_filtered(
    name: str,
    request: CrystalCifRequest,
    session: Session = Depends(current_session),
):
    """Download CIF after applying Draw knobs (supercell, basis, omit)."""
    structure = _require_structure(session, name)
    if _is_default_knobs(
        omit_atom_indices=request.omit_atom_indices,
        nx=request.nx,
        ny=request.ny,
        nz=request.nz,
        basis=request.basis,
    ):
        filtered = structure.copy()
    else:
        filtered = _build_knobbed_structure(
            structure,
            basis=request.basis,
            nx=request.nx,
            ny=request.ny,
            nz=request.nz,
            omit_atom_indices=request.omit_atom_indices,
        )
    try:
        text = filtered.to(fmt="cif")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"CIF export failed: {exc}") from exc
    filename = f"{name}.cif"
    return Response(
        content=text.encode("utf-8"),
        media_type="chemical/x-cif",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/{name}/export/{fmt}")
def export_scene(
    name: str,
    fmt: Literal["3dsmax", "blender"],
    request: SceneExportRequest,
    session: Session = Depends(current_session),
):
    """Build a 3ds Max (.ms) or Blender (.py) script via SceneExporter."""
    structure = _require_structure(session, name)
    structure = _apply_basis(structure, request.basis)

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

    geo = _geometry_from_structure(
        name,
        supercell,
        show_bonds=request.show_bonds,
        bond_threshold=request.bond_threshold,
    )

    omit = normalize_omit_indices(request.omit_atom_indices, geo.n_atoms)
    geo = _filter_geometry_by_omit(geo, omit)

    bz_geo: BZGeometry | None = None
    if request.include_bz:
        raw = CrystalEngine.calculate_brillouin_zone(structure)
        if not raw:
            raise HTTPException(
                status_code=422, detail="Could not build a Brillouin zone for this cell."
            )
        scale = request.bz_scale
        hull = np.asarray(raw["hull_points"], dtype=float) * scale
        simplices = [[int(i) for i in tri] for tri in raw["simplices"]]
        edges = [
            [[float(v) * scale for v in p] for p in edge]
            for edge in raw["edges"]
        ]
        bz_geo = BZGeometry(
            name=name,
            hull_points=[[float(v) for v in row] for row in hull],
            simplices=simplices,
            edges=edges,
            scale=scale,
            style=request.bz_style,
        )

    atoms, bonds, lattice, bz = _scene_export_parts(
        geo,
        include_atoms=request.include_atoms,
        include_cell=request.include_cell,
        include_bz=request.include_bz,
        bz_geometry=bz_geo,
    )

    if fmt == "3dsmax":
        filename = f"{name}_scene.ms"
        media_type = "text/plain"
        writer = SceneExporter.export_3dsmax
    else:
        filename = f"{name}_scene.py"
        media_type = "text/x-python"
        writer = SceneExporter.export_blender

    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / filename
            writer(str(path), atoms, bonds, lattice, bz)
            content = path.read_bytes()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Scene export failed: {exc}") from exc

    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/{name}/export/figure")
def export_figure_route(
    name: str,
    request: CrystalFigureExportRequest,
    session: Session = Depends(current_session),
):
    """Matplotlib PNG/SVG/PDF of the current structure (Draw-equivalent knobs)."""
    structure = _require_structure(session, name)
    structure = _apply_basis(structure, request.basis)

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

    geo = _geometry_from_structure(
        name,
        supercell,
        show_bonds=request.show_bonds,
        show_polyhedra=request.show_polyhedra,
        bond_threshold=request.bond_threshold,
    )

    omit = normalize_omit_indices(request.omit_atom_indices, geo.n_atoms)
    geo = _filter_geometry_by_omit(geo, omit)

    colors = {el: _element_cpk(el) for el in geo.elements}
    atoms = [
        {"element": atom.element, "position": atom.position, "radius": atom.radius}
        for atom in geo.atoms
    ]
    bonds = [(bond.i, bond.j) for bond in geo.bonds]
    polyhedra = (
        [{"vertices": p.vertices, "simplices": p.simplices} for p in geo.polyhedra]
        if request.show_polyhedra
        else None
    )

    camera = None
    if request.use_current_view and request.camera is not None:
        camera = request.camera.model_dump()

    show_cell = request.show_cell and geo.show_cell
    cell = geo.cell if show_cell else None

    try:
        payload = export_crystal_figure(
            atoms=atoms,
            bonds=bonds,
            cell=cell,
            polyhedra=polyhedra,
            show_bonds=request.show_bonds,
            show_cell=show_cell,
            atom_scale=request.atom_scale,
            colors=colors,
            title=request.title,
            fmt=request.fmt,
            camera=camera,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Figure export failed: {exc}") from exc

    media = {"png": "image/png", "svg": "image/svg+xml", "pdf": "application/pdf"}[
        request.fmt
    ]
    filename = f"{name}_figure.{request.fmt}"
    return Response(
        content=payload,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/{name}/push", response_model=CrystalSummary)
def push_crystal_alias(
    name: str,
    request: PushCrystalRequest,
    session: Session = Depends(current_session),
) -> CrystalSummary:
    """Store a copy under ``store_as`` so DFT Suite can pick it by that name."""
    structure = _require_structure(session, name)
    label = _safe_label(request.store_as, name)
    if _is_default_knobs(
        omit_atom_indices=request.omit_atom_indices,
        nx=request.nx,
        ny=request.ny,
        nz=request.nz,
        basis=request.basis,
    ):
        to_store = structure
    else:
        to_store = _build_knobbed_structure(
            structure,
            basis=request.basis,
            nx=request.nx,
            ny=request.ny,
            nz=request.nz,
            omit_atom_indices=request.omit_atom_indices,
        )
    return _store(session, label, to_store)


@router.post("/{name}/relax")
def relax_crystal(
    name: str,
    request: RelaxRequest,
    session: Session = Depends(current_session),
):
    """Relax ions with a pretrained MLIP (CHGNet or M3GNet) and store the result."""
    structure = _require_structure(session, name)
    try:
        result = mlip_engine.relax_structure(
            structure,
            model=request.model,
            fmax=request.fmax,
            steps=request.steps,
            relax_cell=request.relax_cell,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    store_as = request.store_as.strip() or f"{name}_relaxed"
    label = _safe_label(store_as, f"{name}_relaxed")
    summary = _store(session, label, result["structure"])
    geometry = _geometry_from_structure(
        label,
        result["structure"],
        show_bonds=request.show_bonds,
        show_polyhedra=request.show_polyhedra,
        bond_threshold=request.bond_threshold,
    )
    return {
        "summary": summary.model_dump(),
        "geometry": geometry.model_dump(),
        "model": result["model"],
        "fmax": result["fmax"],
        "steps_requested": result["steps_requested"],
        "steps_taken": result["steps_taken"],
        "relax_cell": result["relax_cell"],
        "final_energy_eV": result["final_energy_eV"],
        "stored_as": label,
    }


@router.post("/{name}/gap-predict")
def predict_gap(
    name: str,
    request: GapPredictRequest,
    session: Session = Depends(current_session),
):
    """MEGNet scalar band-gap prediction for a lab stack (not full E(k))."""
    structure = _require_structure(session, name)
    try:
        return mlip_engine.predict_band_gap(structure, fidelity=request.fidelity)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

