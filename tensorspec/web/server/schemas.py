"""Pydantic request/response models for the TensorSpec web API.

These mirror the bounds the UI advertises so that out-of-range values are
rejected at the edge, before anything reaches `core/`.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class WorkspaceItem(BaseModel):
    """One row of the workspace variable tree."""

    name: str
    type: str
    dims: str


class WorkspaceListing(BaseModel):
    items: list[WorkspaceItem]
    session_id: str


class ItemDetail(BaseModel):
    """Metadata shown in the inspector for a selected variable."""

    name: str
    type: str
    dims: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    viewable: bool = False


class DemoSeedRequest(BaseModel):
    """Development fixture used to populate an empty session."""

    name: str = Field(default="demo_arpes_cube", min_length=1, max_length=64)
    n_energy: int = Field(default=48, ge=8, le=256)
    n_kx: int = Field(default=64, ge=8, le=256)
    n_ky: int = Field(default=32, ge=1, le=256)


class BandRequest(BaseModel):
    """Tight-binding parameters. Bounds mirror the DFT Suite spin boxes."""

    path_mode: Literal["auto", "custom", "hexagonal", "rectangular", "square"] = "auto"
    custom_coords: str = Field(default="", max_length=2000)
    custom_labels: str = Field(default="", max_length=500)
    # The desktop app allows 2000, but that runs on the user's own CPU. A
    # shared server caps it and offers the job queue for anything heavier.
    points_per_segment: int = Field(default=100, ge=10, le=500)

    hoppings: list[float] = Field(default=[2.7, 0.0, 0.0, -0.3], max_length=8)
    cutoffs: list[float] = Field(default=[1.6, 2.6, 3.1, 4.5], max_length=8)

    onsite_e: float = Field(default=0.0, ge=-10, le=10)
    shift_s: float = Field(default=-10.0, ge=-50, le=50)
    shift_p: float = Field(default=-2.0, ge=-50, le=50)
    shift_d: float = Field(default=0.0, ge=-50, le=50)

    use_soc: bool = False
    soc_strength: float = Field(default=0.5, ge=0, le=5)
    tb_mode: Literal["Simple Scalar (Isotropic)", "Slater-Koster (Rigorous)"] = "Simple Scalar (Isotropic)"


class BandResult(BaseModel):
    """A computed dispersion, ready to plot."""

    name: str
    k_dist: list[float]
    bands: list[list[float]]
    node_positions: list[float]
    node_labels: list[str]
    n_bands: int
    n_kpoints: int
    fermi_energy: float
    energy_min: float
    energy_max: float
    orbital_labels: list[str]
    elapsed_seconds: float


class StructureOption(BaseModel):
    name: str
    formula: str
    n_sites: int
    shell_keys: list[str]
    default_hoppings: list[float]


class AxisInfo(BaseModel):
    """One dimension of a tensor, as the axis pickers and sliders need it."""

    index: int
    label: str
    unit: str
    size: int
    min: float
    max: float


class TensorAxes(BaseModel):
    name: str
    data_type: str
    ndim: int
    axes: list[AxisInfo]
    default_x: int
    default_y: int
    default_fixed: dict[int, int]


class SliceRequest(BaseModel):
    """Which 2D plane to cut out of an N-dimensional tensor."""

    x_idx: int = Field(ge=0, le=15)
    y_idx: int = Field(ge=0, le=15)
    fixed: dict[int, int] = Field(default_factory=dict)
    # Caps the returned grid. A browser cannot resolve more samples than pixels.
    max_points: int = Field(default=900, ge=32, le=4096)


class ProfileRequest(BaseModel):
    """Crosshair position and integration window for curve extraction."""

    x_idx: int = Field(ge=0, le=15)
    y_idx: int = Field(ge=0, le=15)
    fixed: dict[int, int] = Field(default_factory=dict)
    x_center: int = Field(ge=0)
    y_center: int = Field(ge=0)
    dx: int = Field(default=0, ge=0, le=1024)
    dy: int = Field(default=0, ge=0, le=1024)
    mode: Literal["sum", "mean", "normalized"] = "sum"
    ortho_idx: int | None = None


class Curve(BaseModel):
    axis: list[float]
    values: list[float]
    label: str
    unit: str


class ProfileResponse(BaseModel):
    x: Curve
    y: Curve
    ortho: Curve | None = None
    window: dict[str, int]
    mode: str


class CrystalSummary(BaseModel):
    """Lattice facts shown beside the 3D viewport after a CIF loads."""

    name: str
    formula: str
    spacegroup: str
    n_sites: int
    lattice: dict[str, float]


class GeometryRequest(BaseModel):
    """Supercell and bonding options for a geometry render.

    Bounds mirror the Crystal Suite spin boxes. The supercell cap keeps a
    stray large request from building millions of atoms on a shared server.
    """

    nx: int = Field(default=1, ge=1, le=20)
    ny: int = Field(default=1, ge=1, le=20)
    nz: int = Field(default=1, ge=1, le=20)
    bond_threshold: float = Field(default=1.15, ge=0.5, le=3.0)
    show_bonds: bool = True

    @property
    def cell_count(self) -> int:
        return self.nx * self.ny * self.nz


class Atom(BaseModel):
    element: str
    position: list[float]
    radius: float


class Bond(BaseModel):
    i: int
    j: int


class CrystalGeometry(BaseModel):
    """A renderer-agnostic scene description.

    Positions are Cartesian angstroms and `cell` holds the three lattice
    vectors. Colours are deliberately absent: they are a display choice the
    browser makes, not a property of the crystal.
    """

    name: str
    atoms: list[Atom]
    bonds: list[Bond]
    cell: list[list[float]]
    center: list[float]
    elements: list[str]
    n_atoms: int
