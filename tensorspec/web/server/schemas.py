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
