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

    path_mode: Literal[
        "auto",
        "custom",
        "hexagonal",
        "rectangular",
        "square",
        "primitive_hex_ref",
        "unfold_hex",
    ] = "auto"
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
    # Educational labels for supercell / moiré stacks
    path_kind: str = "standard"
    path_title: str = ""
    path_note: str = ""
    likely_folded: bool = False
    # Popescu–Zunger spectral weights (nk × nband), only for unfold_hex
    weights: list[list[float]] | None = None
    weight_min: float = 0.0
    weight_max: float = 1.0
    unfolded: bool = False
    # Orbital character fat bands (re-projected from cached evecs)
    fat_weights: list[list[float]] | None = None
    fat_target: str = "none"
    fat_n_orbitals: int = 0


class FatBandRequest(BaseModel):
    """Re-project cached eigenvectors onto an orbital / shell / element target."""

    fat_target: str = Field(default="none", max_length=128)


class FatBandResult(BaseModel):
    """Fat-band weights for an already-computed band structure."""

    name: str
    fat_target: str
    fat_weights: list[list[float]] | None = None
    fat_n_orbitals: int = 0
    orbital_labels: list[str] = []
    n_bands: int
    n_kpoints: int


class StructureOption(BaseModel):
    name: str
    formula: str
    n_sites: int
    shell_keys: list[str]
    default_hoppings: list[float]


class QERequest(BaseModel):
    """Validated QE / Wannier90 parameters. No executable paths here."""

    run_name: str = Field(default="run_01", min_length=1, max_length=64)
    ecutwfc: float = Field(default=60.0, ge=20, le=200)
    nbnd: int = Field(default=12, ge=1, le=500)
    kx: int = Field(default=6, ge=1, le=20)
    ky: int = Field(default=6, ge=1, le=20)
    kz: int = Field(default=6, ge=1, le=20)
    use_soc: bool = False
    mlwf_mode: bool = False
    use_mpi: bool = True
    mpi_ranks: int = Field(default=4, ge=1, le=256)


class QEGenerateResponse(BaseModel):
    run_name: str
    run_dir: str
    files: list[str]
    mpi_ranks_capped: int
    max_mpi_ranks: int
    solvers_available: bool


class JobInfo(BaseModel):
    job_id: str
    run_name: str
    status: str
    current_step: int
    total_steps: int
    exit_code: int | None = None
    error: str | None = None
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None


class SolverStatus(BaseModel):
    available: bool
    pw: str | None = None
    wannier90: str | None = None
    pw2wannier90: str | None = None
    mpirun: str | None = None
    pseudo_dir: str | None = None
    max_mpi_ranks: int
    detail: str | None = None


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


class ArpesLoadSummary(BaseModel):
    """Result of uploading a beamline HDF5 into the spectroscopy workspace."""

    name: str
    shape: list[int]
    labels: list[str]
    units: list[str]
    data_type: str
    facility: str = "Unknown"
    measurement_type: str | None = None
    source_file: str | None = None


class InplaneConvertRequest(BaseModel):
    """User-driven angle → k∥ conversion with a chosen Γ center."""

    angle_axis: int = Field(ge=0, le=15)
    energy_axis: int | None = Field(default=None, ge=0, le=15)
    beta_axis: int | None = Field(default=None, ge=0, le=15)
    center: float
    beta_center: float = 0.0
    deg_per_unit: float = Field(default=1.0, gt=0, le=10)
    beta_deg_per_unit: float = Field(default=1.0, gt=0, le=10)
    photon_energy: float = Field(default=80.0, gt=0, le=5000)
    work_function: float = Field(default=4.5, ge=0, le=10)
    energy_mode: str = Field(default="auto")
    e_kin_ref: float | None = Field(default=None, gt=0, le=5000)
    # For preview slice packing
    x_idx: int = Field(default=0, ge=0, le=15)
    y_idx: int = Field(default=1, ge=0, le=15)
    fixed: dict[int, int] = Field(default_factory=dict)
    max_points: int = Field(default=512, ge=32, le=2048)


class InplaneApplyRequest(InplaneConvertRequest):
    store_as: str = Field(default="", max_length=64)
    also_write_processed: bool = True


class KzConvertRequest(BaseModel):
    """Photon-energy → kz conversion with live inner potential."""

    photon_axis: int = Field(ge=0, le=15)
    work_function: float = Field(default=4.5, ge=0, le=10)
    inner_potential: float = Field(default=15.0, ge=0, le=50)
    theta_deg: float = Field(default=0.0, ge=-90, le=90)
    binding_ref: float = Field(default=0.0, ge=-5, le=50)
    include_photon_momentum: bool = False
    photon_incidence_angle: float = Field(default=45.0, ge=0, le=90)
    x_idx: int = Field(default=0, ge=0, le=15)
    y_idx: int = Field(default=1, ge=0, le=15)
    fixed: dict[int, int] = Field(default_factory=dict)
    max_points: int = Field(default=512, ge=32, le=2048)


class KzApplyRequest(KzConvertRequest):
    store_as: str = Field(default="", max_length=64)
    also_write_processed: bool = True


class PerpBZRequest(BaseModel):
    crystal_name: str = Field(min_length=1, max_length=64)
    h: int = Field(default=0, ge=-10, le=10)
    k: int = Field(default=0, ge=-10, le=10)
    l: int = Field(default=1, ge=-10, le=10)
    n_zones: int = Field(default=4, ge=1, le=12)


class PeakSeed(BaseModel):
    center: float
    amplitude: float = Field(gt=0)
    width: float = Field(gt=0)
    sigma: float | None = Field(default=None, gt=0)


class PeakFitCurveRequest(BaseModel):
    """Fit one EDC or MDC extracted from a 2D view of a tensor."""

    x_idx: int = Field(ge=0, le=15)
    y_idx: int = Field(ge=0, le=15)
    fixed: dict[int, int] = Field(default_factory=dict)
    mode: str = Field(default="mdc")
    index: int = Field(ge=0)
    half_width: int = Field(default=0, ge=0, le=50)
    lineshape: str = Field(default="lorentzian")
    analyzer_fwhm: float = Field(default=0.0, ge=0, le=5)
    include_fd: bool = False
    temperature: float = Field(default=10.0, ge=0.01, le=400)
    mu: float = 0.0
    seeds: list[PeakSeed] = Field(default_factory=list)
    n_peaks: int = Field(default=1, ge=1, le=12)
    suggest: bool = False


class PeakFitStackRequest(PeakFitCurveRequest):
    scan_start: int | None = None
    scan_stop: int | None = None
    scan_step: int = Field(default=1, ge=1, le=50)
    propagate_seeds: bool = True
    store: bool = True


class QPResultsRequest(BaseModel):
    """Build δE–E / k_F / m* / FL–MFL curves from a stored peakfit node."""

    peakfit_node: str = Field(default="mdc_peakfit", min_length=1, max_length=64)
    peak: int = Field(default=0, ge=0, le=12)
    e_fermi: float = 0.0
    fit_mass: bool = True
    fit_vf: bool = True
    se_model: str | None = Field(default="fl")  # "fl", "mfl", or null to skip
    se_e_min: float | None = None
    se_e_max: float | None = None
    vf_e_window: float = Field(default=0.08, gt=0, le=2)
    store: bool = True


class GapFitCurveRequest(BaseModel):
    """Fit a Dynes SC/CDW gap model to one EDC."""

    x_idx: int = Field(ge=0, le=15)
    y_idx: int = Field(ge=0, le=15)
    fixed: dict[int, int] = Field(default_factory=dict)
    index: int = Field(ge=0)
    half_width: int = Field(default=0, ge=0, le=50)
    gap_type: str = Field(default="sc")
    temperature: float = Field(default=10.0, ge=0.01, le=400)
    mu: float = 0.0
    analyzer_fwhm: float = Field(default=0.0, ge=0, le=5)
    suggest: bool = True
    delta: float | None = Field(default=None, gt=0, le=2)
    gamma: float | None = Field(default=None, gt=0, le=2)
    amplitude: float | None = Field(default=None, gt=0)


class GapFitStackRequest(GapFitCurveRequest):
    scan_start: int | None = None
    scan_stop: int | None = None
    scan_step: int = Field(default=1, ge=1, le=50)
    propagate_seeds: bool = True
    store: bool = True


class CutOverlayRequest(BaseModel):
    """Project DFT bands and/or resample a sim cube onto an experimental cut."""

    x_idx: int = Field(ge=0, le=15)
    y_idx: int = Field(ge=0, le=15)
    fixed: dict[int, int] = Field(default_factory=dict)
    bands_name: str | None = Field(default=None, max_length=128)
    sim_name: str | None = Field(default=None, max_length=128)
    e_fermi: float = 0.0
    k_component: str = Field(default="kx")
    k_offset: float = 0.0
    band_indices: list[int] | None = None
    sim_x_idx: int | None = Field(default=None, ge=0, le=15)
    sim_y_idx: int | None = Field(default=None, ge=0, le=15)
    sim_fixed: dict[int, int] = Field(default_factory=dict)
    max_points: int = Field(default=512, ge=64, le=2048)


class VolumeViewRequest(BaseModel):
    """Downsampled intensity cube + BZ prism footprint for the 3D cutout viewer."""

    x_idx: int | None = Field(default=None, ge=0, le=15)
    y_idx: int | None = Field(default=None, ge=0, le=15)
    z_idx: int | None = Field(default=None, ge=0, le=15)
    fixed: dict[int, int] = Field(default_factory=dict)
    max_per_axis: int = Field(default=64, ge=16, le=128)
    shape_mode: str = Field(default="auto")  # auto | rectangle | hexagon | crystal
    crystal_name: str | None = Field(default=None, max_length=64)
    h: int = Field(default=0, ge=-10, le=10)
    k: int = Field(default=0, ge=-10, le=10)
    l: int = Field(default=1, ge=-10, le=10)


class SuggestCenterRequest(BaseModel):
    angle_axis: int = Field(ge=0, le=15)
    energy_axis: int | None = Field(default=None, ge=0, le=15)
    fixed: dict[int, int] = Field(default_factory=dict)


class SurfaceBZRequest(BaseModel):
    crystal_name: str = Field(min_length=1, max_length=64)
    h: int = Field(default=0, ge=-10, le=10)
    k: int = Field(default=0, ge=-10, le=10)
    l: int = Field(default=1, ge=-10, le=10)


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
    """Supercell, bonding, and optional CDW options for a geometry render.

    Bounds mirror the Crystal Suite spin boxes. The supercell cap keeps a
    stray large request from building millions of atoms on a shared server.
    """

    nx: int = Field(default=1, ge=1, le=20)
    ny: int = Field(default=1, ge=1, le=20)
    nz: int = Field(default=1, ge=1, le=20)
    bond_threshold: float = Field(default=1.15, ge=0.5, le=3.0)
    show_bonds: bool = True

    # CDW: phase is degrees (the Qt path double-converted; the web path does not).
    cdw_enabled: bool = False
    cdw_target: str = Field(default="All Elements", max_length=32)
    cdw_qx: float = Field(default=0.0, ge=-5, le=5)
    cdw_qy: float = Field(default=0.0, ge=-5, le=5)
    cdw_qz: float = Field(default=0.0, ge=-5, le=5)
    cdw_ax: float = Field(default=0.0, ge=-2, le=2)
    cdw_ay: float = Field(default=0.0, ge=-2, le=2)
    cdw_az: float = Field(default=0.0, ge=-2, le=2)
    cdw_phase: float = Field(default=0.0, ge=0, le=360)

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
    # Heterostructure stacks sit in a 500 Å dummy cell; the frame is meaningless.
    show_cell: bool = True


class BZRequest(BaseModel):
    scale: float = Field(default=1.0, ge=0.1, le=10)
    style: Literal["solid", "skeleton", "both"] = "solid"
    surface: bool = False
    h: int = Field(default=0, ge=-10, le=10)
    k: int = Field(default=0, ge=-10, le=10)
    l: int = Field(default=1, ge=-10, le=10)
    overlay_crystal: bool = True


class BZGeometry(BaseModel):
    name: str
    hull_points: list[list[float]]
    simplices: list[list[int]]
    edges: list[list[list[float]]]
    scale: float
    style: str
    surface_vertices: list[list[float]] | None = None
    surface_simplices: list[list[int]] | None = None
    projection_lines: list[list[list[float]]] | None = None


class TemplateRequest(BaseModel):
    template_name: str = Field(min_length=1, max_length=128)
    store_as: str = Field(default="", max_length=64)


class StackLayerSpec(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    sc_x: int = Field(default=1, ge=1, le=50)
    sc_y: int = Field(default=1, ge=1, le=50)
    z_shift: float = Field(default=0.0, ge=-100, le=100)
    twist: float = Field(default=0.0, ge=-360, le=360)


class StackRequest(BaseModel):
    layers: list[StackLayerSpec] = Field(min_length=1, max_length=20)
    store_as: str = Field(default="heterostructure", max_length=64)
    bond_threshold: float = Field(default=1.15, ge=0.5, le=3.0)
    show_bonds: bool = True
    vacuum: float = Field(default=20.0, ge=5, le=80)


class RelaxRequest(BaseModel):
    """MLIP ionic relaxation for a stacked (or any) crystal in the workspace."""

    model: str = Field(default="chgnet", max_length=32)
    fmax: float = Field(default=0.1, gt=0, le=5)
    steps: int = Field(default=200, ge=1, le=2000)
    relax_cell: bool = False
    store_as: str = Field(default="", max_length=64)
    show_bonds: bool = True
    bond_threshold: float = Field(default=1.15, ge=0.5, le=3.0)


class GapPredictRequest(BaseModel):
    """MEGNet scalar band-gap prediction (not a full E(k) dispersion)."""

    fidelity: Literal["PBE", "GLLB-SC", "HSE", "SCAN"] = "PBE"


class PushCrystalRequest(BaseModel):
    """Copy/rename the active structure under a new workspace name (for DFT)."""

    store_as: str = Field(min_length=1, max_length=64)


class MoireRequest(BaseModel):
    layer1: str = Field(min_length=1, max_length=64)
    layer2: str = Field(min_length=1, max_length=64)
    twist1: float = Field(default=0.0, ge=-360, le=360)
    twist2: float = Field(default=0.0, ge=-360, le=360)
    z_min: float = Field(default=-5.0, ge=-100, le=100)
    z_max: float = Field(default=5.0, ge=-100, le=100)


class MoireResult(BaseModel):
    status: str
    message: str | None = None
    periodicity: float | None = None
    n_cells: int | None = None
    matrix: list[list[float]] | None = None
    envelope: list[list[float]] | None = None


class ExfoliateRequest(BaseModel):
    """Bulk → monolayer (or N-layer) cleave for the Stack & Twist tab."""

    source_name: str = Field(min_length=1, max_length=64)
    mode: Literal["vdw", "miller"] = "vdw"
    h: int = Field(default=0, ge=-10, le=10)
    k: int = Field(default=0, ge=-10, le=10)
    l: int = Field(default=1, ge=-10, le=10)
    num_layers: int = Field(default=1, ge=1, le=10)
    vacuum: float = Field(default=15.0, ge=0, le=100)
    store_as: str = Field(default="", max_length=64)


class ExfoliateResult(BaseModel):
    summary: CrystalSummary
    mode: str
    gap_angstrom: float | None = None
    hkl: list[int] | None = None
    num_layers: int | None = None
    vacuum: float | None = None


class AxisBound(BaseModel):
    min: float = Field(default=-2.0, ge=-20, le=20)
    max: float = Field(default=2.0, ge=-20, le=20)
    steps: int = Field(default=40, ge=1, le=80)


class ArpesSimRequest(BaseModel):
    """Option A / B1 matrix-element simulation parameters.

    Bounds keep shared-server sims finite. The crystal must already live in the
    caller's session workspace; bands are rebuilt as a 2D mesh inside the job.
    """

    crystal_name: str = Field(min_length=1, max_length=64)
    model: Literal["A", "B1"] = "A"
    store_as: str = Field(default="simulated_arpes", min_length=1, max_length=64)

    photon_energy: float = Field(default=90.0, ge=5, le=2000)
    work_function: float = Field(default=4.5, ge=0, le=10)
    inner_potential: float = Field(default=15.0, ge=0, le=30)
    temperature: float = Field(default=10.0, ge=0.1, le=1000)
    incidence_angle: float = Field(default=55.0, ge=0, le=90)
    polarization: str = Field(default="Linear Horizontal (p-pol)", max_length=64)
    lin_pol_angle: float = Field(default=45.0, ge=0, le=360)
    matrix_element_mode: str = Field(default="Full Matrix Elements", max_length=64)

    manip_theta: float = Field(default=0.0, ge=-180, le=180)
    manip_azimuth: float = Field(default=0.0, ge=-180, le=180)
    manip_tilt: float = Field(default=0.0, ge=-90, le=90)
    h: int = Field(default=0, ge=-10, le=10)
    k: int = Field(default=0, ge=-10, le=10)
    l: int = Field(default=1, ge=-10, le=10)
    slit_angle: float = Field(default=0.0, ge=-180, le=180)

    kx: AxisBound = Field(default_factory=AxisBound)
    ky: AxisBound = Field(default_factory=AxisBound)
    energy: AxisBound = Field(default_factory=lambda: AxisBound(min=-2.0, max=0.5, steps=40))

    se_width: float = Field(default=0.010, ge=0.001, le=1)
    res_E: float = Field(default=0.020, ge=0.001, le=1)
    res_k: float = Field(default=0.020, ge=0.001, le=1)

    # TB mesh used to feed Option A (and to build the B1 model).
    mesh_resolution: int = Field(default=20, ge=4, le=40)
    hoppings: list[float] = Field(default=[2.7, 0.0, 0.0, -0.3], max_length=8)
    cutoffs: list[float] = Field(default=[1.6, 2.6, 3.1, 4.5], max_length=8)
    onsite_e: float = Field(default=0.0, ge=-10, le=10)
    tb_mode: Literal["Simple Scalar (Isotropic)", "Slater-Koster (Rigorous)"] = (
        "Simple Scalar (Isotropic)"
    )


class ArpesSimPushRequest(BaseModel):
    name: str = Field(default="", max_length=64)


class FigureExportRequest(BaseModel):
    """Server-side matplotlib PDF/SVG of the current slice + profiles."""

    x_idx: int = Field(ge=0, le=16)
    y_idx: int = Field(ge=0, le=16)
    fixed: dict[int, int] = Field(default_factory=dict)
    x_center: int = Field(default=0, ge=0, le=4096)
    y_center: int = Field(default=0, ge=0, le=4096)
    dx: int = Field(default=0, ge=0, le=1024)
    dy: int = Field(default=0, ge=0, le=1024)
    mode: Literal["sum", "mean", "normalized"] = "sum"
    include_profiles: bool = True
    fmt: Literal["pdf", "svg"] = "pdf"
    title: str = Field(default="", max_length=120)
