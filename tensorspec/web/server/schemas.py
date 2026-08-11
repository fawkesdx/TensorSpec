"""Pydantic request/response models for the TensorSpec web API.

These mirror the bounds the UI advertises so that out-of-range values are
rejected at the edge, before anything reaches `core/`.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


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
    # When true, solve from an uploaded wannier90_hr.dat for this crystal (see POST …/wannier).
    use_wannier: bool = False
    # Optional companion W90 solve drawn dashed-red on BandPlot (same k-path).
    overlay_wannier: bool = False


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
    # Optional W90 companion bands (same k_dist); None when overlay_wannier is false
    overlay_bands: list[list[float]] | None = None


class IsoenergyRequest(BaseModel):
    """2D TB isoenergy cut on a kx–ky mesh (Gaussian DOS density)."""

    energy: float = Field(default=0.0, ge=-50, le=50)
    kx_min: float = Field(default=-2.0, ge=-20, le=20)
    kx_max: float = Field(default=2.0, ge=-20, le=20)
    ky_min: float = Field(default=-2.0, ge=-20, le=20)
    ky_max: float = Field(default=2.0, ge=-20, le=20)
    resolution: int = Field(default=24, ge=4, le=48)
    smear: float = Field(default=0.05, ge=0.001, le=1)

    hoppings: list[float] = Field(default=[2.7, 0.0, 0.0, -0.3], max_length=8)
    cutoffs: list[float] = Field(default=[1.6, 2.6, 3.1, 4.5], max_length=8)

    onsite_e: float = Field(default=0.0, ge=-10, le=10)
    shift_s: float = Field(default=-10.0, ge=-50, le=50)
    shift_p: float = Field(default=-2.0, ge=-50, le=50)
    shift_d: float = Field(default=0.0, ge=-50, le=50)

    use_soc: bool = False
    soc_strength: float = Field(default=0.5, ge=0, le=5)
    tb_mode: Literal["Simple Scalar (Isotropic)", "Slater-Koster (Rigorous)"] = "Simple Scalar (Isotropic)"
    use_wannier: bool = False


class IsoenergyResult(BaseModel):
    """kx–ky isoenergy density map ready for a heatmap."""

    name: str
    kx: list[float]
    ky: list[float]
    intensity: list[list[float]]
    energy: float
    smear: float
    n_bands: int
    resolution: int
    elapsed_seconds: float
    fermi_energy: float = 0.0


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
    # Vacuum slab / Tab-3 stack: kz→1 and assume_isolated='2D'
    slab_mode: bool = False
    # XC functional → QE input_dft (no pseudo filtering)
    functional: Literal["PBE", "LDA", "PBEsol"] = "PBE"
    backend: Literal["local", "einstein_ssh"] = "local"


class SlabPrepareRequest(BaseModel):
    """Cleave a bulk crystal into a vacuum slab for QE (DFT Suite)."""

    preset: str = Field(default="thin_001", max_length=32)
    h: int = Field(default=0, ge=-10, le=10)
    k: int = Field(default=0, ge=-10, le=10)
    l: int = Field(default=1, ge=-10, le=10)
    num_layers: int = Field(default=1, ge=1, le=10)
    vacuum: float = Field(default=15.0, ge=0, le=100)
    store_as: str = Field(default="", max_length=64)
    bond_threshold: float = Field(default=3.2, ge=0.5, le=10.0)


class SlabPrepareResult(BaseModel):
    stored_as: str
    formula: str
    n_sites: int
    hkl: list[int]
    num_layers: int
    vacuum: float
    preset: str
    suggest_slab_qe: bool = True
    lattice_c: float


class StructureOption(BaseModel):
    name: str
    formula: str
    n_sites: int
    shell_keys: list[str]
    default_hoppings: list[float]
    suggest_slab_qe: bool = False
    lattice_c: float | None = None


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


class PeemLoadSummary(BaseModel):
    """Result of loading a PEEM TIF stack or sequence."""

    name: str
    shape: list[int]
    n_frames: int
    data_type: str
    pol_summary: dict[str, int]
    source: str
    loader: str
    csv_attached: bool
    I0_present: bool
    csv_prompt: bool = False
    csv_candidates: list[str] = Field(default_factory=list)


class PeemPairRequest(BaseModel):
    """Pair polarization-tagged raw PEEM frames."""

    mode: Literal["auto", "CP_CM", "LH_LV"] = "auto"


class PeemPairSummary(BaseModel):
    """Summary of a paired PEEM cube written to /processed."""

    name: str
    n_pairs: int
    channel_tags: list[str]
    unpaired_count: int
    mode: str
    has_processed: bool = True
    shape: list[int]


class PeemRoi(BaseModel):
    """Region used to track a PEEM feature during drift correction."""

    kind: Literal["rect", "ellipse", "polygon"]
    x0: float | None = None
    y0: float | None = None
    x1: float | None = None
    y1: float | None = None
    cx: float | None = None
    cy: float | None = None
    rx: float | None = None
    ry: float | None = None
    points: list[list[float]] | None = None


class PeemDriftRequest(BaseModel):
    """Drift-correction settings for raw or processed PEEM data."""

    source: Literal["raw", "processed"] = "raw"
    ref_index: int = Field(ge=0, le=100_000)
    search_radius: int = Field(default=20, ge=1, le=200)
    track_channel: int = Field(default=0, ge=0, le=1)
    roi: PeemRoi


class PeemDriftSummary(BaseModel):
    """Summary of a drift-corrected cube written to /processed."""

    name: str
    source: str
    n_planes: int
    ref_index: int
    search_radius: int
    has_processed: bool = True
    has_drift: bool = True
    max_abs_dx: int
    max_abs_dy: int
    shape: list[int]


class PeemSeparateSummary(BaseModel):
    """Summary after splitting paired /processed into channel children."""

    name: str
    channels: list[str]
    n_frames: int
    has_separated: bool = True
    shape: list[int]  # per-channel (n_frames, y, x) — same for both


class PeemBgRequest(BaseModel):
    """Linear pre-edge background settings for preview or apply."""

    node: str = "raw"
    channel: int = Field(default=0, ge=0, le=1)
    use_roi: bool = False
    roi: PeemRoi | None = None
    e0: float
    e1: float
    ensemble_delta: float | None = None
    ensemble_n: int = Field(default=21, ge=1, le=101)
    seed: int = 0


class PeemBgPreviewResponse(BaseModel):
    """Preview curves for linear pre-edge background (no tree write)."""

    energy: list[float]
    spectrum: list[float]
    bg: list[float]
    bg_std: list[float]
    subtracted: list[float]
    subtracted_std: list[float]
    slope: float
    intercept: float
    energy_source: str
    e0: float
    e1: float
    ensemble_n_valid: int


class PeemBgApplySummary(BaseModel):
    """Summary after writing analysis/background and processed BG child."""

    name: str
    analysis_node: str = "background"
    processed_bg_node: str
    n_frames: int
    shape: list[int]
    has_background: bool = True
    energy_source: str


class PeemMeta(BaseModel):
    """PEEM stack metadata needed by the frame viewer."""

    name: str
    shape: list[int]
    labels: list[str]
    n_frames: int
    frame_names: list[str]
    pol: list[str]
    csv_attached: bool
    I0_present: bool
    I0: float | list[float | None] | None = None
    has_processed: bool = False
    processed_shape: list[int] | None = None
    processed_is_paired: bool = False
    n_processed_frames: int | None = None
    pair_mode: str | None = None
    n_pairs: int | None = None
    channel_tags: list[str] = Field(default_factory=list)
    unpaired_count: int = 0
    has_drift: bool = False
    drift_method: str | None = None
    separated_channels: list[str] = Field(default_factory=list)
    has_background: bool = False
    has_processed_bg: bool = False
    energy_source: str | None = None
    processed_bg_node: str | None = None
    n_bg_frames: int | None = None


class PeemFrame(BaseModel):
    """One PEEM intensity frame and suggested display limits."""

    index: int
    shape: list[int]
    intensity: list[list[float]]
    vmin: float
    vmax: float
    pol: str | None = None
    frame_name: str | None = None
    node: str = "raw"
    pair: int | None = None
    channel: int | None = None
    channel_tag: str | None = None


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
    basis: Literal["conventional", "primitive"] = "conventional"
    bond_threshold: float = Field(default=1.15, ge=0.5, le=3.0)
    show_bonds: bool = True
    show_polyhedra: bool = False

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
    label: str
    position: list[float]
    radius: float


class Bond(BaseModel):
    i: int
    j: int


class Polyhedron(BaseModel):
    center: int
    vertices: list[list[float]]
    simplices: list[list[int]]
    vertex_atom_indices: list[int]


class CrystalGeometry(BaseModel):
    """A renderer-agnostic scene description.

    Positions are Cartesian angstroms and `cell` holds the three lattice
    vectors. Colours are deliberately absent: they are a display choice the
    browser makes, not a property of the crystal.
    """

    name: str
    atoms: list[Atom]
    bonds: list[Bond]
    polyhedra: list[Polyhedron] = []
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


class CrystalCifRequest(BaseModel):
    """Knobs for filtered CIF export (supercell + omit) matching the Draw panel."""

    omit_atom_indices: list[int] = []
    nx: int = Field(default=1, ge=1, le=20)
    ny: int = Field(default=1, ge=1, le=20)
    nz: int = Field(default=1, ge=1, le=20)
    basis: Literal["conventional", "primitive"] = "conventional"

    @property
    def cell_count(self) -> int:
        return self.nx * self.ny * self.nz


class CrystalFigureCamera(BaseModel):
    position: list[float] = Field(min_length=3, max_length=3)
    target: list[float] = Field(min_length=3, max_length=3)
    up: list[float] = Field(default_factory=lambda: [0.0, 1.0, 0.0], min_length=3, max_length=3)


class CrystalFigureExportRequest(BaseModel):
    omit_atom_indices: list[int] = []
    nx: int = Field(default=1, ge=1, le=20)
    ny: int = Field(default=1, ge=1, le=20)
    nz: int = Field(default=1, ge=1, le=20)
    basis: Literal["conventional", "primitive"] = "conventional"
    bond_threshold: float = Field(default=1.15, ge=0.5, le=3.0)
    show_bonds: bool = True
    show_polyhedra: bool = False
    show_cell: bool = True
    atom_scale: float = Field(default=0.5, ge=0.1, le=3.0)
    fmt: Literal["png", "svg", "pdf"] = "png"
    title: str = Field(default="", max_length=128)
    use_current_view: bool = False
    camera: CrystalFigureCamera | None = None

    @property
    def cell_count(self) -> int:
        return self.nx * self.ny * self.nz


class SceneExportRequest(BaseModel):
    """Geometry knobs + which scene parts to include in a DCC script export."""

    omit_atom_indices: list[int] = []
    nx: int = Field(default=1, ge=1, le=20)
    ny: int = Field(default=1, ge=1, le=20)
    nz: int = Field(default=1, ge=1, le=20)
    basis: Literal["conventional", "primitive"] = "conventional"
    bond_threshold: float = Field(default=1.15, ge=0.5, le=3.0)
    show_bonds: bool = True
    include_atoms: bool = True
    include_cell: bool = True
    include_bz: bool = False
    # BZ options when include_bz
    bz_scale: float = Field(default=1.0, ge=0.1, le=10)
    bz_style: Literal["solid", "skeleton", "both"] = "solid"
    bz_h: int = Field(default=0, ge=-10, le=10)
    bz_k: int = Field(default=0, ge=-10, le=10)
    bz_l: int = Field(default=1, ge=-10, le=10)

    @model_validator(mode="after")
    def _one_include(self):
        if not (self.include_atoms or self.include_cell or self.include_bz):
            raise ValueError("Select at least one of Atoms/Bonds, Unit Cell, or Brillouin Zone.")
        return self

    @property
    def cell_count(self) -> int:
        return self.nx * self.ny * self.nz


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
    show_polyhedra: bool = False
    vacuum: float = Field(default=20.0, ge=5, le=80)


class RelaxRequest(BaseModel):
    """MLIP ionic relaxation for a stacked (or any) crystal in the workspace."""

    model: str = Field(default="chgnet", max_length=32)
    fmax: float = Field(default=0.1, gt=0, le=5)
    steps: int = Field(default=200, ge=1, le=2000)
    relax_cell: bool = False
    store_as: str = Field(default="", max_length=64)
    show_bonds: bool = True
    show_polyhedra: bool = False
    bond_threshold: float = Field(default=1.15, ge=0.5, le=3.0)


class GapPredictRequest(BaseModel):
    """MEGNet scalar band-gap prediction (not a full E(k) dispersion)."""

    fidelity: Literal["PBE", "GLLB-SC", "HSE", "SCAN"] = "PBE"


class PushCrystalRequest(BaseModel):
    """Copy/rename the active structure under a new workspace name (for DFT)."""

    store_as: str = Field(min_length=1, max_length=64)
    omit_atom_indices: list[int] = []
    nx: int = Field(default=1, ge=1, le=20)
    ny: int = Field(default=1, ge=1, le=20)
    nz: int = Field(default=1, ge=1, le=20)
    basis: Literal["conventional", "primitive"] = "conventional"

    @property
    def cell_count(self) -> int:
        return self.nx * self.ny * self.nz


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
    backend: Literal["local", "einstein_ssh"] = "local"
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

    # Optional resolution / deflector metadata (UI logging). Chinook uses res_E + ky only.
    deflector_angle: float | None = Field(default=None, ge=-15, le=15)
    slit_size_mm: float | None = Field(default=None, ge=0.1, le=5)
    pass_energy: float | None = Field(default=None, ge=1, le=500)
    res_E_beam: float | None = Field(default=None, ge=0, le=1)
    res_E_extra: float | None = Field(default=None, ge=0, le=1)
    res_E_manual: bool | None = None

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
