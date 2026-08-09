# TensorSpec Roadmap

## UI Architecture: HTML Migration (Active)
The front end is a browser-based HTML interface (FastAPI + `core/`). The PySide6 desktop
GUI has been removed (Phase 5).

**Checkbox meaning:** `[x]` = shipped end-to-end for the HTML app (UI + API + `core/`), unless a
note says otherwise. Deferred items stay unchecked.

- [x] Phase 1 — Static shell: `web/templates/main_browser.html` (ribbon, workspace tree, inspector)
- [x] Phase 2 — Static suite pages: one HTML shell per suite
- [x] Phase 3 — Vertical slices: HTML → FastAPI → `core/` for Crystal, ARPES, and DFT
      (PEEM / XAS / Transport / ML suites remain shells only — see Grand App below).
    - [x] Slice 1 — Session backbone: FastAPI app, per-session workspace, live variable tree
    - [x] Slice 2a — Crystal Suite Tab 1: CIF load, geometry as JSON, three.js viewport
	- [x] Slice 2b — Crystal Suite Tabs 2-4: CDW modulator, stack & twist, Brillouin zone
          (includes bulk vdW / HKL exfoliator on Tab 3; MLIP relax + CIF export + workspace push)
    - [x] Slice 3a — ARPES viewer: server-side slicing, crosshair, EDC/MDC/orthogonal curves
    - [x] Slice 3b — ARPES viewer extras: multi-panel snap grid, cross-dataset crosshair sync,
          profile export (CSV/PDF), and the matrix-element simulator tab
          - [x] Multi-panel viewer with Qt-style spawn Up/Down/Left/Right + detach/reattach
          - [x] Cross-dataset crosshair sync by matching axis labels (same-page panels)
          - [x] Profile CSV export (includes orthogonal curve when enabled)
          - [x] Profile PDF/SVG via server matplotlib; client PNG composite of canvases
          - [x] Matrix-element simulator: Option A + B1 via job queue (crystal → 2D TB mesh)
    - [x] Slice 3c — ARPES load + Process + Analysis (MAESTRO upload; k∥/kz Process tab;
          peakfit / QP / gap / DFT·sim overlay Analysis tab)
    - [x] Slice 3d — ARPES Volume tab: BZ-prism cutout viewer (rect/hex), indentation faces,
          Fermi / energy plane inside the prism
    - [x] Slice 4a — DFT Suite: tight-binding band structures solved server-side and plotted
    - [x] Slice 4b — QE pipeline: input generation, downloadable HPC bundle, job queue,
          live log stream. Commands are argv lists from a server allowlist; the browser
          never supplies executable paths. MPI ranks are capped by TENSORSPEC_MAX_MPI_RANKS.
- [x] Phase 4 — Cross-cutting requirements for the multi-user HTML service
      (deployment target remains a shared server on the LBL VPN).
    - [x] Web service layer (`tensorspec/web/server/`): thin request router, zero physics
    - [x] Per-session workspace: each browser session owns its own data and directory
          (legacy `global_workspace` singleton remains only as a core helper for non-web scripts)
    - [x] Parameter validation: Pydantic schemas mirror the bounds the UI advertises
    - [x] Job queue for long calculations (QE / ARPES sims) so requests never block
          (`web/server/jobs.py`; per-session and global concurrency caps)
    - [x] Live log streaming to the browser (WebSocket) for QE / ARPES job runners
    - [x] three.js viewport: Python emits atom positions, bonds, and BZ facets as JSON;
          the browser renders and rotates them. `crystallography.py` keeps all geometry math
          (three.js is vendored in `web/static/vendor/`, so no CDN is needed on the VPN)
    - [x] Server-side slicing/downsampling so 3D and 4D tensors are never shipped whole
          (`core/tensor_ops.py`; planes travel as framed float32, not JSON text)
    - [x] Command hardening: never build shell commands from raw user input;
          allowlist solver executables and server-control the output directory
          (`web/server/config.py` + `core/dft/qe_pipeline.py`)
- [x] Phase 5 — Qt teardown: deleted `tensorspec/gui/` and Qt-bound plotting backends
      (`matplotlib_engine.py`, `pyvista_engine.py`; `plotting/viewers/` was already gone).
      Dropped PySide6 / PyQt6 / pyvistaqt / QtPy / shiboken6 / pyvista / vtk from `requirements.txt`.
      Headless matplotlib export kept in `plotting/backends/arpes_figure.py`; crystal 3D is three.js.

General Rule for the App
- [ ] Always give option to work with GPU or CPU rendering. In the browser this maps to WebGL-backed interactive rendering vs. server-side static images. In any suite, detect what kind of machine is being used and use the right machinery.

Grand App
- Crystal viewer Suite
	- [x] **Refactor Architectural Modularity:** Decouple monolithic `crystal_viewer.py` into modular architecture (`core/crystallography.py` + HTML/three.js viewport; Qt panel backends removed in Phase 5).
	- [x] File loader panel & "Draw" button.
	- [x] Define Miller indices for bounding the drawing. Define number of unit cells.
	- [x] Draw atoms as spheres scaled to atomic radii.
	- [x] Draw nearest-neighbor connecting sticks.
	- [x] 3D Rotation and camera controls.
	- [x] Interactive Mode: Toggle to view, select, and delete individual atoms/sticks (Continuous Eraser Brush & Camera Lock).
	- [x] PBR/Shiny visual styling (3ds Max style) with color controls.
	- [x] Toggleable crystallographic a, b, c axes display mapped to bounding box.
	- [x] Twisting Multilayer Tab
	- [x] Draw polyhedra/planes connecting atoms instead of just sticks (deletable).
	- [x] CDW Mode: Define atom shifts (dx, dy, dz) and propagate simulated distortions.
	- [x] Brillouin Zone Engine: Bulk BZ, Cleaving surface BZ, kz slicing, and surface termination projections. (Completed: Tab 4 is fully operational, generating the Wigner-Seitz cell, the hovering surface plane, and exporting them natively.)
	- [x] ML Interatomic Potential (MLIP) Relaxation: Tab 3 stack relax via pretrained CHGNet / M3GNet (`matgl`); CIF download + workspace push/rename for DFT Suite.
	- [x] MEGNet band-gap predict (scalar Eg, multi-fidelity PBE/HSE/SCAN/GLLB) on Tab 3 + DFT Suite for lab stacks when full E(k) is heavy.
- DFT Suite
	- [x] Establish Data Pipeline: Pull crystal structure from `workspace.py` (drawn from Crystal Viewer Suite) to perform bulk band structure calculations.
	- [x] Core Math: High-Symmetry K-Path Generator (calculate k-points between high symmetry nodes like Gamma, M, K).
	- [x] Folded vs educational path: Auto = supercell/moiré BZ (folded for twists); optional primitive-hex Γ–K–M reference folded into supercell; **unfold_hex** = TB Popescu–Zunger spectral weights on that path (ARPES-like intensity).
	- [x] Core Math: Pull exact Wigner-Seitz BZ nodes (Gamma, M, K, etc.) directly from Tab 4 of the Crystal Suite via global_workspace to define arbitrary 3D k-paths.
	- [x] Toy Tight Binding (TB) Engine: Built-in simple models (1D chain, 2D square lattice, 2D honeycomb/graphene) to test the solver and plotting.
	- [x] Generalized TB Capability: UI to define custom hopping parameters and on-site energies.Engine dynamically scales Slater-Koster integrals using Harrison's rules and auto-detects materials via database.
	- [x] Multi-Orbital Projection: Calculate eigenvector probabilities for atomic character mapping (Fat Bands). HTML fat-band dropdown (shell / element / orbital) re-projects cached evecs via `POST …/bands/fat`.
	- [ ] k.p Perturbation Capability: Near-band-edge models (e.g., Dirac cone effective mass).
	- [ ] Full DFT Capability: Wrappers to trigger/parse external solvers from Quantum Espresso
		- [x] Abstract QE Input Generator (scf.in, nscf.in, wannier90.win) decoupled from local executables.
		- [x] GUI component for defining pseudo_dir, k-mesh, and functionals (PBE, LDA, HSE).
		- [x] Local subprocess execution runner with thread-safety.
		- [x] Parse resulting wannier90_hr.dat back into the Tight Binding engine.
		- [x] Slab creation for surface slab calculations (drawn from the structure from the crystal suite).
		  DFT Suite: Prepare slab (presets + custom hkl/layers/vacuum) for Tab&nbsp;1/CIF bulk; Tab&nbsp;3 stacks use Slab QE (`kz=1`, `assume_isolated='2D'`) without re-cut.
		- [ ] Semi-infinite structure setup for Green's function calculation (drawn from the crystal suite).
		- [ ] Define which termination to stop at so we know which band belongs to the surface.
		- [ ] Toggle for Wannier90 Hybridization (Atomic Projections vs. Maximally Localized).
		- [ ] Ab-Initio Band Structure Plotter (Parse and overlay raw QE XML eigenvalues).
		- [ ] Ab-initio Relaxation Pipeline: Support for structural optimization (`relax` and `vc-relax`) input generation and execution inside the Quantum ESPRESSO runner.
- ARPES Suite
	- [x] Standalone Kinematics Engine: Convert experimental angles/energies to k_parallel and k_z
	- [x] Hierarchical Simulation Engine Selector (GUI Dropdown & Factory Router)
		- [x] Option A: Phenomenological Three-Step Model
			- [x] Step 1: Matrix element optical excitation mapping
			- [x] Step 2: Mean-free path electron transport modeling
			- [x] Step 3: Classic surface transmission & refraction
		- [x] Option B: One-Step Photoemission Model
			- [x] B1: Tight-Binding + Free Electron Final State (Chinook Engine integration)
			- [ ] B2: Real-space DFT Orbitals + Plane Wave Final State (kMap FFT tomography) — deferred
			- [ ] B3: Full Multiple Scattering & Time-Reversed LEED (SPR-KKR / oscarpes) — deferred
	- [x] Linked Crosshair Data Viewer (kind-aware multi-panel; interactive EDC/MDC; δx/δy integrate; Sync)
	- [x] Data loader — MAESTRO is the active deployment target (other beamlines deferred until needed)
		- [x] MAESTRO (modern Fixed/Swept; Preview XY maps; hv/`mono_eV` scans; Slit Defl Fermi maps; dither/binning attrs; aborted scans; optional Measurement Log CSV; web `POST /api/arpes/load`)
		- [ ] i05 Diamond — deferred
		- [ ] SIS SLS — deferred
		- [ ] ADRESS SLS — deferred
		- [ ] Lorea Alba — deferred
		- [ ] Bloch MaxIV — deferred
	- [x] Implement the `TensorSpec` Hierarchical Data Container (`xarray.DataTree` / NeXus model) to store measurement arrays and comprehensive beamline metadata (`attrs`).
	- [x] Structure tree hierarchy into standardized functional nodes:
		- [x] `/raw`: Raw analyzer intensity array (`data.value`) bound to angular/energy coordinates (`data.energy`, `data.slitangle`).
		- [x] `/raw/motors`: Log multi-axis manipulator variables (`motor1` e.g., polar deflection/theta map, `motor2` e.g., fine piezo scan X/Y, or photon energy $h\nu$ for $k_z$ scans).
		- [x] `/processed`: Store transformed coordinate cubes (e.g., interpolated $E, k_x, k_y$ volumes, curvature analysis, or normalization).
			- [x] In-plane angle → k∥ Process tab (user Γ center via click/drag, Suggest assist, Apply → new dataset + `/processed`)
			- [x] Overlay projected surface BZ from workspace crystal (Crystal Suite Tab 4 machinery)
			- [x] Photon-energy → kz module (Vo slider + perpendicular BZ)
		- [x] `/analysis`: Attach mathematical model outputs (ML-ready peak tables + QP summaries + gap fits).
			- [x] Phase 1 — EDC/MDC peakfit: Lorentzian or Voigt (+ optional FD on EDC); N peaks/seeds; stack fit → `/analysis/mdc_peakfit` or `/edc_peakfit`; ARPES Suite **Analysis** tab (curve + stack plots).
			- [x] Phase 2 — QP result curves from peak tables: δE(Γ)–E, integrated intensity vs E, dispersion E(k), k_F, parabolic m*/m_e, v_F; FL (Γ₀+αω²) / MFL (Γ₀+α|ω|) fits → `/analysis/qp_results`.
			- [x] Phase 3 — Gap tools (Dynes SC/CDW on EDCs → `/analysis/gap_fit`); DFT band polylines + simulated ARPES intensity overlay on experimental cuts (Analysis tab).
		- [x] `/history`: Append sequential audit trail logs of every functional transformation applied to the dataset.
	- [x] Kind-aware general viewer once data is loaded (cut / Fermi map / hv·motor layouts)
		- [x] Dispersion: `energy` × `slitangle` (or k) with EDC/MDC side panels
			- [x] Toggle EDC and MDC on the right and lower panels
			- [x] Crosshair δx / δy integration for EDC and MDC
		- [x] With `motor1`: map panel (slitangle × motor1) + dispersion + motor×energy; linked crosshairs + δx/δy
		- [x] With `motor2`: full multi-panel linked crosshairs + integrate

	**ARPES Suite — available in HTML now (keep in sync when shipping features):**
	- Load: MAESTRO HDF5 (+ optional Measurement Log CSV) via upload / `POST /api/arpes/load`
	- Data Viewer: kind-aware multi-panel layouts (cut / Fermi map / hv scan), Sync crosshair + fixed-dim sliders, EDC/MDC profiles, CSV/PDF/PNG export
	- Process: in-plane → k∥ (Γ click/suggest, surface BZ overlay); photon → kz (Vo, ⊥ BZ); writes `/processed`
	- Analysis: EDC/MDC Lorentzian/Voigt peakfit → `/analysis/*_peakfit`; QP curves (δE–E, k_F, m*, v_F, FL/MFL) → `/analysis/qp_results`; Dynes SC/CDW gap → `/analysis/gap_fit`; DFT bands + sim intensity overlay on cuts
	- Volume: BZ-prism 3D cutout (rectangle or hexagon from crystal/data); indent sectors to reveal interior walls; horizontal E-plane (Fermi surface)
	- Simulator: matrix-element Option A + B1 (job queue + log stream)
	- Deferred (not started): other beamline loaders; sim B2/B3; PEEM / XAS / Transport (see Grand App vision below)

- [ ] PEEM Suite — deferred (vision; first slice = load + view)
	- [ ] loader of tif file stacks
	- [ ] loader of sequences of series of tif files from a folder
	- [ ] stack the CP and CM together or LH and LV together depending on the files
	- [ ] once stacked, build drift-correction options
	- [ ] separate those CP and CM or LH and LV
	- [ ] make the background subtraction button to be applied to all
		- [ ] for background, make it clear to the user what functions we use; refer to Co₃Sn₂S₂ laser ARPES paper as a starter. Other backgrounds can be suggested in later iterations.
		- [ ] UI: plot spectra, toggle background overlay, toggle bg-subtracted spectra separately.
		- [ ] several spectra and their background-related toggles can be plotted together
	- [ ] perform sum rule analysis if it is CP and CM data
	- [ ] analysis of spectra / sum rule switchable: picture-wide | user ROI | pixel-to-pixel (noisy / slower)
		- [ ] ROI shapes: rectangle, ellipse, or custom polygon (straight segments or curved/interpolated from clicked points)
	- [ ] ALWAYS include statistical analysis for background / sum rule (uncertainty often dominated by BG choice): vary plausible backgrounds and report sum-rule spread
	- [ ] real-space PEEM ↔ momentum-microscope data at a given XY: decide how to store / link in metadata
	- [ ] tools to analyze magnetic domain wall size
	- [ ] correlate azimuthal datasets: intensity at same sample position → estimate magnetic moment magnitude and direction
		- [ ] rotate azimuth pictures onto a reference (e.g. azimuth 0); map features with possible non-homogeneous rescale (prefer defect landmarks; magnetic contrast is azimuth-dependent)
	- [ ] tools for % up vs down domain area; temperature series of the same field of view to track domain evolution
	- [ ] other common magnetic-domain shape analyses — iterate in discussion
- [ ] XAS Suite — deferred (thin front-end; shared core with PEEM)
	- [ ] Treat XAS analysis as a 1D sibling of PEEM BG / sum-rule tools (often TEY, not photoemission images). Keep a separate ribbon entry; share `core/` engines (BG, sum rule, stats). Possibly rename later if a better umbrella name emerges.
- [ ] Transport Suite — deferred
	- [ ] Loader for typical Quantum Design PPMS / MPMS files and metadata (extend when other vendors appear)
	- [ ] Support group-specific breakout-box column layouts: parse and assign variables accordingly
	- [ ] Plots: line / combined line; also image-like plots reusing generalized ARPES-style plotters where possible
	- [ ] Axes: keep proper column names when known, generic when not; rename when identified; plot by dimension name (A vs B, multi-column overlays)
	- [ ] Typical transport analyses (gradient, known equations) — survey + suggested analysis toggles + “analyze selected”
		- [ ] also low-level lego blocks: gradient / 2nd derivative / gap / diode / superconducting transition (finite R vs →0), etc., so experts can compose custom pipelines
