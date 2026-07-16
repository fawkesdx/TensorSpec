# AI Interaction Guidelines for TensorSpec

When assisting with this repository, strictly adhere to the following rules:

1.  **Snippet-Only Output:** NEVER output the entire codebase or full script unless explicitly asked to generate a brand new file. Only output the exact classes, functions, or UI blocks that need to be updated or added.
2.  **Contextual Placement:** Always clearly state exactly *where* the provided code block should be inserted or what existing code it replaces (e.g., "Replace the `draw_structure` function" or "Insert this below line 42").
3.  **No Silent Deletions:** Do not remove existing features, buttons, or imports unless specifically instructed to refactor them out. 
4.  **Acknowledge Roadmap:** Always refer back to `ROADMAP.md` to ensure UI additions fit into the planned Tabbed architecture. If the new path emerges during the development, always refer back to the roadmap.md and tell me to update which part into what. do not make a new roadmap from scratch.
5.  **Strict Modularity & Separation of Concerns:** Never write monolithic single-file suites. New features and refactored components must strictly separate logic into three distinct layers:
    * **Core Math & Physics Engine (`tensorspec/core/`):** Pure Python/NumPy/PyMatgen logic (e.g., symmetry parsing, Moiré math, ARPES momentum transformations). Zero GUI or plotting imports allowed.
    * **Rendering & Visualization Backends (`tensorspec/plotting/`):** Dedicated wrapper classes for PyVista, Matplotlib, or PyQtGraph engines.
    * **UI Controllers (`tensorspec/gui/`):** PySide6 layout definitions, widgets, and signal/slot connection routing.
6.  **Hierarchical Data Architecture:** All multi-dimensional spectroscopic data containers must adopt the **Hierarchical Tree Model** (via `xarray.DataTree` aligned with NeXus/HDF5 standards). Never store disconnected arrays. Every data object must structure its nodes as:
    * `/raw`: Immutable experimental intensity matrices, hardware coordinates, and metadata (`attrs`).
    * `/processed`: Transformed datasets (e.g., $E, k$ space, drift-corrected PEEM stacks).
    * `/analysis`: Sub-nodes for mathematical fits (e.g., `/analysis/peakfit`, `/analysis/background`).
    * `/history`: Provenance log tracking all sequential operations and parameters applied to the tree.
7.  **Target Directory Blueprint:** Whenever generating new files or breaking down monolithic scripts, strictly organize code inside this folder structure:
tensorspec/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── workspace.py          # CENTRAL MEMORY: Global dictionary/manager for all active loaded data
│   ├── data_tree.py          # Hierarchical xarray.DataTree structure & /history audit tracking
│   ├── crystallography.py    # Math engine: PyMatgen symmetry, Miller cleavage, CDW, Moiré strain
│   ├── kinematics.py         # Angle/energy to momentum space (k_parallel, k_z) conversions & photon momentum
│   ├── io/                   # Dedicated file loaders
│   │   ├── __init__.py
│   │   ├── arpes_loaders.py  # Readers for MAESTRO, i05 Diamond, SIS/ADRESS SLS, Lorea Alba, Bloch MaxIV 
│   │   └── peem_loaders.py   # TIF stack & sequence folder loaders 
│   ├── dft_engine.py         # Bulk band structure, tight binding, k.p, DFT... 
│   ├── arpes_engine.py       # MAIN ROUTER: Routes calculation to three_step.py or one_step/
│   ├── arpes/                # Nested folder for separated physics engines
│   │   ├── __init__.py
│   │   ├── three_step.py     # Option A: Classic 3-step phenomenological calculations
│   │   └── one_step/         # Option B: Advanced 1-step solver submodules
│   │       ├── __init__.py
│   │       ├── chinook_wrapper.py  # B1: Handles Chinook TB initialization and calculation 
│   │       ├── kmap_solver.py      # B2: Plane-wave FFT from real-space DFT orbitals 
│   │       └── kkr_wrapper.py      # B3: Generates SPR-KKR inputs and parses output via oscarpes 
│   ├── peem_engine.py        # Drift correction, CP/CM & LH/LV separation, sum rule analysis 
│   ├── xas_engine.py         # XAS/XMCD background subtraction and normalization 
│   ├── transport_engine.py   # Transport analysis (magnetoresistance, Hall curves, R-T scaling) 
│   └── ml_engine.py          # Domain clustering, PCA/NMF dimensionality reduction, image analysis 
├── plotting/
│   ├── __init__.py
│   ├── backends/              # Low-level rendering engines
│   │   ├── __init__.py
│   │   ├── matplotlib_engine.py # Safe CPU rendering for 1D lines & static 2D maps
│   │   ├── pyvista_engine.py    # Fast GPU rendering for 3D crystal structures & volumes
│   │   └── pyqtgraph_engine.py  # High-speed real-time 2D image rendering (optional but recommended)
│   └── viewers/               # Reusable Qt widgets for suites to embed
│       ├── __init__.py
│       ├── viewer_1d.py       # LineViewer: 1D spectra, stack overlays, peak fit plotting
│       ├── viewer_2d.py       # ImageViewer: 2D heatmap, contrast levels, live EDC/MDC crosshairs
│       ├── viewer_3d.py       # VolumeSlicer: 3D cube orthogonal slicer & iso-surface rendering
│       └── viewer_4d.py       # HypercubeViewer: 3D VolumeSlicer + 4th dimension timeline/motor slider
└── gui/
    ├── __init__.py
    ├── main_browser.py       # THE BIG GUI: Global Data Workspace Explorer & Suite Launcher Ribbon
    └── suites/               # The 6 independent roadmap suites + ML integration
        ├── __init__.py
        ├── crystal_suite.py  # 1. Crystal Viewer: CIF loader, supercells, CDW, stack/twist, BZ 
        ├── dft_suite.py      # 2. DFT Suite: Band structures, slabs, Green's function surface setup 
        ├── arpes_suite.py    # 3. ARPES Suite: Multi-motor dispersion viewer, linked crosshairs, EDC/MDC 
        ├── peem_suite.py     # 4. PEEM Suite: Stack alignment, drift correction, sum rules 
        ├── xas_suite.py      # 5. XAS Suite: 1D spectral plotting and field/energy normalization 
        ├── transport_suite.py# 6. Transport Suite: Curves, transport parameters, and magneto-transport 
        └── ml_suite.py       # ML Suite: Hyperspectral clustering and PCA/NMF decomposition 

8. **ARPES Multi-Engine Protocol**: 
   When writing physics solvers under `core/arpes/`, never let solver-specific parameters bleed into the main UI. 
   The `arpes_engine.py` must act as a unified Factory Router. It receives a configuration dictionary from the GUI containing the model choice (A, B1, B2, B3) along with experimental variables, routes it to the designated submodule, and parses the output back into an xarray.DataTree structure under `/simulated` .