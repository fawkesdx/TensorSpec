# AI Interaction Guidelines for TensorSpec

When assisting with this repository, strictly adhere to the following rules:

1.  **Snippet-Only Output:** NEVER output the entire codebase or full script unless explicitly asked to generate a brand new file. Only output the exact classes, functions, or UI blocks that need to be updated or added.
2.  **Contextual Placement:** Always clearly state exactly *where* the provided code block should be inserted or what existing code it replaces (e.g., "Replace the `draw_structure` function" or "Insert this below line 42"). if you lose context, always ask what file to be uploaded for better reference.
3.  **No Silent Deletions:** Do not remove existing features, buttons, or imports unless specifically instructed to refactor them out. 
4.  **Acknowledge Roadmap:** Always refer back to `roadmap.md` to ensure UI additions fit into the planned Tabbed architecture. When a feature ships (even if it was not pre-listed), update `roadmap.md` in place — mark checkboxes, expand sub-bullets, and refresh the **ARPES Suite — available in HTML now** (or suite-equivalent) summary block. Also keep `README.md` Key Features aligned. Do not make a new roadmap from scratch; tell the user what you changed in those high-level files.
5.  **Strict Modularity & Separation of Concerns:** Never write monolithic single-file suites. New features and refactored components must strictly separate logic into three distinct layers:
    * **Core Math & Physics Engine (`tensorspec/core/`):** Pure Python/NumPy/PyMatgen logic (e.g., symmetry parsing, Moiré math, ARPES momentum transformations). Zero GUI or plotting imports allowed.
    * **Rendering & Visualization Backends (`tensorspec/plotting/`):** Dedicated wrapper classes for PyVista, Matplotlib, or PyQtGraph engines.
    * **Web UI Layer (`tensorspec/web/`):** HTML templates, CSS, and vanilla JS controllers. No Python UI toolkit (PySide6/Qt) imports are permitted anywhere in this repository.
    * if a long monolithic files need to be separated, always tell me which block to be moved where instead of giving me the whole code so I can follow the logic. when separating files, I want you to tell me what to copy from the old file and what to paste in the new file. I only want to move what I know exist in the old files so we dont lose any feature.
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
|   |   └── loaders/
|   |   |   └── maestro_loader.py
│   │   └── peem_loaders.py   # TIF stack & sequence folder loaders 
│   ├── dft_engine.py         # MAIN ROUTER: Routes calculation to chinook_tb or qe_generator
│   ├── dft/                  # Nested folder for separated DFT physics engines
│   │   ├── __init__.py
│   │   ├── chinook_tb.py     # Tight Binding & Slater-Koster math engine
│   │   └── qe_generator.py   # Quantum Espresso & Wannier90 input file generator
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
│   └── backends/              # Headless figure/data producers (no GUI toolkit)
│       ├── __init__.py
│       ├── matplotlib_engine.py # Static PNG/SVG for 1D lines & 2D maps
│       └── pyvista_engine.py    # Off-screen 3D render / mesh export
└── web/                           # Browser UI + FastAPI service. No physics lives here.
    ├── __init__.py
    ├── server/                    # FastAPI layer: request routing only, zero physics
    │   ├── __init__.py
    │   ├── app.py                 # App factory, static mounts, WebSocket setup
    │   ├── session.py             # Per-session workspace registry (replaces the global singleton)
    │   ├── jobs.py                # Background job queue for QE / ARPES solvers
    │   ├── schemas.py             # Pydantic models mirroring the UI parameter bounds
    │   └── routers/               # One router per suite, mirroring core/ engines
    │       ├── __init__.py
    │       ├── workspace.py       # List / push / pull session data
    │       ├── crystal.py         # CIF load, geometry + BZ facets as JSON
    │       ├── dft.py             # Band structure, QE pipeline, live log stream
    │       └── arpes.py           # Simulation router + server-side tensor slicing
    ├── templates/
    │   ├── main_browser.html      # THE BIG UI: Workspace Explorer & Suite Launcher Ribbon
    │   └── suites/                # One HTML shell per roadmap suite
    │       ├── crystal_suite.html
    │       ├── dft_suite.html
    │       ├── arpes_suite.html
    │       ├── peem_suite.html
    │       ├── xas_suite.html
    │       ├── transport_suite.html
    │       └── ml_suite.html
    └── static/
        ├── css/
        │   ├── base.css           # Design tokens: color, type, spacing
        │   ├── layout.css         # Ribbon / sidebar / inspector shell
        │   └── suite.css          # Suite shell: tabs, form controls, canvas frames
        └── js/
            ├── api.js             # Fetch wrapper for the FastAPI endpoints
            ├── workspace_tree.js  # Renders variable tree + selection events
            ├── inspector.js       # Metadata panel rendering
            ├── suite_launcher.js  # Opens suite panels
            └── viewers/           # Browser replacements for the old Qt viewer widgets
                ├── viewer_1d.js   # LineViewer: 1D spectra, stack overlays, peak fit plotting
                ├── viewer_2d.js   # ImageViewer: 2D heatmap, contrast levels, live EDC/MDC crosshairs
                ├── viewer_3d.js   # three.js crystal / BZ renderer; geometry arrives as JSON
                └── viewer_4d.js   # HypercubeViewer: 3D slicer + 4th dimension timeline/motor slider

8. **ARPES Multi-Engine Protocol**: 
   When writing physics solvers under `core/arpes/`, never let solver-specific parameters bleed into the main UI. 
   The `arpes_engine.py` must act as a unified Factory Router. It receives a configuration dictionary from the UI containing the model choice (A, B1, B2, B3) along with experimental variables, routes it to the designated submodule, and parses the output back into an xarray.DataTree structure under `/simulated` .

9. **HTML-Only UI Protocol:**
   * No `PySide6`, `PyQt6`, `pyvistaqt`, or `QtPy` imports. No `QApplication`, `.show()`, or signal/slot routing.
   * Static-first: plain HTML + CSS + vanilla JS. No build step, no bundler, no framework until explicitly approved.
   * Panels are HTML partials, never monolithic pages — mirror the modularity of rule 5.
   * The Python `core/` layer must stay UI-agnostic: it returns plain data (JSON-serializable dicts, arrays, file paths), never widgets.
   * HTML and JS contain zero physics. Every calculation is an API call into `core/`; the browser only sends parameters and renders returned data.
   * 3D geometry is computed in Python and sent as JSON. three.js draws it; it never derives lattice vectors, bonds, or Brillouin-zone facets on its own.

10. **Multi-User Server Protocol:** The app is served over FastAPI to multiple simultaneous users.
    * No module-level mutable singletons. Session-scoped state only — one user's workspace must never be visible or writable by another.
    * Long-running solvers go to a job queue and report progress; never block a request.
    * Treat every request as untrusted: validate numeric parameters against the same bounds the UI declares, and never interpolate user text into a shell command.