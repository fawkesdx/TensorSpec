# TensorSpec
## Prerequisites
* **Python 3.11 or 3.12** (Python 3.9 and 3.10 are not supported due to modern dependency requirements).
* A virtual environment is highly recommended.
* **A modern browser** (Chrome, Firefox, or Safari) for the user interface.
* No Qt or other desktop toolkit is required.

**A General-Purpose Framework for N-Dimensional Spectroscopic Analysis**
`TensorSpec` is a Python-based framework with a browser-based (HTML/CSS/JS) interface, designed to handle, visualize, and analyze multidimensional spectroscopic data. 
Originally developed for ARPES (Angle-Resolved Photoemission Spectroscopy), it generalizes the concept of "coordinates vs. intensity" to support a wide range of experimental techniques, including XAS, XMCD, PEEM, and other multi-dimensional scanning microscopy.

The goal of this project is to provide a unified data structure for high-dimensional data and seamlessly integrate classical analysis with modern Machine Learning workflows.

## Key Features

### 1. Universal Hierarchical Data Structure (In Development)
Move beyond technique-specific file formats and flat structs. `TensorSpec` utilizes a **Hierarchical Data Tree architecture** powered by `xarray.DataTree` (NeXus/HDF5 aligned) to organize high-dimensional data, metadata, and analytical provenance:
* **Structured Nodes:** Isolates immutable raw beamline data (`/raw`) from calibrated or interpolated data (`/processed`) and fitting outputs (`/analysis/peakfit`).
* **Multi-Technique Coordinates:** Treats intensity arrays as N-dimensional labeled tensors locked to physical coordinates:
  * **ARPES:** $(\theta, \phi, E) \to (k_x, k_y, E)$ or $(k_z, k_y, E)$
  * **sARPES:** $(sx,sy,sz,ARPES)$
  * **tARPES:** $(t_{delay},power_{pump},Eph_{pump},ARPES)$
  * **Nano-ARPES:** $(x, y, ARPES)$
  * **XAS/XMCD:** $(H_{field}, Energy)$
  * **Microscopy/PEEM:** $(x, y, z, Time)$
* **Built-in Provenance Tracking:** Automatically records data processing history and algorithmic parameters directly inside the container tree (`/history`).

### 2. Visualization & Slicing (In Development)
* **Hyperspectral Slicing:** View 2D cuts of 3D/4D data (e.g., momentum-energy cuts in ARPES).
* **Interactive Plotting:**
* **1D & 2D Modes:** Automatically detects data dimensionality to toggle between line plots (XAS/XPS) and image maps (ARPES).

### 3. Analysis & Fitting
* **Momentum conversion (ARPES Suite Process tab):** angle → k∥ (Γ center + surface BZ overlay); photon energy → kz (inner potential Vo + perpendicular BZ). Writes `/processed`.
* **EDC/MDC peakfit (Analysis tab):** Lorentzian or Voigt (analyzer FWHM); optional Fermi–Dirac on EDCs; N peaks with seeds; stack fit stored as `/analysis/mdc_peakfit` or `/analysis/edc_peakfit` (`peakfit_v1`).
* **QP result curves:** from peak tables — δE(Γ) vs E, integrated intensity vs E, E(k), k_F, parabolic m*/m_e, v_F; Fermi-liquid / marginal-FL linewidth fits. Stored as `/analysis/qp_results` (`qp_results_v1`).
* **Gap tools:** Dynes SC/CDW density of states × Fermi–Dirac (± analyzer resolution) on EDCs; stack → Δ(k) under `/analysis/gap_fit` (`gap_fit_v1`).
* **Cut overlays:** DFT path bands as polylines and/or resampled simulated ARPES intensity on experimental cuts (Analysis tab).
* **Still planned:** richer XPS-style backgrounds; image filtering utilities; additional beamline loaders.

### 4. Machine Learning Integration (In Development)
A dedicated module for attaching ML routines to experimental data:
* Clustering for domains classification from spatial scans.
* Dimensionality reduction (PCA/NMF) for hyperspectral datasets.
* Deep learning-based image analysis.
* Transfer learning-based model.
* Peak / QP tables under `/analysis` carry `usable_for_ml` / `usable_for_tb_feedback` attrs for later TB feedback.

### 5. Browser-Based Interface
The user interface is a browser application served by FastAPI (per-session workspace), replacing the earlier PySide6 desktop GUI:
* **Workspace Browser:** Central explorer for active data variables, metadata inspector, and suite launcher ribbon.
* **Suite Panels:** Crystal, DFT, and ARPES suites are live end-to-end (HTML → API → `core/`). PEEM / XAS / Transport / ML shells exist; engines still pending.
* **Static-First front end:** plain HTML/CSS/vanilla JS (no bundler). Physics stays in Python.

## Running the App
From the repo root (with the project venv active):

```bash
uvicorn tensorspec.web.server.app:app --reload --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000/` in a modern browser. Each browser session owns its own workspace.

See `roadmap.md` for the live checklist of shipped vs planned features (including the **ARPES Suite — available in HTML now** block).