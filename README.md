# TensorSpec
## Prerequisites
* **Python 3.11 or 3.12** (Python 3.9 and 3.10 are not supported due to modern dependency requirements).
* A virtual environment is highly recommended.

**A General-Purpose Framework for N-Dimensional Spectroscopic Analysis**
`TensorSpec` is a Python-based library designed to handle, visualize, and analyze multidimensional spectroscopic data. 
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

### 3. Analysis & Fitting (In Development)
* **momentum space conversion:** converting angular data into momentum space for ARPES.
* **Curve Fitting:** Robust routines for fitting XPS peaks (Voigt, Gaussian, Lorentzian) and background subtraction for momentum distribution curve (MDC) and energy distribution curve (EDC).
* **Image Processing:** Standard filtering, background removal, and normalization for spectral maps.

### 4. Machine Learning Integration (In Development)
A dedicated module for attaching ML routines to experimental data:
* Clustering for domains classification from spatial scans.
* Dimensionality reduction (PCA/NMF) for hyperspectral datasets.
* Deep learning-based image analysis.
* Transfer learning-based model.
