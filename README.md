# TensorSpec
**A General-Purpose Framework for N-Dimensional Spectroscopic Analysis**
`TensorSpec` is a Python-based library designed to handle, visualize, and analyze multidimensional spectroscopic data. 
Originally developed for ARPES (Angle-Resolved Photoemission Spectroscopy), it generalizes the concept of "coordinates vs. intensity" to support a wide range of experimental techniques, including XAS, XMCD, PEEM, and other multi-dimensional scanning microscopy.

The goal of this project is to provide a unified data structure for high-dimensional data and seamlessly integrate classical analysis with modern Machine Learning workflows.

## Key Features

### 1. Universal Data Structure (In Development)
Move beyond technique-specific file formats. `TensorSpec` treats intensity data as N-dimensional tensors with associated coordinates:
* **ARPES:** $(\theta, \phi, E)$ or $(k_x, k_y, E)$, $(Eph, \phi, E)$ or $(k_z, k_y, E)$
* **Nano-ARPES:** $(x, y, ARPES)$
* **XAS/XMCD:** $(H\_field, Energy)
* **Microscopy:** $(x, y, z)$

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
