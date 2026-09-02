## File Structure & Module Breakdown

**1. CORE APPLICATION:**
* **`main.py`:** The entry point of the application.
* **`maestroai_gui.py`:** The main switchboard. Handles the workspace list, intercepts file clicks, routes ML commands to the Primary 4D Viewer, and launches floating comparison/3D viewers.

**2. UNIVERSAL MAESTRO UTILITIES (Shared across all future apps):**
* **`maestro_loader.py`:** Handles loading HDF5 files and parsing metadata safely.
* **`maestro_4d_viewer.py`:** A completely isolated, reusable custom QWidget containing the 4D spatial map, dispersion map, 1D profiles, and sliders.
* **`maestro_fermi_viewer.py`:** A standalone QDialog/QWidget for exploring 3D Fermi Maps with Slit/Deflection/Energy cross-slicing.

**3. AI-SPECIFIC UI COMPONENTS:**
* **`maestroai_guides.py`:** Strictly houses text-based HTML educational popups.
* **`maestroai_viewers.py`:** Houses standalone interactive windows specific to the ML pipeline (e.g., `DendrogramDialog`, `AzimuthTemplateViewer`).

**4. MACHINE LEARNING ARCHITECTURES:**
* **`maestroai_models.py`:** Pure PyTorch neural network architectures and loss functions.

**5. WORKER MODULES (Background QThreads):**
* **`maestroai_training_ssl.py`:** PyTorch training loop for Unsupervised models.
* **`maestroai_training_sup.py`:** PyTorch loops for Few-Shot Supervised learning.
* **`maestroai_clustering.py`:** scikit-learn/UMAP clustering math.
* **`maestroai_active_learning.py`:** Uncertainty quantification and map steering.
* **`maestroai_alignment.py`:** 3D volume rotations and cross-correlation math.