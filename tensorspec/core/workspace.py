"""
CENTRAL MEMORY: Global dictionary/manager for all active loaded data.
Strictly pure Python logic. Zero GUI/Plotting imports.
"""
import numpy as np
from pathlib import Path
from tensorspec.core.data_models import TensorData

class WorkspaceManager:
    def __init__(self):
        # The primary dictionary holding all datasets (CIFs, DataTrees, etc.)
        self._data = {}
        
        # Set up a default root directory for the project to prevent saving errors
        self.project_dir = Path.cwd() / "TensorSpec_Workspace"
        self.project_dir.mkdir(parents=True, exist_ok=True)

    def push_crystal_structure(self, name, basis_vectors):
        """
        Stores a parsed crystal structure's local basis vectors.
        This will later be called by the Crystal Suite after loading a CIF.
        """
        self._data[name] = {
            'type': 'crystal_structure',
            'basis': basis_vectors
        }

    def pull_crystal_structure(self, name):
        """
        Retrieves the basis vectors of a stored crystal structure.
        """
        item = self._data.get(name)
        if item and item.get('type') == 'crystal_structure':
            return item['basis']
        return None
    
    def list_crystal_structures(self):
        """
        Returns a list of all currently loaded crystal structure names.
        Filters out band structures to ensure only crystals are returned.
        """
        crystals = []
        for key, value in self._data.items():
            # If it's a dictionary and specifically labeled as a band_structure, skip it
            if isinstance(value, dict) and value.get('type') == 'band_structure':
                continue
            # Otherwise, assume it's a crystal structure from our earlier code
            crystals.append(key)
        return crystals
    
    def push_band_structure(self, name, k_dist, eigenvalues, eigenvectors, k_vecs, node_idx, labels, orbital_positions=None):
        """
        Stores a calculated band structure, its wavefunctions, and basis coordinates.
        """
        self._data[name] = {
            'type': 'band_structure',
            'k_dist': k_dist,               # 1D array for x-axis plotting
            'eigenvalues': eigenvalues,     # Energy bands E(k)
            'eigenvectors': eigenvectors,   # Orbital characters/wavefunctions
            'k_vecs': k_vecs,               # Actual 3D k-vectors for matrix elements
            'node_idx': node_idx,           # High symmetry point indices
            'labels': labels,               # High symmetry labels
            'orbital_positions': orbital_positions or [] # NEW: Atomic basis coords for ARPES ME
        }

    def pull_band_structure(self, name):
        """Retrieves the band structure dictionary."""
        item = self._data.get(name)
        if item and item.get('type') == 'band_structure':
            return item
        return None

    def list_band_structures(self):
        """Returns a list of all currently loaded band structure names."""
        return [k for k, v in self._data.items() if v.get('type') == 'band_structure']

    def save_simulated_arpes(self, name, intensity, kx, ky, E, metadata=None):
        """Saves a simulated ARPES dataset to a compressed numpy archive."""
        # Create the data directory if it doesn't exist
        arpes_dir = self.project_dir / "arpes_data" / "simulated"
        arpes_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = arpes_dir / f"{name}.npz"
        
        # Package everything into a compressed archive
        np.savez_compressed(
            file_path, 
            intensity=intensity, 
            kx=kx, 
            ky=ky, 
            E=E, 
            metadata=metadata if metadata else {}
        )
        print(f"Saved simulated ARPES data to: {file_path}")
        return file_path
    
    def load_simulated_arpes_to_tensor(self, filename) -> TensorData:
        """
        Loads a simulated .npz file and packages it into the agnostic TensorData format.
        """
        file_path = self.project_dir / "arpes_data" / "simulated" / filename
        
        with np.load(file_path, allow_pickle=True) as data:
            intensity = data['intensity']  # Original shape from Chinook: (kx, ky, E)
            
            # For the N-Dimensional viewer, we usually want Energy on the 0th axis
            # Transpose (kx, ky, E) -> (E, kx, ky)
            value_matrix = np.transpose(intensity, (2, 0, 1))
            
            # Package it into the universal format
            return TensorData(
                value=value_matrix,
                axes=[data['E'], data['kx'], data['ky']],
                labels=["Energy", "Slit Angle", "Deflection Angle"],
                units=["eV", "deg", "deg"],
                data_type="Simulated ARPES",
                metadata=data['metadata'].item() if 'metadata' in data else {}
            )

# Instantiate the global singleton to be imported across the application
global_workspace = WorkspaceManager()