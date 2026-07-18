"""
CENTRAL MEMORY: Global dictionary/manager for all active loaded data.
Strictly pure Python logic. Zero GUI/Plotting imports.
"""

class WorkspaceManager:
    def __init__(self):
        # The primary dictionary holding all datasets (CIFs, DataTrees, etc.)
        self._data = {}

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

# Instantiate the global singleton to be imported across the application
global_workspace = WorkspaceManager()