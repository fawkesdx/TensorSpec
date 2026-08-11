"""
CENTRAL MEMORY: Global dictionary/manager for all active loaded data.
Strictly pure Python logic. Zero GUI/Plotting imports.
"""
import numpy as np
from pathlib import Path
from tensorspec.core.data_models import TensorData
from tensorspec.core.data_tree import DataTreeBuilder

class WorkspaceManager:
    def __init__(self, project_dir=None):
        # The primary dictionary holding all datasets (CIFs, DataTrees, etc.)
        self._data = {}
        
        # Set up a root directory for the project to prevent saving errors.
        # Served sessions pass their own directory so users never share files.
        self.project_dir = Path(project_dir) if project_dir else Path.cwd() / "TensorSpec_Workspace"
        self.project_dir.mkdir(parents=True, exist_ok=True)

    def list_items(self):
        """Returns {name: type} for every object currently held in memory."""
        return {name: item.get('type', 'Unknown') for name, item in self._data.items()}

    def push_crystal_structure(self, name, basis_vectors, structure=None):
        """
        Stores a parsed crystal structure's local basis vectors, and optionally
        the full pymatgen Structure so the atoms survive alongside the lattice.
        Callers that only have the basis may omit `structure`.
        """
        self._data[name] = {
            'type': 'crystal_structure',
            'basis': basis_vectors,
            'structure': structure
        }

    def pull_crystal_structure(self, name):
        """
        Retrieves the basis vectors of a stored crystal structure.
        """
        item = self._data.get(name)
        if item and item.get('type') == 'crystal_structure':
            return item['basis']
        return None

    def pull_structure_object(self, name):
        """
        Retrieves the full pymatgen Structure for a stored crystal, or None if
        it was pushed with only its basis vectors.
        """
        item = self._data.get(name)
        if item and item.get('type') == 'crystal_structure':
            return item.get('structure')
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
    
    def push_band_structure(self, name, k_dist, eigenvalues, eigenvectors, k_vecs, node_idx, labels, orbital_positions=None, orbital_labels=None):
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
            'orbital_positions': orbital_positions or [],  # Atomic basis coords for ARPES ME
            'orbital_labels': list(orbital_labels) if orbital_labels is not None else [],
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
    def push_spectroscopy_data(self, name: str, tensor_data: TensorData):
        """
        Converts raw TensorData into a strict xarray.DataTree and stores it in memory.
        """
        tree = DataTreeBuilder.build_from_tensor(name, tensor_data)
        self._data[name] = {
            'type': 'spectroscopy_tree',
            'tree': tree
        }
        print(f"DataTree '{name}' successfully pushed to Global Workspace.")

    def push_tensor_data(self, name: str, tensor_data: TensorData):
        """Alias for push_spectroscopy_data."""
        self.push_spectroscopy_data(name, tensor_data)

    def merge_spectroscopy_raw_attrs(self, name: str, attrs: dict) -> bool:
        """Merge metadata into an existing spectroscopy tree's ``/raw`` node."""
        item = self._data.get(name)
        if not item or item.get('type') != 'spectroscopy_tree':
            return False
        item['tree'] = DataTreeBuilder.merge_raw_attrs(item['tree'], attrs)
        return True

    def write_processed_data(self, name: str, tensor_data: TensorData) -> bool:
        """Write a processed cube into ``/processed`` of an existing spectroscopy tree."""
        item = self._data.get(name)
        if not item or item.get('type') != 'spectroscopy_tree':
            return False
        item['tree'] = DataTreeBuilder.write_processed(item['tree'], tensor_data)
        return True

    def write_processed_child_data(self, name: str, child_name: str, tensor_data: TensorData) -> bool:
        item = self._data.get(name)
        if not item or item.get("type") != "spectroscopy_tree":
            return False
        item["tree"] = DataTreeBuilder.write_processed_child(
            item["tree"], child_name, tensor_data
        )
        return True

    def list_processed_children(self, name: str) -> list[str]:
        item = self._data.get(name)
        if not item or item.get("type") != "spectroscopy_tree":
            return []
        return DataTreeBuilder.list_processed_children(item["tree"])

    def write_analysis_data(self, name: str, node_name: str, dataset) -> bool:
        """Write an analysis Dataset under ``/analysis/<node_name>``."""
        item = self._data.get(name)
        if not item or item.get('type') != 'spectroscopy_tree':
            return False
        item['tree'] = DataTreeBuilder.write_analysis(item['tree'], node_name, dataset)
        return True

    def pull_analysis_data(self, name: str, node_name: str = "mdc_peakfit"):
        """Return an analysis Dataset, or None."""
        item = self._data.get(name)
        if not item or item.get('type') != 'spectroscopy_tree':
            return None
        tree = item['tree']
        leaf = node_name.strip("/")
        try:
            analysis = tree["analysis"]
            if leaf in analysis.children:
                node = analysis[leaf]
            else:
                return None
        except Exception:
            return None
        return node.to_dataset() if hasattr(node, "to_dataset") else node.ds

    def pull_tensor_data(self, name: str, node: str = "raw") -> TensorData:
        """
        Extracts a specific node from a stored DataTree and packages it back 
        into a TensorData object for the DataViewerPanel to consume.
        """
        item = self._data.get(name)
        if not item or item.get('type') != 'spectroscopy_tree':
            return None
            
        tree = item['tree']
        node = node.strip("/")
        try:
            target = tree[node]
        except KeyError:
            print(f"Error: Node '{node}' does not exist in dataset '{name}'.")
            return None

        ds = target.to_dataset() if hasattr(target, "to_dataset") else target
        if "data" not in ds:
            return None
            
        da = ds["data"]
        
        # Package it back to our universal mathematical format
        return TensorData(
            value=da.values,
            axes=[da.coords[dim].values for dim in da.dims],
            labels=list(da.dims),
            units=[da.coords[dim].attrs.get("units", "") for dim in da.dims],
            data_type=da.attrs.get("long_name", "Unknown"),
            metadata=ds.attrs
        )

# Instantiate the global singleton to be imported across the application
global_workspace = WorkspaceManager()