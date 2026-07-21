import xml.etree.ElementTree as ET
import numpy as np
from tensorspec.core.dft.chinook_tb import ChinookTightBindingEngine
from tensorspec.core.dft.qe_generator import QEInputGenerator

class DFTEngineRouter:
    """
    MAIN ROUTER for the DFT Suite. 
    Maintains a single loaded crystal structure and provides unified access 
    to Tight Binding (Chinook) and Ab Initio (Quantum Espresso) engines.
    """
    def __init__(self):
        # We initialize the old engine so the UI doesn't break. 
        # The suite still references self.engine.crystal_structure and self.engine.solve_bands
        self.chinook = ChinookTightBindingEngine()

    @property
    def crystal_structure(self):
        """Proxy to pass the loaded structure to UI and other engines."""
        return self.chinook.crystal_structure

    def load_structure_from_workspace(self, variable_name: str) -> bool:
        """Proxies the workspace loading down to the tight binding engine."""
        return self.chinook.load_structure_from_workspace(variable_name)

    def get_qe_generator(self):
        """Returns the QE Input Generator initialized with the active structure."""
        if not self.crystal_structure:
            raise ValueError("No structure loaded. Cannot initialize QE generator.")
        return QEInputGenerator(self.crystal_structure)

    # -------------------------------------------------------------
    # Proxy Methods for backward compatibility with dft_suite.py
    # -------------------------------------------------------------
    def get_default_hopping(self, formula: str):
        return self.chinook.get_default_hopping(formula)

    def _get_orbital_basis(self, element_symbol):
        return self.chinook._get_orbital_basis(element_symbol)

    def get_auto_kpath(self):
        return self.chinook.get_auto_kpath()

    def get_custom_kpath(self, coords_str, labels_str):
        return self.chinook.get_custom_kpath(coords_str, labels_str)

    def get_kpath_template(self, lattice_type="hexagonal", a=3.0, b=3.0):
        return self.chinook.get_kpath_template(lattice_type, a, b)

    def generate_k_path(self, high_sym_pts, labels, points_per_segment=100):
        return self.chinook.generate_k_path(high_sym_pts, labels, points_per_segment)

    def solve_bands(self, *args, **kwargs):
        return self.chinook.solve_bands(*args, **kwargs)

    def parse_qe_xml(self, filepath: str):
        """
        Parses Quantum ESPRESSO's data-file-schema.xml to extract ab-initio bands.
        Converts eigenvalues from Hartrees to eV.
        """
        tree = ET.parse(filepath)
        root = tree.getroot()

        # Strip XML namespaces for easier searching
        for elem in root.iter():
            if '}' in elem.tag:
                elem.tag = elem.tag.split('}', 1)[1]

        k_pts = []
        eigenvals = []
        fermi_energy = 0.0

        # Attempt to find Fermi Energy
        fermi_elem = root.find('.//fermi_energy')
        if fermi_elem is None:
            fermi_elem = root.find('.//highestOccupiedLevel')
        if fermi_elem is not None:
            fermi_energy = float(fermi_elem.text) * 27.211386  # Hartree to eV

        # Extract K-points and Eigenvalues
        for ks in root.findall('.//ks_energies'):
            k_elem = ks.find('k_point')
            if k_elem is not None:
                k_pts.append([float(x) for x in k_elem.text.split()])

            eig_elem = ks.find('eigenvalues')
            if eig_elem is not None:
                eigenvals.append([float(x) * 27.211386 for x in eig_elem.text.split()])

        k_pts = np.array(k_pts)
        eigenvals = np.array(eigenvals)

        if len(k_pts) == 0:
            raise ValueError("No k-points or eigenvalues found in XML.")

        # Calculate cumulative k-path distances for plotting
        k_dist = [0.0]
        for i in range(1, len(k_pts)):
            dist = np.linalg.norm(k_pts[i] - k_pts[i-1])
            k_dist.append(k_dist[-1] + dist)
        k_dist = np.array(k_dist)

        return k_dist, eigenvals, fermi_energy