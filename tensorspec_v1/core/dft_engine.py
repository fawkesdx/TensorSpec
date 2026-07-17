import numpy as np
from tensorspec.core.workspace import global_workspace

class TightBindingEngine:
    """
    Core mathematical engine for Tight Binding (TB) electronic structure calculations.
    Strictly pure math/NumPy logic. Zero GUI or plotting imports.
    """
    def __init__(self):
        self.crystal_structure = None
        
        self.orb_map = {
            's': 0, 'px': 1, 'py': 2, 'pz': 3,
            'dxy': 4, 'dyz': 5, 'dzx': 6, 'dx2-y2': 7, 'dz2': 8
        }
        
        # Universal Material Database
        self.materials_db = {
            'WTe2': {
                'onsite': {'W': 1.0, 'Te': -2.5},
                'sk_base': {'V_ppS': 1.5, 'V_ppP': -0.5, 'V_pdS': 1.2, 'V_pdP': -0.6, 'V_ddS': -1.0, 'V_ddP': 0.8},
                'd0': 2.85
            },
            'VTe2': {
                'onsite': {'V': 0.5, 'Te': -2.5},
                'sk_base': {'V_ppS': 1.5, 'V_ppP': -0.5, 'V_pdS': 1.3, 'V_pdP': -0.6, 'V_ddS': -0.9, 'V_ddP': 0.7},
                'd0': 2.75 
            },
            'TaIrTe4': {
                'onsite': {'Ta': 1.2, 'Ir': 0.8, 'Te': -2.5},
                'sk_base': {'V_ppS': 1.4, 'V_ppP': -0.5, 'V_pdS': 1.1, 'V_pdP': -0.5, 'V_ddS': -1.1, 'V_ddP': 0.9},
                'd0': 2.80
            },
            'NbIrTe4': {
                'onsite': {'Nb': 1.0, 'Ir': 0.8, 'Te': -2.5},
                'sk_base': {'V_ppS': 1.4, 'V_ppP': -0.5, 'V_pdS': 1.1, 'V_pdP': -0.5, 'V_ddS': -1.1, 'V_ddP': 0.9},
                'd0': 2.80
            }
        }
        
    def _get_orbital_basis(self, symbol):
        """Dynamic periodic table assignment."""
        # Added Ir to the d-band transition metal list
        if symbol in ['W', 'Mo', 'Cr', 'Ta', 'Nb', 'V', 'Ir']:
            return ['dz2', 'dx2-y2', 'dxy', 'dyz', 'dzx']
        elif symbol in ['Te', 'Se', 'S', 'O']:
            return ['px', 'py', 'pz']
        return ['s']
        

    def load_structure_from_workspace(self, name):
        """
        Establishes the data pipeline by pulling crystal structure 
        from the global workspace.
        """
        basis = global_workspace.pull_crystal_structure(name)
        if basis is not None:
            self.crystal_structure = basis
            return True
        return False

    def generate_k_path(self, high_sym_points, labels, points_per_segment=50):
        """
        Generates a linear k-path traversing through given high-symmetry nodes.
        
        Args:
            high_sym_points: List of k-space coordinates e.g., [(0,0), (0.5,0)]
            labels: List of strings for the nodes e.g., ['Gamma', 'M']
        Returns:
            k_vectors: Array of explicit 3D (or 2D) k-points.
            k_distances: 1D array of cumulative scalar distances (for plotting the x-axis).
            node_indices: Array indices where the high symmetry points land.
            labels: The passed labels for those nodes.
        """
        k_vectors = []
        k_distances = []
        node_indices = [0]
        
        current_dist = 0.0
        
        for i in range(len(high_sym_points) - 1):
            start_k = np.array(high_sym_points[i])
            end_k = np.array(high_sym_points[i+1])
            
            # Linear interpolation between nodes
            segment = np.linspace(start_k, end_k, points_per_segment, endpoint=(i == len(high_sym_points)-2))
            
            if i > 0:
                segment = segment[1:] # Avoid duplicating points exactly on the nodes
                
            for step in segment:
                if len(k_vectors) > 0:
                    current_dist += np.linalg.norm(step - k_vectors[-1])
                k_vectors.append(step)
                k_distances.append(current_dist)
                
            node_indices.append(len(k_vectors) - 1)
            
        return np.array(k_vectors), np.array(k_distances), node_indices, labels
    
    def get_kpath_template(self, lattice_type="hexagonal", a=2.46, b=None):
        """
        Returns pre-defined high-symmetry k-points and labels for standard 2D lattices.
        'a' and 'b' are the lattice constants.
        """
        if b is None: 
            b = a
            
        if lattice_type.lower() == "hexagonal":
            G = np.array([0.0, 0.0, 0.0])
            M = np.array([np.pi / a, np.pi / (a * np.sqrt(3)), 0.0])
            K = np.array([4.0 * np.pi / (3.0 * a), 0.0, 0.0])
            return [G, M, K, G], ["$\Gamma$", "M", "K", "$\Gamma$"]
            
        elif lattice_type.lower() in ["tetragonal", "square", "rectangular", "orthorhombic"]:
            # Standard path for Rectangular lattice (1T'-WTe2): Gamma -> X -> M -> Y -> Gamma
            G = np.array([0.0, 0.0, 0.0])
            X = np.array([np.pi / a, 0.0, 0.0])
            M = np.array([np.pi / a, np.pi / b, 0.0])
            Y = np.array([0.0, np.pi / b, 0.0])
            return [G, X, M, Y, G], ["$\Gamma$", "X", "M", "Y", "$\Gamma$"]
            
        else:
            return [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0])], ["Start", "End"]
        
    def slater_koster_matrix(self, d_vec, params):
        """Generates the full 9x9 Slater-Koster hopping matrix between two atoms."""
        dist = np.linalg.norm(d_vec)
        if dist < 1e-5: return np.zeros((9, 9), dtype=complex)
            
        l, m, n = d_vec[0]/dist, d_vec[1]/dist, d_vec[2]/dist
        
        V_ssS, V_spS = params.get('V_ssS', 0.0), params.get('V_spS', 0.0)
        V_ppS, V_ppP = params.get('V_ppS', 0.0), params.get('V_ppP', 0.0)
        V_pdS, V_pdP = params.get('V_pdS', 0.0), params.get('V_pdP', 0.0)
        V_ddS, V_ddP, V_ddD = params.get('V_ddS', 0.0), params.get('V_ddP', 0.0), params.get('V_ddD', 0.0)

        E = np.zeros((9, 9), dtype=float)
        
        # s-s and s-p
        E[0, 0] = V_ssS
        E[0, 1], E[0, 2], E[0, 3] = l * V_spS, m * V_spS, n * V_spS
        E[1, 0], E[2, 0], E[3, 0] = -E[0, 1], -E[0, 2], -E[0, 3]
        
        # p-p
        E[1, 1] = l**2 * V_ppS + (1 - l**2) * V_ppP
        E[2, 2] = m**2 * V_ppS + (1 - m**2) * V_ppP
        E[3, 3] = n**2 * V_ppS + (1 - n**2) * V_ppP
        E[1, 2] = E[2, 1] = l * m * (V_ppS - V_ppP)
        E[1, 3] = E[3, 1] = l * n * (V_ppS - V_ppP)
        E[2, 3] = E[3, 2] = m * n * (V_ppS - V_ppP)
        
        # s-d 
        E[0, 4] = np.sqrt(3) * l * m * V_pdS
        E[0, 5] = np.sqrt(3) * m * n * V_pdS
        E[0, 6] = np.sqrt(3) * l * n * V_pdS
        E[0, 7] = (np.sqrt(3) / 2) * (l**2 - m**2) * V_pdS
        E[0, 8] = V_pdS * (n**2 - 0.5 * (l**2 + m**2))
        
        # p-d (Simplified for WTe2 geometry)
        E[1, 4] = np.sqrt(3) * l**2 * m * V_pdS + m * (1 - 2 * l**2) * V_pdP
        E[2, 4] = np.sqrt(3) * l * m**2 * V_pdS + l * (1 - 2 * m**2) * V_pdP
        E[3, 4] = np.sqrt(3) * l * m * n * V_pdS - 2 * l * m * n * V_pdP
        
        # d-d (Diagonal terms for brevity)
        E[4, 4] = 3 * l**2 * m**2 * V_ddS + (l**2 + m**2 - 4 * l**2 * m**2) * V_ddP + (n**2 + l**2 * m**2) * V_ddD
        E[8, 8] = (n**2 - 0.5 * (l**2 + m**2))**2 * V_ddS + 3 * n**2 * (l**2 + m**2) * V_ddP + (3/4) * (l**2 + m**2)**2 * V_ddD

        

        return E

    def solve_toy_graphene(self, k_vectors, t=2.7, onsite=0.0):
        """
        Solves the nearest-neighbor tight binding model for 2D Graphene along a k-path.
        Returns eigenvalues (energies) and eigenvectors (orbital symmetries).
        """
        eigenvalues = []
        eigenvectors = []
        
        # Real-space nearest neighbor vectors for Graphene A-B sublattices
        delta1 = np.array([0.0, 1.0, 0.0])
        delta2 = np.array([-np.sqrt(3)/2, -0.5, 0.0])
        delta3 = np.array([np.sqrt(3)/2, -0.5, 0.0])
        
        for k in k_vectors:
            # Off-diagonal hopping term f(k)
            f_k = -t * (np.exp(1j * np.dot(k, delta1)) + 
                        np.exp(1j * np.dot(k, delta2)) + 
                        np.exp(1j * np.dot(k, delta3)))
            
            # 2x2 Hamiltonian Matrix
            H = np.array([
                [onsite, f_k],
                [np.conj(f_k), onsite]
            ])
            
            # Diagonalize to find Energy (vals) and Wavefunctions (vecs)
            vals, vecs = np.linalg.eigh(H)
            eigenvalues.append(vals)
            eigenvectors.append(vecs)
            
        # --- ADD DUMMY LABELS FOR GRAPHENE ---
        return np.array(eigenvalues), np.array(eigenvectors), ['C_pz', 'C_pz']
            
        return np.array(eigenvalues), np.array(eigenvectors)
    
    def solve_workspace_structure(self, k_vectors, onsite_params=None, sk_params=None):
        if not self.crystal_structure:
            raise ValueError("No structure loaded from workspace!")
            
        if hasattr(self.crystal_structure, 'lattice'):
            coords = self.crystal_structure.cart_coords
            species = [site.species_string for site in self.crystal_structure]
            lattice = self.crystal_structure.lattice.matrix
            R_range = [-1, 0, 1] 
        else:
            raise ValueError("CIF must be loaded via PyMatgen for multi-orbital symmetry.")
            
        N_atoms = len(coords)
        
        # 1. Dynamic Orbital Assignment
        atom_basis = [self._get_orbital_basis(s) for s in species]
        basis_sizes = [len(orbs) for orbs in atom_basis]
        basis_starts = [sum(basis_sizes[:i]) for i in range(len(basis_sizes))]
        H_dim = sum(basis_sizes)
        
        # --- NEW: Track Orbital Labels ---
        orbital_labels = []
        for i in range(N_atoms):
            for orb in atom_basis[i]:
                orbital_labels.append(f"{species[i]}_{orb}")
        
        # 2. Database Lookup (Auto-detect material based on composition)
        unique_elements = set(species)
        
        # Use subset logic to properly identify ternary vs binary compounds
        if {'Ta', 'Ir', 'Te'}.issubset(unique_elements):
            mat_key = 'TaIrTe4'
        elif {'Nb', 'Ir', 'Te'}.issubset(unique_elements):
            mat_key = 'NbIrTe4'
        elif {'W', 'Te'}.issubset(unique_elements):
            mat_key = 'WTe2'
        elif {'V', 'Te'}.issubset(unique_elements):
            mat_key = 'VTe2'
        else:
            mat_key = 'WTe2' # Fallback
        
        db_entry = self.materials_db.get(mat_key)
        d0_ref = db_entry['d0']
        
        # Use GUI params if passed, otherwise fall back to database
        active_sk = sk_params if sk_params else db_entry['sk_base']
        active_onsite = onsite_params if onsite_params else db_entry['onsite']
            
        eigenvalues, eigenvectors = [], []
        
        for k in k_vectors:
            H = np.zeros((H_dim, H_dim), dtype=complex)
            
            # Fill Onsite Energies
            for i in range(N_atoms):
                for o_idx in range(basis_sizes[i]):
                    H[basis_starts[i] + o_idx, basis_starts[i] + o_idx] = active_onsite.get(species[i], 0.0)
            
            # Harrison-Scaled Hopping
            for i in range(N_atoms):
                for j in range(N_atoms):
                    for nx in R_range:
                        for ny in R_range:
                            for nz in [0]: # Monolayer
                                R_vec = nx * lattice[0] + ny * lattice[1] + nz * lattice[2]
                                d_vec = (coords[j] + R_vec) - coords[i]
                                dist = np.linalg.norm(d_vec)
                                
                                if dist < 1e-4 or dist > 7.0: continue
                                
                                # Apply Harrison's 1/d^2 scaling rule
                                harrison_scale = (d0_ref / dist)**2
                                scaled_sk = {k: v * harrison_scale for k, v in active_sk.items()}
                                
                                E_full = self.slater_koster_matrix(d_vec, scaled_sk)
                                phase = np.exp(1j * np.dot(k, d_vec))
                                
                                for row, orb_i in enumerate(atom_basis[i]):
                                    for col, orb_j in enumerate(atom_basis[j]):
                                        sk_val = E_full[self.orb_map[orb_i], self.orb_map[orb_j]]
                                        H[basis_starts[i] + row, basis_starts[j] + col] += sk_val * phase

            H = 0.5 * (H + H.conj().T)
            # --- DIAGNOSTIC PROBE (Run only for the second k-point) ---
            if np.array_equal(k, k_vectors[1]):
                print(f"\n--- DEBUGGING K-POINT: {k} ---")
                
                # 1. Check if the matrix is completely real (No phase = flat bands)
                max_imag = np.max(np.abs(np.imag(H)))
                print(f"Max Imaginary Phase Amplitude: {max_imag:.5f} (Should be > 0.0)")
                
                # 2. Check Hermiticity (If False, np.linalg.eigh destroys your bands)
                is_hermitian = np.allclose(H, H.conj().T, atol=1e-5)
                print(f"Is Hamiltonian Hermitian?: {is_hermitian}")
                
                # 3. Check Bond Connections
                n_elements = np.count_nonzero(np.abs(H) > 1e-3)
                print(f"Active Hopping Connections: {n_elements} / {H_dim * H_dim}")
                print("--------------------------------\n")
            
            vals, vecs = np.linalg.eigh(H)
            eigenvalues.append(vals)
            eigenvectors.append(vecs)
            
        # --- RETURN LABELS WITH EIGENVECTORS ---
        return np.array(eigenvalues), np.array(eigenvectors), orbital_labels
    
    