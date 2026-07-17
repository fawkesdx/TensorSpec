import numpy as np
import traceback
from pymatgen.core import Structure
from tensorspec.core.workspace import global_workspace

try:
    import chinook.build_lib as build_lib
    import chinook.klib as klib  # The officially documented module
except ImportError:
    build_lib = None
    klib = None

class ChinookTightBindingEngine:
    """
    Core engine for Tight Binding calculations using the Chinook backend.
    Translates PyMatgen structures into Chinook TB models and calculates E(k).
    """
    def __init__(self):
        self.crystal_structure = None
        
        # Materials Database for Slater-Koster hopping parameters
        self.materials_db = {
            'C': { # PyMatgen reduces C2 down to C
                'orbitals': ['C', 'C'],
                'sk_base': {
                    'C-C_nn': -2.7,      # Standard graphene t1 is ~ -2.7 eV
                    'C-C_nnn': -0.2,     # t2 is ~ -0.2 eV
                    'C-C_third': 0.0
                }
            },
            'WTe2': {
                'orbitals': ['W', 'Te'],
                'sk_base': {
                    'W-W': -1.5, 'W-Te': -1.2, 'Te-Te_in_plane': -0.8,
                    'Te-Te_interlayer': -0.3 # New proxy for out-of-plane hopping
                } 
            },
            'VTe2': {
                'orbitals': ['V', 'Te'],
                'sk_base': {'V-V': -1.3, 'V-Te': -1.1, 'Te-Te': -0.7}
            }
        }

    def load_structure_from_workspace(self, variable_name: str) -> bool:
        data = global_workspace.pull_crystal_structure(variable_name)
        if isinstance(data, Structure):
            # --- NEW FIX: Force standardize the structure upon load! ---
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            sga = SpacegroupAnalyzer(data)
            self.crystal_structure = sga.get_primitive_standard_structure()
            return True
        return False

    def get_default_hopping(self, formula: str) -> dict:
        """Returns the default hopping params for a formula, or fallback WTe2 values."""
        if formula in self.materials_db:
            return self.materials_db[formula]['sk_base']
        return {'M-M': -1.5, 'M-X': -1.2, 'X-X': -0.8} # Generic proxy names

    def _get_orbital_basis(self, element_symbol):
        """
        Dynamically returns the orbital basis based on the element.
        Format: [n][l][projection]
        """
        if element_symbol in ["V", "W", "Mo", "Ta", "Ti"]:
            # Transition Metals: 5 d-orbitals using Chinook's strict internal strings
            return ["32xy", "32yz", "32xz", "32ZR", "32XY"] 
        elif element_symbol in ["Te", "Se", "S", "O"]:
            # Chalcogens: n=5, l=0 (s-orbital) and l=1 (3 p-orbitals)
            return ["50", "51x", "51y", "51z"]
        else:
            # Default fallback (Carbon-like)
            return ["20", "21x", "21y", "21z"]

    def export_chinook_dictionary(self, shells=None, onsite_e=0.0, use_soc=False, soc_strength=0.5, tb_mode="Slater-Koster (Rigorous)"):
        if not self.crystal_structure:
            raise ValueError("No structure loaded in DFT engine to export.")

        basis_vectors = self.crystal_structure.lattice.matrix.tolist()
        
        spin_dict = {'bool': False, 'soc': False}
        if use_soc:
            # We are using Native Chinook SOC Generation!
            spin_dict = {'bool': True, 'soc': True, 'lam': {i: soc_strength for i in range(len(self.crystal_structure))}}

        # If user selected Simple Scalar but enabled SOC, we MUST override to SK to prevent the NoneType crash
        if "Scalar" in tb_mode and not use_soc:
            # --- OLD ISOTROPIC LIST MODE (FAST, NO SOC) ---
            atom_orb_indices = {}
            global_idx = 0
            for i, site in enumerate(self.crystal_structure):
                atom_orb_indices[i] = []
                for _ in self._get_orbital_basis(site.species_string):
                    atom_orb_indices[i].append(global_idx)
                    global_idx += 1

            explicit_hopping = []
            for i in range(len(self.crystal_structure)):
                for g_idx in atom_orb_indices[i]:
                    explicit_hopping.append([g_idx, g_idx, 0.0, 0.0, 0.0, complex(onsite_e, 0)])

            sorted_shells = sorted(shells, key=lambda x: x[1]) if shells else []
            for dR_a in [-1, 0, 1]:
                for dR_b in [-1, 0, 1]:
                    for dR_c in [-1, 0, 1]:
                        R_frac = np.array([dR_a, dR_b, dR_c])
                        R_cart = self.crystal_structure.lattice.get_cartesian_coords(R_frac)
                        for i in range(len(self.crystal_structure)):
                            for j in range(len(self.crystal_structure)):
                                if dR_a == 0 and dR_b == 0 and dR_c == 0 and i == j: continue  
                                dist = self.crystal_structure.get_distance(i, j, jimage=[dR_a, dR_b, dR_c])
                                t_val = 0.0
                                for t, r_max in sorted_shells:
                                    if dist <= r_max:
                                        t_val = t
                                        break  
                                if abs(t_val) < 1e-5: continue
                                for idx_a, idx_b in zip(atom_orb_indices[i], atom_orb_indices[j]):
                                    explicit_hopping.append([idx_a, idx_b, R_cart[0], R_cart[1], R_cart[2], complex(t_val, 0)])

            return {
                'type': 'list', 'list': explicit_hopping, 'H': explicit_hopping,
                'a': basis_vectors, 'cutoff': 100.0, 'renorm': 1.0, 'offset': 0.0,
                'tol': 1e-15, 'spin': spin_dict
            }

        # --- NATIVE CHINOOK SLATER-KOSTER MODE ---
        V_dict = {}
        
        # 1. On-site energies (e.g., '32xy' for Atom 0, n=3, l=2)
        for i, site in enumerate(self.crystal_structure):
            for orb in self._get_orbital_basis(site.species_string):
                n, l = orb[0], orb[1]
                if l == '0':
                    # Push s-orbitals deep below the Fermi level (-10.0 eV)
                    V_dict[f"{i}{n}{l}"] = onsite_e - 10.0 
                elif l == '1':
                    # Pull p-orbitals (Te) slightly down so they hybridize nicely
                    V_dict[f"{i}{n}{l}"] = onsite_e - 2.0
                elif l == '2':
                    # Keep d-orbitals (V) right at the Fermi level
                    V_dict[f"{i}{n}{l}"] = onsite_e

        sorted_shells = sorted(shells, key=lambda x: x[1]) if shells else []
        cutoff_max = sorted_shells[-1][1] if sorted_shells else 10.0

        # 2. Map UI Distances to Pairwise SK Bond Strings
        for i, site_i in enumerate(self.crystal_structure):
            for j, site_j in enumerate(self.crystal_structure):
                if i == j:
                    try:
                        # Find distance to the closest adjacent unit cell for t2 self-interaction
                        neighbors = self.crystal_structure.get_neighbors(site_i, cutoff_max)
                        self_dists = [nn.nn_distance for nn in neighbors if nn.index == i and nn.nn_distance > 1e-4]
                        dist = min(self_dists) if self_dists else 100.0
                    except:
                        dist = self.crystal_structure.lattice.a
                else:
                    dist = self.crystal_structure.get_distance(i, j)

                t_val = 0.0
                for t, r_max in sorted_shells:
                    if dist <= r_max:
                        t_val = t
                        break

                if abs(t_val) < 1e-5: continue

                for orb_i in self._get_orbital_basis(site_i.species_string):
                    for orb_j in self._get_orbital_basis(site_j.species_string):
                        n_i, l_i = orb_i[0], orb_i[1]
                        n_j, l_j = orb_j[0], orb_j[1]
                        
                        # Populate Native Slater-Koster Bonds (Sigma, Pi, Delta)
                        V_dict[f"{i}{j}{n_i}{n_j}{l_i}{l_j}S"] = t_val * 1.5
                        if int(l_i) >= 1 and int(l_j) >= 1:
                            V_dict[f"{i}{j}{n_i}{n_j}{l_i}{l_j}P"] = t_val
                        if int(l_i) >= 2 and int(l_j) >= 2:
                            V_dict[f"{i}{j}{n_i}{n_j}{l_i}{l_j}D"] = t_val * 0.8

        print("\n--- CHINOOK DICTIONARY DEBUG ---")
        print(f"Mode: Native Slater-Koster")
        print(f"Total Atoms: {len(self.crystal_structure)}")
        print(f"V_dict Keys Generated: {len(V_dict)}")
        for key, val in V_dict.items(): print(f"  {key} : {val}")
        print("--------------------------------\n")

        return {
            'type': 'SK',               
            'V': V_dict,                
            'a': basis_vectors,                            # <--- Satisfies the Generic Top-Layer Validator
            'avec': np.array(basis_vectors, dtype=float),  # <--- Satisfies the Slater-Koster Math Engine
            'cutoff': float(cutoff_max),
            'renorm': 1.0,
            'offset': 0.0,
            'tol': 1e-15,
            'spin': spin_dict
        }

    def solve_bands(self, k_points, custom_hopping=None, onsite_e=0.0, use_soc=False, soc_strength=0.5, w90_filepath=None, cutoffs=None, tb_mode="Simple Scalar"):
        if build_lib is None or klib is None:
            raise ImportError("Chinook is not installed properly. Cannot calculate bands.")
        if not self.crystal_structure:
            raise ValueError("No structure loaded.")

        if w90_filepath:
            tb_dict = self.export_wannier_dictionary(w90_filepath)
        else:
            shells = []
            if custom_hopping:
                distances = cutoffs if cutoffs else [1.6, 2.6, 3.1, 4.5]
                for i, (key, t_val) in enumerate(custom_hopping.items()):
                    r_max = distances[i] if i < len(distances) else 10.0
                    shells.append((t_val, r_max))

            tb_dict = self.export_chinook_dictionary(
                shells=shells, 
                onsite_e=onsite_e,
                use_soc=use_soc, 
                soc_strength=soc_strength,
                tb_mode=tb_mode # <--- NEW ROUTING PARAMETER
            )

        basis_args = {
            'atoms': list(range(len(self.crystal_structure))),
            'Z': {i: site.specie.number for i, site in enumerate(self.crystal_structure)},
            'pos': [np.array(site.coords, dtype=float) for site in self.crystal_structure],
            'orbs': [self._get_orbital_basis(site.species_string) for site in self.crystal_structure],
            'spin': tb_dict.get('spin', {'bool': False})
        }
        
        import traceback
        
        try:
            print("\n--- CHINOOK BUILD STEPS ---")
            basis = build_lib.gen_basis(basis_args)
            print(f"1. Successfully built basis! (Total Orbitals: {len(basis)})")
            
            tb_model = build_lib.gen_TB(basis, tb_dict) 
            print("2. Successfully built TB Hamiltonian!")
            
            tb_model.Kobj = klib.kpath(k_points) 
            print("3. Successfully built K-path!")
            print("---------------------------\n")
        except Exception as e:
            print("\n!!! FATAL CHINOOK CRASH !!!")
            traceback.print_exc()
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            raise RuntimeError(f"Chinook crashed during initialization: {e}")
        
        # --- DEBUGGING BLOCK 2 START ---
        print("\n--- K-PATH & BASIS DEBUG ---")
        print(f"Basis Positions (Cartesian) passed to Chinook:")
        for p in basis_args['pos']: print(f"  {p}")
        print(f"K-points sample (First 3 points):")
        for kp in k_points[:3]: print(f"  {kp}")
        print("----------------------------\n")
        # --- DEBUGGING BLOCK 2 END ---

        tb_model.solve_H()
        
        eigenvalues = np.real(tb_model.Eband)
        eigenvectors = tb_model.Evec
        
        # Generate safe UI Projection labels directly from the orbital strings
        raw_labels = []
        for site in self.crystal_structure:
            for orb_str in self._get_orbital_basis(site.species_string):
                if orb_str.endswith("0"):
                    orb_name = "s"
                elif orb_str[1] == "1":
                    orb_name = "p" + orb_str[2:]
                elif orb_str[1] == "2":
                  # Prettify the weird Chinook abbreviations
                    if orb_str.endswith("ZR"):
                        orb_name = "dz2"
                    elif orb_str.endswith("XY"):
                        orb_name = "dx2-y2"
                    else:
                        orb_name = "d" + orb_str[2:]
                raw_labels.append(f"{site.species_string}_{orb_name}")

        if use_soc and not w90_filepath: 
            orb_labels = [lbl + "_up" for lbl in raw_labels] + [lbl + "_dn" for lbl in raw_labels]
        else:
            orb_labels = raw_labels

        return eigenvalues, eigenvectors, orb_labels

    
    def get_kpath_template(self, lattice_type="hexagonal", a=3.0, b=3.0):
        if lattice_type == "hexagonal":
            return np.array([[0,0,0], [1/3, 1/3, 0], [0.5, 0, 0], [0,0,0]]), ["$\Gamma$", "K", "M", "$\Gamma$"]
        else:
            return np.array([[0,0,0], [0.5, 0, 0], [0.5, 0.5, 0], [0,0,0]]), ["$\Gamma$", "X", "M", "$\Gamma$"]
    
    def get_auto_kpath(self):
        """Uses PyMatgen to automatically find the standard BZ high-symmetry path."""
        from pymatgen.symmetry.bandstructure import HighSymmKpath
        
        if not self.crystal_structure:
            raise ValueError("No structure loaded to detect k-path.")
            
        kpath = HighSymmKpath(self.crystal_structure)
        kpts_dict = kpath.kpath['kpoints']
        path_segments = kpath.kpath['path']
        
        # Heuristic: If c-axis is large (>10 A), it is a 2D slab/sheet
        is_2d = self.crystal_structure.lattice.c > 10.0
        
        high_sym_pts = []
        labels = []
        
        for segment in path_segments:
            for lbl in segment:
                # Filter out individual 3D points
                if is_2d and abs(kpts_dict[lbl][2]) > 1e-4:
                    continue
                # Prevent immediate duplicate points where segments join
                if not labels or labels[-1] != lbl:
                    labels.append(lbl)
                    # Use the fractional coordinates directly!
                    high_sym_pts.append(kpts_dict[lbl])
                    
        # Format Gamma for matplotlib's LaTeX engine
        labels = ["$\Gamma$" if "Gamma" in lbl or lbl == "\\Gamma" else lbl for lbl in labels]
        
        return np.array(high_sym_pts), labels

    def get_custom_kpath(self, coords_str, labels_str):
        """Parses custom arbitrary path from UI text boxes."""
        try:
            # Parse strings like '0,0,0 ; 0.5,0,0' into float arrays
            pts = [[float(x) for x in pt.split(',')] for pt in coords_str.split(';')]
            lbls = [lbl.strip() for lbl in labels_str.split(';')]
            
            if len(pts) != len(lbls):
                raise ValueError("Number of coordinate points must match number of labels.")
                
            return np.array(pts), lbls
        except Exception as e:
            raise ValueError(f"Custom K-Path format error. Please check your syntax.\n{e}")

    def generate_k_path(self, high_sym_pts, labels, points_per_segment=100):
        k_vecs, k_dist, node_idx = [], [], [0]
        current_dist = 0.0
        for i in range(len(high_sym_pts) - 1):
            start, end = high_sym_pts[i], high_sym_pts[i+1]
            segment = np.linspace(start, end, points_per_segment)[1:] if i > 0 else np.linspace(start, end, points_per_segment)
            k_vecs.append(segment)
            segment_len = np.linalg.norm(end - start)
            k_dist.append(np.linspace(current_dist, current_dist + segment_len, len(segment)))
            current_dist += segment_len
            node_idx.append(len(np.vstack(k_vecs)) - 1)
        return np.vstack(k_vecs), np.concatenate(k_dist), node_idx, labels

    def export_wannier_dictionary(self, w90_filepath):
        """
        Generates the Chinook dictionary required to parse a wannier90_hr.dat file.
        This bypasses the Slater-Koster manual hopping entirely.
        """
        if not self.crystal_structure:
            raise ValueError("Please load a crystal structure from the workspace first to define the basis.")
            
        basis_vectors = self.crystal_structure.lattice.matrix.tolist()
        chinook_atoms = [[site.species_string, site.frac_coords.tolist()] for site in self.crystal_structure]
        
        orbitals = []
        for site in self.crystal_structure:
            orbitals.extend(self._get_orbital_basis(site.species_string))
            
        return {
            'type': 'W90', # This tells Chinook to use the Wannier90 parser
            'a': basis_vectors,
            'atoms': chinook_atoms,
            'orbitals': orbitals,
            'W90_file': w90_filepath # Path to the wannier90_hr.dat file
        }