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
            # --- CRITICAL FIX: DO NOT STANDARDIZE THE LATTICE! ---
            # PyMatgen rotates the crystal, which misaligns the BZ and makes the K-path miss the Dirac cone!
            self.crystal_structure = data
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
        CRITICAL: Must strictly match Wannier90's internal orbital ordering!
        """
        if element_symbol in ["V", "W", "Mo", "Ta", "Ti"]:
            # W90 d-orbital order: dz2, dxz, dyz, dx2-y2, dxy
            return ["32ZR", "32xz", "32yz", "32XY", "32xy"] 
        elif element_symbol in ["Te", "Se", "S", "O"]:
            # W90 p-orbital order: s, pz, px, py
            return ["50", "51z", "51x", "51y"]
        else:
            # Default fallback (Carbon-like)
            return ["20", "21z", "21x", "21y"]

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
            tb_dict, basis_args = self.export_wannier_dictionary(w90_filepath, use_soc)
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
                tb_mode=tb_mode 
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

            # --- DEEP ORBITAL DEBUG ---
            print("\n--- DEEP ORBITAL DEBUG ---")
            print(f"Atomic Numbers (Z) passed: {basis_args['Z']}")
            print(f"Orbitals passed: {basis_args['orbs']}")
            print("Orbitals Chinook actually generated and kept:")
            for b_obj in basis:
                try:
                    # Print the internal dictionary of the Chinook orbital object
                    print(f"  Atom {getattr(b_obj, 'atom', '?')} -> {b_obj.__dict__}")
                except Exception as e:
                    print(f"  {b_obj}")
            print("--------------------------\n")
            
            tb_model = build_lib.gen_TB(basis, tb_dict) 
            print("2. Successfully built TB Hamiltonian!")
            
            tb_model.Kobj = klib.kpath(k_points) 
            print("3. Successfully built K-path!")
            
            # --- NEW: Save the Chinook objects to the engine so the UI can extract them! ---
            self.basis = basis
            self.H_dict = tb_dict
            self.tb_model = tb_model
            
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

    def export_wannier_dictionary(self, w90_filepath, use_soc=False):
        """
        Parses wannier90_hr.dat natively to bypass Chinook's strict/buggy W90 importer.
        Dynamically extracts QE's lattice to reverse basis-vector rotation.
        """
        import os
        if not self.crystal_structure:
            raise ValueError("Please load a crystal structure first.")
            
        print(f"Parsing Wannier90 Hamiltonian natively: {w90_filepath}")
        with open(w90_filepath, 'r') as f:
            lines = f.readlines()
            
        num_wann = int(lines[1].strip())
        nrpts = int(lines[2].strip())
        deg_lines = int(np.ceil(nrpts / 15.0))
        
        # --- 1. EXTRACT WIGNER-SEITZ DEGENERACIES ---
        deg_weights = []
        for line in lines[3:3+deg_lines]:
            deg_weights.extend([int(x) for x in line.split()])
            
        a_mat = self.crystal_structure.lattice.matrix
        
        # --- NATIVE LATTICE ALIGNMENT ---
        # QE defines a different basis (a1, a2) than PyMatgen for hexagonal cells.
        # We find the integer transformation matrix T to perfectly map the W90 cells back to PyMatgen.
        T_mat = np.eye(3)
        work_dir = os.path.dirname(w90_filepath)
        scf_out = os.path.join(work_dir, "scf.out")
        if os.path.exists(scf_out):
            try:
                alat_ang = 1.0
                qe_a = []
                with open(scf_out, 'r') as f:
                    lines_scf = f.readlines()
                for k_idx, line in enumerate(lines_scf):
                    if "lattice parameter (alat)" in line:
                        alat_bohr = float(line.split('=')[1].split()[0])
                        alat_ang = alat_bohr * 0.5291772109
                    elif "crystal axes: (cart. coord. in units of alat)" in line:
                        for m_idx in range(1, 4):
                            coords_str = lines_scf[k_idx+m_idx].split('=')[1].replace('(', '').replace(')', '')
                            parts = coords_str.split()
                            qe_a.append([float(x) * alat_ang for x in parts])
                        break
                if len(qe_a) == 3:
                    A_qe = np.array(qe_a)
                    A_pm_inv = np.linalg.inv(a_mat)
                    T_mat = np.round(np.dot(A_qe, A_pm_inv)).astype(int)
            except Exception as e:
                print(f"Lattice alignment failed: {e}")
        
        flat_atoms = []
        flat_Z = {}
        flat_pos = []
        flat_orbs = []
        
        idx = 0
        for i, site in enumerate(self.crystal_structure):
            for orb in self._get_orbital_basis(site.species_string):
                flat_atoms.append(idx)
                flat_Z[idx] = site.specie.number
                flat_pos.append(np.array(site.coords, dtype=float))
                flat_orbs.append([orb]) 
                idx += 1

        explicit_hopping = []
        data_start = 3 + deg_lines
        
        for line_idx, line in enumerate(lines[data_start:]):
            parts = line.split()
            if len(parts) < 7: continue
            
            # Find the degeneracy weight for this specific R-vector
            r_idx = line_idx // (num_wann * num_wann)
            weight = deg_weights[r_idx]
            
            rx, ry, rz = int(parts[0]), int(parts[1]), int(parts[2])
            i, j = int(parts[3]) - 1, int(parts[4]) - 1 
            t_real, t_imag = float(parts[5]), float(parts[6])
            
            # --- 2. APPLY DEGENERACY WEIGHT ---
            t_ij = complex(t_real, t_imag) / float(weight)
            
            if abs(t_ij) > 1e-6:
                # Map QE cell to PyMatgen cell
                R_qe = np.array([rx, ry, rz])
                R_pm = np.dot(R_qe, T_mat)
                
                # --- 3. FIX INTRACELL PHASE BUG ---
                # Convert Wannier gauge to Bloch gauge by including the precise atomic positions
                tau_i = flat_pos[i]
                tau_j = flat_pos[j]
                
                # True physical hopping vector: R + tau_j - tau_i
                R_cart = np.dot(R_pm, a_mat)
                dR_cart = R_cart + tau_j - tau_i
                
                explicit_hopping.append([i, j, dR_cart[0], dR_cart[1], dR_cart[2], t_ij])
                
        if use_soc:
            spin_dict = {'bool': True, 'soc': True, 'lam': {i: 0.0 for i in range(len(flat_atoms))}}
        else:
            spin_dict = {'bool': False, 'soc': False}
        
        tb_dict = {
            'type': 'list', 
            'list': explicit_hopping, 
            'H': explicit_hopping,
            'a': a_mat.tolist(), 
            'cutoff': 100.0, 
            'renorm': 1.0, 
            'offset': 0.0,
            'tol': 1e-15, 
            'spin': spin_dict
        }
        
        basis_args = {
            'atoms': flat_atoms,
            'Z': flat_Z,
            'pos': flat_pos,
            'orbs': flat_orbs,
            'spin': spin_dict
        }
        
        print(f"Successfully extracted {len(explicit_hopping)} non-zero hopping elements for {num_wann} bands.")
        return tb_dict, basis_args