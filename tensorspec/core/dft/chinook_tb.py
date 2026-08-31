import os
import time
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
        self.A_qe = None
        self._w90_parse_cache = {}
        self._source_cache_key = None
        self._cached_tb_dict = None
        self._cached_basis_args = None
        self._tb_build_cache_key = None
        self._tb_build_cache = None  # (basis, tb_model)
        self._tb_debug = os.environ.get("TENSORSPEC_TB_DEBUG", "").lower() in (
            "1",
            "true",
            "yes",
        )

        # Materials Database for Slater-Koster hopping parameters
        self.materials_db = {
            'C': { # PyMatgen reduces C2 down to C
                'orbitals': ['C', 'C'],
                'sk_base': {
                    'C-C_nn': -2.7,      # Standard graphene t1 is ~ -2.7 eV
                    'C-C_nnn': -0.2,     # t2 is ~ -0.2 eV
                    'C-C_third': 0.0
                },
                # Carbon: p-orbitals form the Dirac cone at E=0, s-orbitals are deep
                'onsite': {'0': -8.0, '1': 0.0} 
            },
            'WTe2': {
                'orbitals': ['W', 'Te'],
                'sk_base': {
                    'W-W': -1.5, 'W-Te': -1.2, 'Te-Te_in_plane': -0.8,
                    'Te-Te_interlayer': -0.3 
                },
                'onsite': {'0': -10.0, '1': -2.0, '2': 0.0}
            },
            'TaIrTe4': {
                'orbitals': ['Ta', 'Ir', 'Te'],
                'sk_base': {'M-M': -1.3, 'M-Te': -1.1, 'Te-Te': -0.7}, # Proxy Slater-Koster bonds
                'onsite': {'0': -10.0, '1': -2.0, '2': 0.0} 
            },
            'MoTe2': {
                'orbitals': ['Mo', 'Te'],
                'sk_base': {'Mo-Mo': -1.4, 'Mo-Te': -1.2, 'Te-Te': -0.8},
                'onsite': {'0': -10.0, '1': -1.5, '2': 0.0}
            },
            'VTe2': {
                'orbitals': ['V', 'Te'],
                'sk_base': {'V-V': -1.3, 'V-Te': -1.1, 'Te-Te': -0.7},
                # VTe2: V d-bands at Fermi level, Te p-bands shifted down
                'onsite': {'0': -10.0, '1': -2.0, '2': 0.0}
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
        from pymatgen.core import Element
        
        # Strip out any oxidation states or numbers (e.g., convert "Te2-" to "Te")
        clean_symbol = ''.join([c for c in element_symbol if c.isalpha()])
        el = Element(clean_symbol)
        
        # Determine the principal quantum number (n) based on the periodic table row
        n = el.row
        
        # Wannier90 strictly orders s;p;d projections as: s, pz, px, py, dz2, dxz, dyz, dx2-y2, dxy
        s_orbs = [f"{n}0"]
        p_orbs = [f"{n}1z", f"{n}1x", f"{n}1y"]
        # d-orbitals are always (n-1) in the valence shell (e.g., 6s -> 5d)
        d_orbs = [f"{n-1}2ZR", f"{n-1}2xz", f"{n-1}2yz", f"{n-1}2XY", f"{n-1}2xy"]
        
        # Mirror the exact projection logic from qe_generator.py
        if el.is_transition_metal or el.number > 30:
            return s_orbs + p_orbs + d_orbs
        else:
            return s_orbs + p_orbs

    def export_chinook_dictionary(self, shells=None, onsite_e=0.0, use_soc=False, soc_strength=0.5, tb_mode="Slater-Koster (Rigorous)", orbital_shifts=None):
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
        
        formula = self.crystal_structure.composition.reduced_formula
        mat_props = self.materials_db.get(formula, self.materials_db['VTe2'])
        
        # Priority: UI Overrides -> Database -> Hardcoded defaults
        onsite_db = orbital_shifts if orbital_shifts else mat_props.get('onsite', {'0': -10.0, '1': -2.0, '2': 0.0})
        
        # 1. On-site energies dynamically mapped per angular momentum (l)
        for i, site in enumerate(self.crystal_structure):
            for orb in self._get_orbital_basis(site.species_string):
                n, l = orb[0], orb[1]
                
                # Fetch the specific orbital energy from the database, apply the global UI shift (onsite_e)
                base_energy = onsite_db.get(l, 0.0)
                V_dict[f"{i}{n}{l}"] = onsite_e + base_energy

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

    def _structure_fingerprint(self):
        s = self.crystal_structure
        return (
            s.composition.reduced_formula,
            tuple(tuple(row) for row in s.lattice.matrix.tolist()),
        )

    def _w90_source_key(self, w90_filepath, use_soc, onsite_e):
        path = os.path.abspath(w90_filepath)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        return (path, mtime, use_soc, float(onsite_e), self._structure_fingerprint())

    def _manual_source_key(
        self,
        custom_hopping,
        onsite_e,
        use_soc,
        soc_strength,
        cutoffs,
        tb_mode,
        orbital_shifts,
    ):
        hop_items = tuple(sorted((custom_hopping or {}).items()))
        shift_items = tuple(sorted((orbital_shifts or {}).items()))
        return (
            self._structure_fingerprint(),
            float(onsite_e),
            use_soc,
            float(soc_strength),
            tb_mode,
            hop_items,
            tuple(cutoffs or ()),
            shift_items,
        )

    def _get_wannier_tb(self, w90_filepath, use_soc, onsite_e):
        key = self._w90_source_key(w90_filepath, use_soc, onsite_e)
        if key in self._w90_parse_cache:
            return self._w90_parse_cache[key], key, True
        tb_dict, basis_args = self.export_wannier_dictionary(
            w90_filepath, use_soc, onsite_e
        )
        self._w90_parse_cache[key] = (tb_dict, basis_args)
        return (tb_dict, basis_args), key, False

    def _build_chinook_tb(self, tb_dict, basis_args, w90_filepath, use_soc):
        if self._tb_debug:
            print("\n--- CHINOOK BUILD STEPS ---")
        basis = build_lib.gen_basis(basis_args)

        if w90_filepath and use_soc:
            orbs_list = basis if isinstance(basis, list) else getattr(basis, "orbitals", [])
            for idx, b_obj in enumerate(orbs_list):
                b_obj.spin = 1.0 if idx < (len(orbs_list) // 2) else -1.0

        if self._tb_debug:
            print(
                f"1. Successfully built basis! "
                f"(Total Orbitals: {len(getattr(basis, 'orbitals', basis))})"
            )
            print("\n--- DEEP ORBITAL DEBUG ---")
            print(f"Atomic Numbers (Z) passed: {basis_args['Z']}")
            print(f"Orbitals passed: {basis_args['orbs']}")
            print("Orbitals Chinook actually generated and kept:")
            for b_obj in basis:
                try:
                    print(f"  Atom {getattr(b_obj, 'atom', '?')} -> {b_obj.__dict__}")
                except Exception:
                    print(f"  {b_obj}")
            print("--------------------------\n")

        tb_model = build_lib.gen_TB(basis, tb_dict)
        if self._tb_debug:
            print("2. Successfully built TB Hamiltonian!")
            print("---------------------------\n")

        self.basis = basis
        self.H_dict = tb_dict
        self.tb_model = tb_model
        return basis, tb_model

    def _diagonalize_tb(self, tb_model, k_points):
        """Chinook-only diag. Do not call GrizzlyME from DFT suite — that API still moves."""
        tb_model.Kobj = klib.kpath(k_points)
        if self._tb_debug:
            print("3. Successfully built K-path!")
            print("\n--- K-PATH & BASIS DEBUG ---")
            print("K-points sample (First 3 points):")
            for kp in k_points[:3]:
                print(f"  {kp}")
            print("----------------------------\n")

        tb_model.solve_H()
        return "chinook"

    def solve_bands(self, k_points, custom_hopping=None, onsite_e=0.0, use_soc=False, soc_strength=0.5, w90_filepath=None, cutoffs=None, tb_mode="Simple Scalar", orbital_shifts=None):
        if build_lib is None or klib is None:
            raise ImportError("Chinook is not installed properly. Cannot calculate bands.")
        if not self.crystal_structure:
            raise ValueError("No structure loaded.")

        t_total = time.perf_counter()
        parse_hit = build_hit = False

        if not w90_filepath:
            self.A_qe = None

        if w90_filepath:
            (tb_dict, basis_args), source_key, parse_hit = self._get_wannier_tb(
                w90_filepath, use_soc, onsite_e
            )
        else:
            source_key = self._manual_source_key(
                custom_hopping,
                onsite_e,
                use_soc,
                soc_strength,
                cutoffs,
                tb_mode,
                orbital_shifts,
            )
            if source_key == self._source_cache_key and self._cached_tb_dict is not None:
                tb_dict, basis_args = self._cached_tb_dict, self._cached_basis_args
                parse_hit = True
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
                    tb_mode=tb_mode,
                    orbital_shifts=orbital_shifts,
                )
                basis_args = {
                    "atoms": list(range(len(self.crystal_structure))),
                    "Z": {
                        i: site.specie.number
                        for i, site in enumerate(self.crystal_structure)
                    },
                    "pos": [
                        np.array(site.coords, dtype=float)
                        for site in self.crystal_structure
                    ],
                    "orbs": [
                        self._get_orbital_basis(site.species_string)
                        for site in self.crystal_structure
                    ],
                    "spin": tb_dict.get("spin", {"bool": False}),
                }
                self._source_cache_key = source_key
                self._cached_tb_dict = tb_dict
                self._cached_basis_args = basis_args

        t_parse = time.perf_counter()

        build_key = source_key
        if build_key == self._tb_build_cache_key and self._tb_build_cache is not None:
            basis, tb_model = self._tb_build_cache
            build_hit = True
        else:
            try:
                basis, tb_model = self._build_chinook_tb(
                    tb_dict, basis_args, w90_filepath, use_soc
                )
                self._tb_build_cache_key = build_key
                self._tb_build_cache = (basis, tb_model)
            except Exception as e:
                print("\n!!! FATAL CHINOOK CRASH !!!")
                traceback.print_exc()
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                raise RuntimeError(f"Chinook crashed during initialization: {e}") from e

        t_build = time.perf_counter()
        backend = self._diagonalize_tb(tb_model, k_points)
        t_diag = time.perf_counter()

        parse_s = t_parse - t_total
        build_s = t_build - t_parse
        diag_s = t_diag - t_build
        total_s = t_diag - t_total
        cache_bits = []
        if parse_hit:
            cache_bits.append("parse")
        if build_hit:
            cache_bits.append("build")
        cache_tag = f" cache={'+'.join(cache_bits)}" if cache_bits else ""
        print(
            f"DFT bands ({backend}): {len(k_points)} k-pts, "
            f"parse={parse_s:.2f}s build={build_s:.2f}s diag={diag_s:.2f}s "
            f"total={total_s:.2f}s{cache_tag}"
        )

        eigenvalues = np.real(tb_model.Eband)
        eigenvectors = tb_model.Evec

        raw_labels = []
        for site in self.crystal_structure:
            for orb_str in self._get_orbital_basis(site.species_string):
                if orb_str.endswith("0"):
                    orb_name = "s"
                elif orb_str[1] == "1":
                    orb_name = "p" + orb_str[2:]
                elif orb_str[1] == "2":
                    if orb_str.endswith("ZR"):
                        orb_name = "dz2"
                    elif orb_str.endswith("XY"):
                        orb_name = "dx2-y2"
                    else:
                        orb_name = "d" + orb_str[2:]
                raw_labels.append(f"{site.species_string}_{orb_name}")

        if use_soc and not w90_filepath:
            orb_labels = [lbl + "_up" for lbl in raw_labels] + [
                lbl + "_dn" for lbl in raw_labels
            ]
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

    def export_wannier_dictionary(self, w90_filepath, use_soc=False, onsite_e=0.0):
        """
        Parses wannier90_hr.dat natively to bypass Chinook's strict/buggy W90 importer.
        Dynamically extracts QE's lattice to reverse basis-vector rotation.
        """
        if not self.crystal_structure:
            raise ValueError("Please load a crystal structure first.")

        t0 = time.perf_counter()
        print(f"Parsing Wannier90 Hamiltonian natively: {w90_filepath}")

        # Header only (comment / num_wann / nrpts / degeneracy weights)
        with open(w90_filepath, "r") as f:
            f.readline()
            num_wann = int(f.readline().strip())
            nrpts = int(f.readline().strip())
            deg_lines = int(np.ceil(nrpts / 15.0))
            deg_weights = []
            for _ in range(deg_lines):
                deg_weights.extend(int(x) for x in f.readline().split())
        deg_weights = np.asarray(deg_weights, dtype=np.float64)
        if len(deg_weights) < nrpts:
            raise ValueError(
                f"wannier90_hr.dat degeneracy block short: got {len(deg_weights)}, need {nrpts}"
            )

        data_start = 3 + deg_lines
        # Columns: R_x R_y R_z i j Re(H) Im(H)  — bulk of file; vectorized
        hop = np.loadtxt(w90_filepath, skiprows=data_start)
        if hop.ndim == 1:
            hop = hop.reshape(1, -1)
        if hop.shape[1] < 7:
            raise ValueError(f"Unexpected wannier90_hr.dat columns: {hop.shape}")

        a_mat = self.crystal_structure.lattice.matrix

        # --- NATIVE LATTICE ALIGNMENT & FERMI SHIFT ---
        # QE defines a different basis (a1, a2) than PyMatgen for hexagonal cells.
        # We find the integer transformation matrix T to perfectly map the W90 cells back to PyMatgen.
        T_mat = np.eye(3)
        ef = 0.0  # NEW: Initialize Fermi Level
        work_dir = os.path.dirname(w90_filepath)
        scf_out = os.path.join(work_dir, "scf.out")
        A_qe_found = False
        if os.path.exists(scf_out):
            try:
                alat_ang = 1.0
                qe_a = []
                with open(scf_out, "r") as f:
                    lines_scf = f.readlines()
                for k_idx, line in enumerate(lines_scf):
                    # --- NEW: Extract Fermi Energy from QE Log ---
                    if "the Fermi energy is" in line:
                        ef = float(line.split("is")[1].split("ev")[0].strip())
                    elif "highest occupied, lowest unoccupied" in line:
                        parts = line.split(":")[-1].split()
                        ef = (float(parts[0]) + float(parts[1])) / 2.0

                    elif "lattice parameter (alat)" in line:
                        alat_bohr = float(line.split("=")[1].split()[0])
                        alat_ang = alat_bohr * 0.5291772109
                    elif "crystal axes: (cart. coord. in units of alat)" in line:
                        for m_idx in range(1, 4):
                            coords_str = (
                                lines_scf[k_idx + m_idx]
                                .split("=")[1]
                                .replace("(", "")
                                .replace(")", "")
                            )
                            parts = coords_str.split()
                            qe_a.append([float(x) * alat_ang for x in parts])
                        break
                if len(qe_a) == 3:
                    A_qe = np.array(qe_a)
                    self.A_qe = A_qe  # SAVE TO SELF FOR K-PATH CALCULATION
                    A_pm_inv = np.linalg.inv(a_mat)
                    T_mat = np.round(np.dot(A_qe, A_pm_inv)).astype(int)
                    A_qe_found = True
            except Exception as e:
                print(f"Lattice alignment failed: {e}")

        # --- FALLBACK: Use wannier90.wout if scf.out is missing ---
        wout_files = [
            f for f in os.listdir(work_dir) if f.endswith(".wout") or f.endswith(".out")
        ]
        wout = (
            os.path.join(work_dir, wout_files[0])
            if wout_files
            else os.path.join(work_dir, "wannier90.wout")
        )

        if (not os.path.exists(scf_out) or not A_qe_found) and os.path.exists(wout):
            try:
                qe_a = []
                with open(wout, "r") as f:
                    lines_wout = f.readlines()
                for i, line in enumerate(lines_wout):
                    if "Lattice Vectors" in line:
                        for j in range(2, 5):
                            parts = lines_wout[i + j].replace("|", "").split()
                            if len(parts) >= 4 and parts[0].startswith("a_"):
                                qe_a.append(
                                    [float(parts[1]), float(parts[2]), float(parts[3])]
                                )
                        break
                if len(qe_a) == 3:
                    A_qe = np.array(qe_a)
                    self.A_qe = A_qe  # SAVE TO SELF FOR K-PATH CALCULATION
                    A_pm_inv = np.linalg.inv(a_mat)
                    T_mat = np.round(np.dot(A_qe, A_pm_inv)).astype(int)
            except Exception as e:
                print(f"wout lattice alignment failed: {e}")

        flat_atoms = []
        flat_Z = {}
        flat_pos = []
        flat_orbs = []

        # --- EXTRACT EXACT WANNIER CENTERS FROM wout ---
        centers = []
        if os.path.exists(wout):
            try:
                with open(wout, "r") as f:
                    lines_wout = f.readlines()
                for line in lines_wout:
                    if "WF centre and spread" in line:
                        parts = line.split("(")[1].split(")")[0].split(",")
                        centers.append([float(x) for x in parts])
            except Exception as e:
                print(f"Failed to read Wannier centers: {e}")

        if len(centers) > 0:
            print(f"Found {len(centers)} exact Wannier centers from .wout!")
            for i in range(num_wann):
                flat_atoms.append(i)
                flat_Z[i] = 1  # Dummy element
                flat_pos.append(np.array(centers[i % len(centers)]))
                flat_orbs.append(["10"])
        else:
            print("Warning: No Wannier centers found. Falling back to CIF projection guess.")
            idx = 0
            for i, site in enumerate(self.crystal_structure):
                for orb in self._get_orbital_basis(site.species_string):
                    flat_atoms.append(idx)
                    flat_Z[idx] = site.specie.number
                    flat_pos.append(np.array(site.coords, dtype=float))
                    flat_orbs.append([orb])
                    idx += 1

        # --- Vectorized hoppings (same physics as former line loop) ---
        R_qe = hop[:, 0:3]
        i_idx = hop[:, 3].astype(np.int64) - 1
        j_idx = hop[:, 4].astype(np.int64) - 1
        t_real = hop[:, 5].astype(np.float64, copy=True)
        t_imag = hop[:, 6].astype(np.float64)

        n_block = num_wann * num_wann
        r_idx = np.arange(hop.shape[0], dtype=np.int64) // n_block
        if r_idx.size and int(r_idx[-1]) >= len(deg_weights):
            raise ValueError(
                f"Hopping rows exceed degeneracy table "
                f"(r_idx max={int(r_idx[-1])}, nrpts={len(deg_weights)})"
            )
        weights = deg_weights[r_idx]

        onsite_mask = (
            (R_qe[:, 0] == 0)
            & (R_qe[:, 1] == 0)
            & (R_qe[:, 2] == 0)
            & (i_idx == j_idx)
        )
        t_real[onsite_mask] = t_real[onsite_mask] - ef + float(onsite_e)

        t_ij = (t_real + 1j * t_imag) / weights
        keep = np.abs(t_ij) > 1e-6

        R_qe_k = R_qe[keep]
        i_k = i_idx[keep]
        j_k = j_idx[keep]
        t_k = t_ij[keep]

        T_mat = np.asarray(T_mat, dtype=np.float64)
        a_mat = np.asarray(a_mat, dtype=np.float64)
        R_pm = R_qe_k @ T_mat
        R_cart = R_pm @ a_mat

        pos = np.asarray(flat_pos, dtype=np.float64)
        n_pos = len(pos)
        if n_pos == 0:
            raise ValueError("No orbital positions for Wannier → Cartesian hop map.")
        tau_i = pos[i_k % n_pos]
        tau_j = pos[j_k % n_pos]
        dR = R_cart + tau_j - tau_i

        # Chinook list H expects Python rows [i, j, dx, dy, dz, complex]
        explicit_hopping = [
            [int(i_k[n]), int(j_k[n]), float(dR[n, 0]), float(dR[n, 1]), float(dR[n, 2]), complex(t_k[n])]
            for n in range(t_k.shape[0])
        ]

        if use_soc:
            # --- CRITICAL FIX: BYPASS CHINOOK'S SOC DUPLICATOR ---
            # Wannier90 already generated the full spinor matrix.
            # If we tell Chinook 'soc': True, it will duplicate indices and crash!
            # Instead, we manually double the basis arrays here and hide SOC from Chinook.
            n_orbs = len(flat_atoms)
            flat_atoms = flat_atoms + [a + n_orbs for a in flat_atoms]

            new_Z = {}
            for idx in range(len(flat_atoms)):
                new_Z[idx] = flat_Z[idx % n_orbs]
            flat_Z = new_Z

            flat_pos = flat_pos + flat_pos
            flat_orbs = flat_orbs + flat_orbs

            spin_dict = {"bool": False, "soc": False}
        else:
            spin_dict = {"bool": False, "soc": False}

        tb_dict = {
            "type": "list",
            "list": explicit_hopping,
            "H": explicit_hopping,
            "a": a_mat.tolist(),
            "cutoff": 100.0,
            "renorm": 1.0,
            "offset": 0.0,
            "tol": 1e-15,
            "spin": spin_dict,
        }

        basis_args = {
            "atoms": flat_atoms,
            "Z": flat_Z,
            "pos": flat_pos,
            "orbs": flat_orbs,
            "spin": spin_dict,
        }

        print(
            f"Successfully extracted {len(explicit_hopping)} non-zero hopping elements "
            f"for {num_wann} bands in {time.perf_counter() - t0:.2f}s "
            f"(read {hop.shape[0]} rows)."
        )
        return tb_dict, basis_args