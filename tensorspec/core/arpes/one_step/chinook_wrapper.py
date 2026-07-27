import numpy as np
import collections
import collections.abc
# Monkey-patch to fix Python 3.10+ compatibility for Chinook
collections.Iterable = collections.abc.Iterable

try:
    import chinook.build_lib as build_lib
    from chinook.ARPES_lib import experiment
    import chinook.electron_configs as econ  
    import chinook.radint_lib as radint_lib  # <-- New import to patch the cutoff bug
    
    # --- PATCH 1: CHINOOK'S MISSING ATOMIC DATABASE & NOISY PRINTS ---
    import sys
    import io
    
    original_Z_eff = econ.Z_eff
    _dummy_stream = io.StringIO() # A permanent black hole for Chinook's print statements
    
    def patched_Z_eff(Z, orb):
        # 1. Temporarily hijack stdout to trap Chinook's hardcoded error messages
        old_stdout = sys.stdout
        sys.stdout = _dummy_stream
        try:
            result = original_Z_eff(Z, orb)
        finally:
            # Restore normal printing immediately after
            sys.stdout = old_stdout
            
        if result is None:
            # Complete Slater's Rule effective charges for heavy elements (Rows 5 & 6)
            z_eff_db = {
                # --- Row 5 ---
                37: 2.20, 38: 2.85, # 5s block (Rb, Sr)
                39: 3.00, 40: 3.65, 41: 4.30, 42: 4.95, 43: 5.60, # 4d block (Y-Tc)
                44: 6.25, 45: 6.90, 46: 7.55, 47: 8.20, 48: 8.85, # 4d block (Ru-Cd)
                49: 5.00, 50: 5.65, 51: 6.30, 52: 6.95, 53: 7.60, 54: 8.25, # 5p block (In-Xe)
                
                # --- Row 6 ---
                55: 2.20, 56: 2.85, # 6s block (Cs, Ba)
                **{z: 3.00 + (z-57)*0.35 for z in range(57, 72)}, # 4f block (La-Lu)
                72: 3.65, 73: 4.30, 74: 4.95, 75: 5.60, 76: 6.25, # 5d block (Hf-Os)
                77: 6.90, 78: 7.55, 79: 8.20, 80: 8.85,           # 5d block (Ir-Hg)
                81: 5.00, 82: 5.65, 83: 6.30, 84: 6.95, 85: 7.60, 86: 8.25  # 6p block (Tl-Rn)
            }
            # Fallback to 4.5 only for extreme actinides (Z > 86)
            return z_eff_db.get(Z, 4.5)  
        return result
        
    econ.Z_eff = patched_Z_eff
    
    # --- PATCH 2: CHINOOK'S UNBOUND LOCAL ERROR BUG ---
    original_find_cutoff = radint_lib.find_cutoff
    
    def safe_find_cutoff(integrand):
        try:
            return original_find_cutoff(integrand)
        except UnboundLocalError:
            # The integrand is a mathematical function object, not an array.
            # If the integral evaluates to effectively zero, Chinook forgets to assign a cutoff.
            # We return a generous physical radial distance (30.0 Bohr radii) to safely bypass.
            return 30.0
            
    radint_lib.find_cutoff = safe_find_cutoff
    # -----------------------------------------------

    CHINOOK_AVAILABLE = True
except Exception as e:
    import traceback
    print("\n" + "="*50)
    print("CHINOOK IMPORT FAILED WITH ERROR:")
    traceback.print_exc()
    print("="*50 + "\n")
    CHINOOK_AVAILABLE = False

class ChinookWrapper:
    """
    The backend bridge to Chinook's ARPES matrix element calculator.
    Translates experimental beamline parameters into physical polarization vectors.
    """
    def __init__(self):
        self.tb_model = None
        self.fermi_shift = 0.0
        self.B_matrix = None  # <-- Add this line

    def build_model(self, tb_dict):
        """
        Safely loads the tight-binding model directly from the workspace and 
        extracts the physical on-site energy shift and reciprocal lattice.
        """
        if not CHINOOK_AVAILABLE:
            print("Chinook not installed. Running in Dummy Mode.")
            return
            
        # 1. Safely extract the pre-built model without dangerously rebuilding it
        if tb_dict.get('tb_model') is not None:
            self.tb_model = tb_dict['tb_model']
        else:
            basis = tb_dict.get('basis', tb_dict.get('Basis', tb_dict.get('chinook_basis')))
            h_dict = tb_dict.get('H_dict', tb_dict.get('hamiltonian_dict', tb_dict.get('hamiltonian')))
            if basis is None or h_dict is None:
                raise ValueError("CRITICAL: Workspace missing Tight-Binding params.")
            self.tb_model = build_lib.gen_TB(basis, h_dict)

        # 2. Extract the true on-site energy
        self.fermi_shift = tb_dict.get('fermi_energy', 0.0)

        # --- ADD THESE DEBUG LINES ---
        print("\n[LOG 4 - CHINOOK WRAPPER]")
        print(f"tb_dict['fermi_energy'] extracted: {tb_dict.get('fermi_energy', 'MISSING')}")
        print(f"Calculated self.fermi_shift applied to axis: {self.fermi_shift}")
        # -----------------------------
        
        # 3. Safely extract Reciprocal Lattice Vectors (B_matrix)
        # Chinook TB_model does not store these natively, so we pull them from the workspace dict
        if 'structure' in tb_dict:
            self.B_matrix = tb_dict['structure'].lattice.reciprocal_lattice.matrix
        elif 'recip_matrix' in tb_dict:
            self.B_matrix = tb_dict['recip_matrix']
        elif 'avec' in tb_dict:
            A_matrix = np.array(tb_dict['avec'])
            self.B_matrix = 2 * np.pi * np.linalg.inv(A_matrix).T
        elif hasattr(self.tb_model, 'Kobj') and self.tb_model.Kobj is not None and hasattr(self.tb_model.Kobj, 'avec'):
            A_matrix = np.array(self.tb_model.Kobj.avec)
            self.B_matrix = 2 * np.pi * np.linalg.inv(A_matrix).T
        else:
            print("WARNING: No lattice vectors found in Tight-Binding dictionary. Falling back to Simple Cubic.")
            self.B_matrix = 2 * np.pi * np.eye(3)

    def run_simulation(self, experiment_kwargs):
        """
        Calculates matrix elements over a transformed detector frame to strictly align
        kx with the analyzer slit and ky with the deflector scan direction.
        """
        # 1. Extract and safely copy k-space bounds (kx, ky, E) so we don't mutate the UI
        kb = {key: list(val) for key, val in experiment_kwargs['k_bounds'].items()}
        
        # Override resolutions for 1D slit measurements without deflection
        if kb['X'][0] == kb['X'][1]: kb['X'][2] = 1
        if kb['Y'][0] == kb['Y'][1]: kb['Y'][2] = 1
            
        num_x, num_y, num_e = int(kb['X'][2]), int(kb['Y'][2]), int(kb['E'][2])

        if not CHINOOK_AVAILABLE or self.tb_model is None:
            return {'intensity_broadened': np.random.rand(num_x, num_y, num_e)}

        # 2. Extract Experimental Parameters
        hv = experiment_kwargs.get('photon_energy', 21.2)
        W = experiment_kwargs.get('work_function', 4.5)
        V0 = experiment_kwargs.get('inner_potential', 15.0)
        T = experiment_kwargs.get('temperature', 10.0)
        inc_angle = experiment_kwargs.get('incidence_angle', 55.0)
        pol_str = experiment_kwargs.get('polarization', "Linear Horizontal")
        me_mode = experiment_kwargs.get('matrix_element_mode', "Full Matrix Elements")
        se_width = experiment_kwargs.get('se_width', 0.01)
        res_e = experiment_kwargs.get('res_E', 0.02)
        res_k = experiment_kwargs.get('res_k', 0.02)

        # 3. Calculate Polarization Vector (A)
        inc_rad = np.radians(inc_angle)
        lin_ang = np.radians(experiment_kwargs.get('lin_pol_angle', 45.0))
        
        if "Horizontal" in pol_str: 
            A_lab = np.array([np.cos(inc_rad), -np.sin(inc_rad), 0.0])
        elif "Vertical" in pol_str: 
            A_lab = np.array([0.0, 0.0, 1.0])
        elif "Arbitrary" in pol_str:
            A_lab = np.cos(lin_ang)*np.array([np.cos(inc_rad), -np.sin(inc_rad), 0.0]) + np.sin(lin_ang)*np.array([0.0, 0.0, 1.0])
        elif "Right" in pol_str: 
            A_lab = (np.array([np.cos(inc_rad), -np.sin(inc_rad), 0.0]) + 1j*np.array([0.0, 0.0, 1.0])) / np.sqrt(2)
        else: 
            A_lab = (np.array([np.cos(inc_rad), -np.sin(inc_rad), 0.0]) - 1j*np.array([0.0, 0.0, 1.0])) / np.sqrt(2)

        # Parse Manipulator Geometry from GUI
        theta_deg = experiment_kwargs.get('manip_theta', 0.0)
        azi_deg = experiment_kwargs.get('manip_azimuth', 0.0)
        tilt_deg = experiment_kwargs.get('manip_tilt', 0.0)
        
        R_base = np.array([
            [1,  0,  0],  # Sample X -> Lab X (Side)
            [0,  0,  1],  # Sample Z -> Lab Y (Forward/Analyzer)
            [0, -1,  0]   # Sample Y -> Lab -Z (Vertical)
        ])
        
        t_rad, a_rad, tilt_rad = np.radians(theta_deg), np.radians(azi_deg), np.radians(tilt_deg)
        R_z = np.array([[np.cos(t_rad), -np.sin(t_rad), 0], [np.sin(t_rad), np.cos(t_rad), 0], [0, 0, 1]])
        R_y = np.array([[np.cos(a_rad), 0, np.sin(a_rad)], [0, 1, 0], [-np.sin(a_rad), 0, np.cos(a_rad)]])
        R_x = np.array([[1, 0, 0], [0, np.cos(tilt_rad), -np.sin(tilt_rad)], [0, np.sin(tilt_rad), np.cos(tilt_rad)]])
        
        R_total = R_z @ R_y @ R_x @ R_base
        R_inv = np.linalg.inv(R_total)
        A_sample = R_inv @ A_lab

        # --- 4. EXACT DETECTOR MESH MAPPING ---
        # Generate the pixel grid strictly in the detector frame
        kx_arr = np.linspace(kb['X'][0], kb['X'][1], num_x)
        ky_arr = np.linspace(kb['Y'][0], kb['Y'][1], num_y)
        K_SLIT, K_DEFL = np.meshgrid(kx_arr, ky_arr, indexing='ij')
        
        k_slit_flat = K_SLIT.flatten(order='C')
        k_defl_flat = K_DEFL.flatten(order='C')
        # Apply the Analyzer Slit Angle mathematically in the Lab Frame
        slit_rad = np.radians(experiment_kwargs.get('slit_angle', 0.0))
        k_lab_x = k_slit_flat * np.cos(slit_rad) - k_defl_flat * np.sin(slit_rad)
        k_lab_z = k_slit_flat * np.sin(slit_rad) + k_defl_flat * np.cos(slit_rad)
        
        # --- FIX: EXACT 3D KINEMATICS & SURFACE REFRACTION ---
        E_kin = max(hv - W, 0.1)
        k_vac_sq = 0.262465 * E_kin
        
        # 1. Calculate momentum along the analyzer axis (Lab Y) in vacuum
        k_lab_y_sq = k_vac_sq - (k_lab_x**2 + k_lab_z**2)
        k_lab_y = np.sqrt(np.clip(k_lab_y_sq, 0.0, None))
        
        K_LAB_VAC = np.vstack([k_lab_x, k_lab_y, k_lab_z])
        
        # 2. Rotate from Lab Frame onto the Macroscopic Sample Surface
        K_SAMPLE = R_inv @ K_LAB_VAC 
        
        # 3. Apply the Inner Potential (V0) acceleration ONLY normal to the surface (Sample Z)
        K_SAMPLE[2] = np.sqrt(K_SAMPLE[2]**2 + 0.262465 * V0)
        # -----------------------------------------------------
        
        # --- 4.5 hkl CLEAVAGE PLANE PROJECTION ---
        hkl = experiment_kwargs.get('hkl', (0, 0, 1)) # Default to (001) surface
        
        from tensorspec.core.crystallography import CrystalEngine
        
        # We NO LONGER ask self.tb_model for its non-existent 'avec' attribute.
        # We use the securely extracted B_matrix from build_model()
        
        # --- FIX: Lock the 0-degree azimuthal baseline to the pure Cartesian Y-axis ---
        Z_surf, Y_surf = CrystalEngine.get_hkl_surface_frame(hkl, self.B_matrix, azimuthal_ref=np.array([0.0, 1.0, 0.0]))
        
        # Construct X_surf to complete the Right-Handed basis (cross product of Y and Z)
        X_surf = np.cross(Y_surf, Z_surf)
        
        # Rotation matrix mapping the (kx, ky) cleavage plane to the 3D Bulk Cartesian frame
        R_hkl_to_bulk = np.column_stack((X_surf, Y_surf, Z_surf))
        
        # Map to Bulk Cartesian Frame for Hamiltonian diagonalization
        K_BULK = R_hkl_to_bulk @ K_SAMPLE
        
        # --- FIX 1: Rotate Polarization Vector into Bulk Frame ---
        # The atomic orbitals are defined in the bulk frame, so the incident light pulse 
        # must be rotated to match, otherwise dipole selection rules will be physically misaligned!
        A_bulk = R_hkl_to_bulk @ A_sample
        
        is_bare = "Off" in me_mode
        is_full = "Full" in me_mode

        # Feed the experiment dictionary
        arpes_dict = {
            'cube': kb,  
            'ang': 0.0,  
            'E': np.linspace(kb['E'][0], kb['E'][1], num_e),
            'hv': hv,
            'W': W,
            'V0': V0,
            'T': T,
            'pol': A_bulk,  # <-- Pass the corrected 3D polarization vector
            'ME': is_full,
            'SE': ['constant', se_width],
            'resolution': {'E': res_e, 'k': res_k}
        }

        # 5. Execute Simulation
        try:
            exp = experiment(self.tb_model, arpes_dict)
            exp.ME = is_full
            
            # THE CACHE KILLER
            if hasattr(exp.TB, 'H'): del exp.TB.H
            if hasattr(exp.TB, 'Eband'): del exp.TB.Eband
            if hasattr(exp.TB, 'evec'): del exp.TB.evec
            
            # Initialize basis rotation and custom K_BULK mesh
            exp.basis = exp.rot_basis()
            class CustomMesh:
                def __init__(self, k):
                    self.kpts = k
            exp.TB.Kobj = CustomMesh(K_BULK.T)
            exp.k = K_BULK.T 
            
            # Generate matrices & diagonalize
            exp.val, exp.vec = exp.TB.solve_H()
            
            # --- NEW: Shift Eigenvalues Natively ---
            # We shift the resulting eigenvalues down to E=0.
            # This guarantees that Chinook's native Fermi-Dirac distributions
            # and our Bare Spectral Function bypass work flawlessly.
            exp.val = exp.val - self.fermi_shift
            # ---------------------------------------
            
            # =====================================================================
            # --- SUPERCHARGE BYPASS FOR BARE SPECTRAL FUNCTION ---
            # =====================================================================
            if is_bare:
                gamma = se_width / 2.0
                
                # --- FIX: Grab the energy axis directly from our dictionary ---
                energy_axis = arpes_dict['E']
                
                # Vectorized Lorentzian Broadening
                # Broadcasting: (Nk, Nbands, 1) against (1, 1, Ne) -> (Nk, Nbands, Ne)
                diff = exp.val[:, :, np.newaxis] - energy_axis[np.newaxis, np.newaxis, :]
                spectral_weight = (gamma / np.pi) / (diff**2 + gamma**2)
                
                # Sum over all bands to get total intensity
                intensity_flat = np.sum(spectral_weight, axis=1)
                
                # --- NEW: Apply Fermi-Dirac Distribution ---
                # Boltzmann constant in eV/K
                kb = 8.617333262e-5 
                
                # Clip the exponent to safely prevent np.exp() overflow at high binding energies
                # T is already parsed from experiment_kwargs earlier in the function
                exponent = np.clip(energy_axis / (kb * T), -100.0, 100.0)
                fermi_dirac = 1.0 / (np.exp(exponent) + 1.0)
                
                # Multiply the flat intensity matrix (Nk, Ne) by the 1D Fermi-Dirac array (Ne,)
                intensity_flat = intensity_flat * fermi_dirac
                
                # Reshape directly back to the 3D detector grid
                intensity_3d = intensity_flat.reshape((num_x, num_y, num_e), order='C')
                
                return {'intensity_broadened': intensity_3d}
            # =====================================================================
            
            # --- FULL MATRIX ELEMENTS ROUTE (Standard Chinook Pipeline) ---
            exp.Eb = exp.val.flatten()
            exp.Ev = exp.vec  
            exp.X = np.zeros((num_y, num_x))
            exp.Y = np.zeros((num_y, num_x))
            
            # --- FIX: ALIGN FINAL STATE PLANE WAVE TO THE BULK FRAME ---
            # We must construct the outgoing photoelectron in the exact same 
            # coordinate system as the initial state orbitals and light polarization.
            k_bulk_para = np.sqrt(K_BULK[0]**2 + K_BULK[1]**2)
            k_bulk_norm = np.sqrt(K_BULK[0]**2 + K_BULK[1]**2 + K_BULK[2]**2)
            
            # Prevent division by zero precisely at the Gamma point
            k_bulk_norm[k_bulk_norm == 0] = 1e-10 
            
            exp.ph = np.arctan2(K_BULK[1], K_BULK[0])
            exp.th = np.arcsin(np.clip(k_bulk_para / k_bulk_norm, -1.0, 1.0))
            # -----------------------------------------------------------
            
            exp.diagonalize = lambda *args, **kwargs: None
            exp.datacube()
            
            output_maps = np.real(exp.spectral())
            
            # --- RESTORED: CORRECT ARRAY UNPACKING ---
            expected_size = num_x * num_y * num_e
            
            # 1. Sum the spin dimensions safely if they exist
            if output_maps.size == 2 * expected_size:
                output_maps = np.sum(output_maps.reshape((2, expected_size)), axis=0)
            elif output_maps.ndim == 4 and output_maps.shape[0] == 2:
                output_maps = np.sum(output_maps, axis=0)
                
            # 2. Chinook natively stores data as (Nk, Ne). Reshape directly to (Nx, Ny, Ne)!
            intensity_3d = output_maps.reshape((num_x, num_y, num_e), order='C')
            # -----------------------------------------
                
            if not is_full: # Dipole Only
                # --- FIX: Include the Z-component and use the aligned Bulk Frame ---
                dipole_dot = A_bulk[0]*K_BULK[0] + A_bulk[1]*K_BULK[1] + A_bulk[2]*K_BULK[2]
                dipole_factor = np.abs(dipole_dot)**2
                dipole_factor = dipole_factor.reshape(num_x, num_y, order='C')
                intensity_3d = intensity_3d * dipole_factor[:, :, np.newaxis]
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Chinook Calculation Error: {e}")

        return {'intensity_broadened': intensity_3d}