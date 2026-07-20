import numpy as np
import collections
import collections.abc
# Monkey-patch to fix Python 3.10+ compatibility for Chinook
collections.Iterable = collections.abc.Iterable

try:
    import chinook.build_lib as build_lib
    from chinook.ARPES_lib import experiment
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

    def build_model(self, tb_dict):
        """
        Safely loads the tight-binding model directly from the workspace and 
        extracts the physical on-site energy shift without modifying the model.
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

        # 2. Extract the true on-site energy (Universal Material Finder)
        self.fermi_shift = 0.0
        h_dict = tb_dict.get('H_dict', {})
        if h_dict.get('type') == 'list' and 'list' in h_dict:
            onsite_vals = []
            max_val = -9999.0
            max_idx = -1
            print(">> DEBUG: Scanning all R=0 on-site energies...")
            
            for hop in h_dict['list']:
                try:
                    # Find all on-site energies (i == j, R = 0)
                    if hop[0] == hop[1] and abs(hop[2]) < 1e-4 and abs(hop[3]) < 1e-4 and abs(hop[4]) < 1e-4:
                        val = np.real(hop[5])
                        onsite_vals.append(val)
                        print(f"   Orbital {hop[0]}: {val:.4f} eV")
                        if val > max_val:
                            max_val = val
                            max_idx = hop[0]
                except Exception:
                    continue
                    
            if onsite_vals:
                self.fermi_shift = -max_val
                print(f">> DEBUG: Auto-detected Fermi Anchor: {self.fermi_shift:.4f} eV")
                print(f">> DEBUG: The script grabbed Orbital {max_idx} as the maximum!")

    def run_simulation(self, experiment_kwargs):
        """
        Calculates matrix elements over a shifted energy domain to center the physics.
        """
        # 1. Extract k-space bounds (kx, ky, E)
        kb = experiment_kwargs['k_bounds']
        num_x, num_y, num_e = kb['X'][2], kb['Y'][2], kb['E'][2]

        if not CHINOOK_AVAILABLE or self.tb_model is None:
            dummy_intensity = np.random.rand(num_x, num_y, num_e)
            return {'intensity_broadened': dummy_intensity}

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
        
        # --- THE TRUE ARPES GEOMETRY MAP ---
        # Lab Y is the Analyzer / Sample Normal. Lab X and Z form the sample plane.
        # Chinook assumes Sample Z is the Normal. We map Sample -> Lab to align them.
        R_base = np.array([
            [1,  0,  0],  # Sample X -> Lab X (Side)
            [0,  0,  1],  # Sample Z -> Lab Y (Forward/Analyzer)
            [0, -1,  0]   # Sample Y -> Lab -Z (Vertical, preserves handedness)
        ])
        
        # Build Manipulator Rotation Stack (Lab Frame)
        t_rad, a_rad, tilt_rad = np.radians(theta_deg), np.radians(azi_deg), np.radians(tilt_deg)
        R_z = np.array([[np.cos(t_rad), -np.sin(t_rad), 0], [np.sin(t_rad), np.cos(t_rad), 0], [0, 0, 1]]) # Theta
        R_y = np.array([[np.cos(a_rad), 0, np.sin(a_rad)], [0, 1, 0], [-np.sin(a_rad), 0, np.cos(a_rad)]]) # Azimuth
        R_x = np.array([[1, 0, 0], [0, np.cos(tilt_rad), -np.sin(tilt_rad)], [0, np.sin(tilt_rad), np.cos(tilt_rad)]]) # Tilt
        
        # Total rotation: Orient Sample to Lab, then apply motors
        R_total = R_z @ R_y @ R_x @ R_base
        R_inv = np.linalg.inv(R_total)
        
        # Transform Lab polarization into the Sample's local frame for Chinook
        A_sample = R_inv @ A_lab

        # 4. Domain & ME setup
        is_bare = "Off" in me_mode
        is_full = "Full" in me_mode

        # --- 5. THE FAIL-SAFE ENERGY SHIFT ---
        # If the UI asks for 0.0 eV, and the Dirac cone is at -1.81 eV, we offset the domain 
        # so Chinook evaluates the matrix at exactly -1.81 eV without touching the matrix itself!
        domain = {
            'X': [kb['X'][0], kb['X'][1], num_x],
            'Y': [kb['Y'][0], kb['Y'][1], num_y],
            'E': [kb['E'][0] - self.fermi_shift, kb['E'][1] - self.fermi_shift, num_e]
        }

        arpes_dict = {
            'cube': domain,
            'hv': hv,
            'W': W,
            'V0': V0,
            'T': T,
            'pol': A_sample,
            'ME': is_full,
            'SE': ['constant', se_width],
            'resolution': {'E': res_e, 'k': res_k}
        }

        # 6. Execute Simulation
        try:
            exp = experiment(self.tb_model, arpes_dict)
            exp.datacube()
            output_maps = np.real(exp.spectral())
            
            if output_maps.ndim == 4 and output_maps.shape[0] == 2:
                intensity_3d = output_maps[1]
            elif output_maps.ndim == 1:
                intensity_3d = output_maps.reshape((num_x, num_y, num_e), order='F')
            else:
                intensity_3d = output_maps
            
            # Geometric Matrix Element Toggles
            if not is_bare and not is_full:
                kx_arr = np.linspace(kb['X'][0], kb['X'][1], num_x)
                ky_arr = np.linspace(kb['Y'][0], kb['Y'][1], num_y)
                KX, KY = np.meshgrid(kx_arr, ky_arr, indexing='ij')
                dipole_factor = np.abs(A_sample[0]*KX + A_sample[1]*KY)**2
                intensity_3d = intensity_3d * dipole_factor[:, :, np.newaxis]
                
            # --- Apply Lab Frame Rotation ---
            import scipy.ndimage
            # Because Azimuth rotates around Lab Y (the Normal), it perfectly spins the in-plane detector image!
            effective_rot = experiment_kwargs.get('manip_azimuth', 0.0) + experiment_kwargs.get('slit_angle', 0.0)
            if effective_rot != 0.0:
                intensity_3d = scipy.ndimage.rotate(intensity_3d, angle=effective_rot, axes=(0, 1), reshape=False, order=1)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Chinook Calculation Error: {e}")

        return {'intensity_broadened': intensity_3d}