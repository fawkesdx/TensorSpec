"""
Option A: Phenomenological Three-Step Model Matrix Element Engine
Strictly pure Python logic. Zero GUI/Plotting imports.

Implements the free-electron final state approximation based on:
Simon Moser, "An experimentalist's guide to the matrix element in angle resolved photoemission"
Journal of Electron Spectroscopy and Related Phenomena 214 (2017) 29-52.
"""

import numpy as np

class ThreeStepWrapper:
    """
    Simulates ARPES intensity using the three-step phenomenological approach.
    Generates a 3D block of intensity I(kx, ky, E) by mapping the tight-binding 
    bands and modulating them with the geometric polarization transition probabilities.
    """
    def __init__(self):
        # Conversion constant: hbar^2 / 2m in eV * Angstrom^2
        self.hbar2_2m = 3.80998 
        
    def run_simulation(self, crystal_data, experiment_kwargs):
        """
        Executes the three-step calculation.
        
        Args:
            crystal_data (dict): Contains 'eigenvalues', 'k_vecs', 'eigenvectors' from DFT.
            experiment_kwargs (dict): hv, work function, inner potential, polarization, bounds.
            
        Returns:
            dict: Simulated 3D intensity map matching the ARPES Suite's expected format.
        """
        # 1. Parse Experimental Parameters
        hv = experiment_kwargs.get('photon_energy', 21.2)
        work_func = experiment_kwargs.get('work_function', 4.5)
        inner_pot = experiment_kwargs.get('inner_potential', 15.0)
        pol_mode = experiment_kwargs.get('polarization', "Linear Horizontal")
        me_mode = experiment_kwargs.get('matrix_element_mode', 'Full')
        
        # NEW: Parse Geometry
        alpha_deg = experiment_kwargs.get('incidence_angle', 55.0)
        theta_deg = experiment_kwargs.get('manip_theta', 0.0)
        azi_deg = experiment_kwargs.get('manip_azimuth', 0.0)
        tilt_deg = experiment_kwargs.get('manip_tilt', 0.0)

        # 2. Parse Domain Bounds (Target 3D Mesh)
        bounds = experiment_kwargs['k_bounds']
        kx_min, kx_max, kx_steps = bounds['X']
        ky_min, ky_max, ky_steps = bounds['Y']
        e_min, e_max, e_steps = bounds['E']

        kx_ax = np.linspace(kx_min, kx_max, int(kx_steps))
        ky_ax = np.linspace(ky_min, ky_max, int(ky_steps))
        e_ax = np.linspace(e_min, e_max, int(e_steps))
        
        # Create 3D phase space: indexing='ij' ensures (kx, ky, E) shape
        KX, KY, E = np.meshgrid(kx_ax, ky_ax, e_ax, indexing='ij')

        # 3. Setup Polarization Vector & Frame Transformation
        inc_rad = np.radians(alpha_deg)
        lin_ang = np.radians(experiment_kwargs.get('lin_pol_angle', 45.0))
        
        if "Horizontal" in pol_mode: 
            eps_lab = np.array([np.cos(inc_rad), -np.sin(inc_rad), 0.0])
        elif "Vertical" in pol_mode: 
            eps_lab = np.array([0.0, 0.0, 1.0])
        elif "Arbitrary" in pol_mode:
            eps_lab = np.cos(lin_ang)*np.array([np.cos(inc_rad), -np.sin(inc_rad), 0.0]) + np.sin(lin_ang)*np.array([0.0, 0.0, 1.0])
        elif "Right" in pol_mode: 
            eps_lab = (np.array([np.cos(inc_rad), -np.sin(inc_rad), 0.0]) + 1j*np.array([0.0, 0.0, 1.0])) / np.sqrt(2)
        else: 
            eps_lab = (np.array([np.cos(inc_rad), -np.sin(inc_rad), 0.0]) - 1j*np.array([0.0, 0.0, 1.0])) / np.sqrt(2)

        # Rotation matrices (same logic as the schematic)
        t_rad, a_rad, tilt_rad = np.radians(theta_deg), np.radians(azi_deg), np.radians(tilt_deg)
        R_z = np.array([[np.cos(t_rad), -np.sin(t_rad), 0], [np.sin(t_rad), np.cos(t_rad), 0], [0, 0, 1]])
        R_y = np.array([[np.cos(a_rad), 0, np.sin(a_rad)], [0, 1, 0], [-np.sin(a_rad), 0, np.cos(a_rad)]])
        R_x = np.array([[1, 0, 0], [0, np.cos(tilt_rad), -np.sin(tilt_rad)], [0, np.sin(tilt_rad), np.cos(tilt_rad)]])
        
        R_total = R_z @ R_y @ R_x
        
        # INVERSE ROTATION: Transform Lab polarization into the Sample's local frame
        R_inv = np.linalg.inv(R_total)
        eps_sample = R_inv @ eps_lab

        # 4. Photoelectron Kinematics (Moser Eq. 21 - 26)
        # Calculate kinetic energy of the emitted photoelectron in vacuum
        E_kin = hv + E - work_func
        E_kin[E_kin < 0] = 0.0  # Zero out unphysical (negative) kinetic energies
        
        # Calculate perpendicular momentum inside the solid (kz), absorbing the inner potential
        kz_inside = np.sqrt(np.maximum((E_kin + inner_pot) / self.hbar2_2m - KX**2 - KY**2, 0.0))
        
        # 5. Polarization Dipole Term I (Now evaluated natively in the sample frame!)
        # Because we rotated eps into the sample frame, KX, KY, and kz map perfectly to px, pz, py.
        # (Assuming your sample mounts with Sample Z pointing out-of-plane, which maps to Lab Y at rest)
        dot_product = eps_sample[0] * KX + eps_sample[1] * kz_inside + eps_sample[2] * KY
        matrix_element = np.abs(dot_product)**2

        # 6. Orbital Term II: (Moser Eq. 41 & 47)
        if me_mode == "Full Matrix Elements":
            # In a full implementation, we calculate the Fourier transform of the specific 
            # atomic orbitals (e.g., d_xy, p_z). As a generic fallback for arbitrary tight-binding 
            # bands without explicitly passed spherical harmonics, we approximate the radial 
            # cross-section falloff (Cooper minimum envelope).
            orbital_form_factor = np.exp(- (KX**2 + KY**2) / 2.0) 
            matrix_element *= orbital_form_factor
        elif me_mode == "Bare Spectral Function (ME Off)":
            matrix_element = np.ones_like(KX)

        # 7. Map Tight Binding Bands into the Spectral Function
        intensity_broadened = np.zeros_like(KX)
        bands = crystal_data.get('eigenvalues', [])
        k_vecs = crystal_data.get('k_vecs', [])
        
        # NEW: Safety check for 1D vs 2D data
        if crystal_data.get('is_2d') is False or len(k_vecs) == 0:
            raise ValueError("The ARPES simulator requires a 2D k-mesh (TB_Bands_2D). You loaded a 1D band path.")
            
        eigenvectors = crystal_data.get('eigenvectors', [])
        orbital_positions = crystal_data.get('orbital_positions', [])
        
        sigma_e = 0.05  # Energy broadening (eV)
        sigma_k = 0.05  # Momentum broadening (1/A)
        
        from scipy.interpolate import griddata

        # Convert to numpy arrays
        bands = np.array(bands)
        k_vecs = np.array(k_vecs)
        eigenvectors = np.array(eigenvectors)
        
        if bands.size > 0 and k_vecs.size > 0:
            num_bands = bands.shape[1]
            
            # Extract the 2D surface of the ARPES grid and the input k-points
            KX_2d = KX[:, :, 0]
            KY_2d = KY[:, :, 0]
            pts = k_vecs[:, :2]
            
            for b_idx in range(num_bands):
                print(f">> DEBUG: Three-Step Engine | Fast Vectorizing Band {b_idx+1}/{num_bands}...")
                
                # 1. Map the band energy directly onto the ARPES grid
                E_band_2d = griddata(pts, bands[:, b_idx], (KX_2d, KY_2d), method='nearest')
                E_band_3d = E_band_2d[:, :, np.newaxis] # Expand to 3D for energy axis
                
                # 2. Single-shot vectorization of the entire energy broadening!
                spectral_weight = np.exp(-((E - E_band_3d)**2) / (2 * sigma_e**2))
                
                # --- SUBLATTICE INTERFERENCE (Moser Eq. 41) ---
                if eigenvectors.size > 0 and len(orbital_positions) > 0 and "ME Off" not in me_mode:
                    # Eigenvectors from DFT are shaped [k_points, basis_orbitals, bands]
                    c_j_all = eigenvectors[:, :, b_idx]
                        
                    phase_sum = np.zeros_like(KX, dtype=complex)
                    num_atoms = len(orbital_positions)
                    orbs_per_atom = c_j_all.shape[1] // num_atoms
                            
                    for atom_idx, R_j in enumerate(orbital_positions):
                        # Group and sum all orbital coefficients belonging to this specific atom
                        atom_c_j = np.sum(c_j_all[:, atom_idx*orbs_per_atom : (atom_idx+1)*orbs_per_atom], axis=1)
                        
                        # Interpolate this atom's total complex weight onto the ARPES grid
                        c_j_interp = griddata(pts, atom_c_j, (KX_2d, KY_2d), method='nearest')
                                
                        # Apply the geometric spatial phase for this atom's coordinates
                        phase = np.exp(-1j * (KX * R_j[0] + KY * R_j[1] + kz_inside * R_j[2]))
                        phase_sum += c_j_interp[:, :, np.newaxis] * phase
                                
                    spectral_weight *= np.abs(phase_sum)**2
                    
                intensity_broadened += spectral_weight
            
            # Modulate the entire phase space by the macroscopic polarization (Term I)
            intensity_broadened *= matrix_element
            
        else:
            # Fallback: Generate a dummy parabolic dispersion if no bands are successfully routed
            dummy_disp = - (KX**2 + KY**2) / (2 * 0.5) 
            spectral_weight = np.exp(-((E - dummy_disp)**2) / (2 * sigma_e**2))
            intensity_broadened = spectral_weight * matrix_element

        # --- Apply Lab Frame Rotation ---
        import scipy.ndimage
        slit_angle = experiment_kwargs.get('slit_angle', 0.0)
        effective_rot = azi_deg + slit_angle
        if effective_rot != 0.0:
            intensity_broadened = scipy.ndimage.rotate(intensity_broadened, angle=-effective_rot, axes=(0, 1), reshape=False, order=1)

        return {
            'intensity_broadened': intensity_broadened,
            'k_axes': (kx_ax, ky_ax),
            'e_axis': e_ax
        }