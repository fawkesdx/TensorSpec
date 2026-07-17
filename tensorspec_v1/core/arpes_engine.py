import numpy as np
from tensorspec.core.workspace import global_workspace

class ARPESGeometryEngine:
    """
    Handles coordinate transformations and vector math for the ARPES 
    Matrix Element Simulator setup.
    """
    def __init__(self):
        # Laboratory Frame: Z = vertical manipulator, Y = forward, X = side way
        self.lab_Z = np.array([0.0, 0.0, 1.0])
        self.lab_Y = np.array([0.0, 1.0, 0.0])
        self.lab_X = np.array([1.0, 0.0, 0.0])
        
        # State Variables (Angles in degrees)
        self.theta = 0.0        # Rotation around Lab Z
        self.beta = 0.0         # Tilt around sample local X
        self.azimuth = 0.0      # Rotation around sample local Y (normal)
        self.slit_angle = 0.0   # Slit rotation around beam axis
        self.beam_angle = 45.0  # Incident angle from lab Y
        self.polarization = 'LH' # LH, LV, CP, CM

        # Initial state orbital symmetry
        self.initial_orbital = 's' # 's', 'px', 'py', 'pz', 'dxy', etc.

        # Simple local crystal basis for Structure Factor demonstration (e.g., 2D square/graphene-like)
        self.crystal_basis_local = [
            np.array([0.0, 0.0, 0.0]),       # Atom 1 at origin
            np.array([1.42, 1.42, 0.0])      # Atom 2 offset in local X-Y
        ]

    def _rot_x(self, angle_deg):
        rad = np.radians(angle_deg)
        return np.array([[1, 0, 0],
                         [0, np.cos(rad), -np.sin(rad)],
                         [0, np.sin(rad), np.cos(rad)]])

    def _rot_y(self, angle_deg):
        rad = np.radians(angle_deg)
        return np.array([[np.cos(rad), 0, np.sin(rad)],
                         [0, 1, 0],
                         [-np.sin(rad), 0, np.cos(rad)]])

    def _rot_z(self, angle_deg):
        rad = np.radians(angle_deg)
        return np.array([[np.cos(rad), -np.sin(rad), 0],
                         [np.sin(rad), np.cos(rad), 0],
                         [0, 0, 1]])

    def get_sample_vectors(self):
        """
        Calculates the sample's normal and in-plane vectors factoring in 
        intrinsic rotations (local axes).
        Returns: tuple of (normal_vec, in_plane_x, in_plane_z)
        """
        # Base state: Normal faces forward (+Y), X is side (+X), Z is up (+Z)
        base_normal = np.array([0.0, 1.0, 0.0])
        base_x = np.array([1.0, 0.0, 0.0])
        base_z = np.array([0.0, 0.0, 1.0])

        # To rotate around local moving axes, we multiply lab matrices in reverse:
        # 1. Azimuth (local Y) -> 2. Beta (local X) -> 3. Theta (Lab Z)
        R = self._rot_z(self.theta) @ self._rot_x(self.beta) @ self._rot_y(self.azimuth)
        
        return R @ base_normal, R @ base_x, R @ base_z
        
    def get_beam_vector(self):
        """
        Calculates incident light vector in the Horizontal (X-Y) plane.
        """
        rad = np.radians(self.beam_angle)
        # Vector points TOWARDS the sample origin from the horizontal plane
        # Assuming analyzer is at +Y, beam comes from a side angle in X-Y
        return np.array([-np.sin(rad), -np.cos(rad), 0.0])
    
    def get_polarization_vectors(self):
        """
        Calculates the electric field (E-vector) direction based on beam trajectory.
        Beam is assumed to be parallel to the ground (X-Y plane).
        """
        beam_dir = self.get_beam_vector()
        
        # LV (Linear Vertical): E-vector points straight up along the manipulator (Z-axis)
        lv_vec = np.array([0.0, 0.0, 1.0])
        
        # LH (Linear Horizontal): E-vector is in the horizontal plane, orthogonal to the beam
        lh_vec = np.cross(lv_vec, beam_dir)
        lh_vec = lh_vec / np.linalg.norm(lh_vec)
        
        if self.polarization == 'LH':
            return [lh_vec]
        elif self.polarization == 'LV':
            return [lv_vec]
        elif self.polarization in ['CP', 'CM']:
            return [lh_vec, lv_vec]
            
        return [lh_vec]

    def get_slit_boundaries(self):
        """
        Calculates the left and right boundary vectors for the 30-degree 
        slit acceptance angle, taking into account the slit rotation.
        Analyzer is fixed on the Lab Y axis.
        """
        analyzer_axis = np.array([0.0, 1.0, 0.0])
        base_slit_dir = np.array([0.0, 0.0, 1.0]) # Default vertical slit
        
        # Rotate slit direction around analyzer axis (Y)
        R_slit = self._rot_y(self.slit_angle)
        slit_dir = R_slit @ base_slit_dir
        
        # The acceptance is +/- 15 degrees from the center (Y axis) along the slit_dir
        # We can use axis-angle rotation around the cross product of Y and slit_dir
        # Cross product of Y and slit_dir gives the axis perpendicular to the slit plane
        rot_axis = np.cross(analyzer_axis, slit_dir)
        
        # Rodrigues' rotation formula for +15 and -15 degrees
        def rodrigues(v, k, theta_deg):
            th = np.radians(theta_deg)
            return v * np.cos(th) + np.cross(k, v) * np.sin(th) + k * np.dot(k, v) * (1 - np.cos(th))
            
        left_bound = rodrigues(analyzer_axis, rot_axis, 15)
        right_bound = rodrigues(analyzer_axis, rot_axis, -15)
        
        return left_bound, right_bound
    
    def load_basis_from_workspace(self, structure_name):
        """
        Attempts to pull a real crystal structure basis from the global workspace 
        to replace the local dummy basis.
        """
        real_basis = global_workspace.pull_crystal_structure(structure_name)
        if real_basis is not None:
            self.crystal_basis_local = real_basis
            return True
        return False
    
    def get_rotated_basis(self):
        """Returns the local crystal basis vectors rotated into the lab frame."""
        # Intrinsic rotations: Azimuth (local Y) -> Beta (local X) -> Theta (Lab Z)
        R = self._rot_z(self.theta) @ self._rot_x(self.beta) @ self._rot_y(self.azimuth)
        return [R @ b for b in self.crystal_basis_local]
    
    def get_orbital_form_factor(self, k_f_lab, R_matrix):
        """
        Calculates the simplified Fourier transform envelope of the initial orbital.
        k_f_lab: Photoelectron momentum vector in the lab frame.
        R_matrix: Total rotation matrix from local sample frame to lab frame.
        """
        # Rotate k_f back into the sample's local frame to evaluate orbital symmetries
        k_f_local = R_matrix.T @ k_f_lab
        
        # Map local lab coordinates to crystal coordinates
        # Local X -> Crystal in-plane x
        # Local Z -> Crystal in-plane y
        # Local Y (Normal) -> Crystal out-of-plane z
        kx = k_f_local[0]
        ky = k_f_local[2]
        kz = k_f_local[1]
        
        # Pure Orbitals
        if self.initial_orbital == 's':
            return 1.0
        elif self.initial_orbital == 'px':
            return kx
        elif self.initial_orbital == 'py':
            return ky
        elif self.initial_orbital == 'pz':
            return kz
        elif self.initial_orbital == 'dxy':
            return kx * ky
        elif self.initial_orbital == 'dxz':
            return kx * kz
        elif self.initial_orbital == 'dyz':
            return ky * kz
        elif self.initial_orbital == 'dx2-y2':
            return kx**2 - ky**2
        elif self.initial_orbital == 'dz2':
            return 3 * kz**2 - 1
            
        # Hybrid Orbitals (Simplified linear combinations)
        elif self.initial_orbital == 'sp2 (x-directed)':
            # e.g., one of the graphene sigma bonds pointing along local x
            return (1.0 / np.sqrt(3)) * 1.0 + (np.sqrt(2) / np.sqrt(3)) * kx
        elif self.initial_orbital == 'sp2 (y-directed)':
            return (1.0 / np.sqrt(3)) * 1.0 + (np.sqrt(2) / np.sqrt(3)) * ky
        elif self.initial_orbital == 'sp3':
            # e.g., tetrahedral bond pointing in [111] direction
            return 0.5 * 1.0 + 0.5 * (kx + ky + kz)
            
        return 1.0 # fallback


    def simulate_matrix_element_map(self, mode="geometric", defl_bounds=(-90, 90), slit_bounds=(-90, 90), res=100):
        """Generates the 2D detector snapshot (Deflection vs Slit) with BZ diffraction."""
        defl_vec = np.linspace(defl_bounds[0], defl_bounds[1], res)
        slit_vec = np.linspace(slit_bounds[0], slit_bounds[1], res)
        Defl, Sl = np.meshgrid(defl_vec, slit_vec, indexing='ij')
        
        defl_rad = np.radians(Defl.ravel())
        sl_rad = np.radians(Sl.ravel())
        
        # 1. k_det from analyzer (Slit = X_det, Deflection = Z_det)
        # Using nested spherical rotations to prevent the diamond clipping
        k_det_x = np.sin(sl_rad)
        k_det_z = np.cos(sl_rad) * np.sin(defl_rad)
        k_det_y = np.cos(sl_rad) * np.cos(defl_rad)
        
        # 2. Rotate detector by Slit Angle around Lab Y
        slit_rot_rad = np.radians(self.slit_angle)
        cg = np.cos(slit_rot_rad)
        sg = np.sin(slit_rot_rad)
        
        kx_lab = cg * k_det_x + sg * k_det_z
        ky_lab = k_det_y
        kz_lab = -sg * k_det_x + cg * k_det_z
        k_lab = np.column_stack([kx_lab, ky_lab, kz_lab])
        
        # 3. Geometric Dipole Transition: |A dot k_lab|^2
        A_vec = self.get_polarization_vectors()[0]
        geom_intensity = np.abs(np.dot(k_lab, A_vec))**2
        
        # 4. Apply R_inv to get k_local
        th_rad = np.radians(self.theta)
        cos_th = np.cos(-th_rad)
        sin_th = np.sin(-th_rad)
        kx_1 = k_lab[:,0]*cos_th - k_lab[:,1]*sin_th
        ky_1 = k_lab[:,0]*sin_th + k_lab[:,1]*cos_th
        kz_1 = k_lab[:,2]
        
        beta_rad = np.radians(-self.beta)
        cb = np.cos(beta_rad)
        sb = np.sin(beta_rad)
        kx_2 = kx_1
        ky_2 = ky_1*cb - kz_1*sb
        kz_2 = ky_1*sb + kz_1*cb
        
        az_rad = np.radians(-self.azimuth)
        ca = np.cos(az_rad)
        sa = np.sin(az_rad)
        kx_local = kx_2*ca + kz_2*sa
        ky_local = ky_2
        kz_local = -kx_2*sa + kz_2*ca
        
        # 5. Map to True Crystal Axes
        cx = kx_local
        cy = kz_local
        cz = ky_local
        
        # 6. Evaluate Orbital Symmetries (Atomic Form Factor)
        orb = self.initial_orbital
        if orb == 's': ff = np.ones_like(cx)
        elif orb == 'px': ff = cx
        elif orb == 'py': ff = cy
        elif orb == 'pz': ff = cz
        elif orb == 'dxy': ff = cx * cy
        elif orb == 'dxz': ff = cx * cz
        elif orb == 'dyz': ff = cy * cz
        elif orb == 'dx2-y2': ff = cx**2 - cy**2
        elif orb == 'dz2': ff = 3 * cz**2 - 1
        elif orb == 'sp2 (x-directed)': ff = (1.0 / np.sqrt(3)) + (np.sqrt(2) / np.sqrt(3)) * cx
        elif orb == 'sp2 (y-directed)': ff = (1.0 / np.sqrt(3)) + (np.sqrt(2) / np.sqrt(3)) * cy
        elif orb == 'sp3': ff = 0.5 + 0.5 * (cx + cy + cz)
        else: ff = np.ones_like(cx)

        # 7. Calculate Full Crystal Lattice Summation (Brillouin Zone Diffraction)
        # Using a generic hexagonal lattice (a=2.46 A) scaled to momentum space
        a1 = np.array([2.46, 0.0])
        a2 = np.array([-1.23, 2.13039])
        
        # Sum over a 15x15 unit cell grid to create sharp BZ diffraction spots
        N_cells = 7
        n_grid = np.arange(-N_cells, N_cells + 1)
        N1, N2 = np.meshgrid(n_grid, n_grid)
        
        # Multiply by typical ARPES k-vector magnitude (e.g. 21.2 eV photon)
        # to map angles correctly to momentum scaling
        k_radius = 2.1 
        cx_scaled = cx * k_radius
        cz_scaled = cz * k_radius

        R_x = N1.ravel() * a1[0] + N2.ravel() * a2[0]
        R_z = N1.ravel() * a1[1] + N2.ravel() * a2[1]
        
        # Phase sum: sum_R exp(i k_local . R)
        phase_sum = np.zeros_like(cx_scaled, dtype=complex)
        for rx, rz in zip(R_x, R_z):
            phase_sum += np.exp(1j * (cx_scaled * rx + cz_scaled * rz))
            
        lattice_sf = np.abs(phase_sum)**2
        if np.max(lattice_sf) > 0:
            lattice_sf = lattice_sf / np.max(lattice_sf)
            
        # 8. Apply UI Mode Check
        if mode == "geometric":
            intensity = geom_intensity
        elif mode == "structure_factor":
            intensity = np.abs(ff)**2
        elif mode == "full_crystal":
            intensity = geom_intensity * np.abs(ff)**2 * lattice_sf
        else:
            intensity = geom_intensity * np.abs(ff)**2
            
        if np.max(intensity) > 1e-6:
            intensity = intensity / np.max(intensity)
            
        return defl_vec, slit_vec, intensity.reshape((res, res))

    def simulate_isoenergy_contour(self, target_energy, mode='full', temp=10.0, broadening=0.1, noise=0.05):
        """Calculates a 2D Constant Energy map (kx vs ky) with exact Hamiltonian phases."""
        if not hasattr(self, 'active_bands') or not self.active_bands.get('is_2d'):
            return None
            
        eigenvalues = self.active_bands['eigenvalues']
        eigenvectors = self.active_bands.get('eigenvectors', None)
        kx = self.active_bands['kx']
        ky = self.active_bands['ky']
        grid_shape = self.active_bands['grid_shape']
        k_vecs = self.active_bands['k_vecs']
        
        spectral_weight = np.zeros(eigenvalues.shape[0])
        
        # Get the sample rotation matrix from the GUI sliders
        R_sample = self._rot_z(self.theta) @ self._rot_x(self.beta) @ self._rot_y(self.azimuth)

        # Call the correct physics array and grab the primary polarization vector
        A_vec = self.get_polarization_vectors()[0]
        
        for b in range(eigenvalues.shape[1]):
            E_k = eigenvalues[:, b]
            band_sw = broadening / ((target_energy - E_k)**2 + broadening**2)
            
            if mode == 'full':
                sf_intensity = np.ones_like(band_sw)
                if eigenvectors is not None:
                    if len(eigenvectors.shape) == 3:
                        c_A = eigenvectors[:, 0, b]
                        c_B = eigenvectors[:, 1, b]
                    else:
                        c_A = eigenvectors[:, b, 0]
                        c_B = eigenvectors[:, b, 1]
                    
                    # PHYSICS FIX: The TB engine already used exact spatial vectors (delta)
                    # to build the Hamiltonian. Therefore, c_A and c_B ALREADY contain 
                    # the full spatial phase shift.
                    sf_intensity = np.abs(c_A + c_B)**2
                
                # Dynamic geometric mapping relative to the beam direction
                
                
                kz_approx = 4.5 
                k_norm = np.sqrt(k_vecs[:, 0]**2 + k_vecs[:, 1]**2 + kz_approx**2)
                
                # Map 2D TB grid to Lab Frame
                kx_local = k_vecs[:, 0] / k_norm
                ky_local = kz_approx / k_norm
                kz_local = k_vecs[:, 1] / k_norm
                
                k_directions_local = np.column_stack([kx_local, ky_local, kz_local])
                k_directions_lab = (R_sample @ k_directions_local.T).T
                
                geom_intensity = np.abs(np.dot(k_directions_lab, A_vec))**2
                
                # Map local lab coordinates to crystal coordinates
                cx = kx_local
                cy = kz_local
                cz = ky_local
                
                orb = self.initial_orbital
                if orb == 's': ff = np.ones_like(cx)
                elif orb == 'px': ff = cx
                elif orb == 'py': ff = cy
                elif orb == 'pz': ff = cz
                elif orb == 'dxy': ff = cx * cy
                elif orb == 'dxz': ff = cx * cz
                elif orb == 'dyz': ff = cy * cz
                elif orb == 'dx2-y2': ff = cx**2 - cy**2
                elif orb == 'dz2': ff = 3 * cz**2 - 1
                elif orb == 'sp2 (x-directed)': ff = (1.0 / np.sqrt(3)) + (np.sqrt(2) / np.sqrt(3)) * cx
                elif orb == 'sp2 (y-directed)': ff = (1.0 / np.sqrt(3)) + (np.sqrt(2) / np.sqrt(3)) * cy
                elif orb == 'sp3': ff = 0.5 + 0.5 * (cx + cy + cz)
                else: ff = np.ones_like(cx)
                
                geom_intensity *= np.abs(ff)**2
                
                if np.max(geom_intensity) > 0:
                    geom_intensity = geom_intensity / np.max(geom_intensity)
                    
                band_sw *= sf_intensity * (geom_intensity + 0.1)
                
            spectral_weight += band_sw
            
        kB = 8.617333262e-5
        T = max(temp, 1e-3)
        exponent = np.clip(target_energy / (kB * T), -700, 700)
        fd = 1.0 / (np.exp(exponent) + 1.0)
        
        intensity = spectral_weight * fd

        if noise > 0:
            intensity += np.abs(np.random.normal(0, noise * np.max(intensity), size=intensity.shape))
            
        return kx, ky, intensity.reshape(grid_shape)
    
    def load_bands_from_workspace(self, name):
        """Pulls the calculated E(k) band structure from the global workspace."""
        data = global_workspace.pull_band_structure(name)
        if data:
            self.active_bands = data
            return True
        return False

    def simulate_arpes_cut(self, E_range=(-5, 2), E_steps=400, mode='full', temp=10.0, broadening=0.1, noise=0.05):
        """
        Simulates the E vs k ARPES spectrum using the Spectral Function A(k, w) 
        and Fermi-Dirac distribution.
        
        mode: 'bands_only' (Ignores Matrix Element) or 'full' (Applies Matrix Element)
        """
        if not hasattr(self, 'active_bands'): 
            return None
            
        k_dist = self.active_bands['k_dist']
        k_vecs = self.active_bands['k_vecs']
        eigenvalues = self.active_bands['eigenvalues'] # Shape: [N_k, N_bands]
        
        E_axis = np.linspace(E_range[0], E_range[1], E_steps)
        intensity_map = np.zeros((E_steps, len(k_dist)))
        
        # 1. Fermi-Dirac Distribution (Numerically Stable)
        kB = 8.617333262e-5 # eV/K
        T = max(temp, 1e-3) # Prevent div by zero
        # Clip exponent to [-700, 700] to prevent np.exp() overflow
        exponent = np.clip(E_axis / (kB * T), -700, 700)
        fermi_dirac = 1.0 / (np.exp(exponent) + 1.0)
        
        pol_vectors = self.get_polarization_vectors()
        R_sample = self._rot_z(self.theta) @ self._rot_x(self.beta) @ self._rot_y(self.azimuth)
        
        for idx, k_f in enumerate(k_vecs):
            # 2. Matrix Element |M|^2
            if mode == 'full':
                # k_f is in crystal coordinates: [kx, ky, kz_normal]
                # Map to local manipulator frame: Local X = kx, Local Y = kz_normal, Local Z = ky
                k_local = np.array([k_f[0], k_f[2], k_f[1]])
                
                # Transform to Lab Frame for the dipole dot product
                k_lab = R_sample @ k_local
                k_lab_norm = k_lab / (np.linalg.norm(k_lab) + 1e-9)
                
                # Calculate form factor using the lab vector
                orbital_envelope = self.get_orbital_form_factor(k_lab_norm, R_sample)
                
                M_squared = 0.0
                for p_vec in pol_vectors:
                    M_squared += np.abs(np.dot(p_vec, k_lab_norm) * orbital_envelope)**2
            else:
                M_squared = 1.0 # Ignore Matrix Elements (Bands Only)
                
            # 3. Spectral Function A(k, w)
            A_kw = np.zeros_like(E_axis)
            for b in range(eigenvalues.shape[1]):
                Ek = eigenvalues[idx, b]
                # Lorentzian broadening based on self-energy imaginary part (Sigma'')
                A_kw += (1.0 / np.pi) * (broadening / ((E_axis - Ek)**2 + broadening**2))
                
            # Combine Physics
            intensity_map[:, idx] = M_squared * A_kw * fermi_dirac
            
        # 4. Experimental Noise
        if noise > 0:
            max_I = np.max(intensity_map) if np.max(intensity_map) > 0 else 1.0
            noise_array = np.random.normal(0, noise * max_I, intensity_map.shape)
            intensity_map += noise_array
            intensity_map = np.clip(intensity_map, 0, None) # No negative intensities
            
        return k_dist, E_axis, intensity_map, self.active_bands['node_idx'], self.active_bands['labels']