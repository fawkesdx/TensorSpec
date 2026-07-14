import numpy as np

class ARPESKinematics:
    """
    Standalone module for converting experimental ARPES angles and kinetic 
    energies into reciprocal space momentum (k-space), with support for 
    Soft X-ray photon momentum corrections.
    """
    
    # Conversion constant: sqrt(2 * m_e) / hbar in units of [eV^(-1/2) * Angstrom^(-1)]
    K_CONST = 0.512316722
    
    # hbar * c in units of [eV * Angstrom] for photon momentum calculations
    HBAR_C = 1973.269804 

    @classmethod
    def angle_to_k_parallel(cls, e_kin, theta, beta=0.0):
        """
        Converts kinetic energy and emission angles to in-plane momentum.
        
        Args:
            e_kin: Kinetic energy of the photoelectron (eV).
            theta: Manipulator polar angle (degrees).
            beta: Manipulator tilt angle (degrees). Default is 0.
            
        Returns:
            k_x, k_y: In-plane momentum components (Å⁻¹).
        """
        # Convert angles to radians
        theta_rad = np.radians(theta)
        beta_rad = np.radians(beta)
        
        # Calculate the magnitude of the momentum vector in vacuum
        k_vacuum = cls.K_CONST * np.sqrt(e_kin)
        
        # Project onto the surface plane (k_x and k_y)
        k_x = k_vacuum * np.sin(theta_rad)
        k_y = k_vacuum * np.cos(theta_rad) * np.sin(beta_rad)
        
        return k_x, k_y

    @classmethod
    def calculate_kz(cls, e_kin, theta, inner_potential, 
                     include_photon_momentum=False, 
                     photon_energy=None, 
                     photon_incidence_angle=45.0):
        """
        Calculates the out-of-plane momentum (k_z) assuming a free-electron 
        final state model, with an optional toggle for photon momentum shift 
        in the SX-ARPES regime.
        
        Args:
            e_kin: Kinetic energy of the photoelectron (eV).
            theta: Emission angle from the surface normal (degrees).
            inner_potential: The inner potential of the sample V_0 (eV).
            include_photon_momentum: Toggle to apply photon momentum shift.
            photon_energy: Incident photon energy (eV). Required if toggle is True.
            photon_incidence_angle: Angle of incoming beam relative to surface normal (degrees).
            
        Returns:
            k_z: Out-of-plane momentum (Å⁻¹).
        """
        theta_rad = np.radians(theta)
        
        # Standard k_z formula incorporating the inner potential
        k_z = cls.K_CONST * np.sqrt(e_kin * (np.cos(theta_rad)**2) + inner_potential)
        
        # Soft X-ray Correction
        if include_photon_momentum:
            if photon_energy is None:
                raise ValueError("photon_energy (eV) must be provided if include_photon_momentum is True.")
                
            # Photon momentum magnitude q = E / (hbar * c)
            q_mag = photon_energy / cls.HBAR_C
            
            # Project photon momentum onto the z-axis.
            # Assuming the beam comes in from the vacuum, its z-momentum is negative.
            # k_final = k_initial + q -> k_initial = k_final - q
            # k_initial_z = k_final_z - (-q_mag * cos(incidence_angle))
            incidence_rad = np.radians(photon_incidence_angle)
            q_z_shift = q_mag * np.cos(incidence_rad)
            
            k_z += q_z_shift
            
        return k_z
        
    @classmethod
    def energy_to_photon(cls, e_kin, work_function, binding_energy):
        """
        Helper method to relate kinetic energy to the incident photon energy.
        E_kin = h*nu - Phi - E_B
        """
        h_nu = e_kin + work_function + binding_energy
        return h_nu