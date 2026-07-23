import h5py
import numpy as np
import os
import traceback

def mock_arpes_signal(motor_values, e_axis, slit_val):
    """
    Generates an ARPES signal derived from a 3D hexagonal tight-binding model.
    Incorporates full kinematic momentum conversion and physical broadening.
    """
    kB = 8.617e-5  # eV/K
    T = 100.0      # Fixed at 100 K per specification
    
    # Extract scanning parameters
    hv = motor_values.get('Photon Energy', 75.0)
    x_pos = motor_values.get('X', 0.0)
    y_pos = motor_values.get('Y', 0.0)
    defl_val = motor_values.get('Deflection', 0.0)
    
    # Lattice Parameters (Angstroms)
    a = 5.0
    c = 10.0
    
    # Base tight-binding parameters (eV)
    t1_base = 1.0
    t2_base = 0.5
    
    # Semi-random spatial variation of hopping parameters
    # Mimics effective mass / interaction variations across the XY sample surface
    t1 = t1_base + 0.1 * np.sin(x_pos * 2.0 * np.pi / 10.0) * np.cos(y_pos * 2.0 * np.pi / 10.0)
    t2 = t2_base + 0.1 * np.cos(x_pos * y_pos * np.pi / 25.0)
    
    # Exact Angle-to-Momentum Conversion
    work_function = 4.5
    inner_potential = 10.0
    
    # Kinetic energy (strictly referenced to Fermi level mapping)
    E_kin = np.maximum(0.0, hv - work_function + e_axis)
    
    # Convert angles to radians
    slit_rad = np.radians(slit_val)
    defl_rad = np.radians(defl_val)
    
    # Cartesian momentum space projection (inverse Angstroms)
    # k_radius depends dynamically on the energy axis (k-warping)
    k_radius = 0.512 * np.sqrt(E_kin)
    kx = k_radius * np.sin(slit_rad) * np.cos(defl_rad)
    ky = k_radius * np.sin(defl_rad)
    
    # Exact kz calculation using the free-electron final state model
    kz = 0.512 * np.sqrt(np.maximum(0.0, E_kin * (np.cos(slit_rad)**2) * (np.cos(defl_rad)**2) + inner_potential))
    
    # 3D Hexagonal Tight-Binding Dispersion
    # Shifted by +4.0 eV to force the band to cross the Fermi level (E=0) optimally
    E_tb = -2.0 * t1 * (np.cos(kx * a) + 2.0 * np.cos(kx * a / 2.0) * np.cos(ky * a * np.sqrt(3.0) / 2.0)) - 2.0 * t2 * np.cos(kz * c)
    E_tb += 4.0 
    
    # Spectral Function Broadening (Lorentzian A(k,w))
    # Gamma encapsulates impurity and electron-phonon scattering at 100K
    Gamma = 0.05 + 0.08 * (e_axis**2) + (kB * T)
    A_kw = (1.0 / np.pi) * Gamma / ((e_axis - E_tb)**2 + Gamma**2)
    
    # Fermi-Dirac Distribution
    fd_dist = 1.0 / (1.0 + np.exp(e_axis / (kB * T)))
    
    measured_signal = A_kw * fd_dist
    
    # Shirley-Style Inelastic Background
    # Accumulates from unoccupied (positive) down into occupied (negative) binding energies
    bg = np.cumsum(measured_signal[::-1])[::-1]
    bg_norm = bg / (np.max(bg) + 1e-6)
    measured_signal += 0.05 * bg_norm
    
    return measured_signal

def create_arpes_hdf5(filepath, motor_dict, e_axis, slit_axis, data_type_name="Unknown"):
    print(f"[{data_type_name}] -> Attempting to create file:\n  {filepath}")
    
    try:
        with h5py.File(filepath, 'w') as f:
            g0 = f.create_group("0D_Data")
            g1 = f.create_group("1D_Data")
            g2 = f.create_group("2D_Data")
            gh = f.create_group("Headers")
            
            dt_swept = np.dtype([('Index', 'i4'), ('Name', 'S32'), ('Value', 'S32'), ('Comment', 'S64')])
            swept_data = np.array([
                (1, b'Scan Mode', data_type_name.encode('utf-8'), b'TB Hexagonal Simulation')
            ], dtype=dt_swept)
            gh.create_dataset("DAQ_Swept", data=swept_data)

            motor_lengths = []
            motor_names = list(motor_dict.keys())
            for name, arr in motor_dict.items():
                g0.create_dataset(name, data=arr)
                motor_lengths.append(len(arr))
            
            axes = list(motor_dict.values()) + [slit_axis]
            grids = np.meshgrid(*axes, indexing='ij')
            target_shape = tuple(motor_lengths) + (len(slit_axis), len(e_axis))
            
            # Pre-allocate hypercube
            data_cube = np.zeros(target_shape, dtype=np.float32)
            print(f"  |-- Calculating {len(axes)+1}D hypercube of shape {target_shape}...")

            it = np.nditer(grids[0], flags=['multi_index'])
            while not it.finished:
                idx = it.multi_index
                coords = [grids[i][idx] for i in range(len(axes))]
                
                current_motor_values = {motor_names[i]: coords[i] for i in range(len(motor_names))}
                slit_val = coords[-1]
                
                spectrum = mock_arpes_signal(current_motor_values, e_axis, slit_val)
                data_cube[idx] = spectrum
                it.iternext()

            n_spectra = np.prod(motor_lengths) if motor_lengths else 1
            flat_shape = (n_spectra, len(slit_axis), len(e_axis))
            
            maestro_flat_data = np.reshape(data_cube, flat_shape)
            ds = g2.create_dataset("Process_000", data=maestro_flat_data, chunks=True, compression="gzip")
            
            ds.attrs['unitNames'] = [b'deg', b'eV']
            ds.attrs['scaleOffset'] = [slit_axis[0], e_axis[0]]
            ds.attrs['scaleDelta']  = [slit_axis[1] - slit_axis[0], e_axis[1] - e_axis[0]]
            
            print(f"  |-- SUCCESS! File written correctly.\n")

    except Exception as e:
        print(f"\n[ERROR] Failed to write {filepath}!")
        print(traceback.format_exc())
        print("-" * 60)

def generate_all_mock_datasets():
    print("=" * 60)
    print("STARTING MOCK GENERATOR (TIGHT-BINDING 3D HEXAGONAL LATTICE)")
    print("=" * 60)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "mock_data_output")
    os.makedirs(output_dir, exist_ok=True)

    # Core Axes Definitions per constraints
    e_axis = np.linspace(-2.0, 0.5, 26)       # -2 to 0.5 eV with step ~0.1 eV
    slit_axis = np.linspace(-15.0, 15.0, 151) # -15 to 15 with step 0.2
    
    # Motor Arrays
    defl_axis = np.linspace(-10.0, 10.0, 101) # -10 to 10 with step 0.2
    hv_axis = np.linspace(50.0, 150.0, 21)    # Photon energy 50-150 eV
    x_axis = np.linspace(0.0, 10.0, 10)       # 10x10 XY scan points
    y_axis = np.linspace(0.0, 10.0, 10)

    # 1. 2D cut: GX cut at kz = 0 (using default hv=75, defl=0)
    create_arpes_hdf5(
        os.path.join(output_dir, "1_mock_2D_GX_cut.h5"),
        motor_dict={}, 
        e_axis=e_axis, slit_axis=slit_axis, data_type_name="1. 2D GX Cut"
    )

    # 2. 3D Fermi surface: FS at kz = 0 using deflection mode
    create_arpes_hdf5(
        os.path.join(output_dir, "2_mock_3D_FS.h5"),
        motor_dict={'Deflection': defl_axis}, 
        e_axis=e_axis, slit_axis=slit_axis, data_type_name="2. 3D Fermi Surface"
    )

    # 3. 4D data: GX cut varying over XY plane (t1/t2 spatial variation)
    create_arpes_hdf5(
        os.path.join(output_dir, "3_mock_4D_XY_cut.h5"),
        motor_dict={'Y': y_axis, 'X': x_axis}, 
        e_axis=e_axis, slit_axis=slit_axis, data_type_name="3. 4D XY Cut"
    )

    # 4. 5D data: FS mapping at each XY position
    create_arpes_hdf5(
        os.path.join(output_dir, "4_mock_5D_XY_FS.h5"),
        motor_dict={'Y': y_axis, 'X': x_axis, 'Deflection': defl_axis}, 
        e_axis=e_axis, slit_axis=slit_axis, data_type_name="4. 5D XY Fermi Surface"
    )

    # 5. 3D data: kz measurement along GX direction (Photon Energy scan)
    create_arpes_hdf5(
        os.path.join(output_dir, "5_mock_3D_kz_cut.h5"),
        motor_dict={'Photon Energy': hv_axis}, 
        e_axis=e_axis, slit_axis=slit_axis, data_type_name="5. 3D kz Cut"
    )

    # 6. 5D data: kz mapping at each XY position
    create_arpes_hdf5(
        os.path.join(output_dir, "6_mock_5D_XY_kz_cut.h5"),
        motor_dict={'Y': y_axis, 'X': x_axis, 'Photon Energy': hv_axis}, 
        e_axis=e_axis, slit_axis=slit_axis, data_type_name="6. 5D XY kz Cut"
    )

    # 7. 4D data: FS mapping combined with Photon Energy scan (Single XY)
    create_arpes_hdf5(
        os.path.join(output_dir, "7_mock_4D_FS_hv.h5"),
        motor_dict={'Photon Energy': hv_axis, 'Deflection': defl_axis}, 
        e_axis=e_axis, slit_axis=slit_axis, data_type_name="7. 4D FS vs Photon Energy"
    )

    # 8. 6D data: FS mapping with Photon Energy scan at each XY position
    # NOTE: This matrix requires ~1.7 GB of memory. Calculation may take a few minutes.
    create_arpes_hdf5(
        os.path.join(output_dir, "8_mock_6D_XY_FS_hv.h5"),
        motor_dict={'Y': y_axis, 'X': x_axis, 'Photon Energy': hv_axis, 'Deflection': defl_axis}, 
        e_axis=e_axis, slit_axis=slit_axis, data_type_name="8. 6D Complete Volumetric Scan"
    )

    print("=" * 60)
    print(f"PROCESS FINISHED. Files successfully generated in:\n{output_dir}")

if __name__ == "__main__":
    generate_all_mock_datasets()