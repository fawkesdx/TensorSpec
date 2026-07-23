import h5py
import numpy as np
import os

class MaestroLoader:
    """
    Modern parser for MAESTRO (ALS) datasets.
    Handles dynamic N-dimensional reshaping, aborted scans, and the new workspace paradigm.
    """
    def __init__(self, filepath):
        self.filepath = str(filepath)
        self.filename = os.path.basename(self.filepath)

    def load(self):
        with h5py.File(self.filepath, 'r') as f:
            # --- 1. STRICT SIGNATURE CHECK ---
            if not all(key in f for key in ["0D_Data", "2D_Data", "Headers"]):
                raise ValueError("Not a MAESTRO file: Missing fundamental HDF5 structure.")
            if "Process_000" not in f["2D_Data"]:
                raise ValueError("Not a MAESTRO file: Missing Process_000 dataset.")

            # --- 2. MAESTRO HEADERS & DETECTOR MODE ---
            mode_string = "Unknown Scan"
            is_fixed = False
            
            if "DAQ_Swept" in f["Headers"]:
                header_data = f["Headers"]["DAQ_Swept"][()]
                mode_string = self._extract_mode(header_data)
            elif "DAQ_Fixed" in f["Headers"]:
                header_data = f["Headers"]["DAQ_Fixed"][()]
                mode_string = self._extract_mode(header_data)
                is_fixed = True
            else:
                raise ValueError("Not a MAESTRO file: Missing DAQ_Swept/DAQ_Fixed tags.")

            # --- 3. MOTORS & RAW DATA ---
            motors = {}
            for motor_name in f["0D_Data"].keys():
                motors[motor_name] = f["0D_Data"][motor_name][()]

            ds = f["2D_Data"]["Process_000"]
            raw_data = ds[()]
            
            # Extract Base Axes Attributes from MAESTRO data
            units = ds.attrs.get('unitNames', [b'eV', b'deg'])
            offsets = ds.attrs.get('scaleOffset', [0.0, 0.0])
            deltas = ds.attrs.get('scaleDelta', [0.01, 0.01])
            
            n_dim_1 = raw_data.shape[-2]
            n_dim_2 = raw_data.shape[-1]
            
            axis_1 = offsets[0] + np.arange(n_dim_1) * deltas[0]
            axis_2 = offsets[1] + np.arange(n_dim_2) * deltas[1]
            
            unit_1 = units[0].decode('utf-8') if isinstance(units[0], bytes) else str(units[0])
            unit_2 = units[1].decode('utf-8') if isinstance(units[1], bytes) else str(units[1])
            
            # --- 4. ABORTED SCAN LOGIC & DYNAMIC GRID CALCULATION ---
            # If the run crashed, the actual points will be fewer than the motor array length
            expected_points = len(list(motors.values())[0]) if motors else 1
            actual_points = raw_data.size // (n_dim_1 * n_dim_2)
            
            is_aborted = False
            if motors and actual_points < expected_points:
                print(f"[{self.filename}] Aborted scan detected! Truncating motors from {expected_points} to {actual_points} points...")
                for k in motors.keys():
                    motors[k] = motors[k][:actual_points]
                is_aborted = True
                mode_string += " (Aborted)"

            # Calculate the N-Dimensional grid shape using unique motor coordinates
            grid_shape = []
            if not is_aborted and motors:
                for arr in motors.values():
                    grid_shape.append(len(np.unique(np.round(arr, 3))))
                
                # Verify the unique grid perfectly matches the total actual points
                if np.prod(grid_shape) == actual_points:
                    target_shape = tuple(grid_shape) + (n_dim_1, n_dim_2)
                else:
                    target_shape = (actual_points, n_dim_1, n_dim_2) # Fallback to 1D list of spectra
            elif is_aborted or motors:
                target_shape = (actual_points, n_dim_1, n_dim_2) # Aborted scans stay flat
            else:
                target_shape = (n_dim_1, n_dim_2) # Pure 2D scan
            
            # Execute the reshape
            try:
                data = np.reshape(raw_data, target_shape)
            except ValueError:
                data = raw_data.squeeze()

            # --- 5. FIX MAESTRO FIXED-MODE ROTATION ---
            # DAQ_Fixed data stores Angle first, Energy second. Transpose to fix.
            if is_fixed and len(motors) == 0:
                data = np.transpose(data, (1, 0))
                axis_1, axis_2 = axis_2, axis_1
                unit_1, unit_2 = unit_2, unit_1

            # Update string dynamically for high-dimension scans
            dimension_count = len(grid_shape) + 2 if not is_aborted else 3
            if dimension_count >= 6 and "Unknown" in mode_string:
                mode_string = f"{dimension_count}D Scan"

            # --- 6. BUILD MODERN WORKSPACE PAYLOAD ---
            axes = {}
            for name, arr in motors.items():
                # Store unique axis steps if perfectly mapped, otherwise store the raw flat array
                if not is_aborted and np.prod(grid_shape) == actual_points:
                    axes[name] = np.unique(np.round(arr, 3))
                else:
                    axes[name] = arr
                
            axis_1_name = "Energy (eV)" if "eV" in unit_1 else "Slit Angle (deg)"
            axis_2_name = "Slit Angle (deg)" if "deg" in unit_2 else "Energy (eV)"
            
            if axis_1_name == axis_2_name:
                axis_1_name, axis_2_name = "Axis Y", "Axis X"

            axes[axis_1_name] = axis_1
            axes[axis_2_name] = axis_2
            
            print(f"[{self.filename}] Successfully loaded via MaestroLoader")
            
            return {
                "name": self.filename.replace('.h5', ''),
                "data": data,
                "axes": axes,
                "mode": mode_string,
                "is_fixed": is_fixed,
                "facility": "MAESTRO"
            }

    def _extract_mode(self, header_data):
        """Helper to extract scan mode from HDF5 bytes."""
        for row in header_data:
            if len(row) >= 4 and b'Mode' in row[1]:
                try: return row[2].decode('utf-8')
                except: pass
        return "Unknown Scan"