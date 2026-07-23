import h5py
import numpy as np
import os

class MockDataLoader:
    """Generic parser for simulated mock N-dimensional data."""
    def __init__(self, filepath):
        self.filepath = str(filepath)
        self.filename = os.path.basename(self.filepath)

    def load(self):
        # 1. UNIQUE SIGNATURE CHECK
        if "mock" not in self.filename.lower():
            raise ValueError("Not recognized as generic mock data.")

        with h5py.File(self.filepath, 'r') as f:
            mode_string = "Unknown Scan"
            is_fixed = False
            
            if "Headers" in f:
                if "DAQ_Swept" in f["Headers"]:
                    header_data = f["Headers"]["DAQ_Swept"][()]
                    mode_string = self._extract_mode(header_data)
                elif "DAQ_Fixed" in f["Headers"]:
                    header_data = f["Headers"]["DAQ_Fixed"][()]
                    mode_string = self._extract_mode(header_data)
                    is_fixed = True
                    
            motors = {}
            if "0D_Data" in f:
                for motor_name in f["0D_Data"].keys():
                    motors[motor_name] = f["0D_Data"][motor_name][()]
                    
            ds = f["2D_Data"]["Process_000"]
            raw_data = ds[()]
            
            units = ds.attrs.get('unitNames', [b'eV', b'deg'])
            offsets = ds.attrs.get('scaleOffset', [0.0, 0.0])
            deltas = ds.attrs.get('scaleDelta', [0.01, 0.01])
            
            n_dim_1 = raw_data.shape[-2]
            n_dim_2 = raw_data.shape[-1]
            
            axis_1 = offsets[0] + np.arange(n_dim_1) * deltas[0]
            axis_2 = offsets[1] + np.arange(n_dim_2) * deltas[1]
            
            unit_1 = units[0].decode('utf-8') if isinstance(units[0], bytes) else str(units[0])
            unit_2 = units[1].decode('utf-8') if isinstance(units[1], bytes) else str(units[1])
            
            # --- Calculate the true N-dimensional grid shape ---
            grid_shape = []
            for m in motors.values():
                grid_shape.append(len(np.unique(np.round(m, 3))))
                
            target_shape = tuple(grid_shape) + (n_dim_1, n_dim_2)
            
            try:
                reshaped_data = np.reshape(raw_data, target_shape)
            except ValueError:
                if raw_data.shape == target_shape or raw_data.shape == (1, *target_shape):
                    reshaped_data = raw_data.squeeze(axis=0) if raw_data.shape[0] == 1 else raw_data
                else:
                    raise ValueError(f"Mock reshape failed: {raw_data.shape} to {target_shape}")

            if is_fixed and len(grid_shape) == 0:
                reshaped_data = np.transpose(reshaped_data, (1, 0))
                axis_1, axis_2 = axis_2, axis_1
                unit_1, unit_2 = unit_2, unit_1

            dimension_count = len(grid_shape) + 2
            if dimension_count >= 6 and "Unknown" in mode_string:
                mode_string = f"{dimension_count}D Scan"

            # Store the unique, unflattened axes for the Workspace UI
            axes = {}
            for name, arr in motors.items():
                axes[name] = np.unique(np.round(arr, 3))
                
            axis_1_name = "Energy (eV)" if "eV" in unit_1 else "Slit Angle (deg)"
            axis_2_name = "Slit Angle (deg)" if "deg" in unit_2 else "Energy (eV)"
            
            if axis_1_name == axis_2_name:
                axis_1_name, axis_2_name = "Axis Y", "Axis X"

            axes[axis_1_name] = axis_1
            axes[axis_2_name] = axis_2
            
            print(f"[{self.filename}] Successfully loaded via MockDataLoader")
            
            return {
                "name": self.filename.replace('.h5', ''),
                "data": reshaped_data,
                "axes": axes,
                "mode": mode_string,
                "is_fixed": is_fixed
            }

    def _extract_mode(self, header_data):
        for row in header_data:
            if len(row) >= 4 and b'Mode' in row[1]:
                try: return row[2].decode('utf-8')
                except: pass
        return "Unknown Scan"