import os
import h5py
import numpy as np
from PySide6.QtCore import QThread, Signal

class LoadWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(str, dict)
    error = Signal(str)

    def __init__(self, path, var_name, process_mode):
        super().__init__()
        self.path = path
        self.var_name = var_name
        self.process_mode = process_mode

    def run(self):
        try:
            self.progress.emit(10, f"Loading metadata from {os.path.basename(self.path)}...")
            with h5py.File(self.path, 'r') as f:
                data = self.recursively_load(f)

                if self.process_mode == "XY Scan":
                    self.progress.emit(20, "Extracting XY Scan Data...")
                    data = self.process_xy(data, f)
                elif self.process_mode == "Fermi Map":
                    self.progress.emit(20, "Extracting 3D Fermi Map...")
                    data = self.process_fermi(data, f)
                elif self.process_mode == "XY Scan Fine":
                    self.progress.emit(70, "Loading Fine Scan (Piezo) Data...")
                    data['kind'] = "XY Scan Fine (Awaiting Logic)"
                else:
                    data['kind'] = "Raw Data (Unrecognized)"

            self.progress.emit(100, "Done!")
            self.finished.emit(self.var_name, data)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
            self.progress.emit(0, "Load Failed.")

    def recursively_load(self, h5_obj):
        res = {}
        for k, item in h5_obj.items():
            if isinstance(item, h5py.Dataset):
                if item.size > 10_000_000:
                    continue
                val = item[()]
                if isinstance(val, bytes): 
                    val = val.decode('utf-8', errors='ignore')
                res[k] = val
            elif isinstance(item, h5py.Group):
                res[k] = self.recursively_load(item)
        return res

    def process_xy(self, data, f):
        x_raw, y_raw = None, None
        for key in ['X', 'Sample X', 'Scan X']:
            if key in data['0D_Data']: 
                x_raw = data['0D_Data'][key]
                break
        for key in ['Y', 'Sample Y', 'Scan Y']:
            if key in data['0D_Data']: 
                y_raw = data['0D_Data'][key]
                break

        if x_raw is not None and y_raw is None: y_raw = np.zeros_like(x_raw)
        elif y_raw is not None and x_raw is None: x_raw = np.zeros_like(y_raw)
        if x_raw is None and y_raw is None: raise KeyError("Could not find spatial motor data.")

        n_points = len(x_raw)
        data_group = f['2D_Data']
        raw_dataset = data_group[list(data_group.keys())[0]]
        shape = raw_dataset.shape
        
        diffs = [abs(dim - n_points) for dim in shape]
        points_axis = np.argmin(diffs)
        actual_points = shape[points_axis]
        
        if actual_points != n_points:
            self.progress.emit(25, f"Aborted scan detected! Truncating to {actual_points} points...")
            x_raw, y_raw = x_raw[:actual_points], y_raw[:actual_points]
            n_points = actual_points
            
        if 'Preview' in data and len(data['Preview']) > 0:
            prev_key = list(data['Preview'].keys())[0]
            nY, nX = data['Preview'][prev_key].shape[0], data['Preview'][prev_key].shape[1]
        else:
            nX_guess = len(np.unique(np.round(x_raw, 3)))
            nY_guess = len(np.unique(np.round(y_raw, 3)))
            if nX_guess * nY_guess == n_points: nX, nY = nX_guess, nY_guess
            else: nX, nY = n_points, 1
                
        data['x'] = np.linspace(x_raw.min(), x_raw.max(), nX)
        data['y'] = np.linspace(y_raw.min(), y_raw.max(), nY)
        
        # --- UNIVERSAL FIXED VS SWEPT DETECTOR ---
        first_E, last_E = 0.0, 1.0
        is_swept = False
        
        for hk in ['DAQ_Fixed', 'DAQ_Swept']:
            if hk in data['Headers']:
                if hk == 'DAQ_Swept': is_swept = True
                for row in data['Headers'][hk]:
                    comment = row[3]
                    if isinstance(comment, bytes): comment = comment.decode('utf-8', errors='ignore')
                    else: comment = str(comment)
                        
                    if 'First Energy' in comment or 'Min Energy' in comment or 'First Energy Channel' in comment: 
                        first_E = float(row[2])
                    elif 'Last Energy' in comment or 'Max Energy' in comment or 'Last Energy Channel' in comment: 
                        last_E = float(row[2])
                break 

        if points_axis == 0:
            dim1, dim2 = shape[1], shape[2]
            layout_style = "Points_First"
        elif points_axis == 2:
            dim1, dim2 = shape[0], shape[1]
            layout_style = "Points_Last"
        else:
            raise ValueError(f"Unrecognized HDF5 shape layout: {shape}")
            
        if is_swept:
            dim_E, dim_A = dim1, dim2
        else:
            dim_A, dim_E = dim1, dim2
        # -----------------------------------------
        
        # FIX: Check for native 1D Swept Energy arrays instead of relying on missing metadata
        if is_swept:
            try:
                # Search for the true 1D scale in the HDF5 dictionary
                energy_keys = [k for k in data.get('1D_Data', {}).keys() if 'Energy' in k]
                if energy_keys:
                    data['E'] = data['1D_Data'][energy_keys[0]]
                else:
                    data['E'] = np.linspace(first_E, last_E, dim_E)
            except Exception as e:
                print(f"Warning: Could not build proper energy axis: {e}")
                data['E'] = np.linspace(first_E, last_E, dim_E) # Fallback
        else:
            data['E'] = np.linspace(first_E, last_E, dim_E)
            
        data['angle'] = (np.arange(dim_A) - int(dim_A / 2)) * 0.048
        
        self.progress.emit(30, "Reading raw data natively...")
        buffer = np.empty(shape, dtype=np.float32)
        raw_dataset.read_direct(buffer)
        self.progress.emit(60, f"Reshaping array to angular grid ({nX}x{nY})...")
        
        if layout_style == "Points_First":
            val_array = buffer.reshape((nY, nX, dim1, dim2))
            _, idx_x = np.unique(np.round(x_raw, 3), return_index=True)
            if len(x_raw[np.sort(idx_x)]) > 1 and x_raw[np.sort(idx_x)][0] > x_raw[np.sort(idx_x)][-1]:
                val_array = np.flip(val_array, axis=1) 
            _, idx_y = np.unique(np.round(y_raw, 3), return_index=True)
            if len(y_raw[np.sort(idx_y)]) > 1 and y_raw[np.sort(idx_y)][0] > y_raw[np.sort(idx_y)][-1]:
                val_array = np.flip(val_array, axis=0) 
                
            if is_swept: transposed_array = np.transpose(val_array, (2, 3, 0, 1))
            else: transposed_array = np.transpose(val_array, (3, 2, 0, 1))
            
        elif layout_style == "Points_Last":
            val_array = buffer.reshape((dim1, dim2, nY, nX))
            if is_swept: transposed_array = np.transpose(val_array, (0, 1, 2, 3))
            else: transposed_array = np.transpose(val_array, (1, 0, 2, 3))
        
        self.progress.emit(85, "Forcing contiguous memory layout...")
        data['value'] = np.ascontiguousarray(transposed_array)
        data['kind'] = "XY Scan (Cleaned)"
        return data
    
    def process_fermi(self, data, f):
        angle_raw = None
        for key in ['Deflection', 'Slit Defl', 'Manipulator Theta', 'Manipulator Phi', 'Tilt', 'ThetaX', 'ThetaY']:
            if key in data['0D_Data']: 
                angle_raw = data['0D_Data'][key]
                break
        if angle_raw is None: raise KeyError("Could not find angular motor data.")

        n_points = len(angle_raw)
        data_group = f['2D_Data']
        raw_dataset = data_group[list(data_group.keys())[0]]
        shape = raw_dataset.shape
        
        diffs = [abs(dim - n_points) for dim in shape]
        points_axis = np.argmin(diffs)
        actual_points = shape[points_axis]
        
        if actual_points != n_points:
            self.progress.emit(25, f"Aborted scan detected! Truncating to {actual_points} points...")
            angle_raw = angle_raw[:actual_points]
            n_points = actual_points
            
        nDefl_guess = len(np.unique(np.round(angle_raw, 3)))
        nDefl = nDefl_guess if nDefl_guess == n_points else n_points

        data['x'] = np.linspace(angle_raw.min(), angle_raw.max(), nDefl)
        data['y'] = np.array([0.0])
        nY, nX = 1, nDefl
        
        # --- UNIVERSAL FIXED VS SWEPT DETECTOR ---
        first_E, last_E = 0.0, 1.0
        is_swept = False
        
        for hk in ['DAQ_Fixed', 'DAQ_Swept']:
            if hk in data['Headers']:
                if hk == 'DAQ_Swept': is_swept = True
                for row in data['Headers'][hk]:
                    comment = row[3]
                    if isinstance(comment, bytes): comment = comment.decode('utf-8', errors='ignore')
                    else: comment = str(comment)
                        
                    if 'First Energy' in comment or 'Min Energy' in comment or 'First Energy Channel' in comment: 
                        first_E = float(row[2])
                    elif 'Last Energy' in comment or 'Max Energy' in comment or 'Last Energy Channel' in comment: 
                        last_E = float(row[2])
                break 
                
        if points_axis == 0:
            dim1, dim2 = shape[1], shape[2]
            layout_style = "Points_First"
        elif points_axis == 2:
            dim1, dim2 = shape[0], shape[1]
            layout_style = "Points_Last"
        else:
            raise ValueError(f"Unrecognized HDF5 shape layout: {shape}")
            
        if is_swept:
            dim_E, dim_A = dim1, dim2
        else:
            dim_A, dim_E = dim1, dim2
        # -----------------------------------------
        
        # FIX: Extract true Energy axis from hidden HDF5 Attributes
        e_axis_built = False
        if 'unitNames' in raw_dataset.attrs and 'scaleOffset' in raw_dataset.attrs and 'scaleDelta' in raw_dataset.attrs:
            try:
                # Decode bytes to strings
                units = [u.decode('utf-8') if isinstance(u, bytes) else u for u in raw_dataset.attrs['unitNames']]
                
                # Find which axis corresponds to Energy (eV)
                eV_idx = -1
                for i, u in enumerate(units):
                    if 'eV' in u or 'Energy' in u:
                        eV_idx = i
                        break
                
                if eV_idx != -1:
                    start_E = float(raw_dataset.attrs['scaleOffset'][eV_idx])
                    step_E = float(raw_dataset.attrs['scaleDelta'][eV_idx])
                    data['E'] = start_E + np.arange(dim_E) * step_E
                    e_axis_built = True
                    print(f"Successfully loaded precise Energy axis from attributes! ({start_E:.2f} eV to {data['E'][-1]:.2f} eV)")
            except Exception as e:
                print(f"Warning: Failed to parse precise Energy attributes: {e}")
                
        # Fallback to standard metadata if attributes are missing (Fixed Mode)
        if not e_axis_built:
            data['E'] = np.linspace(first_E, last_E, dim_E)
            
        data['angle'] = (np.arange(dim_A) - int(dim_A / 2)) * 0.048
        
        self.progress.emit(30, "Reading raw data natively...")
        buffer = np.empty(shape, dtype=np.float32)
        raw_dataset.read_direct(buffer)
        self.progress.emit(60, f"Reshaping dynamic array ({layout_style})...")
        
        if layout_style == "Points_First":
            val_array = buffer.reshape((nY, nX, dim1, dim2))
            _, idx_a = np.unique(np.round(angle_raw, 3), return_index=True)
            a_ordered = angle_raw[np.sort(idx_a)]
            if len(a_ordered) > 1 and a_ordered[0] > a_ordered[-1]:
                val_array = np.flip(val_array, axis=1) 
                
            if is_swept: transposed_array = np.transpose(val_array, (2, 3, 0, 1))
            else: transposed_array = np.transpose(val_array, (3, 2, 0, 1))
            
        elif layout_style == "Points_Last":
            val_array = buffer.reshape((dim1, dim2, nY, nX))
            _, idx_a = np.unique(np.round(angle_raw, 3), return_index=True)
            a_ordered = angle_raw[np.sort(idx_a)]
            if len(a_ordered) > 1 and a_ordered[0] > a_ordered[-1]:
                val_array = np.flip(val_array, axis=3) 
                
            if is_swept: transposed_array = np.transpose(val_array, (0, 1, 2, 3))
            else: transposed_array = np.transpose(val_array, (1, 0, 2, 3))
            
        # FIX: Natively flip the Energy axis (axis=0) because swept mode builds backward
        #if is_swept:
        #    transposed_array = np.flip(transposed_array, axis=0)
        
        self.progress.emit(85, "Forcing contiguous memory layout...")
        data['value'] = np.ascontiguousarray(transposed_array)
        data['kind'] = "Fermi Map (Cleaned)"
        return data