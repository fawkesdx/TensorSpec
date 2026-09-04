import numpy as np
from PySide6.QtCore import QThread, Signal
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates

class AzimuthalTwistWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(np.ndarray, np.ndarray, np.ndarray, str)
    error = Signal(str)

    def __init__(self, xy_data, ref_data, gamma_s, gamma_d, max_shift=30):
        super().__init__()
        self.xy_data, self.ref_data = xy_data, ref_data
        self.gamma_slit, self.gamma_defl = gamma_s, gamma_d
        self.max_shift = max_shift
        self.angles = np.arange(0, 360, 1.0) 

    def run(self):
        try:
            self.progress.emit(5, "Aligning Physical Energy Axes...")
            E_xy, E_ref = self.xy_data['E'].copy(), self.ref_data['E'].copy()
            if np.mean(E_ref) > 0 and np.mean(E_xy) < 0: E_ref = -E_ref
            elif np.mean(E_xy) > 0 and np.mean(E_ref) < 0: E_xy = -E_xy
            
            e_min, e_max = max(np.min(E_xy), np.min(E_ref)), min(np.max(E_xy), np.max(E_ref))
            if e_min >= e_max: raise ValueError("No energy overlap!")
            xy_e_mask = (E_xy >= e_min) & (E_xy <= e_max)
            xy_val = self.xy_data['value'][xy_e_mask, :, :, :]
            dim_E, dim_A_xy, nY, nX = xy_val.shape
            
            ref_squeezed = self.ref_data['value'].squeeze(axis=2)
            A_arr, D_arr = self.ref_data['angle'], self.ref_data['x']
            A_scale = abs(A_arr[-1] - A_arr[0]) / max(1, len(A_arr)-1) or 1.0
            D_scale = abs(D_arr[-1] - D_arr[0]) / max(1, len(D_arr)-1) or 1.0
            aspect_ratio = A_scale / D_scale
            
            sort_idx = np.argsort(E_ref)
            f_interp = interp1d(E_ref[sort_idx], ref_squeezed[sort_idx, :, :], axis=0, bounds_error=False, fill_value=0)
            ref_matched = f_interp(E_xy[xy_e_mask])
            
            self.progress.emit(15, "Precomputing Rotated Templates...")
            templates = []
            ds = np.arange(dim_A_xy) - (dim_A_xy // 2)
            E_coords = np.arange(dim_E)
            
            for theta_deg in self.angles:
                theta = np.deg2rad(theta_deg)
                slit_coords = self.gamma_slit + ds * np.cos(theta)
                defl_coords = self.gamma_defl - ds * aspect_ratio * np.sin(theta)
                
                EE, SS = np.meshgrid(E_coords, slit_coords, indexing='ij')
                _, DD = np.meshgrid(E_coords, defl_coords, indexing='ij')
                
                template = map_coordinates(ref_matched, np.stack([EE, SS, DD]), order=1, mode='constant', cval=0.0)
                t_mean, t_std = np.mean(template), np.std(template)
                if t_std > 0: template = (template - t_mean) / t_std
                templates.append(template)
                
            twist_map, shift_map, score_map = np.zeros((nY, nX)), np.zeros((nY, nX)), np.zeros((nY, nX))
            norm_div, total, curr = dim_E * dim_A_xy, nX * nY, 0
            
            for x in range(nX):
                for y in range(nY):
                    local_cut = xy_val[:, :, y, x]
                    l_mean, l_std = np.mean(local_cut), np.std(local_cut)
                    if l_std == 0: continue
                    local_norm = (local_cut - l_mean) / l_std
                    
                    best_c_score, best_c_idx, best_c_s = -np.inf, 0, 0
                    for i in range(0, 360, 5):
                        temp = templates[i]
                        for s_shift in range(-5, 6):
                            l_start, l_end = max(0, s_shift), min(dim_A_xy, dim_A_xy + s_shift)
                            r_start, r_end = max(0, -s_shift), min(dim_A_xy, dim_A_xy - s_shift)
                            if l_end - l_start == 0: continue
                            score = np.sum(local_norm[:, l_start:l_end] * temp[:, r_start:r_end]) / norm_div
                            if score > best_c_score: best_c_score, best_c_idx, best_c_s = score, i, s_shift
                                
                    best_overall_score, best_angle, best_shift = best_c_score, self.angles[best_c_idx], best_c_s
                    for offset in [-4, -3, -2, -1, 1, 2, 3, 4]:
                        idx = (best_c_idx + offset) % 360
                        temp = templates[idx]
                        for s_shift in range(-self.max_shift, self.max_shift + 1):
                            l_start, l_end = max(0, s_shift), min(dim_A_xy, dim_A_xy + s_shift)
                            r_start, r_end = max(0, -s_shift), min(dim_A_xy, dim_A_xy - s_shift)
                            if l_end - l_start == 0: continue
                            score = np.sum(local_norm[:, l_start:l_end] * temp[:, r_start:r_end]) / norm_div
                            if score > best_overall_score: best_overall_score, best_angle, best_shift = score, self.angles[idx], s_shift
                                
                    twist_map[y, x], shift_map[y, x], score_map[y, x] = best_angle, best_shift, best_overall_score
                    curr += 1
                self.progress.emit(20 + int(80 * curr / total), f"Mapping Twist: Column {x+1}/{nX}")
            self.finished.emit(twist_map, shift_map, score_map, "Azimuthal Twist (In-Plane)")
        except Exception as e:
            self.error.emit(str(e))

class CoupledAzimuthTiltWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(np.ndarray, np.ndarray, np.ndarray, str)
    error = Signal(str)

    def __init__(self, xy_data, ref_data, gamma_s, gamma_d, max_shift=20, max_tilt=15):
        super().__init__()
        self.xy_data, self.ref_data = xy_data, ref_data
        self.gamma_slit, self.gamma_defl = gamma_s, gamma_d
        self.max_shift, self.max_tilt = max_shift, max_tilt
        self.angles = np.arange(0, 360, 1.0) 
        self.tilts = np.arange(-self.max_tilt, self.max_tilt + 1)

    def run(self):
        try:
            self.progress.emit(5, "Aligning Physical Energy Axes...")
            E_xy, E_ref = self.xy_data['E'].copy(), self.ref_data['E'].copy()
            if np.mean(E_ref) > 0 and np.mean(E_xy) < 0: E_ref = -E_ref
            elif np.mean(E_xy) > 0 and np.mean(E_ref) < 0: E_xy = -E_xy
            
            e_min, e_max = max(np.min(E_xy), np.min(E_ref)), min(np.max(E_xy), np.max(E_ref))
            if e_min >= e_max: raise ValueError("No energy overlap!")
            xy_e_mask = (E_xy >= e_min) & (E_xy <= e_max)
            xy_val = self.xy_data['value'][xy_e_mask, :, :, :]
            dim_E, dim_A_xy, nY, nX = xy_val.shape
            
            ref_squeezed = self.ref_data['value'].squeeze(axis=2)
            A_arr, D_arr = self.ref_data['angle'], self.ref_data['x']
            A_scale = abs(A_arr[-1] - A_arr[0]) / max(1, len(A_arr)-1) or 1.0
            D_scale = abs(D_arr[-1] - D_arr[0]) / max(1, len(D_arr)-1) or 1.0
            aspect_ratio = A_scale / D_scale
            
            sort_idx = np.argsort(E_ref)
            f_interp = interp1d(E_ref[sort_idx], ref_squeezed[sort_idx, :, :], axis=0, bounds_error=False, fill_value=0)
            ref_matched = f_interp(E_xy[xy_e_mask])
            
            self.progress.emit(10, "Precomputing Coupled Matrix (Rotations + Tilts)...")
            templates = [[None for _ in range(len(self.tilts))] for _ in range(360)]
            ds = np.arange(dim_A_xy) - (dim_A_xy // 2)
            E_coords = np.arange(dim_E)
            total_temps, curr_temp = 360 * len(self.tilts), 0
            
            for a_idx, theta_deg in enumerate(self.angles):
                theta = np.deg2rad(theta_deg)
                for t_idx, dd in enumerate(self.tilts):
                    slit_coords = self.gamma_slit + ds * np.cos(theta) + dd * np.sin(theta) / aspect_ratio
                    defl_coords = self.gamma_defl - ds * aspect_ratio * np.sin(theta) + dd * np.cos(theta)
                    
                    EE, SS = np.meshgrid(E_coords, slit_coords, indexing='ij')
                    _, DD = np.meshgrid(E_coords, defl_coords, indexing='ij')
                    
                    template = map_coordinates(ref_matched, np.stack([EE, SS, DD]), order=1, mode='constant', cval=0.0)
                    t_mean, t_std = np.mean(template), np.std(template)
                    if t_std > 0: template = (template - t_mean) / t_std
                    templates[a_idx][t_idx] = template
                    curr_temp += 1
                if a_idx % 36 == 0: self.progress.emit(10 + int(20 * curr_temp / total_temps), "Precomputing Coupled Matrix...")
                    
            twist_map, tilt_map, score_map = np.zeros((nY, nX)), np.zeros((nY, nX)), np.zeros((nY, nX))
            norm_div, total, curr = dim_E * dim_A_xy, nX * nY, 0
            
            for x in range(nX):
                for y in range(nY):
                    local_cut = xy_val[:, :, y, x]
                    l_mean, l_std = np.mean(local_cut), np.std(local_cut)
                    if l_std == 0: continue
                    local_norm = (local_cut - l_mean) / l_std
                    
                    best_c_score, best_c_a, best_c_t, best_c_s = -np.inf, 0, 0, 0
                    for a in range(0, 360, 5):
                        for t_idx in range(0, len(self.tilts), 3):
                            temp = templates[a][t_idx]
                            for s_shift in range(-self.max_shift, self.max_shift + 1, 4):
                                l_start, l_end = max(0, s_shift), min(dim_A_xy, dim_A_xy + s_shift)
                                r_start, r_end = max(0, -s_shift), min(dim_A_xy, dim_A_xy - s_shift)
                                if l_end - l_start == 0: continue
                                score = np.sum(local_norm[:, l_start:l_end] * temp[:, r_start:r_end]) / norm_div
                                if score > best_c_score: best_c_score, best_c_a, best_c_t, best_c_s = score, a, t_idx, s_shift
                                    
                    best_overall_score, best_angle, best_tilt = best_c_score, self.angles[best_c_a], self.tilts[best_c_t]
                    for offset_a in [-4, -3, -2, -1, 1, 2, 3, 4]:
                        idx_a = (best_c_a + offset_a) % 360
                        for offset_t in [-2, -1, 1, 2]:
                            idx_t = best_c_t + offset_t
                            if idx_t < 0 or idx_t >= len(self.tilts): continue
                            temp = templates[idx_a][idx_t]
                            for offset_s in [-3, -2, -1, 1, 2, 3]:
                                s_shift = best_c_s + offset_s
                                if s_shift < -self.max_shift or s_shift > self.max_shift: continue
                                l_start, l_end = max(0, s_shift), min(dim_A_xy, dim_A_xy + s_shift)
                                r_start, r_end = max(0, -s_shift), min(dim_A_xy, dim_A_xy - s_shift)
                                if l_end - l_start == 0: continue
                                score = np.sum(local_norm[:, l_start:l_end] * temp[:, r_start:r_end]) / norm_div
                                if score > best_overall_score: 
                                    best_overall_score, best_angle, best_tilt = score, self.angles[idx_a], self.tilts[idx_t]
                                    
                    twist_map[y, x], tilt_map[y, x], score_map[y, x] = best_angle, best_tilt, best_overall_score
                    curr += 1
                self.progress.emit(30 + int(70 * curr / total), f"Mapping Coupled Physics: Column {x+1}/{nX}")
                
            self.finished.emit(twist_map, tilt_map, score_map, "Coupled Azimuth & Deflection Tilt")
        except Exception as e:
            self.error.emit(str(e))

class NormalTiltWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(np.ndarray, np.ndarray, np.ndarray, str)
    error = Signal(str)

    def __init__(self, xy_data, ref_data, gamma_s, gamma_d, max_shift=30):
        super().__init__()
        self.xy_data, self.ref_data = xy_data, ref_data
        self.gamma_slit, self.gamma_defl = gamma_s, gamma_d
        self.max_shift = max_shift

    def run(self):
        try:
            self.progress.emit(5, "Aligning Physical Energy Axes...")
            E_xy, E_ref = self.xy_data['E'].copy(), self.ref_data['E'].copy()
            if np.mean(E_ref) > 0 and np.mean(E_xy) < 0: E_ref = -E_ref
            elif np.mean(E_xy) > 0 and np.mean(E_ref) < 0: E_xy = -E_xy
            
            e_min, e_max = max(np.min(E_xy), np.min(E_ref)), min(np.max(E_xy), np.max(E_ref))
            if e_min >= e_max: raise ValueError("No energy overlap!")
            xy_e_mask = (E_xy >= e_min) & (E_xy <= e_max)
            xy_val = self.xy_data['value'][xy_e_mask, :, :, :]
            dim_E, dim_A, nY, nX = xy_val.shape
            
            ref_squeezed = self.ref_data['value'].squeeze(axis=2)
            ref_A_dim, ref_D_dim = ref_squeezed.shape[1], ref_squeezed.shape[2]
            
            sort_idx = np.argsort(E_ref)
            f_interp = interp1d(E_ref[sort_idx], ref_squeezed[sort_idx, :, :], axis=0, bounds_error=False, fill_value=0)
            ref_matched = f_interp(E_xy[xy_e_mask])
                
            self.progress.emit(15, "Normalizing Reference Volume...")
            ref_norm = np.zeros_like(ref_matched)
            for d in range(ref_D_dim):
                sl = ref_matched[:, :, d]
                m, s = np.mean(sl), np.std(sl)
                if s > 0: ref_norm[:, :, d] = (sl - m) / s
                
            defl_tilt_map, slit_tilt_map, score_map = np.zeros((nY, nX)), np.zeros((nY, nX)), np.zeros((nY, nX))
            total, curr = nX * nY, 0
            
            for x in range(nX):
                for y in range(nY):
                    local_cut = xy_val[:, :, y, x]
                    l_mean, l_std = np.mean(local_cut), np.std(local_cut)
                    if l_std == 0: continue
                    local_norm = (local_cut - l_mean) / l_std
                    
                    best_score, best_d, best_s = -np.inf, 0, 0
                    for d in range(ref_D_dim):
                        for s_shift in range(-self.max_shift, self.max_shift + 1):
                            offset = self.gamma_slit + s_shift - (dim_A // 2)
                            l_start, l_end = max(0, -offset), min(dim_A, ref_A_dim - offset)
                            if l_end <= l_start: continue
                            r_start, r_end = l_start + offset, l_end + offset
                            score = np.sum(local_norm[:, l_start:l_end] * ref_norm[:, r_start:r_end, d]) / (dim_E * dim_A)
                            if score > best_score: best_score, best_d, best_s = score, d, s_shift
                                
                    defl_tilt_map[y, x], slit_tilt_map[y, x], score_map[y, x] = best_d - self.gamma_defl, best_s, best_score
                    curr += 1
                self.progress.emit(20 + int(80 * curr / total), f"Mapping Normal Tilt: Column {x+1}/{nX}")
            self.finished.emit(defl_tilt_map, slit_tilt_map, score_map, "Surface Normal Tilt (Out-of-Plane)")
        except Exception as e:
            self.error.emit(str(e))