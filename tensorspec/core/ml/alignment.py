import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates


def run_azimuthal_twist(xy_data, ref_data, gamma_s, gamma_d, max_shift=30, on_progress=None):
    if on_progress:
        on_progress(5, "Aligning Physical Energy Axes...")
    E_xy, E_ref = xy_data['E'].copy(), ref_data['E'].copy()
    if np.mean(E_ref) > 0 and np.mean(E_xy) < 0:
        E_ref = -E_ref
    elif np.mean(E_xy) > 0 and np.mean(E_ref) < 0:
        E_xy = -E_xy

    e_min, e_max = max(np.min(E_xy), np.min(E_ref)), min(np.max(E_xy), np.max(E_ref))
    if e_min >= e_max:
        raise ValueError("No energy overlap!")
    xy_e_mask = (E_xy >= e_min) & (E_xy <= e_max)
    xy_val = xy_data['value'][xy_e_mask, :, :, :]
    dim_E, dim_A_xy, nY, nX = xy_val.shape

    ref_squeezed = ref_data['value'].squeeze(axis=2)
    A_arr, D_arr = ref_data['angle'], ref_data['x']
    A_scale = abs(A_arr[-1] - A_arr[0]) / max(1, len(A_arr) - 1) or 1.0
    D_scale = abs(D_arr[-1] - D_arr[0]) / max(1, len(D_arr) - 1) or 1.0
    aspect_ratio = A_scale / D_scale

    sort_idx = np.argsort(E_ref)
    f_interp = interp1d(E_ref[sort_idx], ref_squeezed[sort_idx, :, :], axis=0, bounds_error=False, fill_value=0)
    ref_matched = f_interp(E_xy[xy_e_mask])

    if on_progress:
        on_progress(15, "Precomputing Rotated Templates...")
    angles = np.arange(0, 360, 1.0)
    templates = []
    ds = np.arange(dim_A_xy) - (dim_A_xy // 2)
    E_coords = np.arange(dim_E)

    for theta_deg in angles:
        theta = np.deg2rad(theta_deg)
        slit_coords = gamma_s + ds * np.cos(theta)
        defl_coords = gamma_d - ds * aspect_ratio * np.sin(theta)

        EE, SS = np.meshgrid(E_coords, slit_coords, indexing='ij')
        _, DD = np.meshgrid(E_coords, defl_coords, indexing='ij')

        template = map_coordinates(ref_matched, np.stack([EE, SS, DD]), order=1, mode='constant', cval=0.0)
        t_mean, t_std = np.mean(template), np.std(template)
        if t_std > 0:
            template = (template - t_mean) / t_std
        templates.append(template)

    twist_map, shift_map, score_map = np.zeros((nY, nX)), np.zeros((nY, nX)), np.zeros((nY, nX))
    norm_div, total, curr = dim_E * dim_A_xy, nX * nY, 0

    for x in range(nX):
        for y in range(nY):
            local_cut = xy_val[:, :, y, x]
            l_mean, l_std = np.mean(local_cut), np.std(local_cut)
            if l_std == 0:
                continue
            local_norm = (local_cut - l_mean) / l_std

            best_c_score, best_c_idx, best_c_s = -np.inf, 0, 0
            for i in range(0, 360, 5):
                temp = templates[i]
                for s_shift in range(-5, 6):
                    l_start, l_end = max(0, s_shift), min(dim_A_xy, dim_A_xy + s_shift)
                    r_start, r_end = max(0, -s_shift), min(dim_A_xy, dim_A_xy - s_shift)
                    if l_end - l_start == 0:
                        continue
                    score = np.sum(local_norm[:, l_start:l_end] * temp[:, r_start:r_end]) / norm_div
                    if score > best_c_score:
                        best_c_score, best_c_idx, best_c_s = score, i, s_shift

            best_overall_score, best_angle, best_shift = best_c_score, angles[best_c_idx], best_c_s
            for offset in [-4, -3, -2, -1, 1, 2, 3, 4]:
                idx = (best_c_idx + offset) % 360
                temp = templates[idx]
                for s_shift in range(-max_shift, max_shift + 1):
                    l_start, l_end = max(0, s_shift), min(dim_A_xy, dim_A_xy + s_shift)
                    r_start, r_end = max(0, -s_shift), min(dim_A_xy, dim_A_xy - s_shift)
                    if l_end - l_start == 0:
                        continue
                    score = np.sum(local_norm[:, l_start:l_end] * temp[:, r_start:r_end]) / norm_div
                    if score > best_overall_score:
                        best_overall_score, best_angle, best_shift = score, angles[idx], s_shift

            twist_map[y, x], shift_map[y, x], score_map[y, x] = best_angle, best_shift, best_overall_score
            curr += 1
        if on_progress:
            on_progress(20 + int(80 * curr / total), f"Mapping Twist: Column {x + 1}/{nX}")

    return twist_map, shift_map, score_map, "Azimuthal Twist (In-Plane)"


def run_coupled_azimuth_tilt(xy_data, ref_data, gamma_s, gamma_d, max_shift=20, max_tilt=15, on_progress=None):
    if on_progress:
        on_progress(5, "Aligning Physical Energy Axes...")
    E_xy, E_ref = xy_data['E'].copy(), ref_data['E'].copy()
    if np.mean(E_ref) > 0 and np.mean(E_xy) < 0:
        E_ref = -E_ref
    elif np.mean(E_xy) > 0 and np.mean(E_ref) < 0:
        E_xy = -E_xy

    e_min, e_max = max(np.min(E_xy), np.min(E_ref)), min(np.max(E_xy), np.max(E_ref))
    if e_min >= e_max:
        raise ValueError("No energy overlap!")
    xy_e_mask = (E_xy >= e_min) & (E_xy <= e_max)
    xy_val = xy_data['value'][xy_e_mask, :, :, :]
    dim_E, dim_A_xy, nY, nX = xy_val.shape

    ref_squeezed = ref_data['value'].squeeze(axis=2)
    A_arr, D_arr = ref_data['angle'], ref_data['x']
    A_scale = abs(A_arr[-1] - A_arr[0]) / max(1, len(A_arr) - 1) or 1.0
    D_scale = abs(D_arr[-1] - D_arr[0]) / max(1, len(D_arr) - 1) or 1.0
    aspect_ratio = A_scale / D_scale

    sort_idx = np.argsort(E_ref)
    f_interp = interp1d(E_ref[sort_idx], ref_squeezed[sort_idx, :, :], axis=0, bounds_error=False, fill_value=0)
    ref_matched = f_interp(E_xy[xy_e_mask])

    if on_progress:
        on_progress(10, "Precomputing Coupled Matrix (Rotations + Tilts)...")
    angles = np.arange(0, 360, 1.0)
    tilts = np.arange(-max_tilt, max_tilt + 1)
    templates = [[None for _ in range(len(tilts))] for _ in range(360)]
    ds = np.arange(dim_A_xy) - (dim_A_xy // 2)
    E_coords = np.arange(dim_E)
    total_temps, curr_temp = 360 * len(tilts), 0

    for a_idx, theta_deg in enumerate(angles):
        theta = np.deg2rad(theta_deg)
        for t_idx, dd in enumerate(tilts):
            slit_coords = gamma_s + ds * np.cos(theta) + dd * np.sin(theta) / aspect_ratio
            defl_coords = gamma_d - ds * aspect_ratio * np.sin(theta) + dd * np.cos(theta)

            EE, SS = np.meshgrid(E_coords, slit_coords, indexing='ij')
            _, DD = np.meshgrid(E_coords, defl_coords, indexing='ij')

            template = map_coordinates(ref_matched, np.stack([EE, SS, DD]), order=1, mode='constant', cval=0.0)
            t_mean, t_std = np.mean(template), np.std(template)
            if t_std > 0:
                template = (template - t_mean) / t_std
            templates[a_idx][t_idx] = template
            curr_temp += 1
        if a_idx % 36 == 0 and on_progress:
            on_progress(10 + int(20 * curr_temp / total_temps), "Precomputing Coupled Matrix...")

    twist_map, tilt_map, score_map = np.zeros((nY, nX)), np.zeros((nY, nX)), np.zeros((nY, nX))
    norm_div, total, curr = dim_E * dim_A_xy, nX * nY, 0

    for x in range(nX):
        for y in range(nY):
            local_cut = xy_val[:, :, y, x]
            l_mean, l_std = np.mean(local_cut), np.std(local_cut)
            if l_std == 0:
                continue
            local_norm = (local_cut - l_mean) / l_std

            best_c_score, best_c_a, best_c_t, best_c_s = -np.inf, 0, 0, 0
            for a in range(0, 360, 5):
                for t_idx in range(0, len(tilts), 3):
                    temp = templates[a][t_idx]
                    for s_shift in range(-max_shift, max_shift + 1, 4):
                        l_start, l_end = max(0, s_shift), min(dim_A_xy, dim_A_xy + s_shift)
                        r_start, r_end = max(0, -s_shift), min(dim_A_xy, dim_A_xy - s_shift)
                        if l_end - l_start == 0:
                            continue
                        score = np.sum(local_norm[:, l_start:l_end] * temp[:, r_start:r_end]) / norm_div
                        if score > best_c_score:
                            best_c_score, best_c_a, best_c_t, best_c_s = score, a, t_idx, s_shift

            best_overall_score, best_angle, best_tilt = best_c_score, angles[best_c_a], tilts[best_c_t]
            for offset_a in [-4, -3, -2, -1, 1, 2, 3, 4]:
                idx_a = (best_c_a + offset_a) % 360
                for offset_t in [-2, -1, 1, 2]:
                    idx_t = best_c_t + offset_t
                    if idx_t < 0 or idx_t >= len(tilts):
                        continue
                    temp = templates[idx_a][idx_t]
                    for offset_s in [-3, -2, -1, 1, 2, 3]:
                        s_shift = best_c_s + offset_s
                        if s_shift < -max_shift or s_shift > max_shift:
                            continue
                        l_start, l_end = max(0, s_shift), min(dim_A_xy, dim_A_xy + s_shift)
                        r_start, r_end = max(0, -s_shift), min(dim_A_xy, dim_A_xy - s_shift)
                        if l_end - l_start == 0:
                            continue
                        score = np.sum(local_norm[:, l_start:l_end] * temp[:, r_start:r_end]) / norm_div
                        if score > best_overall_score:
                            best_overall_score, best_angle, best_tilt = score, angles[idx_a], tilts[idx_t]

            twist_map[y, x], tilt_map[y, x], score_map[y, x] = best_angle, best_tilt, best_overall_score
            curr += 1
        if on_progress:
            on_progress(30 + int(70 * curr / total), f"Mapping Coupled Physics: Column {x + 1}/{nX}")

    return twist_map, tilt_map, score_map, "Coupled Azimuth & Deflection Tilt"


def run_normal_tilt(xy_data, ref_data, gamma_s, gamma_d, max_shift=30, on_progress=None):
    if on_progress:
        on_progress(5, "Aligning Physical Energy Axes...")
    E_xy, E_ref = xy_data['E'].copy(), ref_data['E'].copy()
    if np.mean(E_ref) > 0 and np.mean(E_xy) < 0:
        E_ref = -E_ref
    elif np.mean(E_xy) > 0 and np.mean(E_ref) < 0:
        E_xy = -E_xy

    e_min, e_max = max(np.min(E_xy), np.min(E_ref)), min(np.max(E_xy), np.max(E_ref))
    if e_min >= e_max:
        raise ValueError("No energy overlap!")
    xy_e_mask = (E_xy >= e_min) & (E_xy <= e_max)
    xy_val = xy_data['value'][xy_e_mask, :, :, :]
    dim_E, dim_A, nY, nX = xy_val.shape

    ref_squeezed = ref_data['value'].squeeze(axis=2)
    ref_A_dim, ref_D_dim = ref_squeezed.shape[1], ref_squeezed.shape[2]

    sort_idx = np.argsort(E_ref)
    f_interp = interp1d(E_ref[sort_idx], ref_squeezed[sort_idx, :, :], axis=0, bounds_error=False, fill_value=0)
    ref_matched = f_interp(E_xy[xy_e_mask])

    if on_progress:
        on_progress(15, "Normalizing Reference Volume...")
    ref_norm = np.zeros_like(ref_matched)
    for d in range(ref_D_dim):
        sl = ref_matched[:, :, d]
        m, s = np.mean(sl), np.std(sl)
        if s > 0:
            ref_norm[:, :, d] = (sl - m) / s

    defl_tilt_map, slit_tilt_map, score_map = np.zeros((nY, nX)), np.zeros((nY, nX)), np.zeros((nY, nX))
    total, curr = nX * nY, 0

    for x in range(nX):
        for y in range(nY):
            local_cut = xy_val[:, :, y, x]
            l_mean, l_std = np.mean(local_cut), np.std(local_cut)
            if l_std == 0:
                continue
            local_norm = (local_cut - l_mean) / l_std

            best_score, best_d, best_s = -np.inf, 0, 0
            for d in range(ref_D_dim):
                for s_shift in range(-max_shift, max_shift + 1):
                    offset = gamma_s + s_shift - (dim_A // 2)
                    l_start, l_end = max(0, -offset), min(dim_A, ref_A_dim - offset)
                    if l_end <= l_start:
                        continue
                    r_start, r_end = l_start + offset, l_end + offset
                    score = np.sum(local_norm[:, l_start:l_end] * ref_norm[:, r_start:r_end, d]) / (dim_E * dim_A)
                    if score > best_score:
                        best_score, best_d, best_s = score, d, s_shift

            defl_tilt_map[y, x], slit_tilt_map[y, x], score_map[y, x] = best_d - gamma_d, best_s, best_score
            curr += 1
        if on_progress:
            on_progress(20 + int(80 * curr / total), f"Mapping Normal Tilt: Column {x + 1}/{nX}")

    return defl_tilt_map, slit_tilt_map, score_map, "Surface Normal Tilt (Out-of-Plane)"
