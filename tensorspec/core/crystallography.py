# File: tensorspec/core/crystallography.py
import os
import json
import warnings
import numpy as np
from scipy.spatial import ConvexHull, KDTree
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.surface import SlabGenerator


class CrystalEngine:
    """
    Core mathematical engine for crystallographic transformations, 
    supercell generation, CDW distortion propagation, and 2D heterostructure stacking.
    """

    @staticmethod
    def get_universal_primitive_matrix(structure: Structure) -> np.ndarray:
        """Universally calculates the primitive lattice matrix without Cartesian rotation."""
        try:
            sga = SpacegroupAnalyzer(structure)
            centering = sga.get_space_group_symbol()[0].upper()
            
            # Universal Transformation Matrices for all 3D Bravais Lattices
            if centering == 'F': P = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
            elif centering == 'I': P = np.array([[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]])
            elif centering == 'A': P = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, -0.5], [0.0, 0.5, 0.5]])
            elif centering == 'B': P = np.array([[0.5, 0.0, -0.5], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5]])
            elif centering == 'C': P = np.array([[0.5, 0.5, 0.0], [-0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
            elif centering == 'R': P = np.array([[2/3, 1/3, 1/3], [-1/3, 1/3, 1/3], [-1/3, -2/3, 1/3]])
            else: P = np.eye(3)
            
            return np.dot(P, structure.lattice.matrix)
        except Exception:
            return structure.lattice.matrix

    @staticmethod
    def get_symmetry_info(structure: Structure) -> dict:
        """Returns space group symbol and conventional-to-primitive volume ratio."""
        try:
            sga = SpacegroupAnalyzer(structure)
            spg_symbol = sga.get_space_group_symbol()
            conv_vol = structure.lattice.volume
            
            P = sga.get_conventional_to_primitive_transformation_matrix()
            prim_matrix = np.dot(P, structure.lattice.matrix)
            prim_vol = abs(np.linalg.det(prim_matrix))
            ratio = int(round(conv_vol / prim_vol))
            return {"spacegroup": spg_symbol, "volume_ratio": ratio}
        except Exception:
            return {"spacegroup": "N/A", "volume_ratio": 1}

    @staticmethod
    def compute_bonds(structure: Structure, thresh_multiplier: float = 1.15,
                      search_radius: float = 4.0, max_atoms: int = 500000) -> dict:
        """
        Determines which atom pairs are bonded using a KDTree neighbour search
        gated by summed atomic radii. Pure geometry: returns index pairs and
        separations so any renderer (PyVista, three.js, exporters) can draw them.

        Returns arrays i, j, distances, and vectors (j minus i), all aligned.
        Empty arrays mean no bonds, including when the cell exceeds max_atoms.
        """
        empty = {"i": np.array([], dtype=int), "j": np.array([], dtype=int),
                 "distances": np.array([]), "vectors": np.empty((0, 3))}

        if len(structure) > max_atoms:
            return empty

        coords = structure.cart_coords
        radii = np.array([s.specie.atomic_radius if s.specie.atomic_radius else 1.2 for s in structure])

        pairs = KDTree(coords).query_pairs(r=search_radius)
        if not pairs:
            return empty

        pairs_arr = np.array(list(pairs))
        i_idx, j_idx = pairs_arr[:, 0], pairs_arr[:, 1]
        vecs = coords[j_idx] - coords[i_idx]
        dists = np.linalg.norm(vecs, axis=1)

        rad_sums = (radii[i_idx] + radii[j_idx]) * thresh_multiplier
        mask = (dists > 0.5) & (dists <= rad_sums)

        return {"i": i_idx[mask], "j": j_idx[mask],
                "distances": dists[mask], "vectors": vecs[mask]}

    @staticmethod
    def apply_cdw_distortion(supercell: Structure, target_el: str, q_vec: tuple, amp_vec: tuple, phase_deg: float) -> Structure:
        """Propagates a Charge Density Wave (CDW) sinusoidal atomic displacement across the lattice."""
        qx, qy, qz = q_vec
        ax, ay, az = amp_vec
        phase = np.radians(phase_deg)
        
        base_frac_coords = supercell.frac_coords.copy()
        new_cart = supercell.cart_coords.copy()
        
        for i, site in enumerate(supercell):
            if target_el == "All Elements" or site.specie.symbol == target_el:
                fx, fy, fz = base_frac_coords[i]
                wave_arg = 2 * np.pi * (qx * fx + qy * fy + qz * fz) + phase
                dx = ax * np.cos(wave_arg)
                dy = ay * np.cos(wave_arg)
                dz = az * np.cos(wave_arg)
                new_cart[i] += np.array([dx, dy, dz])
                
        for i, site in enumerate(supercell):
            supercell.replace(i, site.specie, new_cart[i], coords_are_cartesian=True)
            
        return supercell

    @staticmethod
    def generate_template_structure(template_name: str, db_path: str = "user_templates.json") -> Structure:
        """Generates standard 2D monolayer and stacked structures with vacuum padding."""
        c_vac = 25.0
        
        if "Graphene" in template_name:
            lat = Lattice.hexagonal(2.46, c_vac)
            stacks = {
                "Graphene (Monolayer)": ["A"],
                "Graphene (AB Bilayer)": ["A", "B"],
                "Graphene (AA Bilayer)": ["A", "A"],
                "Graphene (ABA Trilayer)": ["A", "B", "A"],
                "Graphene (ABC 3-Layer)": ["A", "B", "C"],
                "Graphene (ABCA 4-Layer)": ["A", "B", "C", "A"],
                "Graphene (ABCAB 5-Layer)": ["A", "B", "C", "A", "B"],
                "Graphene (ABCABC 6-Layer)": ["A", "B", "C", "A", "B", "C"],
                "Graphene (ABCABCA 7-Layer)": ["A", "B", "C", "A", "B", "C", "A"]
            }
            seq = stacks.get(template_name, ["A"])
            coords = []
            z_spacing = 3.35 / c_vac
            for i, layer in enumerate(seq):
                z = 0.5 + (i * z_spacing)
                if layer == "A": coords.extend([[1/3, 2/3, z], [2/3, 1/3, z]])
                elif layer == "B": coords.extend([[0.0, 0.0, z], [1/3, 2/3, z]])
                elif layer == "C": coords.extend([[0.0, 0.0, z], [2/3, 1/3, z]])
            return Structure(lat, ["C"] * len(coords), coords)
            
        elif "(1T')" in template_name and ("TaIrTe4" in template_name or "NbIrTe4" in template_name):
            is_ta = "TaIrTe4" in template_name
            lat = Lattice.orthorhombic(3.808 if is_ta else 3.78, 12.605 if is_ta else 12.4, c_vac)
            m1 = "Ta" if is_ta else "Nb"
            species = ["Te", "Te", m1, "Te", "Te", "Ir", "Te", "Te", "Ir", "Te", "Te", m1]
            coords = [
                [0.0, 0.0657, 0.4424], [0.5, 0.1518, 0.5591], [0.0, 0.2691, 0.4963], [0.5, 0.3229, 0.4206], 
                [0.0, 0.4120, 0.5741], [0.5, 0.4655, 0.5011], [0.0, 0.5650, 0.4463], [0.5, 0.6533, 0.5526], 
                [0.0, 0.7531, 0.4981], [0.5, 0.8076, 0.4254], [0.0, 0.8933, 0.5800], [0.5, 0.9477, 0.5044], 
            ]
            return Structure(lat, species, coords)
            
        elif template_name == "h-BN":
            lat = Lattice.hexagonal(2.50, c_vac)
            return Structure(lat, ["B", "N"], [[1/3, 2/3, 0.5], [2/3, 1/3, 0.5]])
            
        elif template_name == "MoS2":
            lat = Lattice.hexagonal(3.16, c_vac)
            z_off = 1.56 / c_vac
            return Structure(lat, ["Mo", "S", "S"], [[1/3, 2/3, 0.5], [2/3, 1/3, 0.5 + z_off], [2/3, 1/3, 0.5 - z_off]])
            
        elif template_name == "WSe2":
            lat = Lattice.hexagonal(3.28, c_vac)
            z_off = 1.66 / c_vac
            return Structure(lat, ["W", "Se", "Se"], [[1/3, 2/3, 0.5], [2/3, 1/3, 0.5 + z_off], [2/3, 1/3, 0.5 - z_off]])

        if os.path.exists(db_path):
            with open(db_path, "r") as f:
                custom_templates = json.load(f)
            if template_name in custom_templates:
                return Structure.from_dict(custom_templates[template_name])
        return None

    @staticmethod
    def extract_monolayer_vdw(bulk_struct: Structure) -> tuple[Structure, float]:
        """Automatically detects maximal van der Waals gap and cleaves along Z."""
        bulk_struct.translate_sites(list(range(len(bulk_struct))), [0.01, 0.01, 0.01], to_unit_cell=True)
        sites_sorted = sorted(bulk_struct, key=lambda s: s.frac_coords[2])
        z_fracs = [s.frac_coords[2] for s in sites_sorted]
        n_sites = len(z_fracs)
        if n_sites < 2:
            raise ValueError("Structure has too few atoms to evaluate spacing differences.")

        gaps = []
        for i in range(n_sites):
            diff = z_fracs[i+1] - z_fracs[i] if i < n_sites - 1 else (z_fracs[0] + 1.0) - z_fracs[i]
            gaps.append((diff, i))

        max_gap_diff, split_index = max(gaps, key=lambda x: x[0])
        gap_angstrom = max_gap_diff * bulk_struct.lattice.c

        monolayer_sites = []
        for idx, site in enumerate(sites_sorted):
            shifted_z = site.frac_coords[2] + (1.0 if idx <= split_index else 0.0)
            monolayer_sites.append((site, shifted_z))

        monolayer_sites = sorted(monolayer_sites, key=lambda x: x[1])
        final_sites = [item[0] for item in monolayer_sites[:-1]] if n_sites > 1 else [item[0] for item in monolayer_sites]
        
        new_matrix = bulk_struct.lattice.matrix.copy()
        new_matrix[2] = [0, 0, 25.0]
        new_lat = Lattice(new_matrix)
        
        cart_coords = [s.coords for s in final_sites]
        z_vals = [c[2] for c in cart_coords]
        z_center = (max(z_vals) + min(z_vals)) / 2.0
        final_coords = [[c[0], c[1], c[2] - z_center + 12.5] for c in cart_coords]
        
        mono_struct = Structure(new_lat, [s.specie for s in final_sites], final_coords, coords_are_cartesian=True)
        return mono_struct, gap_angstrom

    @staticmethod
    def extract_monolayer_miller(
        bulk_struct: Structure,
        hkl: tuple,
        num_layers: int = 1,
        vacuum: float = 15.0,
        bond_threshold: float = 3.2,
    ) -> Structure:
        """Cleaves along Miller indices [h k l], keeping complete covalent layers.

        Builds an oversized slab, clusters atoms by a distance graph (BFS), drops
        surface-dust fragments, and keeps ``num_layers`` intact sandwiches with
        the requested vacuum along c. Matches the desktop Tab 3 behaviour.
        """
        if hkl == (0, 0, 0):
            raise ValueError("Miller index [0 0 0] is not a valid cleavage plane.")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        vacuum = float(max(0.0, vacuum))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bulk = bulk_struct.copy()
            bulk.translate_sites(list(range(len(bulk))), [0.01, 0.01, 0.01], to_unit_cell=True)
            # Oversized slab so complete molecules survive the cut.
            safe_thickness = max(30.0, num_layers * 15.0)
            slabgen = SlabGenerator(
                bulk,
                miller_index=hkl,
                min_slab_size=safe_thickness,
                min_vacuum_size=max(vacuum, 1.0),
                center_slab=True,
            )
            slabs = slabgen.get_slabs()
            if not slabs:
                raise ValueError(f"Could not generate slabs for plane {hkl}.")

        raw_slab = slabs[0]
        dist_mat = raw_slab.distance_matrix
        num_sites = len(raw_slab)
        visited: set[int] = set()
        all_clusters: list[list[int]] = []

        for i in range(num_sites):
            if i in visited:
                continue
            queue = [i]
            current: list[int] = []
            while queue:
                curr = queue.pop(0)
                if curr in visited:
                    continue
                visited.add(curr)
                current.append(curr)
                neighbors = np.where(
                    (dist_mat[curr] > 0.1) & (dist_mat[curr] < bond_threshold)
                )[0]
                queue.extend(int(n) for n in neighbors if n not in visited)
            all_clusters.append(current)

        if not all_clusters:
            raise ValueError("No bonded clusters found in the cleaved slab.")

        max_size = max(len(c) for c in all_clusters)
        valid_layers = [c for c in all_clusters if len(c) > max_size * 0.75]
        valid_layers.sort(
            key=lambda indices: float(np.mean([raw_slab[idx].coords[2] for idx in indices]))
        )
        if not valid_layers:
            raise ValueError("No complete layers survived the surface-dust filter.")

        target = valid_layers[: min(num_layers, len(valid_layers))]
        sites_to_keep = [raw_slab[idx] for indices in target for idx in indices]
        if not sites_to_keep:
            raise ValueError("Exfoliation produced an empty structure.")

        coords = [s.coords for s in sites_to_keep]
        z_vals = [c[2] for c in coords]
        slab_thickness = max(z_vals) - min(z_vals)
        total_c = slab_thickness + vacuum
        if total_c <= 0:
            total_c = vacuum if vacuum > 0 else 25.0

        new_matrix = raw_slab.lattice.matrix.copy()
        new_matrix[2] = [0.0, 0.0, total_c]
        new_lat = Lattice(new_matrix)
        z_center = (max(z_vals) + min(z_vals)) / 2.0
        final_coords = [[c[0], c[1], c[2] - z_center + (total_c / 2.0)] for c in coords]
        return Structure(
            new_lat,
            [s.specie for s in sites_to_keep],
            final_coords,
            coords_are_cartesian=True,
        )

    @staticmethod
    def build_heterostructure_stack(layers_data: list[dict], *, vacuum: float = 20.0, pad_xy: float = 3.0) -> Structure:
        """
        Rotates, shifts, and combines multiple 2D structures into a unified stacked supercell.

        Each dictionary in ``layers_data`` must contain:
        ``{'struct': Structure, 'sc_x': int, 'sc_y': int, 'z_shift': float, 'twist': float}``.

        The result uses a tight orthorhombic cell with vacuum (DFT-ready), not a
        500 Å dummy cube.
        """
        all_species, all_coords, layer_tags = [], [], []

        for idx, l in enumerate(layers_data):
            supercell = l['struct'] * (l['sc_x'], l['sc_y'], 1)
            theta = np.radians(l['twist'])
            rot_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [            0,              0, 1]
            ])

            coords = supercell.cart_coords.copy()
            center_xy = np.mean(coords[:, :2], axis=0)
            coords[:, :2] -= center_xy

            rotated_coords = np.dot(coords, rot_matrix.T)
            shifted_coords = rotated_coords + np.array([0, 0, l['z_shift']])
            layer_tag = f"_L{idx + 1}"

            for i, site in enumerate(supercell):
                all_species.append(site.specie.symbol)
                all_coords.append(shifted_coords[i])
                layer_tags.append(f"{site.specie.symbol}{layer_tag}")

        return CrystalEngine.structure_from_cartesian(
            all_species,
            all_coords,
            site_properties={"layer_tag": layer_tags},
            vacuum=vacuum,
            pad_xy=pad_xy,
        )

    @staticmethod
    def structure_from_cartesian(
        species,
        coords,
        *,
        site_properties=None,
        vacuum: float = 20.0,
        pad_xy: float = 3.0,
    ) -> Structure:
        """Build an orthorhombic cell around Cartesian coordinates with vacuum."""
        coords = np.asarray(coords, dtype=float)
        if coords.size == 0:
            raise ValueError("No atomic coordinates to pack into a cell.")
        mins = coords.min(axis=0)
        shift = np.array([
            pad_xy - mins[0],
            pad_xy - mins[1],
            max(vacuum * 0.5, 1.0) - mins[2],
        ])
        shifted = coords + shift
        maxs = shifted.max(axis=0)
        a = float(maxs[0] + pad_xy)
        b = float(maxs[1] + pad_xy)
        c = float(maxs[2] + max(vacuum * 0.5, 1.0))
        # Guard against degenerate in-plane extent (single atom column)
        a = max(a, 5.0)
        b = max(b, 5.0)
        c = max(c, 10.0)
        lattice = Lattice.from_parameters(a, b, c, 90.0, 90.0, 90.0)
        return Structure(
            lattice,
            species,
            shifted,
            coords_are_cartesian=True,
            site_properties=site_properties or {},
        )

    @staticmethod
    def calculate_moire_superlattice(layer1_struct: Structure, layer2_struct: Structure, twist1: float, twist2: float) -> dict:
        """Calculates 2D Moiré superlattice periodicity and checks commensurability."""
        m1 = layer1_struct.lattice.matrix[:2, :2]
        m2 = layer2_struct.lattice.matrix[:2, :2]
        theta = np.radians(twist2 - twist1)
        
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        m2_rot = np.dot(m2, rot_matrix.T)

        try:
            g1 = np.linalg.inv(m1).T
            g2_rot = np.linalg.inv(m2_rot).T
        except np.linalg.LinAlgError:
            return {"status": "error", "message": "Degenerate 2D lattice detected."}

        delta_g = g1 - g2_rot
        try:
            m_moire = np.linalg.inv(delta_g).T
        except np.linalg.LinAlgError:
            return {"status": "perfect_alignment", "message": "Twist is 0° with identical lattices. No Moiré pattern."}

        S_matrix = np.dot(m_moire, np.linalg.inv(m1))
        S_round = np.round(S_matrix)
        is_commensurate = np.all(np.abs(S_matrix - S_round) < 0.05)
        periodicity = float(np.linalg.norm(m_moire[0]))

        if is_commensurate:
            n_cells = int(np.round(np.sqrt(np.abs(np.linalg.det(S_round)))))
            return {"status": "commensurate", "periodicity": periodicity, "n_cells": n_cells, "matrix": m_moire}
        else:
            return {"status": "incommensurate", "periodicity": periodicity, "matrix": m_moire}

    @staticmethod
    def calculate_brillouin_zone(structure: Structure) -> dict:
        """Calculates the 3D Wigner-Seitz cell of the primitive reciprocal lattice."""
        prim_matrix = CrystalEngine.get_universal_primitive_matrix(structure)
        prim_lat = Lattice(prim_matrix)
        recip_lat = prim_lat.reciprocal_lattice
        faces = recip_lat.get_wigner_seitz_cell()
        
        all_pts = []
        export_edges = set()
        for face in faces:
            for pt in face:
                all_pts.append(pt)
            for i in range(len(face)):
                p1 = tuple(np.round(face[i], 5))
                p2 = tuple(np.round(face[(i+1)%len(face)], 5))
                export_edges.add(tuple(sorted([p1, p2])))

        if not all_pts: return {}
        all_pts = np.array(all_pts)
        hull = ConvexHull(all_pts)
        
        return {
            "points": all_pts,
            "hull_points": hull.points,
            "simplices": hull.simplices,
            "edges": export_edges
        }
    
    @staticmethod
    def calculate_surface_projection(bz_points, structure, h, k, l):
        """Projects 3D BZ vertices onto a specific 2D Miller plane and extracts the 3D silhouette."""
        from scipy.spatial import ConvexHull
        import numpy as np
        
        recip_matrix = structure.lattice.reciprocal_lattice.matrix
        normal = h * recip_matrix[0] + k * recip_matrix[1] + l * recip_matrix[2]
        dist = np.linalg.norm(normal)
        if dist == 0: return None
        normal = normal / dist
        
        # Project points onto the plane
        proj_pts = []
        for pt in bz_points:
            d = np.dot(pt, normal)
            proj_pts.append(pt - d * normal)
        proj_pts = np.array(proj_pts)
        
        # Create a local 2D basis
        u = np.cross(normal, [0, 0, 1])
        if np.linalg.norm(u) < 1e-5: u = np.cross(normal, [0, 1, 0])
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        
        pts_2d = np.array([[np.dot(p, u), np.dot(p, v)] for p in proj_pts])
        pts_2d = np.round(pts_2d, decimals=5)
        unique_pts_2d, indices = np.unique(pts_2d, axis=0, return_index=True)
        
        if len(unique_pts_2d) < 3: return None
        
        surf_hull = ConvexHull(unique_pts_2d)
        
        # Extract the exact 3D vertices that form the outer boundary (silhouette)
        hull_indices = indices[surf_hull.vertices]
        surf_vertices_3d = proj_pts[hull_indices]
        silhouette_3d = bz_points[hull_indices]
        
        # Triangulate for rendering
        center = np.mean(surf_vertices_3d, axis=0)
        surf_pts_with_center = np.vstack([surf_vertices_3d, center])
        c_idx = len(surf_pts_with_center) - 1
        
        surf_simplices = []
        n_v = len(surf_vertices_3d)
        for i in range(n_v):
            surf_simplices.append([i, (i+1)%n_v, c_idx])
            
        return {
            "normal": normal.tolist(),
            "origin_plane": surf_pts_with_center.tolist(),
            "simplices": surf_simplices,
            "silhouette_3d": silhouette_3d.tolist(),
            "projected_bounds": surf_vertices_3d.tolist()
        }

    @staticmethod
    def get_hkl_surface_frame(hkl, recip_matrix, azimuthal_ref=None):
        """
        Calculates the orthogonal surface basis for a given (h, k, l) cleavage plane.
        Returns the normal vector (z-axis) and the up vector (y-axis) for camera or simulation alignment.
        """
        import numpy as np
        h, k, l = hkl
        
        # 1. Calculate surface normal
        normal = h * recip_matrix[0] + k * recip_matrix[1] + l * recip_matrix[2]
        dist = np.linalg.norm(normal)
        if dist == 0:
            return np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])
        normal = normal / dist
        
        # 2. Calculate the azimuthal 'up' vector
        if azimuthal_ref is not None:
            up_vector = azimuthal_ref - np.dot(azimuthal_ref, normal) * normal
        else:
            # --- FIX: Lock the default azimuthal frame to the reciprocal lattice ---
            b_star = recip_matrix[1]
            a_star = recip_matrix[0]
            
            # Use b* as the default "up" direction. If the surface normal is parallel to b*, fallback to a*.
            b_star_norm = b_star / np.linalg.norm(b_star)
            default_up = b_star if abs(np.dot(b_star_norm, normal)) < 0.99 else a_star
            
            up_vector = default_up - np.dot(default_up, normal) * normal
            
        up_vector = up_vector / np.linalg.norm(up_vector)
        
        return normal, up_vector