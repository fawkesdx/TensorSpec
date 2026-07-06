# File: tensorspec/plotting/backends/pyvista_engine.py
import os
import numpy as np
from scipy.spatial import KDTree
import pyvista as pv
from pyvistaqt import QtInteractor

# Ensure standard cross-platform OpenGL rendering flags
pv.global_theme.depth_peeling.enabled = False
pv.global_theme.anti_aliasing = "fxaa"
os.environ["QSG_RHI_BACKEND"] = "opengl"


class PyVistaCrystalBackend:
    """
    Dedicated GPU-accelerated 3D rendering backend for TensorSpec.
    Manages VTK scene graphs, PBR materials, adaptive LOD glyphing, and spatial picking.
    """
    def __init__(self, parent=None):
        # Initialize the hardware QtInteractor canvas
        self.plotter = QtInteractor(parent=parent)
        self.plotter.set_background("#1e1e24")
        
        # Internal state tracking for fast eraser picking and toggles
        self.atom_tree = None
        self.bond_tree = None
        self.bond_pairs_list = []
        self.axis_actors = []
        self.box_actors = []
        self.bz_actors = []
        self.moire_actors = []
        self.plane_actor = None

    def clear_scene(self):
        """Wipes the 3D viewport while preserving background styling."""
        self.plotter.clear()
        self.plotter.set_background("#1e1e24")

    # ================= ATOM & BOND RENDERERS =================
    def draw_atoms(self, supercell, active_colors: dict, scale_mod: float = 0.5, is_shiny: bool = False, erased_atoms: set = None):
        """Instantly renders thousands of atoms using hardware GPU glyphing and adaptive LOD."""
        if erased_atoms is None: erased_atoms = set()
        num_atoms = len(supercell)
        
        # Adaptive Level of Detail to protect VRAM on massive supercells
        if num_atoms > 8000: res = 8
        elif num_atoms > 2000: res = 12
        else: res = 40 if is_shiny else 20

        from collections import defaultdict
        atom_coords = defaultdict(list)
        atom_radii = {}

        self.atom_tree = KDTree(supercell.cart_coords)

        for i, site in enumerate(supercell):
            if i in erased_atoms: continue
            tag = supercell.site_properties.get("layer_tag", [site.specie.symbol] * num_atoms)[i]
            atom_coords[tag].append(site.coords)
            atom_radii[tag] = (site.specie.atomic_radius if site.specie.atomic_radius else 1.0) * scale_mod

        for tag, coords in atom_coords.items():
            if not coords: continue
            color = active_colors.get(tag, "#008080")
            rad = atom_radii[tag]

            poly = pv.PolyData(np.array(coords))
            sphere = pv.Sphere(radius=rad, theta_resolution=res, phi_resolution=res)
            glyphs = poly.glyph(geom=sphere, factor=1.0)

            actor = self.plotter.add_mesh(glyphs, color=color, smooth_shading=True, pickable=False, render=False)
            if is_shiny and hasattr(actor, 'prop'):
                actor.prop.interpolation = 'pbr'
                actor.prop.metallic = 0.2
                actor.prop.roughness = 0.15

    def draw_bonds(self, supercell, active_colors: dict, cyl_radius: float = 0.1, thresh_multiplier: float = 1.15, is_shiny: bool = False, erased_bonds: set = None, erased_atoms: set = None):
        """Vectorized C-level bond distance calculation and GPU cylinder glyphing."""
        if len(supercell) > 20000:
            self.bond_tree = None
            return

        if erased_bonds is None: erased_bonds = set()
        if erased_atoms is None: erased_atoms = set()

        coords = supercell.cart_coords
        radii = np.array([s.specie.atomic_radius if s.specie.atomic_radius else 1.2 for s in supercell])
        
        tree = KDTree(coords)
        pairs = tree.query_pairs(r=4.0)
        if not pairs:
            self.bond_tree = None
            return

        pairs_arr = np.array(list(pairs))
        i_idx, j_idx = pairs_arr[:, 0], pairs_arr[:, 1]
        vecs = coords[j_idx] - coords[i_idx]
        dists = np.linalg.norm(vecs, axis=1)
        
        rad_sums = (radii[i_idx] + radii[j_idx]) * thresh_multiplier
        mask = (dists > 0.5) & (dists <= rad_sums)
        
        valid_i, valid_j = i_idx[mask], j_idx[mask]
        valid_dists, valid_vecs = dists[mask], vecs[mask]

        bond_centers, bond_directions, bond_heights = [], [], []
        self.bond_pairs_list = []

        for idx in range(len(valid_i)):
            i, j = valid_i[idx], valid_j[idx]
            if (i, j) in erased_bonds or (j, i) in erased_bonds or i in erased_atoms or j in erased_atoms:
                continue
            bond_centers.append(coords[i] + valid_vecs[idx] / 2.0)
            bond_directions.append(valid_vecs[idx] / valid_dists[idx])
            bond_heights.append(valid_dists[idx])
            self.bond_pairs_list.append((i, j))

        if not bond_centers:
            self.bond_tree = None
            return

        self.bond_tree = KDTree(bond_centers)
        bonds_data = pv.PolyData(np.array(bond_centers))
        bonds_data["scale"] = np.array(bond_heights)
        bonds_data["vectors"] = np.array(bond_directions)

        res = 20 if is_shiny else 8
        source_cyl = pv.Cylinder(center=(0, 0, 0), direction=(1, 0, 0), radius=cyl_radius, height=1.0, resolution=res)
        glyphs = bonds_data.glyph(orient="vectors", scale="scale", geom=source_cyl)

        bond_color = active_colors.get("Bonds", "#FFFFFF")
        actor = self.plotter.add_mesh(glyphs, color=bond_color, smooth_shading=True, pickable=False, render=False)
        if is_shiny and hasattr(actor, 'prop'):
            actor.prop.interpolation = 'pbr'
            actor.prop.metallic = 0.2
            actor.prop.roughness = 0.15

    # ================= LATTICE & AXES RENDERERS =================
    def draw_axes(self, conventional_matrix: np.ndarray = None, primitive_matrix: np.ndarray = None):
        """Draws crystallographic directional arrows mapped to lattice matrices."""
        for actor in self.axis_actors: self.plotter.remove_actor(actor)
        self.axis_actors.clear()

        bounds = self.plotter.bounds
        diag = np.linalg.norm([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])
        origin = np.array([bounds[0], bounds[2], bounds[4]]) - (np.array([1, 1, 1]) * (diag * 0.05))
        arrow_scale = diag * 0.2

        def _draw_arrows(matrix, colors, labels):
            for vec, color, label in zip(matrix, colors, labels):
                length = np.linalg.norm(vec)
                if length == 0: continue
                direction = vec / length
                arrow = pv.Arrow(start=origin, direction=direction, scale=arrow_scale, shaft_radius=0.015, tip_radius=0.04)
                a_act = self.plotter.add_mesh(arrow, color=color, smooth_shading=True, pickable=False, render=False)
                t_act = self.plotter.add_point_labels([origin + direction * (arrow_scale * 1.15)], [label], text_color=color, font_size=20, shape_opacity=0.0, show_points=False, render=False)
                self.axis_actors.extend([a_act, t_act])

        if conventional_matrix is not None:
            _draw_arrows(conventional_matrix, ["red", "green", "blue"], ["a", "b", "c"])
        if primitive_matrix is not None:
            _draw_arrows(primitive_matrix, ["#ff6666", "#66ff66", "#6666ff"], ["a_prim", "b_prim", "c_prim"])

    def draw_lattice_boxes(self, conventional_matrix: np.ndarray = None, primitive_matrix: np.ndarray = None):
        """Draws slanted unit cell wireframe boundaries."""
        for actor in self.box_actors: self.plotter.remove_actor(actor)
        self.box_actors.clear()

        def _draw_slanted(matrix, color):
            origin = np.zeros(3)
            corners = [
                origin, origin + matrix[0], origin + matrix[1], origin + matrix[0] + matrix[1],
                origin + matrix[2], origin + matrix[0] + matrix[2], origin + matrix[1] + matrix[2], origin + matrix[0] + matrix[1] + matrix[2]
            ]
            edges = [(0,1), (0,2), (1,3), (2,3), (4,5), (4,6), (5,7), (6,7), (0,4), (1,5), (2,6), (3,7)]
            for p1, p2 in edges:
                line = pv.Line(corners[p1], corners[p2])
                actor = self.plotter.add_mesh(line, color=color, line_width=2, render=False)
                self.box_actors.append(actor)

        if conventional_matrix is not None: _draw_slanted(conventional_matrix, "#FF5733")
        if primitive_matrix is not None: _draw_slanted(primitive_matrix, "#33FF57")

    # ================= SPECIALIZED RECIPROCAL & MOIRÉ RENDERERS =================
    def draw_brillouin_zone(self, bz_points: np.ndarray, bz_simplices: np.ndarray, solid: bool = True):
        """Renders 3D Wigner-Seitz reciprocal cell and reciprocal axis vectors."""
        for actor in self.bz_actors: self.plotter.remove_actor(actor)
        self.bz_actors.clear()

        faces_pv = np.column_stack((np.full(len(bz_simplices), 3), bz_simplices)).flatten()
        bz_mesh = pv.PolyData(bz_points, faces_pv)

        if solid:
            actor = self.plotter.add_mesh(bz_mesh, color="#FF00FF", opacity=0.25, show_edges=True, edge_color="white", line_width=2, render=False)
        else:
            actor = self.plotter.add_mesh(bz_mesh, style="wireframe", color="#FF00FF", line_width=3, render=False)
        self.bz_actors.append(actor)

        origin = np.zeros(3)
        arrow_scale = np.max(bz_points) * 1.25
        for vec, color, label in zip(np.eye(3) * arrow_scale, ["#ff6666", "#66ff66", "#6666ff"], ["k_x", "k_y", "k_z"]):
            arrow = pv.Arrow(start=origin, direction=vec / np.linalg.norm(vec), scale=np.linalg.norm(vec), shaft_radius=0.015, tip_radius=0.04)
            a_act = self.plotter.add_mesh(arrow, color=color, render=False)
            l_act = self.plotter.add_point_labels([origin + vec * 1.15], [label], text_color=color, font_size=24, shape_opacity=0.0, show_points=False, render=False)
            self.bz_actors.extend([a_act, l_act])

    def draw_moire_envelope(self, m_moire: np.ndarray, z_min: float, z_max: float):
        """Renders a glowing golden bounding envelope around stacked 2D supercells."""
        for actor in self.moire_actors: self.plotter.remove_actor(actor)
        self.moire_actors.clear()

        v0, v1, v2 = np.zeros(3), np.array([m_moire[0,0], m_moire[0,1], 0]), np.array([m_moire[1,0], m_moire[1,1], 0])
        v3 = v1 + v2
        b0, b1, b2, b3 = [np.array([v[0], v[1], z_min]) for v in (v0, v1, v2, v3)]
        t0, t1, t2, t3 = [np.array([v[0], v[1], z_max]) for v in (v0, v1, v2, v3)]

        edges = [(b0,b1), (b0,b2), (b1,b3), (b2,b3), (t0,t1), (t0,t2), (t1,t3), (t2,t3), (b0,t0), (b1,t1), (b2,t2), (b3,t3)]
        for p1, p2 in edges:
            line = pv.Line(p1, p2)
            actor = self.plotter.add_mesh(line, color="#FFD700", line_width=4, render=False)
            self.moire_actors.append(actor)

    def set_camera_preset(self, preset: str):
        """Snaps hardware camera to quick isometric or crystallographic axes."""
        if preset == 'x': self.plotter.view_yz()
        elif preset == 'y': self.plotter.view_xz()
        elif preset == 'z': self.plotter.view_xy()
        elif preset == 'iso': self.plotter.view_isometric()
        self.plotter.render()