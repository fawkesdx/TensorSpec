# File: tensorspec/plotting/backends/matplotlib_engine.py
import matplotlib
matplotlib.use('qtagg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class MatplotlibCrystalBackend:
    """
    Crash-proof CPU rendering backend for legacy systems.
    Uses Matplotlib to simulate 3D graphics safely without VTK/OpenGL.
    """
    def __init__(self, parent=None):
        self.figure = Figure(facecolor="#1e1e24")
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.ax.set_facecolor("#1e1e24")
        self.ax.axis('off')
        
        # Duck-typing: Pretend to be a PyVista plotter so the GUI doesn't crash
        self.plotter = self.canvas
        self.plotter.render = self.canvas.draw_idle

    def clear_scene(self):
        self.ax.clear()
        self.ax.set_facecolor("#1e1e24")
        self.ax.axis('off')

    def draw_atoms(self, supercell, active_colors: dict, scale_mod: float = 0.5, is_shiny: bool = False, erased_atoms: set = None):
        if erased_atoms is None: erased_atoms = set()
        
        for i, site in enumerate(supercell):
            if i in erased_atoms: continue
            tag = supercell.site_properties.get("layer_tag", [site.specie.symbol] * len(supercell))[i]
            color = active_colors.get(tag, "#008080")
            rad = (site.specie.atomic_radius if site.specie.atomic_radius else 1.0) * scale_mod
            
            x, y, z = site.coords
            # Multiply radius to look decent in matplotlib scatter sizing
            self.ax.scatter(x, y, z, s=(rad * 400), c=color, edgecolors='black', depthshade=True)

    def draw_bonds(self, supercell, active_colors: dict, cyl_radius: float = 0.1, thresh_multiplier: float = 1.15, is_shiny: bool = False, erased_bonds: set = None, erased_atoms: set = None):
        if len(supercell) > 2000: return # Protect CPU from freezing on massive arrays
        if erased_bonds is None: erased_bonds = set()
        if erased_atoms is None: erased_atoms = set()

        bond_color = active_colors.get("Bonds", "#FFFFFF")
        for i in range(len(supercell)):
            if i in erased_atoms: continue
            for j in range(i + 1, len(supercell)):
                if j in erased_atoms or (i, j) in erased_bonds: continue
                
                a, b = supercell[i], supercell[j]
                dist = np.linalg.norm(a.coords - b.coords)
                ra = a.specie.atomic_radius if a.specie.atomic_radius else 1.2
                rb = b.specie.atomic_radius if b.specie.atomic_radius else 1.2
                
                if 0.5 < dist <= (ra + rb) * thresh_multiplier:
                    self.ax.plot([a.coords[0], b.coords[0]], 
                                 [a.coords[1], b.coords[1]], 
                                 [a.coords[2], b.coords[2]], 
                                 color=bond_color, linewidth=2)

    def draw_axes(self, conventional_matrix=None, primitive_matrix=None):
        pass # Simplified for CPU rendering

    def draw_lattice_boxes(self, conventional_matrix=None, primitive_matrix=None):
        pass # Simplified for CPU rendering

    def draw_brillouin_zone(self, bz_points, bz_simplices, solid=True):
        pass # Simplified for CPU rendering

    def draw_moire_envelope(self, m_moire, z_min, z_max):
        pass # Simplified for CPU rendering

    def set_camera_preset(self, preset: str):
        if preset == 'x': self.ax.view_init(elev=0, azim=0)
        elif preset == 'y': self.ax.view_init(elev=0, azim=90)
        elif preset == 'z': self.ax.view_init(elev=90, azim=-90)
        elif preset == 'iso': self.ax.view_init(elev=35, azim=45)
        self.canvas.draw_idle()