from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
import numpy as np

# Try to load PyVista; fallback to Matplotlib if GPU/OpenCore fails
try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

class MatrixSetupViewer(QWidget):
    """
    3D visualization wrapper for the ARPES manipulator setup.
    Automatically detects if PyVista (GPU) is available, otherwise falls back to Matplotlib (CPU).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.use_pyvista = PYVISTA_AVAILABLE
        
        if self.use_pyvista:
            self.plotter = QtInteractor(self)
            self.layout.addWidget(self.plotter.interactor)
            self.plotter.set_background('white')
            self.plotter.add_axes()
        else:
            # Safe Matplotlib 3D Fallback
            self.figure = Figure()
            self.canvas = FigureCanvas(self.figure)
            self.ax = self.figure.add_subplot(111, projection='3d')
            self.layout.addWidget(self.canvas)
            
    def update_geometry(self, engine):
        """Redraws the 3D scene using the active backend."""
        if self.use_pyvista:
            self._update_pyvista(engine)
        else:
            self._update_matplotlib(engine)

    def _update_pyvista(self, engine):
        """High-performance GPU rendering for modern machines."""
        self.plotter.clear_actors()

        # Manipulator & Sample
        manipulator = pv.Cylinder(center=(0, 0, 2), direction=(0, 0, 1), radius=0.2, height=4.0)
        self.plotter.add_mesh(manipulator, color='lightgray', name='manipulator')

        normal, in_plane_x, in_plane_z = engine.get_sample_vectors()
        sample_disk = pv.Cylinder(center=(0, 0, 0), direction=normal, radius=1.0, height=0.1)
        self.plotter.add_mesh(sample_disk, color='gold', name='sample')
        
        normal_arrow = pv.Arrow(start=(0, 0, 0), direction=normal, scale=2.0)
        self.plotter.add_mesh(normal_arrow, color='red', name='sample_normal')

        # Beam & Polarization
        beam_dir = engine.get_beam_vector()
        beam_start = -4.0 * beam_dir 
        beam_arrow = pv.Arrow(start=beam_start, direction=beam_dir, scale=4.0)
        self.plotter.add_mesh(beam_arrow, color='cyan', name='beam')
        
        pol_vectors = engine.get_polarization_vectors()
        for idx, p_vec in enumerate(pol_vectors):
            pol_arrow = pv.Arrow(start=beam_start, direction=p_vec, scale=1.0)
            color = 'magenta' if engine.polarization in ['LH', 'LV'] else 'orange'
            self.plotter.add_mesh(pol_arrow, color=color, name=f'polarization_{idx}')

        # Analyzer Slit
        analyzer = pv.Cone(center=(0, 4, 0), direction=(0, -1, 0), height=3.0, radius=1.0)
        self.plotter.add_mesh(analyzer, color='blue', opacity=0.1, name='analyzer')
        
        left_bound, right_bound = engine.get_slit_boundaries()
        points = np.array([[0, 0, 0], left_bound * 4.0, right_bound * 4.0])
        faces = np.array([3, 0, 1, 2])
        self.plotter.add_mesh(pv.PolyData(points, faces), color='green', opacity=0.4, name='slit_fan')

        if not hasattr(self, '_camera_set'):
            self.plotter.reset_camera()
            self._camera_set = True
        self.plotter.render()

    def _update_matplotlib(self, engine):
        """Safe CPU rendering for older Macs (OpenCore 2013)."""
        self.ax.clear()
        
        # Set static limits to prevent the camera from auto-zooming crazily
        self.ax.set_xlim([-4, 4])
        self.ax.set_ylim([-4, 4])
        self.ax.set_zlim([-4, 4])
        self.ax.set_xlabel('Lab X (Sideways)')
        self.ax.set_ylabel('Lab Y (Forward)')
        self.ax.set_zlabel('Lab Z (Up)')
        self.ax.set_title("3D Setup (Safe CPU Mode)")

        # 1. Draw Manipulator Rod (Thick line along Z-axis)
        self.ax.plot([0, 0], [0, 0], [0, 4], color='lightgray', linewidth=5, label='Manipulator')

        # 2. Draw Sample Normal Vector
        normal, in_plane_x, in_plane_z = engine.get_sample_vectors()
        self.ax.quiver(0, 0, 0, normal[0], normal[1], normal[2], 
                       color='red', length=2.0, normalize=True, label='Sample Normal')
        
        # 3. Draw Incident Beam and Polarization
        beam_dir = engine.get_beam_vector()
        beam_start = -4.0 * beam_dir 
        self.ax.quiver(beam_start[0], beam_start[1], beam_start[2], 
                       beam_dir[0], beam_dir[1], beam_dir[2], 
                       color='cyan', length=4.0, normalize=True, label='Beam')
        
        pol_vectors = engine.get_polarization_vectors()
        for p_vec in pol_vectors:
            color = 'magenta' if engine.polarization in ['LH', 'LV'] else 'orange'
            self.ax.quiver(beam_start[0], beam_start[1], beam_start[2], 
                           p_vec[0], p_vec[1], p_vec[2], 
                           color=color, length=1.5, normalize=True)

        # 4. Draw Analyzer Axis & Slit Acceptance Fan
        self.ax.plot([0, 0], [0, 4], [0, 0], color='blue', linestyle='--', label='Analyzer Axis')
        
        left_bound, right_bound = engine.get_slit_boundaries()
        L = left_bound * 4.0
        R = right_bound * 4.0
        
        # Draw the triangle fan for the slit
        self.ax.plot([0, L[0]], [0, L[1]], [0, L[2]], color='green', linewidth=2)
        self.ax.plot([0, R[0]], [0, R[1]], [0, R[2]], color='green', linewidth=2)
        self.ax.plot([L[0], R[0]], [L[1], R[1]], [L[2], R[2]], color='green', linewidth=2, label='Slit 30°')

        # Enforce 1:1:1 aspect ratio so rotations don't look warped
        try:
            self.ax.set_box_aspect([1, 1, 1])
        except AttributeError:
            pass # Fails safely on very old matplotlib versions

        self.canvas.draw()