import os
import sys
import platform
import numpy as np
from scipy.spatial import ConvexHull

# --- Cross-Platform Environment Standardization ---
if platform.system() == "Darwin":  
    os.environ["QT_API"] = "pyside6"
    os.environ["QT_MAC_WANTS_LAYER"] = "1"
    BASIC_GRAPHICS = True 
else:  
    os.environ["QT_API"] = os.environ.get("QT_API", "pyside6")
    BASIC_GRAPHICS = False

# 1. IMPORT PYVISTA FIRST (Allows VTK to inject Windows OpenGL Surface Formats)
import pyvista as pv
from pyvistaqt import QtInteractor
pv.global_theme.depth_peeling.enabled = False
pv.global_theme.anti_aliasing = "fxaa"

# 2. PYSIDE IMPORTS & APP INITIALIZATION
# 2. PYSIDE IMPORTS & APP INITIALIZATION
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QFileDialog, QLabel, 
                               QSpinBox, QDoubleSpinBox, QComboBox, QColorDialog,
                               QStackedWidget, QTabWidget, QCheckBox, QGroupBox, QGridLayout, QSlider, QFrame, QSplitter)
from PySide6.QtCore import Qt
from PySide6.QtGui import QSurfaceFormat # <--- NEW IMPORT
import vtk # <--- NEW IMPORT

# --- NEW FIX: Force Qt6 to use OpenGL to prevent C++ VTK segfaults on Windows ---
os.environ["QSG_RHI_BACKEND"] = "opengl"

# --- NEW FIX: Manually define the 3D Surface Format (Bypassing VTK's missing method) ---
fmt = QSurfaceFormat()
fmt.setRenderableType(QSurfaceFormat.OpenGL)
fmt.setVersion(3, 2)
fmt.setProfile(QSurfaceFormat.CoreProfile)
fmt.setSwapBehavior(QSurfaceFormat.DoubleBuffer)
fmt.setDepthBufferSize(24)
QSurfaceFormat.setDefaultFormat(fmt)

# 3. IMPORT PYMATGEN
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN

# Default Fallback Colors
CPK_COLORS = {
    "H": "#FFFFFF", "C": "#333333", "N": "#2233FF", "O": "#FF2200",
    "V": "#999999", "Te": "#FF8C00", "Fe": "#E06633", "Cu": "#C88033"
}

class StackLayerWidget(QFrame):
    def __init__(self, name, struct, default_z):
        super().__init__()
        self.struct = struct
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QFrame { background-color: #2b2b36; border-radius: 5px; padding: 5px; margin-bottom: 5px; }")
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Label
        self.lbl_name = QLabel(f"<b>{name}</b>")
        self.lbl_name.setFixedWidth(100)
        layout.addWidget(self.lbl_name)
        
        # Supercell size (X, Y only for 2D slabs)
        layout.addWidget(QLabel("SC:"))
        self.spin_x = QSpinBox(); self.spin_x.setRange(1, 100); self.spin_x.setValue(1)
        self.spin_y = QSpinBox(); self.spin_y.setRange(1, 100); self.spin_y.setValue(1)
        layout.addWidget(self.spin_x); layout.addWidget(self.spin_y)
        
        # Z-Shift
        layout.addWidget(QLabel("z (Å):"))
        self.spin_z = QDoubleSpinBox()
        self.spin_z.setRange(-100.0, 100.0)
        self.spin_z.setSingleStep(0.5)
        self.spin_z.setValue(default_z)
        layout.addWidget(self.spin_z)
        
        # Twist Angle
        layout.addWidget(QLabel("θ (°):"))
        self.spin_twist = QDoubleSpinBox()
        self.spin_twist.setRange(-360.0, 360.0)
        self.spin_twist.setSingleStep(1.0)
        self.spin_twist.setValue(0.0)
        layout.addWidget(self.spin_twist)
        
        # Action Buttons
        self.btn_up = QPushButton("▲")
        self.btn_up.setFixedWidth(25)
        self.btn_down = QPushButton("▼")
        self.btn_down.setFixedWidth(25)
        self.btn_delete = QPushButton("X")
        self.btn_delete.setStyleSheet("background-color: #d9534f; color: white;")
        self.btn_delete.setFixedWidth(25)
        
        layout.addWidget(self.btn_up)
        layout.addWidget(self.btn_down)
        layout.addWidget(self.btn_delete)

class TensorSpecApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TensorSpec - Crystal Suite")
        self.resize(1300, 800) 

        self.current_structure = None
        self.active_colors = {"Bonds": "#d3d3d3"} # Stores current colors for rendering

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # --- NEW: Draggable Splitter ---
        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)

        # ================= L E F T   P A N E L  (T A B S) =================
        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(320) # No longer fixed, just a minimum!
        self.splitter.addWidget(self.tabs)

        # --- TAB 1: VIEW & EDIT ---
        self.tab1 = QWidget()
        self.ui_panel = QVBoxLayout(self.tab1)
        
        self.btn_load = QPushButton("Load CIF File")
        self.btn_load.clicked.connect(self.load_file)
        self.ui_panel.addWidget(self.btn_load)
        
        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ui_panel.addWidget(self.lbl_file)

        self.ui_panel.addWidget(QLabel("Graphics Backend:"))
        self.combo_backend = QComboBox()
        self.combo_backend.addItems(["CPU Safe Mode (Matplotlib)", "GPU Fast Mode (PyVista)"])
        self.combo_backend.currentIndexChanged.connect(self.switch_backend)
        self.ui_panel.addWidget(self.combo_backend)

        # --- Supercell & Scaling Group ---
        group_geom = QGroupBox("Geometry & Scaling")
        geom_layout = QVBoxLayout(group_geom)
        
        geom_layout.addWidget(QLabel("Supercell (X, Y, Z):"))
        self.spin_x = QSpinBox(); self.spin_x.setValue(1); self.spin_x.setMinimum(1)
        self.spin_y = QSpinBox(); self.spin_y.setValue(1); self.spin_y.setMinimum(1)
        self.spin_z = QSpinBox(); self.spin_z.setValue(1); self.spin_z.setMinimum(1)
        for spin in [self.spin_x, self.spin_y, self.spin_z]:
            geom_layout.addWidget(spin)

        geom_layout.addWidget(QLabel("Atom Radius Scale:"))
        self.spin_radius = QDoubleSpinBox()
        self.spin_radius.setRange(0.1, 5.0); self.spin_radius.setValue(0.5); self.spin_radius.setSingleStep(0.1)
        geom_layout.addWidget(self.spin_radius)

        geom_layout.addWidget(QLabel("Bond/Stick Thickness:"))
        self.spin_bond_thick = QDoubleSpinBox()
        self.spin_bond_thick.setRange(0.01, 1.0); self.spin_bond_thick.setValue(0.10); self.spin_bond_thick.setSingleStep(0.02)
        geom_layout.addWidget(self.spin_bond_thick)
        self.ui_panel.addWidget(group_geom)

        # --- Appearance & Colors Group (Dynamic) ---
        self.group_colors = QGroupBox("Styles & Colors")
        self.colors_layout = QGridLayout(self.group_colors)
        
        self.colors_layout.addWidget(QLabel("Connections:"), 0, 0)
        self.combo_style = QComboBox()
        self.combo_style.addItems(["Bonds (Sticks)", "Polyhedra (Planes)", "None"])
        self.colors_layout.addWidget(self.combo_style, 0, 1)
        self.ui_panel.addWidget(self.group_colors)

       # --- Interactive Mode ---
        self.chk_edit_mode = QCheckBox("Enable Interactive Delete")
        self.chk_edit_mode.setStyleSheet("color: #d9534f; font-weight: bold;")
        self.chk_edit_mode.stateChanged.connect(self.toggle_edit_mode)
        self.ui_panel.addWidget(self.chk_edit_mode)

        # --- NEW: Camera & View Controls ---
        self.group_camera = QGroupBox("Camera & Projection")
        self.cam_layout = QVBoxLayout(self.group_camera)
        
        # Perspective vs Orthogonal
        self.combo_projection = QComboBox()
        self.combo_projection.addItems(["Perspective Projection", "Orthogonal Projection"])
        self.combo_projection.currentIndexChanged.connect(self.update_projection)
        self.cam_layout.addWidget(self.combo_projection)
        
        # Quick Axis Views
        self.cam_layout.addWidget(QLabel("Quick View:"))
        quick_view_layout = QHBoxLayout()
        for axis, label in [('x', '+a'), ('y', '+b'), ('z', '+c'), ('iso', '111')]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked, a=axis: self.set_camera_view(a))
            quick_view_layout.addWidget(btn)
        self.cam_layout.addLayout(quick_view_layout)

        # Azimuth and Elevation
        azel_layout = QHBoxLayout()
        azel_layout.addWidget(QLabel("Azimuth:"))
        self.spin_azimuth = QDoubleSpinBox()
        self.spin_azimuth.setRange(-360.0, 360.0); self.spin_azimuth.setSingleStep(5.0)
        self.spin_azimuth.valueChanged.connect(self.update_azel)
        azel_layout.addWidget(self.spin_azimuth)
        
        azel_layout.addWidget(QLabel("Elevation:"))
        self.spin_elevation = QDoubleSpinBox()
        self.spin_elevation.setRange(-90.0, 90.0); self.spin_elevation.setSingleStep(5.0)
        self.spin_elevation.valueChanged.connect(self.update_azel)
        azel_layout.addWidget(self.spin_elevation)
        
        self.cam_layout.addLayout(azel_layout)
        self.ui_panel.addWidget(self.group_camera)

        # --- NEW: Crystallography Tools ---
        self.group_cryst = QGroupBox("Crystallography Tools")
        self.cryst_layout = QVBoxLayout(self.group_cryst)
        
        # Miller Indices [h k l]
        hkl_layout = QHBoxLayout()
        hkl_layout.addWidget(QLabel("View [h k l]:"))
        self.spin_h = QDoubleSpinBox(); self.spin_h.setRange(-10, 10); self.spin_h.setSingleStep(0.1)
        self.spin_k = QDoubleSpinBox(); self.spin_k.setRange(-10, 10); self.spin_k.setSingleStep(0.1)
        self.spin_l = QDoubleSpinBox(); self.spin_l.setRange(-10, 10); self.spin_l.setSingleStep(0.1)
        hkl_layout.addWidget(self.spin_h); hkl_layout.addWidget(self.spin_k); hkl_layout.addWidget(self.spin_l)
        
        self.btn_align_hkl = QPushButton("Align")
        self.btn_align_hkl.clicked.connect(self.align_to_hkl)
        hkl_layout.addWidget(self.btn_align_hkl)
        self.cryst_layout.addLayout(hkl_layout)

        # Depth Cut Plane
        plane_layout = QHBoxLayout()
        self.chk_show_plane = QCheckBox("Show Cut Plane")
        self.chk_show_plane.stateChanged.connect(self.toggle_cut_plane) # Changed from update_cut_plane
        plane_layout.addWidget(self.chk_show_plane)
        
        self.combo_plane_color = QComboBox()
        self.combo_plane_color.addItems(["cyan", "magenta", "yellow", "white", "gray"])
        self.combo_plane_color.currentTextChanged.connect(self.update_cut_plane)
        plane_layout.addWidget(self.combo_plane_color)
        
        # NEW: Dropdown to select what the plane aligns to
        self.combo_plane_orient = QComboBox()
        self.combo_plane_orient.addItems(["Lock to Camera", "Lock to [h k l]"])
        self.combo_plane_orient.currentIndexChanged.connect(self.lock_plane_normal)
        plane_layout.addWidget(self.combo_plane_orient)
        self.cryst_layout.addLayout(plane_layout)
        
        depth_layout = QHBoxLayout()
        depth_layout.addWidget(QLabel("Depth:"))
        self.slider_plane_depth = QSlider(Qt.Horizontal)
        self.slider_plane_depth.setRange(-100, 100)
        self.slider_plane_depth.setValue(0)
        self.slider_plane_depth.valueChanged.connect(self.update_cut_plane)
        depth_layout.addWidget(self.slider_plane_depth)
        self.cryst_layout.addLayout(depth_layout)
        
        self.ui_panel.addWidget(self.group_cryst)

        # --- NEW: Advanced Rendering & Bonds ---
        self.group_render = QGroupBox("Advanced Rendering")
        self.render_layout = QVBoxLayout(self.group_render)

        # Bond Threshold Spinner
        bond_thresh_layout = QHBoxLayout()
        bond_thresh_layout.addWidget(QLabel("Bond Threshold Multiplier:"))
        self.spin_bond_thresh = QDoubleSpinBox()
        self.spin_bond_thresh.setRange(0.1, 5.0)
        self.spin_bond_thresh.setSingleStep(0.1)
        self.spin_bond_thresh.setValue(1.15) # Your cleaned-up default
        bond_thresh_layout.addWidget(self.spin_bond_thresh)
        self.render_layout.addLayout(bond_thresh_layout)

        # 3D Shiny PBR Toggle
        self.chk_shiny = QCheckBox("Enable High Quality 3D (Shiny/PBR)")
        self.chk_shiny.setChecked(False) 
        self.chk_shiny.stateChanged.connect(self.toggle_shiny_materials) # NEW: Live update
        self.render_layout.addWidget(self.chk_shiny)

        # Show a, b, c crystallographic axes
        self.chk_show_axes = QCheckBox("Show a, b, c Axes")
        self.chk_show_axes.setChecked(True)
        self.chk_show_axes.stateChanged.connect(self.draw_pyvista_axes) # NEW: Live update
        self.render_layout.addWidget(self.chk_show_axes)

        # --- NEW: Unit Cell Wireframe Boundaries ---
        self.chk_show_conventional = QCheckBox("Show Conventional Cell Box")
        self.chk_show_conventional.setStyleSheet("color: #FF5733; font-weight: bold;") # Light coral
        self.chk_show_conventional.stateChanged.connect(self.update_lattice_boxes)
        self.ui_panel.addWidget(self.chk_show_conventional)

        self.chk_show_primitive = QCheckBox("Show Primitive Cell Box")
        self.chk_show_primitive.setStyleSheet("color: #33FF57; font-weight: bold;") # Vibrant green
        self.chk_show_primitive.stateChanged.connect(self.update_lattice_boxes)
        self.ui_panel.addWidget(self.chk_show_primitive)

        self.ui_panel.addWidget(self.group_render)

        # --- Action Buttons ---
        self.btn_draw = QPushButton("Draw Structure")
        self.btn_draw.clicked.connect(self.draw_structure)
        self.btn_draw.setStyleSheet("background-color: #2b5c8f; font-weight: bold; color: white; padding: 5px;")
        self.ui_panel.addWidget(self.btn_draw)

        self.btn_save = QPushButton("Save High-Res Image")
        self.btn_save.clicked.connect(self.save_image)
        self.btn_save.setStyleSheet("background-color: #28a745; font-weight: bold; color: white; padding: 5px;")
        self.ui_panel.addWidget(self.btn_save)

        # --- NEW: 3ds Max Export Button ---
        self.btn_export_max = QPushButton("Export to 3ds Max")
        self.btn_export_max.setStyleSheet("background-color: #0F6A8B; color: white; font-weight: bold; padding: 5px;")
        self.btn_export_max.clicked.connect(self.export_3dsmax_script)
        self.ui_panel.addWidget(self.btn_export_max)

        # --- NEW: Blender Export Button ---
        self.btn_export_blend = QPushButton("Export to Blender")
        self.btn_export_blend.setStyleSheet("background-color: #E87D0D; color: white; font-weight: bold; padding: 5px;")
        self.btn_export_blend.clicked.connect(self.export_blender_script)
        self.ui_panel.addWidget(self.btn_export_blend)

        self.ui_panel.addStretch()
        self.tabs.addTab(self.tab1, "1. View & Edit")

        # --- TAB 2: CDW MODULATOR ---
        self.tab2 = QWidget()
        tab2_layout = QVBoxLayout(self.tab2)
        
        self.chk_cdw_enable = QCheckBox("Enable CDW Distortion")
        self.chk_cdw_enable.setStyleSheet("font-weight: bold; color: #2b5c8f;")
        self.chk_cdw_enable.stateChanged.connect(self.draw_structure)
        tab2_layout.addWidget(self.chk_cdw_enable)
        
        tab2_layout.addWidget(QLabel("Target Atom Type:"))
        self.combo_cdw_element = QComboBox()
        self.combo_cdw_element.addItem("All Elements")
        self.combo_cdw_element.currentIndexChanged.connect(self.draw_structure)
        tab2_layout.addWidget(self.combo_cdw_element)
        
        # Wavevector Modulation Group
        group_q = QGroupBox("Modulation Wavevector q (rlu)")
        q_layout = QVBoxLayout(group_q)
        self.spin_qx = QDoubleSpinBox(); self.spin_qx.setRange(-5.0, 5.0); self.spin_qx.setSingleStep(0.05); self.spin_qx.setDecimals(3)
        self.spin_qy = QDoubleSpinBox(); self.spin_qy.setRange(-5.0, 5.0); self.spin_qy.setSingleStep(0.05); self.spin_qy.setDecimals(3)
        self.spin_qz = QDoubleSpinBox(); self.spin_qz.setRange(-5.0, 5.0); self.spin_qz.setSingleStep(0.05); self.spin_qz.setDecimals(3)
        for spin, lbl in [(self.spin_qx, "q_a :"), (self.spin_qy, "q_b :"), (self.spin_qz, "q_c :")]:
            q_layout.addWidget(QLabel(lbl))
            q_layout.addWidget(spin)
            spin.valueChanged.connect(self.draw_structure)
        tab2_layout.addWidget(group_q)
        
        # Displacement Amplitude Group
        group_amp = QGroupBox("Displacement Amplitude A (Å)")
        amp_layout = QVBoxLayout(group_amp)
        self.spin_ax = QDoubleSpinBox(); self.spin_ax.setRange(-2.0, 2.0); self.spin_ax.setSingleStep(0.01); self.spin_ax.setDecimals(3)
        self.spin_ay = QDoubleSpinBox(); self.spin_ay.setRange(-2.0, 2.0); self.spin_ay.setSingleStep(0.01)
        self.spin_az = QDoubleSpinBox(); self.spin_az.setRange(-2.0, 2.0); self.spin_az.setSingleStep(0.01)
        for spin, lbl in [(self.spin_ax, "Δx (along a):"), (self.spin_ay, "Δy (along b):"), (self.spin_az, "Δz (along c):")]:
            amp_layout.addWidget(QLabel(lbl))
            amp_layout.addWidget(spin)
            spin.valueChanged.connect(self.draw_structure)
        tab2_layout.addWidget(group_amp)
        
        # Phase Adjuster
        tab2_layout.addWidget(QLabel("Phase Shift φ (degrees):"))
        self.spin_cdw_phase = QDoubleSpinBox()
        self.spin_cdw_phase.setRange(0.0, 360.0); self.spin_cdw_phase.setSingleStep(15.0)
        self.spin_cdw_phase.valueChanged.connect(self.draw_structure)
        tab2_layout.addWidget(self.spin_cdw_phase)
        
        tab2_layout.addStretch()
        self.tabs.addTab(self.tab2, "2. CDW Modulator")

        # --- TAB 3: STACK, TWIST, MOIRÉ ---
        self.tab3 = QWidget()
        tab3_layout = QVBoxLayout(self.tab3)
        
        self.stack_layers = [] # Internal list to track loaded layers
        
        self.btn_add_layer = QPushButton("+ Add Layer from CIF")
        self.btn_add_layer.setStyleSheet("background-color: #5bc0de; color: black; font-weight: bold; padding: 5px;")
        self.btn_add_layer.clicked.connect(self.add_stack_layer)
        tab3_layout.addWidget(self.btn_add_layer)

        # Scroll area for dynamic layers
        from PySide6.QtWidgets import QScrollArea
        self.layer_scroll = QScrollArea()
        self.layer_scroll.setWidgetResizable(True)
        self.layer_container = QWidget()
        self.layer_layout = QVBoxLayout(self.layer_container)
        self.layer_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.layer_scroll.setWidget(self.layer_container)
        tab3_layout.addWidget(self.layer_scroll)
        
        self.btn_draw_stack = QPushButton("Draw Heterostructure Stack")
        self.btn_draw_stack.setStyleSheet("background-color: #2b5c8f; color: white; font-weight: bold; padding: 8px;")
        self.btn_draw_stack.clicked.connect(self.draw_stack)
        tab3_layout.addWidget(self.btn_draw_stack)
        
        # --- PHASE 3: Moiré Identification Hook ---
        self.btn_moire = QPushButton("Identify Moiré Superlattice")
        self.btn_moire.setStyleSheet("background-color: #8A2BE2; color: white; font-weight: bold; padding: 8px;")
        self.btn_moire.clicked.connect(self.identify_moire)
        tab3_layout.addWidget(self.btn_moire)
        
        self.lbl_moire_status = QLabel("Moiré Status: Waiting for calculation...")
        self.lbl_moire_status.setWordWrap(True)
        self.lbl_moire_status.setStyleSheet("background-color: #1e1e24; color: #d3d3d3; padding: 5px; border-radius: 3px;")
        tab3_layout.addWidget(self.lbl_moire_status)
        
        # ADD TAB 3 FIRST
        self.tabs.addTab(self.tab3, "3. Stack & Twist")

        # --- TAB 4: BRILLOUIN ZONE (Placeholder) ---
        self.tab4 = QWidget()
        tab4_layout = QVBoxLayout(self.tab4)
        tab4_layout.addWidget(QLabel("Wigner-Seitz cells and surface projections will go here."))
        tab4_layout.addStretch()
        
        # ADD TAB 4 SECOND
        self.tabs.addTab(self.tab4, "4. Brillouin Zone")


        # ================= R I G H T   P A N E L (Stacked Viewers) =================
        print("     [Trace] Building right panel stack...")
        self.viewer_stack = QStackedWidget()
        self.splitter.addWidget(self.viewer_stack)
        self.splitter.setSizes([380, 920]) # Default startup sizes (Left panel slightly wider)

        # --- INITIALIZE PYVISTA FIRST ---
        print("     [Trace] Initializing PyVista QtInteractor (VTK)...")
        
        # Initialize without passing a parent so VTK doesn't prematurely hook an invalid HWND.
        self.plotter = QtInteractor() 
        self.plotter.set_background("#1e1e24")
        self.plotter.iren.add_observer("EndInteractionEvent", self.sync_ui_to_camera)  
        
        # Add it directly to the StackedWidget. 
        # Qt safely handles reparenting and defers C++ window creation until render time.
        self.viewer_stack.addWidget(self.plotter)
        print("     [Trace] PyVista Widget added successfully.")

        # --- INITIALIZE MATPLOTLIB SECOND ---
        print("     [Trace] Initializing Matplotlib Figure...")
        
        # FIX 2: Import Matplotlib locally ONLY AFTER PyVista has claimed the hardware context
        import matplotlib
        matplotlib.use('qtagg')
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        self.figure = Figure(facecolor="#1e1e24")
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.ax.set_facecolor("#1e1e24")
        self.ax.axis('off')
        
        self.canvas.mpl_connect('pick_event', self.on_matplotlib_pick)
        self.canvas.mpl_connect('button_press_event', self.start_matplotlib_eraser)
        self.canvas.mpl_connect('button_release_event', self.stop_matplotlib_eraser)
        self.canvas.mpl_connect('motion_notify_event', self.matplotlib_eraser_brush)
        self.canvas.mpl_connect('button_release_event', self.sync_ui_to_camera) 
        
        self.viewer_stack.insertWidget(0, self.canvas) 
        print("     [Trace] Matplotlib Widget added successfully.")
        
        self.viewer_stack.setCurrentIndex(0) 
        print("     [Trace] __init__ completed perfectly.")

    # ================= C O L O R   &   U I   L O G I C =================
    def build_dynamic_color_panel(self, elements):
        """Builds VESTA-style color pickers based on the elements in the CIF"""
        # Clear ALL widgets from the layout EXCEPT the persistent combo_style dropdown
        for i in reversed(range(self.colors_layout.count())):
            item = self.colors_layout.itemAt(i)
            widget = item.widget()
            if widget:
                if widget == self.combo_style:
                    continue # Save the dropdown box
                widget.setParent(None) # Nuke everything else (including old labels!)
                
        # Re-add the clean label for the combo box on row 0
        self.colors_layout.addWidget(QLabel("Connections:"), 0, 0)
        self.colors_layout.addWidget(self.combo_style, 0, 1)

        row = 1
        self.active_colors = {"Bonds": "#d3d3d3"} # Reset and initialize bonds

        for el in elements:
            if el == "Bonds": continue
            
            # Strip the layer tag (e.g., "C_L1" -> "C") to find standard CPK color defaults
            base_element = el.split('_')[0] if '_' in el else el
            default_color = CPK_COLORS.get(base_element, "#008080")
            
            # If Layer 2, subtly alter the default color slightly so they don't look identical initially
            if "_L2" in el and base_element in CPK_COLORS:
                # Let's make Layer 2 automatically a distinct deep slate blue if it's Carbon
                if base_element == "C": default_color = "#5bc0de" 
                else: default_color = "#ffc107" # Amber/Gold fallback for distinguishing layer 2
            
            current_color = self.active_colors.get(el, default_color)
            self.active_colors[el] = current_color
            
            lbl = QLabel(f"{el} Color:")
            btn = QPushButton()
            btn.setStyleSheet(f"background-color: {current_color}; border: 1px solid white;")
            btn.clicked.connect(lambda checked, key=el, b=btn: self.pick_color(key, b))
            
            self.colors_layout.addWidget(lbl, row, 0)
            self.colors_layout.addWidget(btn, row, 1)
            row += 1

        # Add Bonds color picker
        lbl_bond = QLabel("Bond Color:")
        btn_bond = QPushButton()
        btn_bond.setStyleSheet(f"background-color: {self.active_colors['Bonds']}; border: 1px solid white;")
        btn_bond.clicked.connect(lambda checked, key="Bonds", b=btn_bond: self.pick_color(key, b))
        
        self.colors_layout.addWidget(lbl_bond, row, 0)
        self.colors_layout.addWidget(btn_bond, row, 1)

    def pick_color(self, key, button):
        color = QColorDialog.getColor()
        if color.isValid():
            hex_color = color.name()
            self.active_colors[key] = hex_color
            button.setStyleSheet(f"background-color: {hex_color}; border: 1px solid white;")
            self.draw_structure() # Auto-redraw to apply

    def switch_backend(self, index):
        self.viewer_stack.setCurrentIndex(index)
        self.draw_structure() 

    def load_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '', "CIF files (*.cif)")
        if fname:
            self.lbl_file.setText(fname.split('/')[-1])
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.current_structure = Structure.from_file(fname)
                    
                    # Extract unique elements and build UI
                    unique_elements = sorted(list(set([site.specie.symbol for site in self.current_structure])))
                    self.build_dynamic_color_panel(unique_elements)

                    # Populate CDW target selection dropdown dynamically
                    self.combo_cdw_element.clear()
                    self.combo_cdw_element.addItem("All Elements")
                    self.combo_cdw_element.addItems(unique_elements)
                    
            except Exception as e:
                self.lbl_file.setText("Error loading CIF")

    def save_image(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png)")
        if fname:
            if self.combo_backend.currentIndex() == 0:
                # Matplotlib: Boost from 300 up to 1200 DPI for extreme print quality
                self.figure.savefig(fname, dpi=1200, bbox_inches='tight', transparent=True)
            else:
                # PyVista: Force VTK to render a massive 4K off-screen buffer (3840 x 2160)
                # Note: You can change this to [7680, 4320] if you need native 8K resolution!
                self.plotter.screenshot(fname, transparent_background=True, window_size=[3840, 2160])

    def draw_structure(self, *args):
        if self.current_structure is None: return
        backend = self.combo_backend.currentIndex()
        
        # 1. Start Fresh ONLY if the "Draw Structure" button is clicked, or if no supercell exists
        if self.sender() == getattr(self, 'btn_draw', None) or not hasattr(self, 'active_supercell'):
            self.active_supercell = self.current_structure * (self.spin_x.value(), self.spin_y.value(), self.spin_z.value())
            self.base_cart_coords = self.active_supercell.cart_coords.copy()
            self.base_frac_coords = self.active_supercell.frac_coords.copy()
            self.erased_atoms = set()
            self.erased_bonds = set()

        new_cart = self.base_cart_coords.copy()
        
        # 2. Apply CDW Distortion Propagation to the pristine base coordinates
        if hasattr(self, 'chk_cdw_enable') and self.chk_cdw_enable.isChecked():
            target_el = self.combo_cdw_element.currentText()
            qx, qy, qz = self.spin_qx.value(), self.spin_qy.value(), self.spin_qz.value()
            ax, ay, az = self.spin_ax.value(), self.spin_ay.value(), self.spin_az.value()
            phase = np.radians(self.spin_cdw_phase.value())
            
            for i, site in enumerate(self.active_supercell):
                if target_el == "All Elements" or site.specie.symbol == target_el:
                    fx, fy, fz = self.base_frac_coords[i]
                    wave_argument = 2 * np.pi * (qx * fx + qy * fy + qz * fz) + phase
                    
                    dx = ax * np.cos(wave_argument)
                    dy = ay * np.cos(wave_argument)
                    dz = az * np.cos(wave_argument)
                    new_cart[i] += np.array([dx, dy, dz])

        # 3. Safely update the active supercell without losing its data
        for i, site in enumerate(self.active_supercell):
            self.active_supercell.replace(
                i, 
                site.specie, 
                new_cart[i], 
                coords_are_cartesian=True
            )

        if backend == 0: self.draw_matplotlib(self.active_supercell)
        else: self.draw_pyvista(self.active_supercell)

        self.update_projection()

    def toggle_edit_mode(self, *args):
        """Toggles the interactive eraser brush using direct VTK hardware picking."""
        if self.combo_backend.currentIndex() == 0:  # Matplotlib backend
            if self.chk_edit_mode.isChecked():
                self.ax.disable_mouse_rotation()
            else:
                self.ax.mouse_init()
        elif self.combo_backend.currentIndex() == 1:  # PyVista backend
            if self.chk_edit_mode.isChecked():
                self.is_erasing = False
                
                # 1. Lock the camera: left-click drag will no longer rotate the view!
                self.plotter.enable_trackball_actor_style()
                
                # 2. Attach continuous brush listeners
                self.press_obs = self.plotter.iren.add_observer("LeftButtonPressEvent", self.start_erase)
                self.move_obs = self.plotter.iren.add_observer("MouseMoveEvent", self.do_erase)
                self.release_obs = self.plotter.iren.add_observer("LeftButtonReleaseEvent", self.stop_erase)
                self.right_obs = self.plotter.iren.add_observer("RightButtonPressEvent", self.exit_eraser_mode)
            else:
                # Restore standard camera rotation
                self.plotter.enable_trackball_style()
                
                # Safely remove all brush listeners
                if hasattr(self, 'press_obs'):
                    self.plotter.iren.remove_observer(self.press_obs)
                    self.plotter.iren.remove_observer(self.move_obs)
                    self.plotter.iren.remove_observer(self.release_obs)
                    self.plotter.iren.remove_observer(self.right_obs)
                    del self.press_obs
                    del self.move_obs
                    del self.release_obs
                    del self.right_obs
                self.plotter.update()

    def start_erase(self, obj, event):
        """Triggers when left mouse button is pressed to start sweeping."""
        self.is_erasing = True
        self.do_erase(obj, event)

    def stop_erase(self, obj, event):
        """Triggers when left mouse button is released."""
        self.is_erasing = False

    def do_erase(self, obj, event):
        """Sweeps and deletes any atom or bond using 3D Spatial GPU Intersection."""
        if not getattr(self, 'is_erasing', False) or not getattr(self.chk_edit_mode, 'isChecked', lambda: False)(): 
            return
            
        import vtk
        click_pos = obj.GetEventPosition()
        
        # Switch to CellPicker to get the physical 3D intersection coordinate
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005) 
        picker.Pick(click_pos[0], click_pos[1], 0, self.plotter.renderer)
        
        if picker.GetActor():
            pick_pos = picker.GetPickPosition()
            erased_something = False
            
            # 1. Check if we hit an Atom
            if hasattr(self, 'atom_tree'):
                dist, idx = self.atom_tree.query(pick_pos)
                # If click was within ~1 Angstrom of an atom center
                if dist < 1.0: 
                    if idx not in self.erased_atoms:
                        self.erased_atoms.add(idx)
                        erased_something = True
                        
            # 2. Check if we hit a Bond (Only if an atom wasn't just deleted to prevent double-kills)
            if not erased_something and hasattr(self, 'bond_tree') and self.bond_tree is not None:
                dist, idx = self.bond_tree.query(pick_pos)
                # Tighter tolerance for thin bonds
                if dist < 0.6: 
                    pair = self.bond_pairs_list[idx]
                    if pair not in getattr(self, 'erased_bonds', set()):
                        self.erased_bonds.add(pair)
                        erased_something = True
                        
            # 3. Fast GPU Redraw (keeping the camera exactly where it is)
            if erased_something:
                self.draw_pyvista(self.active_supercell, keep_camera=True)

    def exit_eraser_mode(self, *args):
        """Quickly escapes eraser mode via right-click."""
        if getattr(self, 'chk_edit_mode', None):
            self.chk_edit_mode.setChecked(False) # This auto-triggers toggle_edit_mode

    def export_3dsmax_script(self, *args):
        """Generates a pymxs Python script to recreate the exact trimmed crystal in 3ds Max."""
        if not hasattr(self, 'active_supercell'):
            print("Please draw a structure first!")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, "Export 3ds Max Script", "", "Python Files (*.py)")
        if not file_path: return

        # 1. Gather valid atoms (ignoring erased ones)
        atoms_data = []
        scale_mod = self.spin_radius.value()
        for i, site in enumerate(self.active_supercell):
            if i in getattr(self, 'erased_atoms', set()): continue
            radius = float((site.specie.atomic_radius if site.specie.atomic_radius else 1.0) * scale_mod)
            color = self.active_colors.get(site.specie.symbol, "#008080")
            coords = site.coords
            atoms_data.append((float(coords[0]), float(coords[1]), float(coords[2]), radius, color))

        # 2. Gather valid bonds (ignoring erased ones)
        bonds_data = []
        if self.combo_style.currentIndex() == 0:  # If bonds are active
            cyl_radius = float(self.spin_bond_thick.value())
            bond_color = self.active_colors.get("Bonds", "#FFFFFF")
            coords = self.active_supercell.cart_coords
            radii = np.array([s.specie.atomic_radius if s.specie.atomic_radius else 1.2 for s in self.active_supercell])
            thresh = self.spin_bond_thresh.value()
            dist_mat = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=-1)
            threshold_mat = (radii[:, np.newaxis] + radii[np.newaxis, :]) * thresh
            valid_pairs = np.triu((dist_mat > 0.5) & (dist_mat <= threshold_mat), k=1)
            
            for i, j in np.argwhere(valid_pairs):
                if (i, j) in getattr(self, 'erased_bonds', set()) or \
                   i in getattr(self, 'erased_atoms', set()) or j in getattr(self, 'erased_atoms', set()):
                    continue
                p1, p2 = coords[i], coords[j]
                bonds_data.append((float(p1[0]), float(p1[1]), float(p1[2]), 
                                   float(p2[0]), float(p2[1]), float(p2[2]), 
                                   cyl_radius, bond_color))

        # 3. Gather lattice boundaries
        lattice_data = []
        if self.current_structure is not None and (self.chk_show_conventional.isChecked() or self.chk_show_primitive.isChecked()):
            def get_edges(matrix, color):
                a, b, c = matrix[0], matrix[1], matrix[2]
                v = np.zeros((8, 3))
                v[0] = [0, 0, 0]
                v[1], v[2], v[3] = a, b, c
                v[4], v[5], v[6] = a + b, a + c, b + c
                v[7] = a + b + c
                edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]
                for p1_idx, p2_idx in edges:
                    p1, p2 = v[p1_idx], v[p2_idx]
                    lattice_data.append((float(p1[0]), float(p1[1]), float(p1[2]), 
                                         float(p2[0]), float(p2[1]), float(p2[2]), color))

            if self.chk_show_conventional.isChecked():
                try:
                    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                    sga = SpacegroupAnalyzer(self.current_structure)
                    conv_struct = sga.get_conventional_standard_structure()
                    get_edges(conv_struct.lattice.matrix, "#FF5733")
                except Exception:
                    get_edges(self.current_structure.lattice.matrix, "#FF5733")

            if self.chk_show_primitive.isChecked():
                try:
                    prim_struct = self.current_structure.get_primitive_structure()
                    get_edges(prim_struct.lattice.matrix, "#33FF57")
                except Exception:
                    get_edges(self.current_structure.lattice.matrix, "#33FF57")

        # Format arrays with line breaks for clean reading
        atoms_formatted = ",\n        ".join(str(a) for a in atoms_data)
        bonds_formatted = ",\n        ".join(str(b) for b in bonds_data)
        lattice_formatted = ",\n        ".join(str(l) for l in lattice_data)

        # 4. Generate the actual 3ds Max Python script text
        script_content = f'''import pymxs
rt = pymxs.runtime

def hex_to_color(hex_str):
    h = hex_str.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def get_material(hex_color):
    mat_name = "Mat_" + hex_color.replace('#', '')
    mat = rt.getNodeByName(mat_name)
    if not mat:
        mat = rt.PhysicalMaterial()
        mat.name = mat_name
        r, g, b = hex_to_color(hex_color)
        mat.Base_Color = rt.Color(r, g, b)
        mat.roughness = 0.1 # Shiny finish
    return mat

with pymxs.undo(True, 'Create Crystal'):
    # --- ATOMS (Instanced for Memory/Poly optimization) ---
    atoms = [
        {atoms_formatted}
    ]
    base_atoms = {{}}
    for ax, ay, az, rad, color in atoms:
        key = (color, rad)
        if key not in base_atoms:
            sphere = rt.Sphere(radius=rad, segs=32)
            sphere.material = get_material(color)
            sphere.pos = rt.Point3(ax, ay, az)
            base_atoms[key] = sphere
        else:
            inst = rt.instance(base_atoms[key])
            inst.pos = rt.Point3(ax, ay, az)

    # --- BONDS (Unique Objects to avoid scaling artifacts) ---
    bonds = [
        {bonds_formatted}
    ]
    for x1, y1, z1, x2, y2, z2, rad, color in bonds:
        p1 = rt.Point3(x1, y1, z1)
        p2 = rt.Point3(x2, y2, z2)
        dist = rt.distance(p1, p2)
        
        cyl = rt.Cylinder(radius=rad, height=dist, sides=16)
        cyl.material = get_material(color)
        cyl.pos = p1
        vec = p2 - p1
        cyl.dir = rt.normalize(vec) 

    # --- LATTICE BOUNDARIES (Unique Objects) ---
    lattice_edges = [
        {lattice_formatted}
    ]
    for x1, y1, z1, x2, y2, z2, color in lattice_edges:
        p1 = rt.Point3(x1, y1, z1)
        p2 = rt.Point3(x2, y2, z2)
        dist = rt.distance(p1, p2)
        
        cyl = rt.Cylinder(radius=0.05, height=dist, sides=12)
        cyl.material = get_material(color)
        cyl.pos = p1
        vec = p2 - p1
        cyl.dir = rt.normalize(vec)

rt.redrawViews()
print("Crystal successfully imported!")
'''
        
        # 5. Save to file
        with open(file_path, 'w') as f:
            f.write(script_content)
        
        print(f"3ds Max Script saved to {file_path}")
    

    def export_blender_script(self, *args):
        """Generates a Blender Python script to recreate the exact crystal structure."""
        if not hasattr(self, 'active_supercell'):
            print("Please draw a structure first!")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Blender Script", "", "Python Files (*.py)")
        if not file_path: return

        # 1. Gather Atoms
        atoms_data = []
        scale_mod = self.spin_radius.value()
        for i, site in enumerate(self.active_supercell):
            if i in getattr(self, 'erased_atoms', set()): continue
            radius = float((site.specie.atomic_radius if site.specie.atomic_radius else 1.0) * scale_mod)
            color = self.active_colors.get(site.specie.symbol, "#008080")
            atoms_data.append((float(site.coords[0]), float(site.coords[1]), float(site.coords[2]), radius, color))

        # 2. Gather Bonds
        bonds_data = []
        if self.combo_style.currentIndex() == 0:
            cyl_radius = float(self.spin_bond_thick.value())
            bond_color = self.active_colors.get("Bonds", "#FFFFFF")
            coords = self.active_supercell.cart_coords
            radii = np.array([s.specie.atomic_radius if s.specie.atomic_radius else 1.2 for s in self.active_supercell])
            thresh = self.spin_bond_thresh.value()
            dist_mat = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=-1)
            threshold_mat = (radii[:, np.newaxis] + radii[np.newaxis, :]) * thresh
            valid_pairs = np.triu((dist_mat > 0.5) & (dist_mat <= threshold_mat), k=1)
            
            for i, j in np.argwhere(valid_pairs):
                if (i, j) in getattr(self, 'erased_bonds', set()) or i in getattr(self, 'erased_atoms', set()) or j in getattr(self, 'erased_atoms', set()):
                    continue
                bonds_data.append((float(coords[i][0]), float(coords[i][1]), float(coords[i][2]), 
                                   float(coords[j][0]), float(coords[j][1]), float(coords[j][2]), cyl_radius, bond_color))

        # 3. Gather Lattice
        lattice_data = []
        if self.current_structure is not None and (self.chk_show_conventional.isChecked() or self.chk_show_primitive.isChecked()):
            def get_edges(matrix, color):
                a, b, c = matrix[0], matrix[1], matrix[2]
                v = np.zeros((8, 3))
                v[1], v[2], v[3] = a, b, c
                v[4], v[5], v[6], v[7] = a + b, a + c, b + c, a + b + c
                edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]
                for p1, p2 in edges:
                    lattice_data.append((float(v[p1][0]), float(v[p1][1]), float(v[p1][2]), 
                                         float(v[p2][0]), float(v[p2][1]), float(v[p2][2]), color))

            if self.chk_show_conventional.isChecked():
                try:
                    sga = __import__('pymatgen.symmetry.analyzer').symmetry.analyzer.SpacegroupAnalyzer(self.current_structure)
                    get_edges(sga.get_conventional_standard_structure().lattice.matrix, "#FF5733")
                except Exception:
                    get_edges(self.current_structure.lattice.matrix, "#FF5733")
            if self.chk_show_primitive.isChecked():
                try:
                    get_edges(self.current_structure.get_primitive_structure().lattice.matrix, "#33FF57")
                except Exception:
                    get_edges(self.current_structure.lattice.matrix, "#33FF57")

        # Format strings
        atoms_formatted = ",\n        ".join(str(a) for a in atoms_data)
        bonds_formatted = ",\n        ".join(str(b) for b in bonds_data)
        lattice_formatted = ",\n        ".join(str(l) for l in lattice_data)

        # 4. Generate Blender Script
        script_content = f'''import bpy
import mathutils

# Create Crystal Collection
coll_name = "TensorSpec_Crystal"
if coll_name not in bpy.data.collections:
    crystal_coll = bpy.data.collections.new(coll_name)
    bpy.context.scene.collection.children.link(crystal_coll)
else:
    crystal_coll = bpy.data.collections[coll_name]

def hex_to_rgba(hex_str):
    h = hex_str.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4)) + (1.0,)

def get_material(hex_color):
    mat_name = "Mat_" + hex_color.replace('#', '')
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Base Color'].default_value = hex_to_rgba(hex_color)
            bsdf.inputs['Roughness'].default_value = 0.15
            bsdf.inputs['Metallic'].default_value = 0.2
    return mat

def create_base_sphere(name, radius, color):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, segments=32, ring_count=16)
    obj = bpy.context.active_object
    obj.data.materials.append(get_material(color))
    bpy.ops.object.shade_smooth()
    mesh = obj.data
    bpy.data.objects.remove(obj) # Keep mesh data, delete placeholder
    return mesh

# --- ATOMS (Linked Instances) ---
atoms = [
    {atoms_formatted}
]
base_meshes = {{}}
for ax, ay, az, rad, color in atoms:
    key = (color, rad)
    if key not in base_meshes:
        base_meshes[key] = create_base_sphere("AtomMesh", rad, color)
    
    obj = bpy.data.objects.new("Atom", base_meshes[key])
    obj.location = (ax, ay, az)
    crystal_coll.objects.link(obj)

# --- Helper for Cylinders ---
def create_cylinder_between_points(p1, p2, rad, color, verts):
    vec = p2 - p1
    dist = vec.length
    if dist == 0: return
    
    bpy.ops.mesh.primitive_cylinder_add(radius=rad, depth=dist, vertices=verts)
    obj = bpy.context.active_object
    obj.data.materials.append(get_material(color))
    bpy.ops.object.shade_smooth()
    
    obj.location = (p1 + p2) / 2
    rot_quat = mathutils.Vector((0, 0, 1)).rotation_difference(vec)
    obj.rotation_euler = rot_quat.to_euler()
    
    bpy.context.collection.objects.unlink(obj)
    crystal_coll.objects.link(obj)

# --- BONDS (Unique) ---
bonds = [
    {bonds_formatted}
]
for x1, y1, z1, x2, y2, z2, rad, color in bonds:
    p1 = mathutils.Vector((x1, y1, z1))
    p2 = mathutils.Vector((x2, y2, z2))
    create_cylinder_between_points(p1, p2, rad, color, 16)

# --- LATTICE BOUNDARIES (Unique) ---
lattice = [
    {lattice_formatted}
]
for x1, y1, z1, x2, y2, z2, color in lattice:
    p1 = mathutils.Vector((x1, y1, z1))
    p2 = mathutils.Vector((x2, y2, z2))
    create_cylinder_between_points(p1, p2, 0.05, color, 12)

print("Crystal imported successfully to Blender!")
'''
        with open(file_path, 'w') as f:
            f.write(script_content)
        
        print(f"Blender Script saved to {file_path}")

    def on_pyvista_pick(self, mesh):
        """Safely removes ONLY the clicked actor without continuous hovering."""
        if not self.chk_edit_mode.isChecked() or mesh is None: return
        for actor in list(self.plotter.renderer.actors.values()):
            if getattr(actor, 'mapper', None) and actor.mapper.dataset == mesh:
                # Write to memory bank so it stays deleted on CDW redraws
                if hasattr(actor, '_atom_index') and hasattr(self, 'erased_atoms'): 
                    self.erased_atoms.add(actor._atom_index)
                if hasattr(actor, '_bond_pair') and hasattr(self, 'erased_bonds'): 
                    self.erased_bonds.add(actor._bond_pair)
                
                self.plotter.remove_actor(actor)
                break
        self.plotter.update()

    def on_matplotlib_pick(self, event):
        """Fallback single click selection."""
        if self.chk_edit_mode.isChecked():
            event.artist.remove()
            self.canvas.draw()

    def start_matplotlib_eraser(self, event):
        if self.chk_edit_mode.isChecked() and event.button == 1:
            self.is_eraser_dragging = True

    def stop_matplotlib_eraser(self, event):
        if event.button == 1:
            self.is_eraser_dragging = False

    def matplotlib_eraser_brush(self, event):
        """Wipes out atoms and bonds under the crosshair continuously while dragging."""
        if self.chk_edit_mode.isChecked() and getattr(self, 'is_eraser_dragging', False) and event.inaxes == self.ax:
            # Check if mouse is hovering over an atom collection
            for collection in list(self.ax.collections):
                contained, _ = collection.contains(event)
                if contained:
                    collection.remove()
                    self.canvas.draw_idle()
                    return
            # Check if mouse is hovering over a bond line
            for line in list(self.ax.lines):
                contained, _ = line.contains(event)
                if contained:
                    line.remove()
                    self.canvas.draw_idle()
                    return

    # ================= M A T P L O T L I B   R E N D E R =================
    def draw_matplotlib(self, supercell):
        self.ax.clear()
        self.ax.set_facecolor("#1e1e24")
        self.ax.axis('off')

        scale_mod = self.spin_radius.value()
        style = self.combo_style.currentIndex()

        for site in supercell:
            x, y, z = site.coords
            radius = (site.specie.atomic_radius if site.specie.atomic_radius else 1.0) * scale_mod
            atom_color = self.active_colors.get(site.specie.symbol, "#008080")
            self.ax.scatter(x, y, z, s=(radius * 800), c=atom_color, edgecolors='black', picker=True)

        if style == 0: self.draw_matplotlib_bonds(supercell)
        elif style == 1: self.draw_matplotlib_polyhedra(supercell)
        self.canvas.draw()

    def draw_matplotlib_bonds(self, supercell):
        line_thick = self.spin_bond_thick.value() * 40 # Scale thickness for Matplotlib
        bond_color = self.active_colors["Bonds"]
        for i in range(len(supercell)):
            for j in range(i + 1, len(supercell)):
                a, b = supercell[i], supercell[j]
                dist = np.linalg.norm(a.coords - b.coords)
                ra = a.specie.atomic_radius if a.specie.atomic_radius else 1.2
                rb = b.specie.atomic_radius if b.specie.atomic_radius else 1.2
                if 0.5 < dist <= (ra + rb) * 1.35:
                    self.ax.plot([a.coords[0], b.coords[0]], [a.coords[1], b.coords[1]], [a.coords[2], b.coords[2]], 
                                 color=bond_color, linewidth=line_thick, picker=True)

    def draw_matplotlib_polyhedra(self, supercell):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        nn_finder = CrystalNN(distance_cutoffs=None, x_diff_weight=0.0, porous_adjustment=False)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, site in enumerate(supercell):
                if site.specie.symbol in ["Te", "O", "S", "Se"]: continue 
                try:
                    neighbors = nn_finder.get_nn_info(supercell, i)
                    pts = np.array([n['site'].coords for n in neighbors if tuple(n['image']) == (0,0,0)])
                    if len(pts) >= 4:
                        hull = ConvexHull(pts)
                        faces = [pts[simplex] for simplex in hull.simplices]
                        poly_color = self.active_colors.get(site.specie.symbol, "#008080")
                        poly = Poly3DCollection(faces, alpha=0.3, facecolor=poly_color, edgecolor='black', picker=True)
                        self.ax.add_collection3d(poly)
                except:
                    continue

    # ================= P Y V I S T A   R E N D E R =================
    def draw_pyvista(self, supercell, keep_camera=False):
        # Save camera state so erasing doesn't jerk the screen
        cam_pos = self.plotter.camera_position if keep_camera else None
        
        self.plotter.clear()
        self.plotter.set_background("#1e1e24")
        
        # --- NEW: Build Atom Spatial Tree for high-speed eraser picking ---
        from scipy.spatial import KDTree
        self.atom_tree = KDTree(supercell.cart_coords)
        scale_mod = self.spin_radius.value()
        style = self.combo_style.currentIndex()
        is_shiny = self.chk_shiny.isChecked()

        # --- OPTIMIZED ATOM DRAWING (GPU GLYPHING + ADAPTIVE LOD) ---
        is_shiny = self.chk_shiny.isChecked()
        num_atoms = len(supercell)
        
        # Dynamically drop resolution for massive supercells to save VRAM
        if num_atoms > 8000: res = 8
        elif num_atoms > 2000: res = 12
        else: res = 40 if is_shiny else 20

        # 1. Group atoms by their tag/layer to bulk-apply colors
        from collections import defaultdict
        atom_coords = defaultdict(list)
        atom_radii = {}

        for i, site in enumerate(supercell):
            if i in getattr(self, 'erased_atoms', set()): continue

            if "layer_tag" in supercell.site_properties:
                tag = supercell.site_properties["layer_tag"][i]
            else:
                tag = site.specie.symbol

            atom_coords[tag].append(site.coords)
            atom_radii[tag] = (site.specie.atomic_radius if site.specie.atomic_radius else 1.0) * scale_mod

        # 2. Instantly stamp out each group on the GPU
        for tag, coords in atom_coords.items():
            if not coords: continue
            
            color = self.active_colors.get(tag, "#008080")
            rad = atom_radii[tag]

            poly = pv.PolyData(np.array(coords))
            sphere = pv.Sphere(radius=rad, theta_resolution=res, phi_resolution=res)

            # factor=1.0 ensures it uses our exact radius without auto-scaling
            glyphs = poly.glyph(geom=sphere, factor=1.0)

            self.plotter.add_mesh(glyphs, color=color, smooth_shading=True, pickable=False, render=False)

        if style == 0: self.draw_pyvista_bonds(supercell)
        elif style == 1: self.draw_pyvista_polyhedra(supercell)
        
        self.toggle_shiny_materials()
        if self.chk_show_axes.isChecked(): self.draw_pyvista_axes()
        
        self.update_lattice_boxes()

        # Restore camera or reset it based on the flag
        if keep_camera and cam_pos:
            self.plotter.camera_position = cam_pos
        else:
            self.plotter.reset_camera()
            
        self.plotter.render()

    def draw_pyvista_bonds(self, supercell):
        # --- MACRO-SCALE KILL SWITCH ---
        # If there are more than 20,000 atoms, skip bond calculations entirely. 
        # It prevents RAM freezing and bonds are visually indistinguishable at this scale anyway.
        if len(supercell) > 20000:
            print(f"Skipping bonds: Structure too massive ({len(supercell)} atoms).")
            self.bond_tree = None
            return

        cyl_radius = self.spin_bond_thick.value()
        bond_color = self.active_colors.get("Bonds", "#FFFFFF")
        
        coords = supercell.cart_coords
        radii = np.array([s.specie.atomic_radius if s.specie.atomic_radius else 1.2 for s in supercell])
        thresh_multiplier = self.spin_bond_thresh.value()
        
        # --- OPTIMIZATION 1: Use Scipy KDTree for high-speed spatial neighbor queries ---
        from scipy.spatial import KDTree
        tree = KDTree(coords)
        
        # Find all pairs within a conservative maximum possible bond distance
        pairs = tree.query_pairs(r=4.0)
        
        if not pairs:
            self.bond_tree = None
            return

        # --- OPTIMIZATION 2: C-Level Vectorized Bond Math ---
        # Convert pairs to NumPy arrays to calculate all distances instantly
        pairs_arr = np.array(list(pairs))
        i_idx = pairs_arr[:, 0]
        j_idx = pairs_arr[:, 1]
        
        vecs = coords[j_idx] - coords[i_idx]
        dists = np.linalg.norm(vecs, axis=1)
        
        # Create a boolean mask for valid bonds based on atomic radii thresholds
        rad_sums = (radii[i_idx] + radii[j_idx]) * thresh_multiplier
        mask = (dists > 0.5) & (dists <= rad_sums)
        
        # Filter down to only the valid bonds instantly
        valid_i = i_idx[mask]
        valid_j = j_idx[mask]
        valid_dists = dists[mask]
        valid_vecs = vecs[mask]

        bond_centers = []
        bond_directions = []
        bond_heights = []
        self.bond_pairs_list = []
        
        erased_bonds = getattr(self, 'erased_bonds', set())
        erased_atoms = getattr(self, 'erased_atoms', set())

        # Now apply the manual eraser filter (on a vastly reduced list)
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
            
        # --- NEW: Build Bond Spatial Tree ---
        self.bond_tree = KDTree(bond_centers)

        # --- OPTIMIZATION 3: Hardware GPU Glyphing with Adaptive LOD ---
        bonds_data = pv.PolyData(np.array(bond_centers))
        bonds_data["scale"] = np.array(bond_heights)
        bonds_data["vectors"] = np.array(bond_directions)
        
        # Adaptive Level of Detail for cylinders
        num_bonds = len(bond_centers)
        if num_bonds > 20000: res = 6
        elif num_bonds > 5000: res = 8
        else: res = 20 if getattr(self, 'chk_shiny', None) and self.chk_shiny.isChecked() else 8
        
        source_cylinder = pv.Cylinder(center=(0, 0, 0), direction=(1, 0, 0), 
                                      radius=cyl_radius, height=1.0, resolution=res)
        
        # Instantly stamp out all cylinders natively on the GPU core
        glyphs = bonds_data.glyph(orient="vectors", scale="scale", geom=source_cylinder)
        
        # Add the entire bond grid as a single ultra-fast actor
        actor = self.plotter.add_mesh(glyphs, color=bond_color, smooth_shading=True, pickable=False, render=False)
        actor._bond_pair = (-1, -1)
    
    def draw_pyvista_axes(self, *args):
        """Draws or removes the a, b, c lattice vectors dynamically."""
        if self.combo_backend.currentIndex() == 0 or self.current_structure is None: return
        
        import pyvista as pv
        import numpy as np
        
        if not hasattr(self, 'axis_actors'):
            self.axis_actors = []
        for actor in self.axis_actors:
            self.plotter.remove_actor(actor)
        self.axis_actors.clear()
        
        if not self.chk_show_axes.isChecked():
            self.plotter.render()
            self.plotter.update()
            return
            
        matrix = self.current_structure.lattice.matrix
        bounds = self.plotter.bounds
        
        # Calculate the diagonal size of the entire drawn crystal
        diag = np.linalg.norm([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])
        
        # Spawn axes at the bottom-left-front corner of the bounds
        origin = np.array([bounds[0], bounds[2], bounds[4]])
        
        # Pull the origin away from the crystal slightly so it doesn't clip
        origin = origin - (np.array([1, 1, 1]) * (diag * 0.05))
        
        colors = ["red", "green", "blue"]
        labels = ["a", "b", "c"]
        
        # Make the arrows 20% the size of the whole crystal
        arrow_scale = diag * 0.2 
        
        for vec, color, label in zip(matrix, colors, labels):
            length = np.linalg.norm(vec)
            if length == 0: continue
            
            direction = vec / length
            
            # Thinner shafts and tips so they don't look bulky
            arrow = pv.Arrow(start=origin, direction=direction, scale=arrow_scale, 
                             shaft_radius=0.015, tip_radius=0.04)
            
            arrow_actor = self.plotter.add_mesh(arrow, color=color, smooth_shading=True, pickable=False, render=False)
            self.axis_actors.append(arrow_actor)
            
            label_pos = origin + direction * (arrow_scale * 1.15)
            text_actor = self.plotter.add_point_labels([label_pos], [label], text_color=color, 
                                          font_size=20, shape_opacity=0.0, show_points=False, render=False)
            self.axis_actors.append(text_actor)
            
        self.plotter.render()
        self.plotter.update()
    
    def update_lattice_boxes(self, *args):
        """Draws or removes the unit cell wireframes independently without redrawing atoms."""
        if self.combo_backend.currentIndex() == 0 or self.current_structure is None: return

        # 1. Clean up old wireframe actors
        if not hasattr(self, 'lattice_actors'):
            self.lattice_actors = []
        for actor in self.lattice_actors:
            self.plotter.remove_actor(actor)
        self.lattice_actors.clear()

        # If both are unchecked, just render the cleared scene and exit
        if not self.chk_show_conventional.isChecked() and not self.chk_show_primitive.isChecked():
            self.plotter.render()
            return

        def draw_lattice_box(matrix, color):
            a, b, c = matrix[0], matrix[1], matrix[2]
            v = np.zeros((8, 3))
            v[0] = [0, 0, 0]
            v[1], v[2], v[3] = a, b, c
            v[4], v[5], v[6] = a + b, a + c, b + c
            v[7] = a + b + c
            edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]
            for p1, p2 in edges:
                line = pv.Line(v[p1], v[p2])
                actor = self.plotter.add_mesh(line, color=color, line_width=3, render=False)
                self.lattice_actors.append(actor) # Save actor to memory bank for easy deletion

        # 2. Render Conventional standard cell boundary
        if self.chk_show_conventional.isChecked():
            try:
                from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                sga = SpacegroupAnalyzer(self.current_structure)
                conv_struct = sga.get_conventional_standard_structure()
                draw_lattice_box(conv_struct.lattice.matrix, "#FF5733")
            except Exception:
                draw_lattice_box(self.current_structure.lattice.matrix, "#FF5733")

        # 3. Render Primitive cell boundary 
        if self.chk_show_primitive.isChecked():
            try:
                prim_struct = self.current_structure.get_primitive_structure()
                draw_lattice_box(prim_struct.lattice.matrix, "#33FF57")
            except Exception:
                draw_lattice_box(self.current_structure.lattice.matrix, "#33FF57")
                
        self.plotter.render()

    def draw_pyvista_polyhedra(self, supercell):
        nn_finder = CrystalNN(distance_cutoffs=None, x_diff_weight=0.0, porous_adjustment=False)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, site in enumerate(supercell):
                if site.specie.symbol in ["Te", "O", "S", "Se"]: continue
                try:
                    neighbors = nn_finder.get_nn_info(supercell, i)
                    pts = np.array([n['site'].coords for n in neighbors if tuple(n['image']) == (0,0,0)])
                    if len(pts) >= 4:
                        hull = ConvexHull(pts)
                        faces = np.column_stack((np.full(len(hull.simplices), 3), hull.simplices)).flatten()
                        poly = pv.PolyData(pts, faces)
                        poly_color = self.active_colors.get(site.specie.symbol, "#008080")
                        self.plotter.add_mesh(poly, color=poly_color, opacity=0.4, show_edges=True, pickable=True)
                except:
                    continue
    
    # ================= C A M E R A   C O N T R O L S =================
    def update_projection(self):
        """Switches between 3D Perspective and flat 2D Orthogonal projection."""
        is_ortho = self.combo_projection.currentIndex() == 1
        # Matplotlib projection switch
        self.ax.set_proj_type('ortho' if is_ortho else 'persp')
        self.canvas.draw_idle()
        # PyVista projection switch
        if is_ortho:
            self.plotter.camera.enable_parallel_projection()
        else:
            self.plotter.camera.disable_parallel_projection()
        self.plotter.render()

    def set_camera_view(self, view_type):
        """Snaps the camera to predefined crystallographic axes."""
        if self.combo_backend.currentIndex() == 0: # Matplotlib
            if view_type == 'x': self.ax.view_init(elev=0, azim=0)
            elif view_type == 'y': self.ax.view_init(elev=0, azim=90)
            elif view_type == 'z': self.ax.view_init(elev=90, azim=-90)
            elif view_type == 'iso': self.ax.view_init(elev=35.264, azim=45)
            self.canvas.draw_idle()
        else: # PyVista
            if view_type == 'x': self.plotter.view_yz() # Looking down X (+a)
            elif view_type == 'y': self.plotter.view_xz() # Looking down Y (+b)
            elif view_type == 'z': self.plotter.view_xy() # Looking down Z (+c)
            elif view_type == 'iso': self.plotter.view_isometric()
            self.plotter.render()
            
        # Sync the UI boxes to match the newly snapped view
        self.sync_ui_to_camera()

    def update_azel(self):
        """Physically calculates the spherical coordinates to rotate the camera based on input boxes."""
        az = self.spin_azimuth.value()
        el = self.spin_elevation.value()
        
        if self.combo_backend.currentIndex() == 0: # Matplotlib handles Az/El natively
            self.ax.view_init(elev=el, azim=az)
            self.canvas.draw_idle()
        else: # PyVista requires manual spherical-to-Cartesian coordinate mapping
            fp = np.array(self.plotter.camera.focal_point)
            dist = self.plotter.camera.distance
            
            az_rad = np.radians(az)
            el_rad = np.radians(el)
            
            # Map spherical degrees to 3D Cartesian coordinates (Z is Up)
            dx = dist * np.cos(el_rad) * np.cos(az_rad)
            dy = dist * np.cos(el_rad) * np.sin(az_rad)
            dz = dist * np.sin(el_rad)
            
            self.plotter.camera.position = fp + np.array([dx, dy, dz])
            self.plotter.camera.up = (0, 0, 1) # Lock the Z axis as "up" so it doesn't tilt sideways
            self.plotter.render()
    
    def align_to_hkl(self):
        """Calculates the physical Cartesian vector from Miller Indices and snaps the camera."""
        if self.current_structure is None: return
        h, k, l = self.spin_h.value(), self.spin_k.value(), self.spin_l.value()
        if h == 0 and k == 0 and l == 0: return
        
       # Convert (h k l) plane indices to real 3D Cartesian normal vector
        recip_matrix = self.current_structure.lattice.reciprocal_lattice.matrix
        cart_vec = h * recip_matrix[0] + k * recip_matrix[1] + l * recip_matrix[2]
        
        if self.combo_backend.currentIndex() == 0: # Matplotlib
            dist = np.linalg.norm(cart_vec)
            az = np.degrees(np.arctan2(cart_vec[1], cart_vec[0]))
            el = np.degrees(np.arcsin(cart_vec[2] / dist))
            self.ax.view_init(elev=el, azim=az)
            self.canvas.draw_idle()
        else: # PyVista
            fp = np.array(self.plotter.camera.focal_point)
            dist = self.plotter.camera.distance
            norm_vec = cart_vec / np.linalg.norm(cart_vec)
            self.plotter.camera.position = fp + norm_vec * dist
            self.plotter.camera.up = (0, 0, 1)
            self.plotter.render()
            
        self.sync_ui_to_camera()

    def toggle_cut_plane(self):
        """When the plane is turned on, lock its orientation in space."""
        if self.chk_show_plane.isChecked():
            self.lock_plane_normal()
        self.update_cut_plane()

    def toggle_shiny_materials(self, *args):
        """Applies a High-Quality PBR (Physically Based Rendering) look."""
        # Safety check: Only run this if we are in PyVista mode
        if getattr(self, 'combo_backend', None) and self.combo_backend.currentIndex() == 0:
            return 
        
        is_shiny = self.chk_shiny.isChecked()
            
        # Iterate through the direct PyVista actors dictionary
        for name, actor in self.plotter.actors.items():
            # Apply to everything that has a visual 3D property (atoms, bonds)
            if hasattr(actor, 'prop') and actor.prop is not None:
                if is_shiny:
                    # The ultimate 3ds Max look: Physically Based Rendering
                    actor.prop.interpolation = 'pbr'
                    actor.prop.metallic = 0.2      # Slight subsurface metal feel
                    actor.prop.roughness = 0.15    # Extremely polished and sharp reflections
                else:
                    # Default PyVista flat Gouraud shading
                    actor.prop.interpolation = 'Gouraud' 
                    actor.prop.metallic = 0.0
                    actor.prop.roughness = 1.0
                    
        # Force the graphics engine and Qt Window to redraw the materials immediately
        self.plotter.render()
        if hasattr(self.plotter, 'app'):
            self.plotter.app.processEvents()

    def lock_plane_normal(self, *args):
        """Memorizes the exact 3D orientation so the camera can move freely afterward."""
        if self.current_structure is None or self.combo_backend.currentIndex() == 0: return
        
        self.plane_center_base = np.array(self.plotter.camera.focal_point)
        
        if self.combo_plane_orient.currentIndex() == 0: # Lock to Camera
            fp = np.array(self.plotter.camera.focal_point)
            pos = np.array(self.plotter.camera.position)
            normal = pos - fp
        else: # Lock to [h k l]
            h, k, l = self.spin_h.value(), self.spin_k.value(), self.spin_l.value()
            if h == 0 and k == 0 and l == 0: 
                normal = np.array([0,0,1])
            else:
                # --- FIX: Use reciprocal lattice for mathematically correct plane normals ---
                recip_matrix = self.current_structure.lattice.reciprocal_lattice.matrix
                normal = h * recip_matrix[0] + k * recip_matrix[1] + l * recip_matrix[2]
                
        dist = np.linalg.norm(normal)
        self.plane_normal = normal / dist if dist != 0 else np.array([0,0,1])
        
        if self.chk_show_plane.isChecked():
            self.update_cut_plane()

    def update_cut_plane(self):
        """Draws a transparent plane along the LOCKED normal, adjustable by depth."""
        if self.combo_backend.currentIndex() == 0: return 
            
        # 1. Clean up old plane
        if hasattr(self, 'plane_actor') and self.plane_actor is not None:
            self.plotter.remove_actor(self.plane_actor)
            self.plane_actor = None
            
        if not self.chk_show_plane.isChecked() or self.current_structure is None:
            self.plotter.render()
            return
            
        # Fallback in case normal isn't locked yet
        if getattr(self, 'plane_normal', None) is None:
            self.lock_plane_normal()
            
        # 2. Use the LOCKED normal and base center
        normal = self.plane_normal
        base_center = self.plane_center_base
        
        # 3. Calculate dynamic depth mapping
        depth_val = self.slider_plane_depth.value() / 100.0 
        bounds = self.plotter.bounds
        diag = np.linalg.norm([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])
        
        # Slide the center along the locked normal vector
        center = base_center + normal * (depth_val * diag / 2.0)
        
        # 4. Generate Plane Geometry
        plane = pv.Plane(center=center, direction=normal, i_size=diag*1.5, j_size=diag*1.5)
        color = self.combo_plane_color.currentText()
        
        self.plane_actor = self.plotter.add_mesh(plane, color=color, opacity=0.35, pickable=False)
        self.plotter.render()

    def sync_ui_to_camera(self, *args):
        """Reads the actual camera position after a mouse drag and updates the UI boxes silently."""
        vec = None
        if self.combo_backend.currentIndex() == 0: # Matplotlib
            az = self.ax.azim
            el = self.ax.elev
            # Approximate cartesian vector for Matplotlib sync
            az_rad, el_rad = np.radians(az), np.radians(el)
            vec = np.array([np.cos(el_rad)*np.cos(az_rad), np.cos(el_rad)*np.sin(az_rad), np.sin(el_rad)])
        else: # PyVista
            fp = np.array(self.plotter.camera.focal_point)
            pos = np.array(self.plotter.camera.position)
            vec = pos - fp
            dist = np.linalg.norm(vec)
            if dist == 0: return
            
            # Map 3D Cartesian back to Spherical degrees
            az = np.degrees(np.arctan2(vec[1], vec[0]))
            el = np.degrees(np.arcsin(vec[2] / dist))

        # --- Sync Azimuth & Elevation ---
        self.spin_azimuth.blockSignals(True)
        self.spin_elevation.blockSignals(True)
        self.spin_azimuth.setValue(az)
        self.spin_elevation.setValue(el)
        self.spin_azimuth.blockSignals(False)
        self.spin_elevation.blockSignals(False)
        
       # --- NEW: Sync Fractional [h k l] ---
        if self.current_structure is not None and vec is not None:
            # Map physical view vector back to plane indices using the Reciprocal Inverse
            recip_inv_matrix = self.current_structure.lattice.reciprocal_lattice.inv_matrix
            frac_vec = np.dot(vec, recip_inv_matrix)
            
            # Normalize so the largest component is exactly 1 or -1
            max_val = np.max(np.abs(frac_vec))
            if max_val > 0: frac_vec = frac_vec / max_val
            
            self.spin_h.blockSignals(True)
            self.spin_k.blockSignals(True)
            self.spin_l.blockSignals(True)
            
            self.spin_h.setValue(frac_vec[0])
            self.spin_k.setValue(frac_vec[1])
            self.spin_l.setValue(frac_vec[2])
            
            self.spin_h.blockSignals(False)
            self.spin_k.blockSignals(False)
            self.spin_l.blockSignals(False)
    
    # ================= T A B  3:  S T A C K  &  T W I S T =================
    def add_stack_layer(self):
        # Create a popup dialog to choose between CIF or Template
        from PySide6.QtWidgets import QDialog, QRadioButton, QDialogButtonBox
        dialog = QDialog(self)
        dialog.setWindowTitle("Add New Layer")
        layout = QVBoxLayout(dialog)
        
        radio_cif = QRadioButton("Load from CIF File")
        radio_cif.setChecked(True)
        layout.addWidget(radio_cif)
        
        radio_template = QRadioButton("Load from Template:")
        layout.addWidget(radio_template)
        
        combo_template = QComboBox()
        combo_template.addItems([
            "Graphene (Monolayer)", "Graphene (AB Bilayer)", "Graphene (AA Bilayer)", "Graphene (ABA Trilayer)",
            "Graphene (ABC 3-Layer)", "Graphene (ABCA 4-Layer)", "Graphene (ABCAB 5-Layer)", "Graphene (ABCABC 6-Layer)", "Graphene (ABCABCA 7-Layer)",
            "hBN (Monolayer)", "MoS2 (1H)", "WS2 (1H)", "MoSe2 (1H)", "WSe2 (1H)", 
            "TaS2 (1H)", "TaSe2 (1H)", "VTe2 (1H)", "TaS2 (1T)", "TaSe2 (1T)", "VTe2 (1T)",
            "WTe2 (1T')", "MoTe2 (1T')", "TaIrTe4 (1T')", "NbIrTe4 (1T')",
            "Phosphorene", "Silicene"
        ])
        layout.addWidget(combo_template)
        
        # Link radio button to combo box
        combo_template.setEnabled(False)
        radio_template.toggled.connect(combo_template.setEnabled)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        if dialog.exec() == QDialog.Accepted:
            struct = None
            name = ""
            if radio_cif.isChecked():
                fname, _ = QFileDialog.getOpenFileName(self, 'Open CIF', '', "CIF files (*.cif)")
                if not fname: return
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        struct = Structure.from_file(fname)
                    name = fname.split('/')[-1].replace('.cif', '')
                except Exception as e:
                    print(f"Error loading CIF: {e}")
                    return
            else:
                name = combo_template.currentText()
                struct = self.generate_template_structure(name)
                
            if struct:
                self.append_layer_to_ui(name, struct)
    
    def append_layer_to_ui(self, name, struct):
        if len(self.stack_layers) == 0:
            default_z = 0.0
        else:
            highest_z = max([layer.spin_z.value() for layer in self.stack_layers])
            default_z = highest_z + 3.4
            
        layer_widget = StackLayerWidget(name, struct, default_z)
        layer_widget.btn_delete.clicked.connect(lambda: self.remove_stack_layer(layer_widget))
        
        self.layer_layout.addWidget(layer_widget)
        self.stack_layers.append(layer_widget)
        
        unique_elements = sorted(list(set([site.specie.symbol for site in struct])))
        for el in unique_elements:
            if el not in self.active_colors:
                self.active_colors[el] = CPK_COLORS.get(el, "#008080")
    
    def generate_template_structure(self, template_name):
        from pymatgen.core import Lattice, Structure
        
        # Enforce a massive Z-axis vacuum (25 Å) so layers don't mathematically interact in 3D algorithms
        c_vac = 25.0 
        
        if "Graphene" in template_name:
            a = 2.46
            lat = Lattice.hexagonal(a, c_vac)
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
                if layer == "A":
                    coords.extend([[1/3, 2/3, z], [2/3, 1/3, z]])
                elif layer == "B":
                    coords.extend([[0.0, 0.0, z], [1/3, 2/3, z]])
                elif layer == "C":
                    coords.extend([[0.0, 0.0, z], [2/3, 1/3, z]])
                    
            return Structure(lat, ["C"] * len(coords), coords)
            
        elif template_name == "hBN (Monolayer)":
            lat = Lattice.hexagonal(2.50, c_vac)
            return Structure(lat, ["B", "N"], [[1/3, 2/3, 0.5], [2/3, 1/3, 0.5]])
            
        elif "(1H)" in template_name:
            params = {
                "MoS2 (1H)": (3.16, 1.58, "Mo", "S"), 
                "WS2 (1H)": (3.15, 1.57, "W", "S"), 
                "MoSe2 (1H)": (3.29, 1.66, "Mo", "Se"), 
                "WSe2 (1H)": (3.28, 1.65, "W", "Se"), 
                "TaS2 (1H)": (3.31, 1.55, "Ta", "S"), 
                "TaSe2 (1H)": (3.43, 1.64, "Ta", "Se"),
                "VTe2 (1H)": (3.64, 1.65, "V", "Te")
            }
            a, z_dist, metal, chalc = params.get(template_name, (3.16, 1.58, "Mo", "S"))
            lat = Lattice.hexagonal(a, c_vac)
            z_frac = z_dist / c_vac
            coords = [[0, 0, 0.5], [1/3, 2/3, 0.5 + z_frac], [1/3, 2/3, 0.5 - z_frac]]
            return Structure(lat, [metal, chalc, chalc], coords)
            
        elif "(1T)" in template_name and not "1T'" in template_name:
            params = {
                "TaS2 (1T)": (3.36, 1.50, "Ta", "S"), 
                "TaSe2 (1T)": (3.48, 1.60, "Ta", "Se"), 
                "VTe2 (1T)": (3.64, 1.65, "V", "Te")
            }
            a, z_dist, metal, chalc = params.get(template_name, (3.36, 1.50, "Ta", "S"))
            lat = Lattice.hexagonal(a, c_vac)
            z_frac = z_dist / c_vac
            coords = [[0, 0, 0.5], [1/3, 2/3, 0.5 + z_frac], [2/3, 1/3, 0.5 - z_frac]]
            return Structure(lat, [metal, chalc, chalc], coords)
            
        elif "(1T')" in template_name:
            if "TaIrTe4" in template_name or "NbIrTe4" in template_name:
                # These materials have a DOUBLED unit cell along the b-axis (12 atoms)
                is_ta = "TaIrTe4" in template_name
                lat = Lattice.orthorhombic(3.77 if is_ta else 3.78, 12.4, c_vac)
                m1 = "Ta" if is_ta else "Nb"
                species = [m1, "Ir", m1, "Ir"] + ["Te"] * 8
                z_frac = 1.75 / c_vac
                coords = [
                    # Metals (y is halved compared to WTe2, then duplicated at +0.5)
                    [0.0, 0.125, 0.5], [0.5, 0.375, 0.5], 
                    [0.0, 0.625, 0.5], [0.5, 0.875, 0.5],
                    # Top Te
                    [0.5, 0.075, 0.5 + z_frac], [0.0, 0.425, 0.5 + z_frac],
                    [0.5, 0.575, 0.5 + z_frac], [0.0, 0.925, 0.5 + z_frac],
                    # Bottom Te
                    [0.0, 0.175, 0.5 - z_frac], [0.5, 0.325, 0.5 - z_frac],
                    [0.0, 0.675, 0.5 - z_frac], [0.5, 0.825, 0.5 - z_frac]
                ]
                return Structure(lat, species, coords)
            else:
                # Standard 1T' (WTe2, MoTe2) with single unit cell (6 atoms)
                is_w = "WTe2" in template_name
                lat = Lattice.orthorhombic(3.47 if is_w else 3.45, 6.32 if is_w else 6.30, c_vac)
                m = "W" if is_w else "Mo"
                species = [m, m, "Te", "Te", "Te", "Te"]
                z_frac = 1.75 / c_vac
                coords = [
                    [0.0, 0.25, 0.5], [0.5, 0.75, 0.5], 
                    [0.5, 0.15, 0.5 + z_frac], [0.0, 0.85, 0.5 + z_frac],
                    [0.0, 0.35, 0.5 - z_frac], [0.5, 0.65, 0.5 - z_frac]
                ]
                return Structure(lat, species, coords)

        elif template_name == "Phosphorene":
            lat = Lattice.orthorhombic(3.31, 4.38, c_vac)
            z_frac = 1.05 / c_vac
            coords = [
                [0.0, 0.0, 0.5 - z_frac], [0.5, 0.5, 0.5 - z_frac],
                [0.0, 0.17, 0.5 + z_frac], [0.5, 0.67, 0.5 + z_frac]
            ]
            return Structure(lat, ["P"] * 4, coords)

        elif template_name == "Silicene":
            lat = Lattice.hexagonal(3.86, c_vac)
            z_frac = 0.46 / c_vac
            coords = [[1/3, 2/3, 0.5 + z_frac], [2/3, 1/3, 0.5 - z_frac]]
            return Structure(lat, ["Si", "Si"], coords)

        return None

    def remove_stack_layer(self, widget):
        self.layer_layout.removeWidget(widget)
        widget.deleteLater()
        self.stack_layers.remove(widget)

    def draw_stack(self):
        if not self.stack_layers: return
        
        all_species = []
        all_coords = []
        
        # 1. Iterate through every layer in the UI
        for idx, layer in enumerate(self.stack_layers):
            struct = layer.struct
            sc_x, sc_y = layer.spin_x.value(), layer.spin_y.value()
            
            # Multiply 2D plane
            supercell = struct * (sc_x, sc_y, 1)
            
            z_shift = layer.spin_z.value()
            theta = np.radians(layer.spin_twist.value())
            
            # 2. Build mathematical transformation matrix (Z-axis rotation)
            rot_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [            0,              0, 1]
            ])
            
            # 3. Apply Matrix Rotation, then Z-Translation
            coords = supercell.cart_coords.copy()
            
            # Center the sheets so they overlap perfectly in the middle
            center_xy = np.mean(coords[:, :2], axis=0)
            coords[:, :2] -= center_xy
            
            rotated_coords = np.dot(coords, rot_matrix.T)
            
            # Shift Z (subtracting 12.5 to remove the c_vac vacuum padding from the templates)
            shifted_coords = rotated_coords + np.array([0, 0, z_shift - 12.5])
            
            # --- FIX: Store Layer ID as a property, keep symbol chemically valid ---
            layer_tag = f"_L{idx + 1}"
            for i, site in enumerate(supercell):
                all_species.append(site.specie.symbol) 
                all_coords.append(shifted_coords[i])
                
                # Add custom tag to a parallel list if we didn't define it yet
                if not hasattr(self, '_temp_layer_tags'): self._temp_layer_tags = []
                self._temp_layer_tags.append(f"{site.specie.symbol}{layer_tag}")
                
        # 4. Create a giant dummy structure so PyVista/Matplotlib code can be reused instantly
        from pymatgen.core import Lattice
        dummy_lattice = Lattice.cubic(500.0) # Massive bounds so the unit cell doesn't clip
        
        # Pass the tags in as a site_property
        self.active_supercell = Structure(
            dummy_lattice, 
            all_species, 
            all_coords, 
            coords_are_cartesian=True,
            site_properties={"layer_tag": getattr(self, '_temp_layer_tags', [])}
        )
        
        # Clear temporary tags list for next time
        if hasattr(self, '_temp_layer_tags'):
            del self._temp_layer_tags
        
        # Temporarily nullify single-crystal variables to prevent bounding box artifacts
        self.current_structure = None 
        self.erased_atoms = set()
        self.erased_bonds = set()
        
        # 5. Route to existing render engines
        if self.combo_backend.currentIndex() == 0: 
            self.draw_matplotlib(self.active_supercell)
        else: 
            self.draw_pyvista(self.active_supercell)
            
        # --- FIX: Force Tab 1's color panel to recognize the tagged elements from properties ---
        if "layer_tag" in self.active_supercell.site_properties:
            unique_elements = sorted(list(set(self.active_supercell.site_properties["layer_tag"])))
            self.build_dynamic_color_panel(unique_elements)
            
        self.update_projection()
        self.set_camera_view('z') # Automatically look straight down at the stack!
    
    def identify_moire(self):
        """Calculates the Moiré superlattice using reciprocal space vectors and checks commensurability."""
        if len(self.stack_layers) != 2:
            self.lbl_moire_status.setText("<b style='color: #d9534f;'>Error:</b> Moiré calculation currently requires exactly 2 layers.")
            self.lbl_moire_status.setStyleSheet("background-color: #3b2a2a; color: white; padding: 5px;")
            return

        import numpy as np
        
        # 1. Extract 2D Lattice Matrices (XY plane only)
        layer1, layer2 = self.stack_layers[0], self.stack_layers[1]
        m1 = layer1.struct.lattice.matrix[:2, :2]
        m2 = layer2.struct.lattice.matrix[:2, :2]
        
        # 2. Calculate relative twist angle
        theta = np.radians(layer2.spin_twist.value() - layer1.spin_twist.value())
        
        # Apply 2D rotation to Top Layer (Layer 2)
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        m2_rot = np.dot(m2, rot_matrix.T)

        # 3. Enter Reciprocal Space: G = (M^-1)^T
        # (We omit the 2*pi factor because it cancels out when converting back to real space)
        try:
            g1 = np.linalg.inv(m1).T
            g2_rot = np.linalg.inv(m2_rot).T
        except np.linalg.LinAlgError:
            self.lbl_moire_status.setText("Error: Degenerate 2D lattice detected.")
            return

        # 4. Find the Moiré Reciprocal Vector Difference (ΔG = G1 - G2)
        delta_g = g1 - g2_rot

        # 5. Revert back to Real Space to get the Moiré Superlattice Vectors
        try:
            m_moire = np.linalg.inv(delta_g).T
        except np.linalg.LinAlgError:
            self.lbl_moire_status.setText("<b style='color: #5bc0de;'>Perfect Alignment:</b> Twist is 0° with identical lattices. No Moiré pattern.")
            self.lbl_moire_status.setStyleSheet("background-color: #2b3b4a; color: white; padding: 5px;")
            return

        # 6. Commensurability Check: Does the Moiré cell divide perfectly into Layer 1?
        # S = M_moire * M1^-1
        S_matrix = np.dot(m_moire, np.linalg.inv(m1))
        S_round = np.round(S_matrix)
        
        # If the actual matrix is within 0.5% strain of a perfect integer matrix, it's commensurate!
        is_commensurate = np.all(np.abs(S_matrix - S_round) < 0.05)
        
        L_m = np.linalg.norm(m_moire[0]) # Periodicity in Ångströms

        if is_commensurate:
            n_cells = int(np.round(np.sqrt(np.abs(np.linalg.det(S_round)))))
            msg = f"<b style='color: #33FF57;'>🟢 COMMENSURATE SUPERCELL</b><br>" \
                  f"Periodicity (L<sub>M</sub>): <b>{L_m:.2f} Å</b><br>" \
                  f"Matches exactly <b>{n_cells} × {n_cells}</b> base unit cells."
            self.lbl_moire_status.setStyleSheet("background-color: #2a3b2a; color: white; padding: 5px; border-radius: 3px;")
        else:
            msg = f"<b style='color: #FFD700;'>🟡 INCOMMENSURATE ENVELOPE</b><br>" \
                  f"Periodicity (L<sub>M</sub>): <b>{L_m:.2f} Å</b><br>" \
                  f"Requires artificial strain to enforce periodic boundaries."
            self.lbl_moire_status.setStyleSheet("background-color: #3b3a2a; color: white; padding: 5px; border-radius: 3px;")

        self.lbl_moire_status.setText(msg)
        
        # 7. Route the 2D bounding box to PyVista for 3D drawing
        self.draw_moire_box(m_moire)

    def draw_moire_box(self, m_moire):
        """Draws a glowing gold bounding box representing the calculated Moiré envelope."""
        if self.combo_backend.currentIndex() == 0: return # Skip for Matplotlib
        
        import pyvista as pv
        
        # Clean up any old boxes
        if not hasattr(self, 'moire_actors'):
            self.moire_actors = []
        for actor in self.moire_actors:
            self.plotter.remove_actor(actor)
        self.moire_actors.clear()

        # Expand 2D Moiré matrix into 3D coordinates
        v0 = np.array([0, 0, 0])
        v1 = np.array([m_moire[0, 0], m_moire[0, 1], 0])
        v2 = np.array([m_moire[1, 0], m_moire[1, 1], 0])
        v3 = v1 + v2
        
        # Calculate exactly how tall the bounding box needs to be to wrap the layers
        z_vals = [layer.spin_z.value() for layer in self.stack_layers]
        z_min = min(z_vals) - 1.5
        z_max = max(z_vals) + 1.5

        # Define bottom and top planes
        b0, b1, b2, b3 = v0.copy(), v1.copy(), v2.copy(), v3.copy()
        b0[2] = b1[2] = b2[2] = b3[2] = z_min

        t0, t1, t2, t3 = v0.copy(), v1.copy(), v2.copy(), v3.copy()
        t0[2] = t1[2] = t2[2] = t3[2] = z_max

        # Construct the 12 wireframe edges
        edges = [
            (b0, b1), (b0, b2), (b1, b3), (b2, b3),
            (t0, t1), (t0, t2), (t1, t3), (t2, t3),
            (b0, t0), (b1, t1), (b2, t2), (b3, t3)
        ]

        # Draw to screen!
        for p1, p2 in edges:
            line = pv.Line(p1, p2)
            actor = self.plotter.add_mesh(line, color="#FFD700", line_width=4, render=False) # Glowing Gold
            self.moire_actors.append(actor)

        self.plotter.render()

if __name__ == '__main__':
    print("1. Python script started successfully!")
    
    app = QApplication.instance()
    if not app: 
        app = QApplication(sys.argv)
    print("2. QApplication initialized!")
    
    try:
        print("3. Attempting to build TensorSpecApp...")
        window = TensorSpecApp()
        print("4. App built successfully! Attempting to show window...")
        window.show()
        print("5. Window shown! Entering event loop...")
        sys.exit(app.exec())
    except Exception as e:
        print(f"CRITICAL ERROR CAUGHT: {e}")