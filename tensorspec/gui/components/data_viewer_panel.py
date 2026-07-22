import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, 
    QComboBox, QCheckBox, QGroupBox, QScrollArea, QFrame
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches

from tensorspec.core.data_models import TensorData


class MplCanvas(FigureCanvas):
    """Clean Matplotlib canvas embedded in PySide6."""
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, layout='tight')
        super().__init__(fig)


class DataViewerPanel(QWidget):
    """
    Agnostic N-Dimensional Data Viewer.
    Renders 1D, 2D, 3D, 4D, or N-D TensorData seamlessly by dynamically 
    spawning sliders for non-projected axes.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tensor_data: TensorData = None
        self.slider_widgets = {}  # Maps axis index -> QSlider object
        self.slider_labels = {}   # Maps axis index -> QLabel object
        
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        # --- TOP CONTROLS: Axis Projections ---
        proj_group = QGroupBox("Projection Setup")
        proj_layout = QHBoxLayout(proj_group)

        proj_layout.addWidget(QLabel("Plot X-Axis:"))
        self.combo_x_axis = QComboBox()
        self.combo_x_axis.currentIndexChanged.connect(self._on_axis_selection_changed)
        proj_layout.addWidget(self.combo_x_axis)

        proj_layout.addWidget(QLabel("Plot Y-Axis:"))
        self.combo_y_axis = QComboBox()
        self.combo_y_axis.currentIndexChanged.connect(self._on_axis_selection_changed)
        proj_layout.addWidget(self.combo_y_axis)

        proj_layout.addSpacing(20)

        proj_layout.addWidget(QLabel("Colormap:"))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(['magma', 'viridis', 'inferno', 'plasma', 'cividis', 'gray', 'twilight_shifted', 'PiYG'])
        self.combo_cmap.currentTextChanged.connect(self.update_plot)
        proj_layout.addWidget(self.combo_cmap)

        proj_layout.addWidget(QLabel("Contrast %:"))
        self.slider_contrast = QSlider(Qt.Orientation.Horizontal)
        self.slider_contrast.setRange(1, 200)
        self.slider_contrast.setValue(100)
        self.slider_contrast.valueChanged.connect(self.update_plot)
        proj_layout.addWidget(self.slider_contrast)

        main_layout.addWidget(proj_group)

        # --- MIDDLE: Matplotlib Plot Canvas ---
        self.canvas = MplCanvas(self)
        fig = self.canvas.figure
        
        # GridSpec: Main 2D slice on left, EDC/MDC profiles on right/bottom
        gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1])
        self.ax_main = fig.add_subplot(gs[0, 0])
        self.ax_profile_y = fig.add_subplot(gs[0, 1], sharey=self.ax_main)
        self.ax_profile_x = fig.add_subplot(gs[1, 0], sharex=self.ax_main)

        # Initialize blank image handle
        self.im_main = self.ax_main.imshow(np.zeros((10, 10)), origin='lower', aspect='auto', cmap='magma')
        self.line_prof_x, = self.ax_profile_x.plot([], [], 'b-')
        self.line_prof_y, = self.ax_profile_y.plot([], [], 'r-')

        # Crosshairs
        self.vline = self.ax_main.axvline(0, color='cyan', linestyle='--', alpha=0.7)
        self.hline = self.ax_main.axhline(0, color='cyan', linestyle='--', alpha=0.7)

        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        main_layout.addWidget(self.canvas)

        # --- BOTTOM: Dynamic Sliders Area ---
        self.sliders_group = QGroupBox("Dimension Slices & Integration")
        self.sliders_container = QVBoxLayout(self.sliders_group)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self.sliders_group)
        scroll.setMaximumHeight(180)
        
        main_layout.addWidget(scroll)

    def load_data(self, tensor_data: TensorData):
        """Bridge method: Pass a TensorData object to populate the viewer."""
        self.tensor_data = tensor_data

        # 1. Populate Axis Dropdowns
        self.combo_x_axis.blockSignals(True)
        self.combo_y_axis.blockSignals(True)
        self.combo_x_axis.clear()
        self.combo_y_axis.clear()

        for idx, (label, unit) in enumerate(zip(tensor_data.labels, tensor_data.units)):
            display_str = f"{label} ({unit})" if unit else label
            self.combo_x_axis.addItem(display_str, userData=idx)
            self.combo_y_axis.addItem(display_str, userData=idx)

        # Default: Axis 1 -> X, Axis 0 -> Y (or 0 for X if 1D)
        if tensor_data.ndim > 1:
            self.combo_x_axis.setCurrentIndex(1)
            self.combo_y_axis.setCurrentIndex(0)
        else:
            self.combo_x_axis.setCurrentIndex(0)
            self.combo_y_axis.setCurrentIndex(0)

        self.combo_x_axis.blockSignals(False)
        self.combo_y_axis.blockSignals(False)

        # 2. Rebuild Slicing Sliders
        self._rebuild_dynamic_sliders()

        # 3. Refresh Plot
        self.update_plot()

    def _rebuild_dynamic_sliders(self):
        """Spawns sliders for every dimension NOT currently selected for X or Y plot axes."""
        # Clear old sliders from UI
        while self.sliders_container.count():
            item = self.sliders_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.slider_widgets.clear()
        self.slider_labels.clear()

        if not self.tensor_data:
            return

        x_idx = self.combo_x_axis.currentData()
        y_idx = self.combo_y_axis.currentData()

        # Create a slider for all remaining dimensions
        for dim_idx in range(self.tensor_data.ndim):
            if dim_idx == x_idx or dim_idx == y_idx:
                continue  # Skip axes currently shown on 2D image

            axis_arr = self.tensor_data.axes[dim_idx]
            label = self.tensor_data.labels[dim_idx]
            unit = self.tensor_data.units[dim_idx]

            row_layout = QHBoxLayout()

            lbl = QLabel(f"{label}: {axis_arr[len(axis_arr)//2]:.3f} {unit}")
            lbl.setMinimumWidth(180)
            row_layout.addWidget(lbl)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, len(axis_arr) - 1)
            slider.setValue(len(axis_arr) // 2)
            
            # Capture dim_idx in lambda default arg
            slider.valueChanged.connect(lambda val, d=dim_idx: self._on_slider_moved(d, val))

            row_layout.addWidget(slider)
            
            self.sliders_container.addLayout(row_layout)
            self.slider_widgets[dim_idx] = slider
            self.slider_labels[dim_idx] = lbl

    def _on_axis_selection_changed(self):
        """Triggered when user changes X or Y plot dropdowns."""
        self._rebuild_dynamic_sliders()
        self.update_plot()

    def _on_slider_moved(self, dim_idx: int, val_idx: int):
        """Triggered when an N-D slicing slider moves."""
        axis_arr = self.tensor_data.axes[dim_idx]
        label = self.tensor_data.labels[dim_idx]
        unit = self.tensor_data.units[dim_idx]
        
        self.slider_labels[dim_idx].setText(f"{label}: {axis_arr[val_idx]:.3f} {unit}")
        self.update_plot()

    def update_plot(self):
        """Slices the N-D array down to 2D using active slider indices and renders it."""
        if not self.tensor_data:
            return

        x_idx = self.combo_x_axis.currentData()
        y_idx = self.combo_y_axis.currentData()
        
        # Build slicing tuple for N-D array
        slices = []
        for dim_idx in range(self.tensor_data.ndim):
            if dim_idx == x_idx or dim_idx == y_idx:
                slices.append(slice(None))  # Keep full axis
            else:
                slider_val = self.slider_widgets[dim_idx].value()
                slices.append(slider_val)   # Take slice at slider index

        sliced_data = self.tensor_data.value[tuple(slices)]

        # Handle axis transpositions for 2D display
        if x_idx < y_idx:
            sliced_data = sliced_data.T

        # Update Extent
        x_arr = self.tensor_data.axes[x_idx]
        y_arr = self.tensor_data.axes[y_idx]
        
        dx = (x_arr[-1] - x_arr[0]) / max(1, len(x_arr) - 1) if len(x_arr) > 1 else 0.1
        dy = (y_arr[-1] - y_arr[0]) / max(1, len(y_arr) - 1) if len(y_arr) > 1 else 0.1
        
        extent = (x_arr[0] - dx/2, x_arr[-1] + dx/2, y_arr[0] - dy/2, y_arr[-1] + dy/2)
        
        self.im_main.set_data(sliced_data)
        self.im_main.set_extent(extent)
        self.im_main.set_cmap(self.combo_cmap.currentText())

        # Contrast scaling
        c_scale = self.slider_contrast.value() / 100.0
        vmin, vmax = np.nanmin(sliced_data), np.nanmax(sliced_data)
        self.im_main.set_clim(vmin, vmin + (vmax - vmin) * c_scale + 1e-8)

        # Labels & Titles
        self.ax_main.set_xlabel(f"{self.tensor_data.labels[x_idx]} ({self.tensor_data.units[x_idx]})")
        self.ax_main.set_ylabel(f"{self.tensor_data.labels[y_idx]} ({self.tensor_data.units[y_idx]})")
        self.ax_main.set_title(f"Dataset: {self.tensor_data.data_type}")

        self.canvas.draw_idle()

    def _on_canvas_click(self, event):
        """Allows clicking on the plot to snap crosshairs/sliders."""
        if not event.inaxes or event.inaxes != self.ax_main or not self.tensor_data:
            return

        x_idx = self.combo_x_axis.currentData()
        y_idx = self.combo_y_axis.currentData()

        # Update crosshairs
        self.vline.set_xdata([event.xdata])
        self.hline.set_ydata([event.ydata])
        self.canvas.draw_idle()