import os
import h5py
import numpy as np
import pickle

from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFileDialog, QListWidget, QComboBox, 
                             QLabel, QTabWidget, QCheckBox, QGroupBox, QSplitter, 
                             QMessageBox, QProgressBar, QStatusBar, QSpinBox, QDoubleSpinBox,
                             QLineEdit, QMenu, QDialog, QApplication)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor

from tensorspec.gui.components.ml_tabs.active_learning_panel import ActiveLearningPanel
from tensorspec.gui.components.ml_tabs.alignment_panel import AlignmentPanel
from tensorspec.gui.components.ml_tabs.layout import scrollable, split_panel, tab_group
from tensorspec.gui.components.ml_tabs.simulate_al_panel import SimulateALPanel
from tensorspec.gui.ml_session import MLSession
from tensorspec.gui.maestroai.model_warehouse_tab import ModelWarehouseTab
from tensorspec.gui.maestroai.build_pipeline_tab import BuildPipelineTab
from tensorspec.gui.maestroai.train_model_tab import TrainModelTab

# --- UNIVERSAL COMPONENTS ---
from .maestro_loader import LoadWorker
from .maestro_fermi_viewer import FermiViewerWindow

# --- AI WORKERS ---
from .maestroai_training_ssl import TrainWorker
from .maestroai_training_sup import SupTrainWorker, SupTestWorker
from .maestroai_clustering import ClusterWorker

# --- AI VIEWERS & GUIDES ---
from .maestroai_guides import (MasterGuideDialog, SSLGuideDialog, ClusterGuideDialog, 
                              SupGuideDialog)
from .maestroai_viewers import MplCanvas, DendrogramDialog


class MaestroAIApp(QMainWindow):
    # Lets the main browser drop this window from its registry on close. Kept as
    # a plain QMainWindow rather than a FloatingViewerWindow because the suite
    # owns its own status bar and progress bar.
    window_closed = Signal(str)

    def __init__(self, win_id="ML Suite"):
        super().__init__()
        self.win_id = win_id
        self.setWindowTitle("MaestroAI Suite")
        self.setMinimumSize(1000, 620)
        self.resize(1500, 900)
        self.session = MLSession()
        self.workspace = self.session.workspace
        self.current_folder = ""
        self.current_view_data = None
        self.init_ui()
        self.session.viewer = self.viewer
        self.session.status_changed.connect(self._on_session_status)

    def _on_session_status(self, value, message):
        self.prog_bar.setValue(value)
        self.status.showMessage(message)
        if 0 < value < 100:
            self.prog_bar.setVisible(True)
        elif value >= 100 or value == 0:
            self.prog_bar.setVisible(False)

    def closeEvent(self, event):
        self.window_closed.emit(self.win_id)
        super().closeEvent(event)

    def init_ui(self):
        main_widget = QWidget()
        layout = QHBoxLayout(main_widget)
        
        # --- LEFT: BROWSER & WORKSPACE ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        self.btn_master_help = QPushButton("🚀 Getting Started: Workflow Guide")
        self.btn_master_help.setStyleSheet("font-weight: bold; color: #ff7f0e; padding: 8px; font-size: 15px;")
        self.btn_master_help.clicked.connect(self.show_master_guide)
        left_layout.addWidget(self.btn_master_help)
        left_layout.addSpacing(10)
        
        left_layout.addWidget(QLabel("<b>1. Files on Disk</b>"))
        
        dir_layout = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Paste folder path and press Enter...")
        self.path_input.returnPressed.connect(self.load_directory_from_input) 
        
        self.btn_dir = QPushButton("Browse")
        self.btn_dir.clicked.connect(self.select_directory)
        
        dir_layout.addWidget(self.path_input)
        dir_layout.addWidget(self.btn_dir)
        left_layout.addLayout(dir_layout)
        
        self.disk_list = QListWidget()
        self.btn_load = QPushButton("Load to Workspace")
        self.btn_load.clicked.connect(self.request_load)
        
        self.workspace_list = QListWidget()
        self.workspace_list.itemClicked.connect(self.activate_data)
        
        # --- NEW: Right-Click Menu for Floating Viewers ---
        self.workspace_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.workspace_list.customContextMenuRequested.connect(self.show_workspace_menu)
        
        session_layout = QHBoxLayout()
        self.btn_save_session = QPushButton("Save ML Session")
        self.btn_save_session.clicked.connect(self.save_session)
        self.btn_load_session = QPushButton("Load ML Session")
        self.btn_load_session.clicked.connect(self.load_session)
        session_layout.addWidget(self.btn_save_session)
        session_layout.addWidget(self.btn_load_session)
        
        left_layout.addWidget(self.disk_list)
        left_layout.addWidget(self.btn_load)
        left_layout.addWidget(QLabel("<b>2. RAM Workspace</b>"))
        left_layout.addWidget(self.workspace_list)
        left_layout.addLayout(session_layout)

        # --- MIDDLE: INTERACTIVE VIEWER ---
        mid_panel = QWidget()
        mid_layout = QVBoxLayout(mid_panel)
        
        # We just drop the isolated reusable widget right in!
        from tensorspec.gui.components.data_viewer_panel import DataViewerPanel
        self.viewer = DataViewerPanel()
        mid_layout.addWidget(self.viewer)

        # --- RIGHT: MACHINE LEARNING ---
        # Pages are collected as they are built and grouped into a small set of
        # top-level tabs at the end of this method; ten flat tabs overflowed the
        # bar in a pane this narrow.
        
        # --- NEW REIMAGINED TABS ---
        self.model_warehouse_tab = ModelWarehouseTab(self)
        self.build_pipeline_tab = BuildPipelineTab(self)
        self.train_model_tab = TrainModelTab(self)
        
        model_pages = [
            ("Model Warehouse", self.model_warehouse_tab),
            ("Build Pipeline", scrollable(self.build_pipeline_tab)),
            ("Train Model", self.train_model_tab),
        ]
        
        # Tab A: Train SSL
        train_controls = QWidget()
        train_layout = QVBoxLayout(train_controls)
        
        self.btn_ssl_help = QPushButton("📖 New to SSL Training? Click Here for a Guide")
        self.btn_ssl_help.setStyleSheet("font-weight: bold; color: #1f77b4; padding: 6px; font-size: 14px;")
        self.btn_ssl_help.clicked.connect(self.show_ssl_guide)
        train_layout.addWidget(self.btn_ssl_help)
        
        model_selection_layout = QVBoxLayout()
        group_gen = QGroupBox("1A. Generative (Reconstruction)")
        layout_gen = QVBoxLayout(group_gen)
        self.chk_ae = QCheckBox("Autoencoder (CNN)"); self.chk_ae.setChecked(True)
        self.chk_beta = QCheckBox("Beta-VAE (CNN)")
        self.chk_mae = QCheckBox("MAE (CNN)")
        self.chk_vit_mae = QCheckBox("ViT-MAE (Vision Transformer)")
        layout_gen.addWidget(self.chk_ae); layout_gen.addWidget(self.chk_beta)
        layout_gen.addWidget(self.chk_mae); layout_gen.addWidget(self.chk_vit_mae)

        group_con = QGroupBox("1B. Contrastive (Negative Sampling)")
        layout_con = QVBoxLayout(group_con)
        self.chk_simclr = QCheckBox("SimCLR")
        self.chk_moco = QCheckBox("MoCo (Momentum Contrast)")
        layout_con.addWidget(self.chk_simclr); layout_con.addWidget(self.chk_moco)
        
        group_dist = QGroupBox("1C. Distillation (No Negatives)")
        layout_dist = QVBoxLayout(group_dist)
        self.chk_byol = QCheckBox("BYOL (Bootstrap Your Own Latent)")
        layout_dist.addWidget(self.chk_byol)
        
        group_clust = QGroupBox("1D. Clustering")
        layout_clust = QVBoxLayout(group_clust)
        self.chk_swav = QCheckBox("SwAV (Swapping Assignments)")
        layout_clust.addWidget(self.chk_swav)
        
        model_selection_layout.addWidget(group_gen); model_selection_layout.addWidget(group_con)
        model_selection_layout.addWidget(group_dist); model_selection_layout.addWidget(group_clust)
        
        hyper_group = QGroupBox("2. Hyperparameters:")
        hyper_layout = QHBoxLayout(hyper_group)
        self.spin_epochs = QSpinBox(); self.spin_epochs.setRange(1, 500); self.spin_epochs.setValue(15)
        self.spin_lr = QDoubleSpinBox(); self.spin_lr.setRange(0.0001, 0.1); self.spin_lr.setSingleStep(0.001)
        self.spin_lr.setDecimals(4); self.spin_lr.setValue(0.0010)
        hyper_layout.addWidget(QLabel("Epochs:")); hyper_layout.addWidget(self.spin_epochs)
        hyper_layout.addWidget(QLabel("Learn Rate:")); hyper_layout.addWidget(self.spin_lr)
        
        self.btn_train = QPushButton("Start Queue Training")
        self.btn_train.clicked.connect(self.start_training)
        
        loss_view_layout = QHBoxLayout()
        loss_view_layout.addWidget(QLabel("View Loss For:"))
        self.combo_loss_view = QComboBox()
        self.combo_loss_view.currentTextChanged.connect(self.on_loss_view_changed)
        loss_view_layout.addWidget(self.combo_loss_view)
        
        self.loss_canvas = MplCanvas(self, width=4, height=3, is_3d=True)
        self.ax_loss = self.loss_canvas.axes
        self.ax_loss.set_title("Training Loss")
        self.loss_history_dict = {} 
        self.current_training_model = None
        
        train_layout.addLayout(model_selection_layout) 
        train_layout.addWidget(hyper_group)
        train_layout.addWidget(self.btn_train)
        train_layout.addLayout(loss_view_layout) 
        train_layout.addStretch()
        ssl_page = split_panel(train_controls, self.loss_canvas, sizes=(420, 380))

        # Tab B: Clustering
        cluster_controls = QWidget()
        cluster_layout = QVBoxLayout(cluster_controls)
        
        self.btn_cluster_help = QPushButton("📊 What do these Algorithms do? Click Here")
        self.btn_cluster_help.setStyleSheet("font-weight: bold; color: #d62728; padding: 6px; font-size: 14px;")
        self.btn_cluster_help.clicked.connect(self.show_cluster_guide)
        cluster_layout.addWidget(self.btn_cluster_help)
        
        cluster_ctrl_group = QGroupBox("Clustering Controls:")
        c_layout = QVBoxLayout(cluster_ctrl_group)
        
        self.combo_embed = QComboBox()
        self.combo_parent_filter = QComboBox()
        self.combo_parent_filter.addItem("None (Run on Entire Map)")
        
        self.combo_algo = QComboBox()
        self.combo_algo.addItems(["Hierarchical", "K-Means", "Gaussian Mixture", "DBSCAN", "HDBSCAN"])
        self.combo_algo.currentTextChanged.connect(self.toggle_cluster_params)

        self.combo_metric = QComboBox()
        self.combo_metric.addItems(["euclidean", "cosine", "correlation"])
        
        h_k = QHBoxLayout()
        self.spin_k = QSpinBox(); self.spin_k.setRange(2, 50); self.spin_k.setValue(5)
        h_k.addWidget(QLabel("Clusters (k):")); h_k.addWidget(self.spin_k)
        
        h_eps = QHBoxLayout()
        self.spin_eps = QDoubleSpinBox(); self.spin_eps.setRange(0.01, 10.0); self.spin_eps.setSingleStep(0.1); self.spin_eps.setValue(1.50)
        h_eps.addWidget(QLabel("DBSCAN eps:")); h_eps.addWidget(self.spin_eps)
        
        # --- NEW: UMAP First Checkbox ---
        self.chk_umap_first = QCheckBox("Run UMAP First (HDBSCAN / Hierarchical)")
        self.chk_umap_first.setChecked(False)
        
        # --- NEW: Normalization Checkbox ---
        self.chk_normalize = QCheckBox("Normalize EDC Intensity (Max = 1.0)")
        self.chk_normalize.setChecked(True)

        c_layout.addWidget(QLabel("Target Embedding:")); c_layout.addWidget(self.combo_embed)
        c_layout.addWidget(QLabel("Parent Domain Filter:")); c_layout.addWidget(self.combo_parent_filter)
        c_layout.addWidget(QLabel("Algorithm:")); c_layout.addWidget(self.combo_algo)
        c_layout.addWidget(QLabel("Distance Metric:")); c_layout.addWidget(self.combo_metric)
        c_layout.addLayout(h_k); c_layout.addLayout(h_eps)
        c_layout.addWidget(self.chk_umap_first)
        c_layout.addWidget(self.chk_normalize)
        
        self.btn_cluster = QPushButton("Run Clustering & UMAP")
        self.btn_cluster.clicked.connect(self.run_clustering)
        
        self.btn_dendro = QPushButton("View Hierarchical Dendrogram & Spectra")
        self.btn_dendro.setStyleSheet("background-color: #e6e6fa; color: #333;") 
        self.btn_dendro.clicked.connect(self.show_dendrogram)
        
        self.btn_save_labels = QPushButton("Save Labels to CSV/TXT")
        self.btn_save_labels.clicked.connect(self.save_labels)
        
        umap_ctrl_layout = QHBoxLayout()
        umap_ctrl_layout.addWidget(QLabel("UMAP Click Action:"))
        self.combo_umap_plot_type = QComboBox()
        self.combo_umap_plot_type.addItems(["Full Dispersion 2D", "Integrated EDC 1D", "Integrated MDC 1D"])
        umap_ctrl_layout.addWidget(self.combo_umap_plot_type)
        umap_ctrl_layout.addStretch()

        self.umap_canvas = MplCanvas(self, width=4, height=4, is_3d=False)
        self.ax_umap, self.ax_band = self.umap_canvas.axes
        self.umap_canvas.mpl_connect('pick_event', self.on_umap_pick)
        
        cluster_layout.addWidget(cluster_ctrl_group); cluster_layout.addWidget(self.btn_cluster)
        cluster_layout.addWidget(self.btn_dendro); cluster_layout.addWidget(self.btn_save_labels)
        cluster_layout.addLayout(umap_ctrl_layout)
        cluster_layout.addStretch()
        cluster_page = split_panel(cluster_controls, self.umap_canvas, sizes=(400, 400))

        # Tab C: Supervised Few-Shot
        sup_tab = QWidget()
        sup_layout = QVBoxLayout(sup_tab)

        self.btn_sup_help = QPushButton("🧑‍🏫 How does Few-Shot Learning work? Click Here")
        self.btn_sup_help.setStyleSheet("font-weight: bold; color: #2ca02c; padding: 6px; font-size: 14px;")
        self.btn_sup_help.clicked.connect(self.show_sup_guide)
        sup_layout.addWidget(self.btn_sup_help)

        sup_ctrl_group = QGroupBox("1. Define Target Labels")
        sup_ctrl_layout = QHBoxLayout(sup_ctrl_group)
        self.spin_sup_classes = QSpinBox(); self.spin_sup_classes.setRange(2, 10); self.spin_sup_classes.setValue(3)
        self.btn_create_classes = QPushButton("Generate Label Buttons")
        self.btn_create_classes.clicked.connect(self.create_sup_buttons)
        sup_ctrl_layout.addWidget(QLabel("Number of Distinct Labels:")); sup_ctrl_layout.addWidget(self.spin_sup_classes)
        sup_ctrl_layout.addWidget(self.btn_create_classes)

        self.sup_btn_group = QGroupBox("2. Collect Data (Move sliders to target, then click!)")
        self.sup_btn_layout = QVBoxLayout(self.sup_btn_group)
        self.sup_buttons = []
        self.sup_data = {} 
        self.sup_coords = {} 

        sup_act_group = QGroupBox("3. Train & Infer")
        sup_act_layout = QVBoxLayout(sup_act_group)
        
        sup_io_layout = QHBoxLayout()
        self.btn_sup_save = QPushButton("Save Training Set")
        self.btn_sup_save.clicked.connect(self.save_sup_data)
        self.btn_sup_load = QPushButton("Load Training Set")
        self.btn_sup_load.clicked.connect(self.load_sup_data)
        sup_io_layout.addWidget(self.btn_sup_save); sup_io_layout.addWidget(self.btn_sup_load)
        
        self.btn_sup_train = QPushButton("Train Model (Few-Shot)")
        self.btn_sup_train.clicked.connect(self.train_supervised)
        self.btn_sup_test = QPushButton("Test (Classify Entire Map)")
        self.btn_sup_test.clicked.connect(self.test_supervised)
        
        self.btn_sup_save_results = QPushButton("Save Classification Results")
        self.btn_sup_save_results.clicked.connect(self.save_sup_results)
        
        self.btn_sup_reset = QPushButton("Reset All Training Data")
        self.btn_sup_reset.clicked.connect(self.reset_supervised)
        
        sup_act_layout.addLayout(sup_io_layout)
        sup_act_layout.addWidget(self.btn_sup_train); sup_act_layout.addWidget(self.btn_sup_test)
        sup_act_layout.addWidget(self.btn_sup_save_results); sup_act_layout.addWidget(self.btn_sup_reset)

        sup_layout.addWidget(sup_ctrl_group); sup_layout.addWidget(self.sup_btn_group); sup_layout.addWidget(sup_act_group)
        sup_layout.addStretch()
        sup_page = scrollable(sup_tab)

        # Tab D: Active Learning
        self.al_panel = ActiveLearningPanel(self.session)
        al_page = self.al_panel

        # Tab E: Simulate Active Learning
        self.sim_panel = SimulateALPanel(self.session)
        sim_page = self.sim_panel

        # Tab F: 3D Alignment
        self.align_panel = AlignmentPanel(self.session)
        align_page = self.align_panel

        # Group the pages by workflow stage. SSL Training produces the
        # embeddings Clustering consumes, and Clustering produces the domains
        # Steer consumes, so those lead. The Models group holds the not-yet
        # implemented placeholders and sits near the end so the suite does not
        # open on a stub.
        right_panel = QTabWidget()
        right_panel.setDocumentMode(True)
        right_panel.setMovable(True)
        right_panel.setUsesScrollButtons(True)
        # No elide: truncated labels ("Mo...", "Bu...") are unreadable, so keep
        # full names and let the scroll arrows handle any overflow.
        right_panel.setElideMode(Qt.TextElideMode.ElideNone)

        for group_label, pages in (
            ("Train", [("SSL Training", ssl_page),
                       ("Supervised Learning", sup_page)]),
            ("Cluster", [("Clustering", cluster_page)]),
            ("Align", [("3D Alignment", align_page)]),
            ("Steer", [("Active Learning", al_page),
                       ("Simulate AL", sim_page)]),
            ("Models", model_pages),
        ):
            right_panel.addTab(tab_group(pages), group_label)

        right_panel.setCurrentIndex(0)
        
        # Each tab scrolls its own control column, so the tab bar itself must not
        # sit inside a scroll area or it scrolls out of reach.
        left_panel.setMinimumWidth(220)
        mid_panel.setMinimumWidth(320)
        right_panel.setMinimumWidth(380)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel); splitter.addWidget(mid_panel); splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)
        splitter.setCollapsible(1, False)
        splitter.setSizes([260, 680, 560])
        layout.addWidget(splitter)
        self.setCentralWidget(main_widget)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.prog_bar = QProgressBar()
        self.prog_bar.setVisible(False)
        self.status.addPermanentWidget(self.prog_bar)

    # --- UI Logic Methods ---
    def show_master_guide(self):
        MasterGuideDialog(self).exec()
    def show_ssl_guide(self):
        SSLGuideDialog(self).exec()
    def show_cluster_guide(self):
        ClusterGuideDialog(self).exec()
    def show_sup_guide(self):
        SupGuideDialog(self).exec()

    def select_directory(self):
        start_path = self.current_folder if self.current_folder else ""
        path = QFileDialog.getExistingDirectory(self, "Select Folder", start_path)
        if path:
            self.path_input.setText(path)
            self.load_directory(path)

    def load_directory_from_input(self):
        path = self.path_input.text().strip()
        self.load_directory(path)

    def load_directory(self, path):
        if os.path.isdir(path):
            self.current_folder = path
            self.disk_list.clear()
            self.disk_list.addItems([f for f in sorted(os.listdir(path)) if f.endswith('.h5')])
        else:
            QMessageBox.warning(self, "Invalid Path", "The specified folder does not exist. Please check the path and try again.")
            
    def _refresh_viewer_ml_layers(self, select_layer: str | None = None):
        """Push ML domain/label arrays into the middle DataViewer without resetting the grid."""
        if not self.current_view_data:
            return
        self.viewer.sync_ml_layers(self.current_view_data)
        if select_layer:
            self.viewer.focus_spatial_layer(select_layer)

    def _convert_to_tensor_data(self, data):
        from tensorspec.core.data_models import TensorData
        import numpy as np
        if data is None: return None
        layers = {}
        for k, v in data.items():
            if k.startswith("Labels_") or k.startswith("domains_") or k.startswith("probs_"):
                layers[k] = v
        
        # Maestro4DViewer used E, A, Y, X
        axes = [data.get('E', np.array([0])), data.get('angle', np.array([0])), data.get('y', np.array([0])), data.get('x', np.array([0]))]
        labels = ["Energy", "Angle", "Y", "X"]
        units = ["eV", "deg", "mm", "mm"]
        return TensorData(
            value=data['value'],
            axes=axes,
            labels=labels,
            units=units,
            data_type=data.get('kind', 'Maestro Data'),
            metadata={'layers': layers}
        )

    def load_workspace_to_viewer(self):
        if 'raw' in self.current_workspace:
            data = self.current_workspace['raw']
            self.current_view_data = data
            td = self._convert_to_tensor_data(data)
            self.viewer.load_data(td)
        else:
            QMessageBox.warning(self, "No Data", "Workspace is missing 'raw' data array.")
    
    # --- NEW: Workspace Context Menu for Floating Viewers ---
    def show_workspace_menu(self, pos):
        item = self.workspace_list.itemAt(pos)
        if not item: return
        data = self.workspace[item.text()]
        
        menu = QMenu(self)
        if data.get('kind') == "XY Scan (Cleaned)":
            compare_action = menu.addAction("Open in Floating Comparison Window")
            action = menu.exec(self.workspace_list.mapToGlobal(pos))
            if action == compare_action:
                self.open_floating_viewer(data, item.text())
        elif data.get('kind') == "Fermi Map (Cleaned)":
            open_action = menu.addAction("Open 3D Fermi Viewer")
            action = menu.exec(self.workspace_list.mapToGlobal(pos))
            if action == open_action:
                self.fermi_dialog = FermiViewerWindow(data, self)
                self.fermi_dialog.show()

    def open_floating_viewer(self, data, title):
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Comparison Viewer: {title}")
        dialog.resize(1000, 600)
        layout = QVBoxLayout(dialog)
        from tensorspec.gui.components.data_viewer_panel import DataViewerPanel
        floating_viewer = DataViewerPanel()
        td = self._convert_to_tensor_data(data)
        floating_viewer.load_data(td)
        layout.addWidget(floating_viewer)
        dialog.show()
    # --------------------------------------------------------

    def save_session(self):
        if not self.current_view_data:
            QMessageBox.warning(self, "No Data", "Please select a loaded file from the Workspace first.")
            return
            
        session_data = {}
        for key, val in self.current_view_data.items():
            if key.startswith("embeddings_") or key.startswith("domains_") or key == "Supervised Probabilities":
                session_data[key] = val
                
        if not session_data:
            QMessageBox.warning(self, "No ML Data", "There are no embeddings or clustering labels to save yet. Run some analysis first!")
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "Save ML Session", "MAESTRO_Session.pkl", "Pickle Files (*.pkl)")
        if not path: return
        
        try:
            with open(path, 'wb') as f:
                pickle.dump(session_data, f)
            self.status.showMessage(f"✅ Saved lightweight ML session to {os.path.basename(path)}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save session:\n{str(e)}")

    def load_session(self):
        if not self.current_view_data:
            QMessageBox.warning(self, "No Raw Data", "Please load and activate the raw .h5 file in the Workspace first so we have somewhere to put the labels!")
            return
            
        path, _ = QFileDialog.getOpenFileName(self, "Load ML Session", "", "Pickle Files (*.pkl)")
        if not path: return
        
        try:
            with open(path, 'rb') as f:
                session_data = pickle.load(f)
                
            for key, val in session_data.items():
                self.current_view_data[key] = val
                
            self.combo_parent_filter.clear()
            self.combo_parent_filter.addItem("None (Run on Entire Map)")
            for k in self.current_view_data.keys():
                if k.startswith('domains_'):
                    unique_clusters = np.unique(self.current_view_data[k])
                    for c in unique_clusters:
                        if c != -1: 
                            self.combo_parent_filter.addItem(f"{k} -> Cluster {c}")

            # Send data to the viewer component
            self.viewer.load_data(self._convert_to_tensor_data(self.current_view_data))
            self.session.activate(self.current_view_data)
            self.session.notify_domains()
            
            # Now let the UI dropdowns fetch the newly added modes directly from the viewer
            self.combo_embed.clear()
            for k in self.current_view_data.keys():
                if k.startswith("embeddings_"):
                    self.combo_embed.addItem(k)
            self.combo_embed.addItem("Integrated EDC (from Viewer)")
            self.combo_embed.addItem("Integrated MDC (from Viewer)")

            self.status.showMessage(f"✅ Successfully injected ML session into {self.workspace_list.currentItem().text()}", 5000)
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load session:\n{str(e)}")

    def request_load(self):
        item = self.disk_list.currentItem()
        if not item: return
        file_path = os.path.join(self.current_folder, item.text())
        var_name = "MAESTRO_" + item.text().replace('.h5', '').replace('-', '_')

        try:
            with h5py.File(file_path, 'r') as f:
                d0_keys = list(f['0D_Data'].keys()) if '0D_Data' in f else []
                has_spatial = any(k in d0_keys for k in ['X', 'Y', 'Sample X', 'Sample Y', 'Scan X', 'Scan Y'])
                has_angle = any(k in d0_keys for k in ['Deflection', 'Slit Defl', 'Manipulator Theta', 'Manipulator Phi', 'Tilt'])
                
                scan_info = ""
                if 'Headers' in f and 'Low_Level_Scan' in f['Headers']:
                    scan_info = str(f['Headers']['Low_Level_Scan'][0][2])
                
                if "XY Scan Fine" in scan_info: mode = "XY Scan Fine"
                elif "XY Scan" in scan_info or has_spatial: mode = "XY Scan"
                elif "Fermi" in scan_info or has_angle: mode = "Fermi Map"
                else: mode = "Raw"
        except:
            mode = "Raw"

        if mode == "Raw":
            reply = QMessageBox.question(self, "Unknown Type", f"Could not identify {item.text()} as a Spatial Scan. Load as Raw Data?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No: return

        # Start Background Load
        self.prog_bar.setVisible(True)
        self.btn_load.setEnabled(False)
        self.loader = LoadWorker(file_path, var_name, mode)
        self.loader.progress.connect(self.update_status)
        self.loader.finished.connect(self.on_load_finish)
        
        # --- ADD THIS LINE TO PREVENT SILENT CRASHES ---
        self.loader.error.connect(lambda e: QMessageBox.critical(self, "Loader Error", str(e)))
        
        self.loader.start()

    def update_status(self, val, msg):
        self.prog_bar.setValue(val); self.status.showMessage(msg)

    def on_load_finish(self, var_name, data):
        self.session.add_dataset(var_name, data)
        if self.workspace_list.findItems(var_name, Qt.MatchFlag.MatchExactly) == []:
            self.workspace_list.addItem(var_name)
        self.prog_bar.setVisible(False)
        self.btn_load.setEnabled(True)
        self.status.showMessage("Ready.", 3000)

    def activate_data(self, item):
        data = self.workspace[item.text()]
        
        # --- NEW: Safety Catch for 3D Fermi Maps ---
        if data.get('kind') == "Fermi Map (Cleaned)":
            reply = QMessageBox.question(self, "3D Data Detected", 
                                         "This is a 3D Fermi Map. The main dashboard is designed for 4D Spatial Scans.\n\nWould you like to open this map in a floating 3D viewer instead?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.fermi_dialog = FermiViewerWindow(data, self)
                self.fermi_dialog.show()
            return # Stop here so we don't crash the 4D viewer!
        # --------------------------------------------
            
        if data.get('kind') == "XY Scan (Cleaned)":
            self.current_view_data = data
            self.session.activate(data)
            
            # --- THE API HOOK --- 
            self.viewer.load_data(self._convert_to_tensor_data(data))
            
            self.session.notify_domains()
            
            self.combo_embed.blockSignals(True)
            self.combo_embed.clear()
            for k in data.keys():
                if k.startswith("embeddings_"): self.combo_embed.addItem(k)
            self.combo_embed.addItem("Integrated EDC (from Viewer)")
            self.combo_embed.addItem("Integrated MDC (from Viewer)")
            self.combo_embed.blockSignals(False)

    def start_training(self):
        if not self.current_view_data: 
            QMessageBox.warning(self, "No Data", "Select a loaded file in the Workspace first.")
            return
        selected_models = []
        if self.chk_ae.isChecked(): selected_models.append("Autoencoder")
        if self.chk_beta.isChecked(): selected_models.append("Beta-VAE")
        if self.chk_mae.isChecked(): selected_models.append("MAE")
        if self.chk_vit_mae.isChecked(): selected_models.append("ViT-MAE")
        if self.chk_simclr.isChecked(): selected_models.append("SimCLR")
        if self.chk_moco.isChecked(): selected_models.append("MoCo")
        if self.chk_byol.isChecked(): selected_models.append("BYOL")
        if self.chk_swav.isChecked(): selected_models.append("SwAV")
        if not selected_models: return

        self.btn_train.setEnabled(False)
        self.status.showMessage("Initializing PyTorch Training Queue...")
        self.loss_history_dict.clear(); self.combo_loss_view.clear()
        
        # --- NEW: Lock in the target dictionary so it doesn't matter if the user clicks away ---
        self.active_train_target = self.current_view_data 
        
        e, lr = self.spin_epochs.value(), self.spin_lr.value()
        self.trainer = TrainWorker(self.active_train_target['value'], epochs=e, lr=lr, selected_models=selected_models)
        self.trainer.progress.connect(self.update_live_loss)
        self.trainer.model_changed.connect(self.on_model_changed)
        self.trainer.finished.connect(self.on_train_finish)
        self.trainer.start()

    def on_model_changed(self, model_name):
        self.current_training_model = model_name
        self.loss_history_dict[model_name] = {'epochs': [], 'losses': []}
        self.combo_loss_view.blockSignals(True)
        if self.combo_loss_view.findText(model_name) == -1: self.combo_loss_view.addItem(model_name)
        self.combo_loss_view.setCurrentText(model_name)
        self.combo_loss_view.blockSignals(False)
        self.redraw_loss_plot(model_name)

    def update_live_loss(self, epoch, loss):
        if not self.current_training_model: return
        self.loss_history_dict[self.current_training_model]['epochs'].append(epoch)
        self.loss_history_dict[self.current_training_model]['losses'].append(loss)
        if self.combo_loss_view.currentText() == self.current_training_model:
            self.redraw_loss_plot(self.current_training_model)

    def on_loss_view_changed(self, model_name):
        if model_name: self.redraw_loss_plot(model_name)

    def redraw_loss_plot(self, model_name):
        self.ax_loss.clear(); self.ax_loss.set_title(f"Training Loss: {model_name}")
        self.ax_loss.set_xlabel("Epoch"); self.ax_loss.set_ylabel("Loss")
        if model_name in self.loss_history_dict:
            epochs = self.loss_history_dict[model_name]['epochs']
            losses = self.loss_history_dict[model_name]['losses']
            self.ax_loss.plot(epochs, losses, 'r-', linewidth=2, marker='o', markersize=4)
        self.ax_loss.relim(); self.ax_loss.autoscale_view()
        self.loss_canvas.draw_idle()

    def on_train_finish(self, new_embeddings_dict):
        for key, emb in new_embeddings_dict.items():
            # 1. Save the embeddings into the exact file that started the training!
            self.active_train_target[key] = emb 
            
            # 2. Only update the UI dropdowns if the user is STILL looking at that specific file
            if self.current_view_data is self.active_train_target:
                if self.combo_embed.findText(key) == -1: self.combo_embed.addItem(key)
                self.combo_embed.setCurrentText(list(new_embeddings_dict.keys())[-1]) 
                
        self.btn_train.setEnabled(True)
        self.status.showMessage("All selected models finished training!", 5000)

    def show_dendrogram(self):
        if not self.current_view_data:
            QMessageBox.warning(self, "No Data", "Please load a file first.")
            return
        algo = self.combo_algo.currentText()
        if algo != "Hierarchical":
            QMessageBox.information(self, "Wrong Algorithm", "The Dendrogram is only mathematically possible when the 'Hierarchical' algorithm is selected in the dropdown!")
            return
        embed_key = self.combo_embed.currentText()
        k = self.spin_k.value()
        if k > 15:
            QMessageBox.warning(self, "Too Many Branches", "Please lower k to 15 or less to visually render the spectra.")
            return
        if embed_key in ["Integrated EDC (from Viewer)", "Integrated MDC (from Viewer)"]:
            QMessageBox.information(self, "Use ML Embeddings", "The Dendrogram preview works best with Neural Network embeddings. Please select an embedding like SimCLR or MAE.")
            return
        else:
            embeds = self.current_view_data[embed_key]

        val = self.current_view_data['value']
        E_arr = self.current_view_data['E']
        A_arr = self.current_view_data['angle']
        
        # --- API HOOK ---
        contrast_scale = self.viewer.get_dispersion_contrast()
        gamma_scale = 1.0

        self.status.showMessage("Calculating Hierarchical Tree (This may take a few seconds)...")
        QApplication.processEvents() 
        dialog = DendrogramDialog(embeds, val, k, E_arr, A_arr, contrast_scale, gamma_scale, self)
        self.status.showMessage("Ready.", 3000)
        dialog.exec()

    def toggle_cluster_params(self, algo_name):
        if algo_name == "DBSCAN":
            self.spin_k.setEnabled(False); self.spin_eps.setEnabled(True)
            self.chk_umap_first.setEnabled(False)  # <-- Grays out the checkbox
        else:
            self.spin_k.setEnabled(True); self.spin_eps.setEnabled(False)
            self.chk_umap_first.setEnabled(True)   # <-- Reactivates it for everything else

    def run_clustering(self):
        if not self.current_view_data: return
        embed_key = self.combo_embed.currentText()
        parent_filter = self.combo_parent_filter.currentText()
        algo = self.combo_algo.currentText()
        k = self.spin_k.value(); eps = self.spin_eps.value()
        
        if embed_key in ["Integrated EDC (from Viewer)", "Integrated MDC (from Viewer)"]:
            val = self.current_view_data['value']
            dim_E, dim_A, nY, nX = val.shape
            
            # --- API HOOK ---
            e_c, de, a_c, da = self.viewer.get_slider_values()
            
            e1, e2 = max(0, e_c - de), min(dim_E, e_c + de + 1)
            a1, a2 = max(0, a_c - da), min(dim_A, a_c + da + 1)
            
            # Slice BOTH axes to restrict the AI exactly to the viewer's window
            if "EDC" in embed_key:
                sliced = np.sum(val[e1:e2, a1:a2, :, :], axis=1) 
                embeds = sliced.transpose(1, 2, 0).reshape(nY * nX, e2 - e1)
            else:
                sliced = np.sum(val[e1:e2, a1:a2, :, :], axis=0) 
                embeds = sliced.transpose(1, 2, 0).reshape(nY * nX, a2 - a1)
            row_max = embeds.max(axis=1, keepdims=True) + 1e-8
            embeds = embeds / row_max
        else:
            embeds = self.current_view_data[embed_key]
        
        self.active_mask_indices = None 
        if parent_filter != "None (Run on Entire Map)":
            domain_key, cluster_val = parent_filter.split(" -> Cluster ")
            cluster_val = int(cluster_val)
            labels = self.current_view_data[domain_key]
            self.active_mask_indices = np.where(labels == cluster_val)[0]
            if len(self.active_mask_indices) < 5:
                QMessageBox.warning(self, "Too Small", "Selected cluster has too few pixels to sub-cluster.")
                return
            embeds = embeds[self.active_mask_indices]

        # Grab the metric and the new checkbox states
        metric = self.combo_metric.currentText()
        do_umap_first = self.chk_umap_first.isChecked()
        do_normalize = self.chk_normalize.isChecked() 
        
        self.btn_cluster.setEnabled(False); self.prog_bar.setVisible(True)
        self.status.showMessage(f"Running {algo}...")
        
        # --- NEW: Lock the target dictionary for multitasking ---
        self.active_cluster_target = self.current_view_data
        
        # Pass all params to your updated ClusterWorker
        self.cluster_worker = ClusterWorker(
            embeds, algo, k, eps, metric=metric, use_umap_first=do_umap_first, normalize_edcs=do_normalize
        )
        self.cluster_worker.progress.connect(self.update_status)
        self.cluster_worker.finished.connect(lambda l, u: self.on_cluster_finish(l, u, embed_key, algo))
        self.cluster_worker.error.connect(lambda e: self.status.showMessage(f"Error: {e}"))
        self.cluster_worker.start()

    def on_cluster_finish(self, labels, umap_res, embed_key, algo):
        domain_key = f"domains_{embed_key}"
        
        # 1. Save data to the LOCKED target
        if self.active_mask_indices is not None:
            dim_E, dim_A, nY, nX = self.active_cluster_target['value'].shape
            total_pixels = nY * nX
            full_labels = np.full(total_pixels, -1, dtype=int)
            full_labels[self.active_mask_indices] = labels
            parent_txt = self.combo_parent_filter.currentText().replace(' -> ', '_sub')
            domain_key = f"{domain_key}_{parent_txt}"
            self.active_cluster_target[domain_key] = full_labels
        else:
            self.active_cluster_target[domain_key] = labels
            
        # 2. Only update UI if the user is STILL looking at this exact file
        if self.current_view_data is self.active_cluster_target:
            self._refresh_viewer_ml_layers(domain_key)
            
            self.session.notify_domains()
                
            self.combo_parent_filter.clear()
            self.combo_parent_filter.addItem("None (Run on Entire Map)")
            for k in self.current_view_data.keys():
                if k.startswith('domains_'):
                    unique_clusters = np.unique(self.current_view_data[k])
                    for c in unique_clusters:
                        if c != -1: self.combo_parent_filter.addItem(f"{k} -> Cluster {c}")
            
            self.ax_umap.clear(); self.ax_band.clear()
            self.scatter = self.ax_umap.scatter(umap_res[:,0], umap_res[:,1], c=labels, cmap='tab20', vmin=-0.5, vmax=19.5, s=15, alpha=0.8, picker=5)
            self.ax_umap.set_title(f"{algo}: {embed_key}\n(Click a point!)")
            self.ax_umap.set_xlabel("UMAP 1"); self.ax_umap.set_ylabel("UMAP 2")
            self.ax_band.axis('off')
            
            self.umap_canvas.draw_idle()

        self.btn_cluster.setEnabled(True); self.prog_bar.setVisible(False)
        self.status.showMessage(f"Clustering complete.", 5000)
    def on_umap_pick(self, event):
        if event.mouseevent.inaxes != self.ax_umap or not self.current_view_data: return
        ind = event.ind[0] 
        if hasattr(self, 'active_mask_indices') and self.active_mask_indices is not None: global_ind = self.active_mask_indices[ind]
        else: global_ind = ind

        data = self.current_view_data
        val = data['value']
        dim_E, dim_A, nY, nX = val.shape
        flat_bands = val.transpose(2, 3, 0, 1).reshape((nY * nX, dim_E, dim_A))
        band = flat_bands[global_ind] 
        
        E_arr, A_arr = data['E'], data['angle']
        px_A = (A_arr[-1] - A_arr[0]) / max(1, dim_A-1)
        px_E = (E_arr[-1] - E_arr[0]) / max(1, dim_E-1)
        extent = (A_arr[0]-px_A/2, A_arr[-1]+px_A/2, E_arr[0]-px_E/2, E_arr[-1]+px_E/2)
        
        self.ax_band.clear(); self.ax_band.axis('on')
        plot_type = self.combo_umap_plot_type.currentText()
        if "EDC" in plot_type:
            edc = np.sum(band, axis=1)
            self.ax_band.plot(edc, E_arr, 'r-', linewidth=2) 
            self.ax_band.set_xlabel("Intensity"); self.ax_band.set_ylabel("Energy (eV)")
        elif "MDC" in plot_type:
            mdc = np.sum(band, axis=0)
            self.ax_band.plot(A_arr, mdc, 'b-', linewidth=2)
            self.ax_band.set_xlabel("Angle (Degrees)"); self.ax_band.set_ylabel("Intensity")
        else:
            self.ax_band.imshow(band, aspect='auto', cmap='magma', origin='lower', extent=extent)
            self.ax_band.set_xlabel("Angle (Degrees)"); self.ax_band.set_ylabel("Energy (eV)")
        
        embed_key = self.combo_embed.currentText()
        domain_key = f"domains_{embed_key}"
        if hasattr(self, 'active_mask_indices') and self.active_mask_indices is not None:
            parent_txt = self.combo_parent_filter.currentText().replace(' -> ', '_sub')
            domain_key = f"{domain_key}_{parent_txt}"
            
        labels = data.get(domain_key, [])
        cluster_id = labels[global_ind] if len(labels) > global_ind else "Unknown"
        self.ax_band.set_title(f"Global Index: {global_ind} | Cluster: {cluster_id}")
        self.umap_canvas.draw_idle()

    def save_labels(self):
        if not self.current_view_data:
            QMessageBox.warning(self, "No Data", "Please load a file first.")
            return
        domain_keys = [k for k in self.current_view_data.keys() if k.startswith('domains_')]
        if not domain_keys:
            QMessageBox.warning(self, "No Labels", "No clustering labels found. Run clustering first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Labels", "MAESTRO_Cluster_Labels.csv", "CSV Files (*.csv);;Text Files (*.txt)")
        if not path: return
        try:
            X_arr, Y_arr = self.current_view_data['x'], self.current_view_data['y']
            X_grid, Y_grid = np.meshgrid(X_arr, Y_arr)
            x_flat, y_flat = X_grid.flatten(), Y_grid.flatten()
            header = "X,Y"
            cols = [x_flat, y_flat]
            for k in domain_keys:
                clean_name = k.replace("domains_", "")
                header += f",label_{clean_name}"
                cols.append(self.current_view_data[k])
            out_matrix = np.column_stack(cols)
            np.savetxt(path, out_matrix, delimiter=',', header=header, comments='', fmt='%g')
            self.status.showMessage(f"✅ Saved labels to {os.path.basename(path)}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save file:\n{str(e)}")

    def create_sup_buttons(self):
        for btn in self.sup_buttons:
            self.sup_btn_layout.removeWidget(btn); btn.deleteLater()
        self.sup_buttons.clear(); self.sup_data.clear(); self.sup_coords.clear() 

        num_classes = self.spin_sup_classes.value()
        for i in range(num_classes):
            self.sup_data[i] = []; self.sup_coords[i] = [] 
            btn = QPushButton(f"Assign Target Coordinate to Label {i+1} (Count: 0)")
            btn.clicked.connect(lambda checked, idx=i: self.add_sup_data(idx))
            self.sup_btn_layout.addWidget(btn); self.sup_buttons.append(btn)

    def add_sup_data(self, idx):
        if not self.current_view_data: return
        
        # --- API HOOK ---
        x_c, y_c = self.viewer.get_current_coords()
        
        val = self.current_view_data['value']
        band = val[:, :, y_c, x_c] 
        self.sup_data[idx].append(band); self.sup_coords[idx].append((x_c, y_c)) 
        count = len(self.sup_data[idx])
        self.sup_buttons[idx].setText(f"Assign Target Coordinate to Label {idx+1} (Count: {count})")

    def reset_supervised(self):
        for i in range(len(self.sup_buttons)):
            self.sup_data[i] = []; self.sup_coords[i] = [] 
            self.sup_buttons[i].setText(f"Assign Target Coordinate to Label {i+1} (Count: 0)")
        self.trained_sup_model = None
        self.status.showMessage("Supervised training data cleared.")
    
    def save_sup_data(self):
        if not hasattr(self, 'sup_coords') or not any(self.sup_coords.values()):
            QMessageBox.warning(self, "No Data", "No training data to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Training Set", "MAESTRO_Sup_Training.csv", "CSV Files (*.csv)")
        if not path: return
        try:
            with open(path, 'w') as f:
                f.write("Label_Index,X_Index,Y_Index\n")
                for label_idx, coords in self.sup_coords.items():
                    for x_c, y_c in coords: f.write(f"{label_idx},{x_c},{y_c}\n")
            self.status.showMessage(f"✅ Saved training data to {os.path.basename(path)}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def load_sup_data(self):
        if not self.current_view_data:
            QMessageBox.warning(self, "No Data", "Load an ARPES scan into the workspace first.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load Training Set", "", "CSV Files (*.csv)")
        if not path: return
        try:
            data = np.loadtxt(path, delimiter=',', skiprows=1, dtype=int)
            if data.ndim == 1: data = data.reshape(1, -1) 
            max_label = np.max(data[:, 0])
            if len(self.sup_buttons) <= max_label:
                self.spin_sup_classes.setValue(max_label + 1)
                self.create_sup_buttons()
            val = self.current_view_data['value']
            nY, nX = val.shape[2], val.shape[3]
            for row in data:
                label_idx, x_c, y_c = row[0], row[1], row[2]
                if y_c >= nY or x_c >= nX: continue 
                band = val[:, :, y_c, x_c]
                self.sup_data[label_idx].append(band)
                self.sup_coords[label_idx].append((x_c, y_c))
            for i in range(len(self.sup_buttons)):
                count = len(self.sup_data[i])
                self.sup_buttons[i].setText(f"Assign Target Coordinate to Label {i+1} (Count: {count})")
            self.status.showMessage(f"✅ Loaded training data from {os.path.basename(path)}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Could not load file:\n{str(e)}\n\nMake sure it is a valid Training CSV.")

    def train_supervised(self):
        X, Y = [], []
        for label_idx, bands in self.sup_data.items():
            for b in bands: X.append(b); Y.append(label_idx)
        if len(X) == 0:
            QMessageBox.warning(self, "No Data", "Please assign coordinates to labels first!")
            return
        self.btn_sup_train.setEnabled(False); self.status.showMessage("Training Few-Shot CNN...")
        X_arr, Y_arr = np.array(X), np.array(Y)
        num_classes = len(self.sup_data.keys())
        self.sup_train_worker = SupTrainWorker(X_arr, Y_arr, num_classes)
        self.sup_train_worker.progress.connect(self.update_status)
        self.sup_train_worker.finished.connect(self.on_sup_train_finish)
        self.sup_train_worker.start()

    def on_sup_train_finish(self, trained_model):
        self.trained_sup_model = trained_model
        self.btn_sup_train.setEnabled(True)
        self.status.showMessage("Supervised Model Trained Successfully!", 5000)

    def test_supervised(self):
        if not hasattr(self, 'trained_sup_model') or self.trained_sup_model is None:
            QMessageBox.warning(self, "No Model", "You must Train the model first!")
            return
            
        # --- NEW: Lock the target dictionary ---
        self.active_sup_test_target = self.current_view_data
        val = self.active_sup_test_target['value'] 
        
        self.btn_sup_test.setEnabled(False)
        self.status.showMessage("Running Full Map Inference...")
        self.sup_test_worker = SupTestWorker(self.trained_sup_model, val)
        self.sup_test_worker.progress.connect(self.update_status)
        self.sup_test_worker.finished.connect(self.on_sup_test_finish)
        self.sup_test_worker.start()

    def on_sup_test_finish(self, prob_map):
        mode_name = "Supervised Probabilities"
        
        # 1. Save data strictly to the LOCKED target
        self.active_sup_test_target[mode_name] = prob_map
        
        # 2. Only update UI if the user is STILL looking at this exact file
        if self.current_view_data is self.active_sup_test_target:
            self._refresh_viewer_ml_layers(mode_name)
        
        self.btn_sup_test.setEnabled(True)
        self.status.showMessage("Inference Complete! View updated.", 5000)

    def save_sup_results(self):
        if not self.current_view_data or "Supervised Probabilities" not in self.current_view_data:
            QMessageBox.warning(self, "No Results", "No classification results to save. Please run 'Test (Classify Entire Map)' first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Classification Results", "MAESTRO_Supervised_Results.csv", "CSV Files (*.csv)")
        if not path: return
        try:
            prob_map = self.current_view_data["Supervised Probabilities"]
            nY, nX, nC = prob_map.shape
            X_arr, Y_arr = self.current_view_data['x'], self.current_view_data['y']
            X_grid, Y_grid = np.meshgrid(X_arr, Y_arr)
            x_flat, y_flat = X_grid.flatten(), Y_grid.flatten()
            predicted_labels = np.argmax(prob_map, axis=2).flatten() + 1
            max_probs = np.max(prob_map, axis=2).flatten() * 100
            header = "X,Y,Predicted_Label,Confidence_Percent"
            cols = [x_flat, y_flat, predicted_labels, max_probs]
            for c in range(nC):
                header += f",Prob_L{c+1}"
                cols.append(prob_map[:, :, c].flatten() * 100)
            out_matrix = np.column_stack(cols)
            np.savetxt(path, out_matrix, delimiter=',', header=header, comments='', fmt='%g')
            self.status.showMessage(f"✅ Saved classification results to {os.path.basename(path)}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save file:\n{str(e)}")

