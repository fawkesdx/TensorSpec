import os
import h5py
import numpy as np
import pickle
import matplotlib.patches as patches

from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFileDialog, QListWidget, QComboBox, 
                             QLabel, QTabWidget, QCheckBox, QGroupBox, QSplitter, QScrollArea, 
                             QMessageBox, QProgressBar, QStatusBar, QSpinBox, QDoubleSpinBox,
                             QLineEdit, QMenu, QDialog, QApplication)
import psutil
import time
from collections import deque
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCursor

from tensorspec.gui.maestroai.model_warehouse_tab import ModelWarehouseTab
from tensorspec.gui.maestroai.build_pipeline_tab import BuildPipelineTab
from tensorspec.gui.maestroai.train_model_tab import TrainModelTab

class DiagnosticsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        # --- Data Storage for Plotting ---
        self.max_history = 5000
        self.time_data = deque(maxlen=self.max_history)
        self.sys_data = deque(maxlen=self.max_history)
        self.app_data = deque(maxlen=self.max_history)
        self.start_time = time.time()
        
        # --- Dynamic Sampling State & Thresholds ---
        self.last_record_time = 0.0
        self.last_record_sys = 0.0
        self.last_record_app = 0.0
        
        self.delta_trigger_app = 0.05  
        self.delta_trigger_sys = 0.50  
        self.max_idle_seconds = 60.0   
        
        # --- UI Elements ---
        sys_group = QGroupBox("Total System Memory Utilization")
        sys_layout = QVBoxLayout(sys_group)
        self.sys_label = QLabel("System RAM usage: Loading...")
        self.sys_bar = QProgressBar()
        self.sys_bar.setRange(0, 100)
        sys_layout.addWidget(self.sys_label)
        sys_layout.addWidget(self.sys_bar)
        layout.addWidget(sys_group)
        
        app_group = QGroupBox("MaestroAI Process Memory Consumption")
        app_layout = QVBoxLayout(app_group)
        self.app_label = QLabel("Python Process RAM usage: Loading...")
        app_layout.addWidget(self.app_label)
        layout.addWidget(app_group)
        
        # --- Live Plot Canvas ---
        self.plot_canvas = MplCanvas(self, width=4, height=4, is_3d=False)
        self.plot_canvas.figure.clf()
        
        # FIX: Create two independent axes that share the same X-axis
        self.ax_sys = self.plot_canvas.figure.add_subplot(111)
        self.ax_app = self.ax_sys.twinx() 
        
        layout.addWidget(self.plot_canvas, 1)
        
        self.process = psutil.Process(os.getpid())
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_diagnostics)
        self.timer.start(2000)
        
    def update_diagnostics(self):
        try:
            current_time = time.time()
            virtual_mem = psutil.virtual_memory()
            total_gb = virtual_mem.total / (1024 ** 3)
            sys_used_gb = virtual_mem.used / (1024 ** 3)
            percent = virtual_mem.percent
            
            mem_info = self.process.memory_info()
            app_used_gb = mem_info.rss / (1024 ** 3)
            
            self.sys_label.setText(f"System Total: {sys_used_gb:.2f} GB / {total_gb:.1f} GB ({percent}%)")
            self.sys_bar.setValue(int(percent))
            self.app_label.setText(f"MaestroAI App Allocation: {app_used_gb:.3f} GB of RAM occupied.")
            
            time_since_last = current_time - self.last_record_time
            sys_change = abs(sys_used_gb - self.last_record_sys)
            app_change = abs(app_used_gb - self.last_record_app)
            
            # THE FIX 2: Check if time_data is empty so it ALWAYS draws the initial point
            if (len(self.time_data) == 0 or 
                app_change >= self.delta_trigger_app or 
                sys_change >= self.delta_trigger_sys or 
                time_since_last >= self.max_idle_seconds):
                
                elapsed_hours = (current_time - self.start_time) / 3600.0 
                
                self.time_data.append(elapsed_hours)
                self.sys_data.append(sys_used_gb)
                self.app_data.append(app_used_gb)
                
                self.last_record_time = current_time
                self.last_record_sys = sys_used_gb
                self.last_record_app = app_used_gb
                
                # Redraw
                self.ax_sys.clear()
                self.ax_app.clear()
                
                # FIX: Force the twin axis to stay on the right after being cleared
                self.ax_app.yaxis.set_label_position("right")
                self.ax_app.yaxis.tick_right()
                
                # Plot System RAM on the Left Axis (Blue)
                line1 = self.ax_sys.plot(self.time_data, self.sys_data, 'b-', label="System RAM", linewidth=2)
                self.ax_sys.set_ylabel("System RAM (GB)", color='blue', fontweight='bold')
                self.ax_sys.tick_params(axis='y', labelcolor='blue')
                self.ax_sys.set_ylim(0, total_gb + 5) 
                
                # Plot App RAM on the Right Axis (Red)
                line2 = self.ax_app.plot(self.time_data, self.app_data, 'r-', label="MaestroAI RAM", linewidth=2)
                self.ax_app.set_ylabel("App RAM (GB)", color='red', fontweight='bold')
                self.ax_app.tick_params(axis='y', labelcolor='red')
                
                # Dynamically scale the right axis to give the App line breathing room
                current_max_app = max(self.app_data) if self.app_data else 1.0
                self.ax_app.set_ylim(0, max(10.0, current_max_app * 1.5)) 
                
                # Formatting
                self.ax_sys.set_title("Memory Consumption History (Dynamic Sampling)")
                self.ax_sys.set_xlabel("Time Elapsed (Hours)")
                self.ax_sys.grid(True, linestyle='--', alpha=0.6)
                
                # Combine legends into one box
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                self.ax_sys.legend(lines, labels, loc="upper left")
                
                self.plot_canvas.draw_idle()
                
        except Exception as e:
            # Print the error to your IDE terminal just in case something else breaks!
            print(f"Diagnostics Error: {e}")

# --- UNIVERSAL COMPONENTS ---
from .maestro_loader import LoadWorker
from .maestro_fermi_viewer import FermiViewerWindow

# --- AI WORKERS ---
from .maestroai_training_ssl import TrainWorker
from .maestroai_training_sup import SupTrainWorker, SupTestWorker
from .maestroai_clustering import ClusterWorker
from .maestroai_active_learning import ActiveLearningWorker, SimulateALWorker
from .maestroai_alignment import AzimuthalTwistWorker, CoupledAzimuthTiltWorker, NormalTiltWorker

# --- AI VIEWERS & GUIDES ---
from .maestroai_guides import (MasterGuideDialog, SSLGuideDialog, ClusterGuideDialog, 
                              SupGuideDialog, ActiveLearningGuideDialog, SimulateALGuideDialog,
                              AlignmentGuideDialog)
from .maestroai_viewers import MplCanvas, DendrogramDialog, AzimuthTemplateViewer


class MaestroAIApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MaestroAI Suite")
        self.setMinimumSize(1000, 620)
        self.resize(1500, 900)
        self.workspace = {}
        self.current_folder = ""
        self.current_view_data = None
        self.init_ui()

    @staticmethod
    def _scrollable(widget):
        """Wrap a control column so it can shrink below its natural height."""
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        return scroll

    @classmethod
    def _split_tab(cls, controls, canvas, sizes=(300, 460)):
        """Tab body: scrollable controls above a canvas, divided by a drag handle.

        The canvas gets the stretch so it absorbs any extra height, and the
        controls stay reachable via the scroll area when the pane is short.
        """
        # Inset the canvas to line up with the control column, which the scroll
        # area indents by its own layout margin.
        canvas_holder = QWidget()
        holder_layout = QVBoxLayout(canvas_holder)
        holder_layout.setContentsMargins(9, 0, 9, 9)
        holder_layout.addWidget(canvas)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(cls._scrollable(controls))
        splitter.addWidget(canvas_holder)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(1, False)
        splitter.setSizes(list(sizes))

        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(splitter)
        return tab

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
        right_panel = QTabWidget()
        right_panel.setDocumentMode(True)
        right_panel.setMovable(True)
        right_panel.setUsesScrollButtons(True)
        # No elide: with this many tabs, truncated labels ("Mo...", "Bu...") are
        # unreadable, so keep full names and let the scroll arrows handle overflow.
        right_panel.setElideMode(Qt.TextElideMode.ElideNone)
        
        # --- NEW REIMAGINED TABS ---
        self.model_warehouse_tab = ModelWarehouseTab(self)
        self.build_pipeline_tab = BuildPipelineTab(self)
        self.train_model_tab = TrainModelTab(self)
        
        right_panel.addTab(self.model_warehouse_tab, "Model Warehouse")
        right_panel.addTab(self._scrollable(self.build_pipeline_tab), "Build Pipeline")
        right_panel.addTab(self.train_model_tab, "Train Model")
        
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
        right_panel.addTab(self._split_tab(train_controls, self.loss_canvas, sizes=(420, 380)),
                           "SSL Training")

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
        right_panel.addTab(self._split_tab(cluster_controls, self.umap_canvas, sizes=(400, 400)),
                           "Clustering")

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
        right_panel.addTab(self._scrollable(sup_tab), "Supervised Learning")

        # Tab D: Active Learning
        al_controls = QWidget()
        al_layout = QVBoxLayout(al_controls)

        self.btn_al_help = QPushButton("🧭 What is Active Learning? Click Here")
        self.btn_al_help.setStyleSheet("font-weight: bold; color: #8c564b; padding: 6px; font-size: 14px;")
        self.btn_al_help.clicked.connect(self.show_al_guide)
        al_layout.addWidget(self.btn_al_help)

        al_ctrl_group = QGroupBox("Active Learning Controls:")
        al_c_layout = QVBoxLayout(al_ctrl_group)
        
        self.combo_gp_domain = QComboBox() 
        al_c_layout.addWidget(QLabel("Target Clustered Domain:"))
        al_c_layout.addWidget(self.combo_gp_domain)

        self.combo_al_algo = QComboBox()
        self.combo_al_algo.addItems([
            "Bayesian Network (GPU)", "Deep Ensembles (GPU)", "Evidential Deep Learning (GPU)",
            "Gaussian Process (CPU)", "Random Forest (CPU)"
        ])
        al_c_layout.addWidget(QLabel("Steering Algorithm:"))
        al_c_layout.addWidget(self.combo_al_algo)

        self.btn_run_al = QPushButton("Calculate Next Scan Suggestions")
        self.btn_run_al.setStyleSheet("font-weight: bold; color: #8c564b; padding: 6px; font-size: 14px;")
        self.btn_run_al.clicked.connect(self.run_active_learning)

        self.al_canvas = MplCanvas(self, width=5, height=4, is_3d=False, orientation='vertical')
        self.ax_al_pred, self.ax_al_uncert = self.al_canvas.axes

        al_layout.addWidget(al_ctrl_group); al_layout.addWidget(self.btn_run_al)
        al_layout.addStretch()
        right_panel.addTab(self._split_tab(al_controls, self.al_canvas, sizes=(260, 540)),
                           "Active Learning")

        # Tab E: Simulate Active Learning
        sim_controls = QWidget()
        sim_layout = QVBoxLayout(sim_controls)

        self.btn_sim_help = QPushButton("🎮 How to use the AL Simulator? Click Here")
        self.btn_sim_help.setStyleSheet("font-weight: bold; color: #17becf; padding: 6px; font-size: 14px;")
        self.btn_sim_help.clicked.connect(lambda: SimulateALGuideDialog(self).exec())
        sim_layout.addWidget(self.btn_sim_help)

        sim_ctrl_group = QGroupBox("Simulation Controls:")
        sim_c_layout = QVBoxLayout(sim_ctrl_group)
        
        self.combo_sim_domain = QComboBox() 
        sim_c_layout.addWidget(QLabel("Ground Truth Domain (From Clustering):"))
        sim_c_layout.addWidget(self.combo_sim_domain)

        self.combo_sim_algo = QComboBox()
        self.combo_sim_algo.addItems([
            "Bayesian Network (GPU)", "Deep Ensembles (GPU)", "Evidential Deep Learning (GPU)",
            "Gaussian Process (CPU)", "Random Forest (CPU)"
        ])
        sim_c_layout.addWidget(QLabel("Steering Algorithm:"))
        sim_c_layout.addWidget(self.combo_sim_algo)

        h_sim = QHBoxLayout()
        self.spin_sim_points = QSpinBox(); self.spin_sim_points.setRange(2, 50); self.spin_sim_points.setValue(10)
        h_sim.addWidget(QLabel("Initial Random Seed Points:")); h_sim.addWidget(self.spin_sim_points)
        
        self.spin_sim_ff = QSpinBox(); self.spin_sim_ff.setRange(1, 500); self.spin_sim_ff.setValue(10)
        h_sim.addWidget(QLabel("Fast-Forward Steps:")); h_sim.addWidget(self.spin_sim_ff)
        sim_c_layout.addLayout(h_sim)

        btn_layout = QHBoxLayout()
        self.btn_sim_init = QPushButton("1. Initialize Simulation")
        self.btn_sim_init.setStyleSheet("font-weight: bold; color: #1f77b4;")
        self.btn_sim_init.clicked.connect(self.run_sim_init)
        
        self.btn_sim_step = QPushButton("2. Simulate 1 Step")
        self.btn_sim_step.setStyleSheet("font-weight: bold; color: #d62728;")
        self.btn_sim_step.setEnabled(False)
        self.btn_sim_step.clicked.connect(self.run_sim_step)
        
        self.btn_sim_ff = QPushButton("3. Fast-Forward")
        self.btn_sim_ff.setStyleSheet("font-weight: bold; color: #9467bd;")
        self.btn_sim_ff.setEnabled(False)
        self.btn_sim_ff.clicked.connect(self.run_sim_ff)
        
        btn_layout.addWidget(self.btn_sim_init); btn_layout.addWidget(self.btn_sim_step); btn_layout.addWidget(self.btn_sim_ff)

        self.sim_canvas = MplCanvas(self, width=5, height=4, is_3d=False, orientation='vertical_3')
        self.ax_sim_truth, self.ax_sim_pred, self.ax_sim_uncert = self.sim_canvas.axes

        sim_layout.addWidget(sim_ctrl_group); sim_layout.addLayout(btn_layout)
        sim_layout.addStretch()
        right_panel.addTab(self._split_tab(sim_controls, self.sim_canvas, sizes=(300, 500)),
                           "Simulate AL")

        self.sim_measured_mask = None
        self.sim_next_idx = None

        # Tab F: 3D Alignment
        align_controls = QWidget()
        align_layout = QVBoxLayout(align_controls)

        self.btn_align_help = QPushButton("📐 Guide to Alignment (Azimuth & Tilt)")
        self.btn_align_help.setStyleSheet("font-weight: bold; color: #bcbd22; padding: 6px; font-size: 14px;")
        self.btn_align_help.clicked.connect(lambda: AlignmentGuideDialog(self).exec())
        align_layout.addWidget(self.btn_align_help)

        align_ctrl_group = QGroupBox("Alignment Controls:")
        align_c_layout = QVBoxLayout(align_ctrl_group)
        
        self.combo_align_ref = QComboBox() 
        align_c_layout.addWidget(QLabel("Select Reference 3D Fermi Map:"))
        align_c_layout.addWidget(self.combo_align_ref)

        self.btn_inspect_ref = QPushButton("Inspect Selected Reference Map")
        self.btn_inspect_ref.setStyleSheet("font-weight: bold; color: #1f77b4; padding: 4px;")
        self.btn_inspect_ref.clicked.connect(self.open_fermi_viewer)
        align_c_layout.addWidget(self.btn_inspect_ref)

        self.btn_inspect_azimuth = QPushButton("Visualize Azimuthal Cuts")
        self.btn_inspect_azimuth.setStyleSheet("font-weight: bold; color: #9467bd; padding: 4px;")
        self.btn_inspect_azimuth.clicked.connect(self.open_azimuth_viewer)
        align_c_layout.addWidget(self.btn_inspect_azimuth)

        self.combo_align_mode = QComboBox()
        self.combo_align_mode.addItems([
            "Azimuthal Twist (In-Plane)", "Surface Normal Tilt (Out-of-Plane)", "Coupled Azimuth & Deflection Tilt"
        ])
        align_c_layout.addWidget(QLabel("Alignment Mode:"))
        align_c_layout.addWidget(self.combo_align_mode)
        
        gamma_layout = QHBoxLayout()
        self.spin_gamma_s = QDoubleSpinBox(); self.spin_gamma_s.setRange(-45, 45); self.spin_gamma_s.setDecimals(2)
        self.spin_gamma_d = QDoubleSpinBox(); self.spin_gamma_d.setRange(-45, 45); self.spin_gamma_d.setDecimals(2)
        gamma_layout.addWidget(QLabel("Γ Slit (°):")); gamma_layout.addWidget(self.spin_gamma_s)
        gamma_layout.addWidget(QLabel("Γ Defl (°):")); gamma_layout.addWidget(self.spin_gamma_d)
        align_c_layout.addWidget(QLabel("Center of Rotation (Γ-Point Target):"))
        align_c_layout.addLayout(gamma_layout)

        self.btn_run_align = QPushButton("Run Global Alignment Search")
        self.btn_run_align.setStyleSheet("font-weight: bold; color: #2ca02c; padding: 6px; font-size: 14px;")
        self.btn_run_align.clicked.connect(self.run_alignment)

        self.align_canvas = MplCanvas(self, width=5, height=4, is_3d=False, orientation='horizontal_3')
        self.ax_align_1, self.ax_align_2, self.ax_align_3 = self.align_canvas.axes

        align_layout.addWidget(align_ctrl_group); align_layout.addWidget(self.btn_run_align)
        align_layout.addStretch()
        right_panel.addTab(self._split_tab(align_controls, self.align_canvas, sizes=(380, 420)),
                           "3D Alignment")

        # --- NEW: Add the Diagnostics Tab ---
        self.diagnostics_tab = DiagnosticsTab(self)
        right_panel.addTab(self.diagnostics_tab, "Diagnostics")
        
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
    def show_al_guide(self):
        ActiveLearningGuideDialog(self).exec()

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
                
                if key.startswith("domains_"):
                    self.combo_gp_domain.addItem(key)
                    self.combo_sim_domain.addItem(key)

            self.combo_parent_filter.clear()
            self.combo_parent_filter.addItem("None (Run on Entire Map)")
            for k in self.current_view_data.keys():
                if k.startswith('domains_'):
                    unique_clusters = np.unique(self.current_view_data[k])
                    for c in unique_clusters:
                        if c != -1: 
                            self.combo_parent_filter.addItem(f"{k} -> Cluster {c}")

            # Send data to the viewer component
            self.viewer.set_data(self.current_view_data)
            
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
        self.workspace[var_name] = data
        if self.workspace_list.findItems(var_name, Qt.MatchFlag.MatchExactly) == []:
            self.workspace_list.addItem(var_name)
            self.combo_align_ref.addItem(var_name)
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
            
            # --- THE API HOOK --- 
            self.viewer.set_data(data)
            
            self.combo_gp_domain.blockSignals(True) 
            self.combo_sim_domain.blockSignals(True) 
            
            self.combo_gp_domain.clear(); self.combo_sim_domain.clear()       
            for k in data.keys():
                if k.startswith("domains_"):
                    self.combo_gp_domain.addItem(k); self.combo_sim_domain.addItem(k) 
                    
            self.combo_gp_domain.blockSignals(False)
            self.combo_sim_domain.blockSignals(False)
            
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
            # --- API HOOK ---
            self.viewer.add_overlay_mode(domain_key)
            
            if self.combo_gp_domain.findText(domain_key) == -1:
                self.combo_gp_domain.addItem(domain_key)
                self.combo_sim_domain.addItem(domain_key)
                
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
            # --- API HOOK ---
            self.viewer.add_overlay_mode(mode_name)
        
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

    def run_active_learning(self):
        if not self.current_view_data: return
        domain_key = self.combo_gp_domain.currentText()
        if not domain_key: 
            QMessageBox.warning(self, "No Domains", "Please run Clustering first to define the phases!")
            return
        algo = self.combo_al_algo.currentText()
        labels_2d = self.current_view_data[domain_key]
        x_arr, y_arr = self.current_view_data['x'], self.current_view_data['y']
        self.btn_run_al.setEnabled(False); self.prog_bar.setVisible(True)
        self.status.showMessage(f"Initializing {algo}...")
        self.al_worker = ActiveLearningWorker(x_arr, y_arr, labels_2d, algo)
        self.al_worker.setStackSize(32 * 1024 * 1024)
        self.al_worker.progress.connect(self.update_status)
        self.al_worker.finished.connect(self.on_al_finish)
        self.al_worker.error.connect(lambda e: self.status.showMessage(f"Active Learning Error: {e}"))
        self.al_worker.start()

    def on_al_finish(self, pred_map, uncert_map, new_x, new_y, bounds):
        self.ax_al_pred.clear(); self.ax_al_uncert.clear()
        extent = (new_x[0], new_x[-1], new_y[0], new_y[-1])
        self.ax_al_pred.imshow(pred_map, origin='lower', extent=extent, cmap='tab20', alpha=0.8)
        self.ax_al_pred.set_title("Extended Phase Prediction")
        import matplotlib.patches as patches
        self.ax_al_uncert.imshow(uncert_map, origin='lower', extent=extent, cmap='magma')
        self.ax_al_uncert.set_title("Uncertainty Heatmap\n(Bright = Scan Next)")
        for ax in [self.ax_al_pred, self.ax_al_uncert]:
            rect = patches.Rectangle((bounds[0], bounds[2]), bounds[1]-bounds[0], bounds[3]-bounds[2], linewidth=2, edgecolor='white', facecolor='none', linestyle='--')
            ax.add_patch(rect)
            ax.set_xlabel("X Position"); ax.set_ylabel("Y Position")
        self.al_canvas.draw_idle()
        self.btn_run_al.setEnabled(True); self.prog_bar.setVisible(False)
        self.status.showMessage("Steering Suggestions Computed!", 5000)
    
    def run_sim_init(self):
        if not self.current_view_data: return
        domain_key = self.combo_sim_domain.currentText()
        if not domain_key: 
            QMessageBox.warning(self, "No Domains", "Please run Clustering first to define the ground truth!")
            return
        labels_2d = self.current_view_data[domain_key]
        total_pixels = labels_2d.size
        self.sim_measured_mask = np.zeros(total_pixels, dtype=bool)
        self.sim_auto_steps = 0 
        valid_indices = np.where(labels_2d.flatten() != -1)[0]
        n_start = self.spin_sim_points.value()
        start_indices = np.random.choice(valid_indices, n_start, replace=False)
        self.sim_measured_mask[start_indices] = True
        self.btn_sim_step.setEnabled(False); self.btn_sim_ff.setEnabled(False)
        self.execute_sim_worker()

    def run_sim_ff(self):
        self.sim_auto_steps = self.spin_sim_ff.value() - 1 
        self.run_sim_step()

    def run_sim_step(self):
        if self.sim_next_idx is None or self.sim_next_idx == -1: return
        self.sim_measured_mask[self.sim_next_idx] = True
        self.btn_sim_step.setEnabled(False); self.btn_sim_ff.setEnabled(False); self.btn_sim_init.setEnabled(False)
        self.execute_sim_worker()

    def execute_sim_worker(self):
        domain_key = self.combo_sim_domain.currentText()
        algo = self.combo_sim_algo.currentText()
        labels_2d = self.current_view_data[domain_key]
        x_arr, y_arr = self.current_view_data['x'], self.current_view_data['y']
        self.prog_bar.setVisible(True)
        if self.sim_auto_steps > 0: self.status.showMessage(f"Fast-Forwarding {algo} ({self.sim_auto_steps} steps remaining)...")
        else: self.status.showMessage(f"Simulating {algo}...")
        self.sim_worker = SimulateALWorker(x_arr, y_arr, labels_2d, algo, self.sim_measured_mask)
        self.sim_worker.setStackSize(32 * 1024 * 1024) 
        self.sim_worker.progress.connect(self.update_status)
        self.sim_worker.finished.connect(self.on_sim_finish)
        self.sim_worker.error.connect(lambda e: self.status.showMessage(f"Simulation Error: {e}"))
        self.sim_worker.start()

    def on_sim_finish(self, pred_map, uncert_map, next_idx):
        self.ax_sim_truth.clear(); self.ax_sim_pred.clear(); self.ax_sim_uncert.clear()
        x_arr, y_arr = self.current_view_data['x'], self.current_view_data['y']
        X_grid, Y_grid = np.meshgrid(x_arr, y_arr)
        extent = (x_arr[0], x_arr[-1], y_arr[0], y_arr[-1])

        domain_key = self.combo_sim_domain.currentText()
        truth_map_1d = self.current_view_data[domain_key]
        truth_map_2d = truth_map_1d.reshape(X_grid.shape)

        self.ax_sim_truth.imshow(truth_map_2d, origin='lower', extent=extent, cmap='tab20', alpha=0.8, vmin=-0.5, vmax=19.5)
        self.ax_sim_pred.imshow(pred_map, origin='lower', extent=extent, cmap='tab20', alpha=0.8, vmin=-0.5, vmax=19.5)
        self.ax_sim_uncert.imshow(uncert_map, origin='lower', extent=extent, cmap='magma')

        measured_x = X_grid.flatten()[self.sim_measured_mask]
        measured_y = Y_grid.flatten()[self.sim_measured_mask]
        for ax in [self.ax_sim_truth, self.ax_sim_pred, self.ax_sim_uncert]:
            ax.scatter(measured_x, measured_y, c='white', s=10, edgecolors='black', label='Measured Points')
            ax.set_xlabel("X Position"); ax.set_ylabel("Y Position")

        self.sim_next_idx = next_idx
        if next_idx != -1:
            nxt_x = X_grid.flatten()[next_idx]
            nxt_y = Y_grid.flatten()[next_idx]
            self.ax_sim_uncert.scatter([nxt_x], [nxt_y], c='lime', s=80, marker='X', edgecolors='black', label='Next Scan Suggestion')
            self.ax_sim_uncert.legend(loc='upper right', fontsize=8)

        total_measured = np.sum(self.sim_measured_mask)
        self.ax_sim_truth.set_title("Ground Truth Domain Map")
        self.ax_sim_pred.set_title(f"Simulation Phase Prediction\n(Trained on {total_measured} points)")
        self.ax_sim_uncert.set_title("Simulation Uncertainty Heatmap\n(Green X = Next Step)")

        self.sim_canvas.figure.tight_layout()
        self.sim_canvas.draw_idle()
        
        if hasattr(self, 'sim_auto_steps') and self.sim_auto_steps > 0:
            self.sim_auto_steps -= 1
            from PySide6.QtCore import QTimer
            QTimer.singleShot(100, self.run_sim_step)
        else:
            self.btn_sim_init.setEnabled(True); self.btn_sim_step.setEnabled(True); self.btn_sim_ff.setEnabled(True)
            self.prog_bar.setVisible(False)
            self.status.showMessage("Simulation step complete!", 5000)

    def run_alignment(self):
        if not self.current_view_data:
            QMessageBox.warning(self, "No Target Map", "Please activate an XY Scan in the workspace first.")
            return
        ref_name = self.combo_align_ref.currentText()
        if not ref_name:
            QMessageBox.warning(self, "No Reference Map", "Please load a standard 3D Fermi map into the workspace.")
            return
        mode = self.combo_align_mode.currentText()
        gamma_s_deg, gamma_d_deg = self.spin_gamma_s.value(), self.spin_gamma_d.value()
        ref_data = self.workspace[ref_name]
        gamma_s_px = int(np.argmin(np.abs(ref_data['angle'] - gamma_s_deg)))
        gamma_d_px = int(np.argmin(np.abs(ref_data['x'] - gamma_d_deg)))
        
        self.btn_run_align.setEnabled(False); self.prog_bar.setVisible(True)
        self.status.showMessage(f"Initializing {mode} Search...")
        
        if "Coupled" in mode: self.align_worker = CoupledAzimuthTiltWorker(self.current_view_data, ref_data, gamma_s_px, gamma_d_px)
        elif "Azimuth" in mode: self.align_worker = AzimuthalTwistWorker(self.current_view_data, ref_data, gamma_s_px, gamma_d_px)
        else: self.align_worker = NormalTiltWorker(self.current_view_data, ref_data, gamma_s_px, gamma_d_px)
            
        self.align_worker.progress.connect(self.update_status)
        self.align_worker.finished.connect(self.on_align_finish)
        self.align_worker.error.connect(lambda e: self.status.showMessage(f"Alignment Error: {e}"))
        self.align_worker.start()

    def on_align_finish(self, map1, map2, map3, mode):
        self.align_canvas.figure.clf()
        self.ax_align_1, self.ax_align_2, self.ax_align_3 = self.align_canvas.figure.subplots(1, 3)
        x_arr, y_arr = self.current_view_data['x'], self.current_view_data['y']
        nX, nY = len(x_arr), len(y_arr)
        px_x = (x_arr[-1] - x_arr[0]) / max(1, nX-1) if nX > 1 else 0.1
        px_y = (y_arr[-1] - y_arr[0]) / max(1, nY-1) if nY > 1 else 0.1
        extent = (x_arr[0]-px_x/2, x_arr[-1]+px_x/2, y_arr[0]-px_y/2, y_arr[-1]+px_y/2)
        
        if "Coupled" in mode:
            title1, key1, cmap1 = r"Best Azimuth Rotation ($\phi^\circ$)", 'domains_Align_Azimuth_Coupled', 'hsv'
            title2, key2, cmap2 = r"Deflection Tilt Shift (pixels)", 'domains_Align_Defl_Coupled', 'PiYG'
            title3, key3, cmap3 = "Match Quality Score", 'domains_Align_Score_Coupled', 'magma'
        elif "Azimuth" in mode:
            title1, key1, cmap1 = r"Best Azimuth Rotation ($\phi^\circ$)", 'domains_Align_Azimuth', 'hsv'
            title2, key2, cmap2 = r"Momentum Slit Shift (pixels)", 'domains_Align_Slit_Shift', 'PiYG'
            title3, key3, cmap3 = "Match Quality Score", 'domains_Align_Score', 'magma'
        else: 
            title1, key1, cmap1 = r"Deflection Tilt Shift (pixels)", 'domains_Align_Defl_Tilt', 'PiYG'
            title2, key2, cmap2 = r"Momentum Slit Shift (pixels)", 'domains_Align_Slit_Tilt', 'PiYG'
            title3, key3, cmap3 = "Match Quality Score", 'domains_Align_Score_Tilt', 'magma'
            
        for ax, data_map, title, cmap in zip([self.ax_align_1, self.ax_align_2, self.ax_align_3], [map1, map2, map3], [title1, title2, title3], [cmap1, cmap2, cmap3]):
            im = ax.imshow(data_map, origin='lower', extent=extent, cmap=cmap, aspect='auto')
            ax.set_title(title, fontsize=10); ax.set_xlabel("X Position"); ax.set_ylabel("Y Position")
            self.align_canvas.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
        self.align_canvas.figure.tight_layout(); self.align_canvas.draw_idle()
        self.current_view_data[key1], self.current_view_data[key2], self.current_view_data[key3] = map1.flatten(), map2.flatten(), map3.flatten()
        
        # --- API HOOK ---
        for k in [key1, key2, key3]:
            self.viewer.add_overlay_mode(k)

        self.btn_run_align.setEnabled(True); self.prog_bar.setVisible(False)
        self.status.showMessage(f"{mode} successfully computed!", 5000)
    
    def open_fermi_viewer(self):
        ref_name = self.combo_align_ref.currentText()
        if not ref_name:
            QMessageBox.warning(self, "No Reference Map", "Please load a standard 3D Fermi map into the workspace.")
            return
        ref_data = self.workspace[ref_name]
        if len(ref_data['value'].shape) != 4 or ref_data['value'].shape[2] != 1:
            QMessageBox.warning(self, "Invalid Format", "The selected item is an XY Spatial Scan, not a 3D Deflection Map!")
            return
        self.fermi_dialog = FermiViewerWindow(ref_data, self); self.fermi_dialog.show()

    def open_azimuth_viewer(self):
        ref_name = self.combo_align_ref.currentText()
        if not ref_name:
            QMessageBox.warning(self, "No Reference Map", "Please load a standard 3D Fermi map into the workspace.")
            return
        gamma_s_deg = self.spin_gamma_s.value()
        gamma_d_deg = self.spin_gamma_d.value()
        self.az_viewer = AzimuthTemplateViewer(self.workspace[ref_name], gamma_s_deg, gamma_d_deg, self)
        self.az_viewer.show()