from PySide6.QtWidgets import (QWidget, QVBoxLayout, QTabWidget, QFormLayout, 
                               QComboBox, QSpinBox, QDoubleSpinBox, QPushButton)

class TrainModelTab(QWidget):
    def __init__(self, parent_app, parent=None):
        super().__init__(parent)
        self.parent_app = parent_app
        self.init_ui()
        
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Tab 1: Fine-Tune Mode
        self.fine_tune_tab = QWidget()
        self.init_fine_tune_tab()
        self.tab_widget.addTab(self.fine_tune_tab, "Fine-Tune Mode")
        
        # Tab 2: Supervised Head Mode
        self.supervised_head_tab = QWidget()
        self.init_supervised_head_tab()
        self.tab_widget.addTab(self.supervised_head_tab, "Supervised Head Mode")
        
    def init_fine_tune_tab(self):
        layout = QFormLayout(self.fine_tune_tab)
        
        self.ft_base_model_combo = QComboBox()
        layout.addRow("Base Model:", self.ft_base_model_combo)
        
        self.ft_freeze_layers_spin = QSpinBox()
        self.ft_freeze_layers_spin.setRange(0, 100)
        layout.addRow("Freeze all except last N layers:", self.ft_freeze_layers_spin)
        
        self.ft_training_data_combo = QComboBox()
        layout.addRow("Training Data:", self.ft_training_data_combo)
        
        self.ft_learning_rate_spin = QDoubleSpinBox()
        self.ft_learning_rate_spin.setRange(0.0001, 1.0)
        self.ft_learning_rate_spin.setSingleStep(0.001)
        self.ft_learning_rate_spin.setDecimals(4)
        layout.addRow("Learning Rate:", self.ft_learning_rate_spin)
        
        self.ft_epochs_spin = QSpinBox()
        self.ft_epochs_spin.setRange(1, 1000)
        layout.addRow("Epochs:", self.ft_epochs_spin)
        
        # --- Remote Compute Integration ---
        self.ft_compute_target = QComboBox()
        self.ft_compute_target.addItems(["Local Device"])
        # Load from ~/.tensorspec_clusters.json
        import os, json
        conf = os.path.expanduser('~/.tensorspec_clusters.json')
        if os.path.exists(conf):
            try:
                with open(conf, 'r') as f:
                    clusters = json.load(f)
                    self.ft_compute_target.addItems([c['name'] for c in clusters])
            except:
                pass
        layout.addRow("Compute Target:", self.ft_compute_target)
        
        self.ft_start_btn = QPushButton("Start Training")
        self.ft_start_btn.clicked.connect(self.start_training_job)
        layout.addRow(self.ft_start_btn)
        
    def start_training_job(self):
        target = self.ft_compute_target.currentText()
        if target == "Local Device":
            print("Training locally...")
        else:
            print(f"Dispatching training job to cluster: {target}")
            # Here we would use JobDispatcher to submit the job
        
    def init_supervised_head_tab(self):
        layout = QFormLayout(self.supervised_head_tab)
        
        self.sh_base_model_combo = QComboBox()
        layout.addRow("Base Model:", self.sh_base_model_combo)
        
        self.sh_reference_file_combo = QComboBox()
        layout.addRow("Labeled Reference File:", self.sh_reference_file_combo)
        
        self.sh_train_btn = QPushButton("Train Head")
        layout.addRow(self.sh_train_btn)
