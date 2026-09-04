from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QComboBox, QListWidget, 
    QPushButton, QHBoxLayout, QSpinBox, QRadioButton, QButtonGroup, 
    QLabel
)

class BuildPipelineTab(QWidget):
    def __init__(self, parent_app):
        super().__init__()
        self.parent_app = parent_app
        
        main_layout = QVBoxLayout()
        
        # Input Block
        input_group = QGroupBox("Input Data")
        input_layout = QVBoxLayout()
        self.input_combo = QComboBox()
        self.input_combo.addItems(["Workspace Data 1", "Workspace Data 2"]) # Placeholder
        input_layout.addWidget(self.input_combo)
        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)
        
        # Preprocessing Block
        prep_group = QGroupBox("Preprocessing")
        prep_layout = QVBoxLayout()
        self.prep_list = QListWidget()
        
        prep_controls_layout = QHBoxLayout()
        self.prep_combo = QComboBox()
        self.prep_combo.addItems([
            "Normalize", "Flatten", "Crop", 
            "Resize to 224x224", "Background Subtract"
        ])
        self.add_prep_btn = QPushButton("Add")
        self.add_prep_btn.clicked.connect(self.add_preprocessing_layer)
        
        prep_controls_layout.addWidget(self.prep_combo)
        prep_controls_layout.addWidget(self.add_prep_btn)
        
        prep_layout.addWidget(self.prep_list)
        prep_layout.addLayout(prep_controls_layout)
        prep_group.setLayout(prep_layout)
        main_layout.addWidget(prep_group)
        
        # Model Block
        model_group = QGroupBox("Model Warehouse")
        model_layout = QVBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Model A", "Model B"]) # Placeholder
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)
        
        # Clustering Block
        cluster_group = QGroupBox("Clustering")
        cluster_layout = QHBoxLayout()
        
        self.cluster_combo = QComboBox()
        self.cluster_combo.addItems(["KMeans", "DBSCAN"])
        
        cluster_layout.addWidget(QLabel("Algorithm:"))
        cluster_layout.addWidget(self.cluster_combo)
        
        cluster_layout.addWidget(QLabel("Params (e.g. n_clusters):"))
        self.cluster_spinbox = QSpinBox()
        self.cluster_spinbox.setMinimum(1)
        self.cluster_spinbox.setMaximum(100)
        self.cluster_spinbox.setValue(3)
        cluster_layout.addWidget(self.cluster_spinbox)
        
        cluster_group.setLayout(cluster_layout)
        main_layout.addWidget(cluster_group)
        
        # Output Block
        output_group = QGroupBox("Output Strategy")
        output_layout = QHBoxLayout()
        
        self.unsup_radio = QRadioButton("Unsupervised (Label Map)")
        self.sup_radio = QRadioButton("Supervised (1D array)")
        self.unsup_radio.setChecked(True)
        
        self.output_btngroup = QButtonGroup()
        self.output_btngroup.addButton(self.unsup_radio)
        self.output_btngroup.addButton(self.sup_radio)
        
        output_layout.addWidget(self.unsup_radio)
        output_layout.addWidget(self.sup_radio)
        output_group.setLayout(output_layout)
        main_layout.addWidget(output_group)
        
        # Run Button
        run_layout = QHBoxLayout()
        self.compute_target = QComboBox()
        self.compute_target.addItems(["Local Device"])
        import os, json
        conf = os.path.expanduser('~/.tensorspec_clusters.json')
        if os.path.exists(conf):
            try:
                with open(conf, 'r') as f:
                    clusters = json.load(f)
                    self.compute_target.addItems([c['name'] for c in clusters])
            except:
                pass
                
        run_layout.addWidget(QLabel("Compute Target:"))
        run_layout.addWidget(self.compute_target)
        
        self.run_btn = QPushButton("Run Pipeline")
        self.run_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        self.run_btn.clicked.connect(self.run_pipeline)
        run_layout.addWidget(self.run_btn)
        main_layout.addLayout(run_layout)
        
        self.setLayout(main_layout)

    def add_preprocessing_layer(self):
        layer = self.prep_combo.currentText()
        self.prep_list.addItem(layer)

    def run_pipeline(self):
        target = self.compute_target.currentText()
        if target == "Local Device":
            print("Running pipeline locally... (Placeholder)")
        else:
            print(f"Dispatching pipeline to cluster: {target} ... (Placeholder)")
