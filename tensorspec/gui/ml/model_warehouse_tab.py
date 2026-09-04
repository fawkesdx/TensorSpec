import os
import torch
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QTextEdit, QPushButton, QMessageBox
)
from PySide6.QtCore import Qt

class ModelWarehouseTab(QWidget):
    def __init__(self, parent_app=None):
        super().__init__(parent_app)
        self.parent_app = parent_app

        self.local_models = [
            "SimpleCAE", "CNNVAE", "CNNMaskedAutoencoder", "SimCLRModel",
            "MoCoModel", "BYOLModel", "SwAVModel", "ViTMAE", "SupervisedCNN"
        ]

        self.pretrained_models = {
            "DINOv2 ViT-S/14": {
                "repo": "facebookresearch/dinov2",
                "model_name": "dinov2_vits14",
                "description": "DINOv2: Learning Robust Visual Features without Supervision (ViT-Small, patch size 14)."
            },
            "DINOv2 ViT-B/14": {
                "repo": "facebookresearch/dinov2",
                "model_name": "dinov2_vitb14",
                "description": "DINOv2: Learning Robust Visual Features without Supervision (ViT-Base, patch size 14)."
            }
        }

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # Left side: Model List
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QListWidget.SingleSelection)

        # Add local models
        local_label = QListWidgetItem("--- Local Models ---")
        local_label.setFlags(Qt.NoItemFlags) # Make it unselectable header
        self.model_list.addItem(local_label)

        for model in self.local_models:
            self.model_list.addItem(model)

        # Add pretrained models
        pretrained_label = QListWidgetItem("--- Pretrained Models ---")
        pretrained_label.setFlags(Qt.NoItemFlags) # Make it unselectable header
        self.model_list.addItem(pretrained_label)

        for model in self.pretrained_models.keys():
            self.model_list.addItem(model)

        self.model_list.itemSelectionChanged.connect(self.on_model_selected)

        main_layout.addWidget(self.model_list, 1)

        # Right side: Properties and Action
        right_layout = QVBoxLayout()

        self.properties_text = QTextEdit()
        self.properties_text.setReadOnly(True)
        self.properties_text.setPlaceholderText("Select a model to view properties...")
        right_layout.addWidget(self.properties_text, 1)

        self.action_button = QPushButton("Load")
        self.action_button.setEnabled(False)
        self.action_button.clicked.connect(self.on_action_clicked)
        right_layout.addWidget(self.action_button)

        self.status_label = QLabel("")
        right_layout.addWidget(self.status_label)

        main_layout.addLayout(right_layout, 2)

    def on_model_selected(self):
        selected_items = self.model_list.selectedItems()
        if not selected_items:
            return

        item_text = selected_items[0].text()

        if item_text in self.pretrained_models:
            details = self.pretrained_models[item_text]
            self.properties_text.setText(f"Model: {item_text}\n\nDescription:\n{details['description']}")
            self.action_button.setText("Download")
            self.action_button.setEnabled(True)
            self.status_label.setText("Status: Ready to download")
        elif item_text in self.local_models:
            self.properties_text.setText(f"Model: {item_text}\n\nDescription:\nLocal model architecture.")
            self.action_button.setText("Load")
            self.action_button.setEnabled(True)
            self.status_label.setText("Status: Ready to load")
        else:
            self.properties_text.clear()
            self.action_button.setEnabled(False)
            self.status_label.setText("")

    def on_action_clicked(self):
        selected_items = self.model_list.selectedItems()
        if not selected_items:
            return

        item_text = selected_items[0].text()

        if item_text in self.pretrained_models:
            details = self.pretrained_models[item_text]
            self.status_label.setText(f"Status: Downloading {item_text}...")
            
            # Force UI update if parent has processEvents (e.g., QApplication)
            if hasattr(self.parent_app, 'processEvents'):
                self.parent_app.processEvents()

            try:
                cache_dir = 'tensorspec/gui/ml/model_warehouse/pretrained'
                os.makedirs(cache_dir, exist_ok=True)
                torch.hub.set_dir(cache_dir)

                # Load model
                model = torch.hub.load(details['repo'], details['model_name'])
                self.status_label.setText(f"Status: Successfully downloaded {item_text}")
                QMessageBox.information(self, "Success", f"{item_text} downloaded successfully.")
            except Exception as e:
                self.status_label.setText(f"Status: Error downloading {item_text}")
                QMessageBox.critical(self, "Error", f"Failed to download {item_text}:\n{str(e)}")
        elif item_text in self.local_models:
            self.status_label.setText(f"Status: Loaded {item_text} locally.")
