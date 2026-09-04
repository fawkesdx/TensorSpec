"""Self-supervised training controls plus live loss canvas."""
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from tensorspec.gui.components.ml_tabs.layout import split_panel
from tensorspec.gui.ml.maestroai_guides import SSLGuideDialog
from tensorspec.gui.ml.maestroai_training_ssl import TrainWorker
from tensorspec.gui.ml.maestroai_viewers import MplCanvas


class SSLTrainingPanel(QWidget):
    """Self-supervised training controls plus live loss canvas."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.loss_history_dict = {}
        self.current_training_model = None
        self.active_train_target = None
        self._build()

    def _build(self):
        controls = QWidget()
        train_layout = QVBoxLayout(controls)

        self.btn_ssl_help = QPushButton("📖 New to SSL Training? Click Here for a Guide")
        self.btn_ssl_help.setStyleSheet(
            "font-weight: bold; color: #1f77b4; padding: 6px; font-size: 14px;"
        )
        self.btn_ssl_help.clicked.connect(self.show_ssl_guide)
        train_layout.addWidget(self.btn_ssl_help)

        model_selection_layout = QVBoxLayout()
        group_gen = QGroupBox("1A. Generative (Reconstruction)")
        layout_gen = QVBoxLayout(group_gen)
        self.chk_ae = QCheckBox("Autoencoder (CNN)")
        self.chk_ae.setChecked(True)
        self.chk_beta = QCheckBox("Beta-VAE (CNN)")
        self.chk_mae = QCheckBox("MAE (CNN)")
        self.chk_vit_mae = QCheckBox("ViT-MAE (Vision Transformer)")
        layout_gen.addWidget(self.chk_ae)
        layout_gen.addWidget(self.chk_beta)
        layout_gen.addWidget(self.chk_mae)
        layout_gen.addWidget(self.chk_vit_mae)

        group_con = QGroupBox("1B. Contrastive (Negative Sampling)")
        layout_con = QVBoxLayout(group_con)
        self.chk_simclr = QCheckBox("SimCLR")
        self.chk_moco = QCheckBox("MoCo (Momentum Contrast)")
        layout_con.addWidget(self.chk_simclr)
        layout_con.addWidget(self.chk_moco)

        group_dist = QGroupBox("1C. Distillation (No Negatives)")
        layout_dist = QVBoxLayout(group_dist)
        self.chk_byol = QCheckBox("BYOL (Bootstrap Your Own Latent)")
        layout_dist.addWidget(self.chk_byol)

        group_clust = QGroupBox("1D. Clustering")
        layout_clust = QVBoxLayout(group_clust)
        self.chk_swav = QCheckBox("SwAV (Swapping Assignments)")
        layout_clust.addWidget(self.chk_swav)

        model_selection_layout.addWidget(group_gen)
        model_selection_layout.addWidget(group_con)
        model_selection_layout.addWidget(group_dist)
        model_selection_layout.addWidget(group_clust)

        hyper_group = QGroupBox("2. Hyperparameters:")
        hyper_layout = QHBoxLayout(hyper_group)
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 500)
        self.spin_epochs.setValue(15)
        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setRange(0.0001, 0.1)
        self.spin_lr.setSingleStep(0.001)
        self.spin_lr.setDecimals(4)
        self.spin_lr.setValue(0.0010)
        hyper_layout.addWidget(QLabel("Epochs:"))
        hyper_layout.addWidget(self.spin_epochs)
        hyper_layout.addWidget(QLabel("Learn Rate:"))
        hyper_layout.addWidget(self.spin_lr)

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

        train_layout.addLayout(model_selection_layout)
        train_layout.addWidget(hyper_group)
        train_layout.addWidget(self.btn_train)
        train_layout.addLayout(loss_view_layout)
        train_layout.addStretch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(split_panel(controls, self.loss_canvas, sizes=(420, 380)))

    def show_ssl_guide(self):
        SSLGuideDialog(self).exec()

    def start_training(self):
        if not self.session.current_view_data:
            QMessageBox.warning(
                self, "No Data", "Select a loaded file in the Workspace first."
            )
            return
        selected_models = []
        if self.chk_ae.isChecked():
            selected_models.append("Autoencoder")
        if self.chk_beta.isChecked():
            selected_models.append("Beta-VAE")
        if self.chk_mae.isChecked():
            selected_models.append("MAE")
        if self.chk_vit_mae.isChecked():
            selected_models.append("ViT-MAE")
        if self.chk_simclr.isChecked():
            selected_models.append("SimCLR")
        if self.chk_moco.isChecked():
            selected_models.append("MoCo")
        if self.chk_byol.isChecked():
            selected_models.append("BYOL")
        if self.chk_swav.isChecked():
            selected_models.append("SwAV")
        if not selected_models:
            return

        self.btn_train.setEnabled(False)
        self.session.set_status(1, "Initializing PyTorch Training Queue...")
        self.loss_history_dict.clear()
        self.combo_loss_view.clear()

        self.active_train_target = self.session.current_view_data

        e, lr = self.spin_epochs.value(), self.spin_lr.value()
        self.trainer = TrainWorker(
            self.active_train_target["value"],
            epochs=e,
            lr=lr,
            selected_models=selected_models,
        )
        self.trainer.progress.connect(self.update_live_loss)
        self.trainer.model_changed.connect(self.on_model_changed)
        self.trainer.finished.connect(self.on_train_finish)
        self.trainer.start()

    def on_model_changed(self, model_name):
        self.current_training_model = model_name
        self.loss_history_dict[model_name] = {"epochs": [], "losses": []}
        self.combo_loss_view.blockSignals(True)
        if self.combo_loss_view.findText(model_name) == -1:
            self.combo_loss_view.addItem(model_name)
        self.combo_loss_view.setCurrentText(model_name)
        self.combo_loss_view.blockSignals(False)
        self.redraw_loss_plot(model_name)

    def update_live_loss(self, epoch, loss):
        if not self.current_training_model:
            return
        self.loss_history_dict[self.current_training_model]["epochs"].append(epoch)
        self.loss_history_dict[self.current_training_model]["losses"].append(loss)
        if self.combo_loss_view.currentText() == self.current_training_model:
            self.redraw_loss_plot(self.current_training_model)

    def on_loss_view_changed(self, model_name):
        if model_name:
            self.redraw_loss_plot(model_name)

    def redraw_loss_plot(self, model_name):
        self.ax_loss.clear()
        self.ax_loss.set_title(f"Training Loss: {model_name}")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        if model_name in self.loss_history_dict:
            epochs = self.loss_history_dict[model_name]["epochs"]
            losses = self.loss_history_dict[model_name]["losses"]
            self.ax_loss.plot(
                epochs, losses, "r-", linewidth=2, marker="o", markersize=4
            )
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()
        self.loss_canvas.draw_idle()

    def on_train_finish(self, new_embeddings_dict):
        for key, emb in new_embeddings_dict.items():
            self.active_train_target[key] = emb
        if self.session.current_view_data is self.active_train_target:
            self.session.notify_embeddings()
        self.btn_train.setEnabled(True)
        self.session.set_status(100, "All selected models finished training!")
