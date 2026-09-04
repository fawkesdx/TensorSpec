"""Clustering controls plus UMAP / band canvas."""
import os

import numpy as np
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGroupBox,
    QHBoxLayout, QLabel, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from tensorspec.gui.components.ml_tabs.layout import split_panel
from tensorspec.gui.ml.maestroai_clustering import ClusterWorker
from tensorspec.gui.ml.maestroai_guides import ClusterGuideDialog
from tensorspec.gui.ml.maestroai_viewers import DendrogramDialog, MplCanvas


class ClusterPanel(QWidget):
    """Clustering controls plus UMAP / band canvas."""

    _INTEGRATED_ITEMS = (
        "Integrated EDC (from Viewer)",
        "Integrated MDC (from Viewer)",
    )

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.active_mask_indices = None
        self.active_cluster_target = None
        self._build()
        self.session.embeddings_changed.connect(self.set_embedding_keys)
        self.set_embedding_keys([])

    def _build(self):
        cluster_controls = QWidget()
        cluster_layout = QVBoxLayout(cluster_controls)

        self.btn_cluster_help = QPushButton("📊 What do these Algorithms do? Click Here")
        self.btn_cluster_help.setStyleSheet(
            "font-weight: bold; color: #d62728; padding: 6px; font-size: 14px;"
        )
        self.btn_cluster_help.clicked.connect(self.show_cluster_guide)
        cluster_layout.addWidget(self.btn_cluster_help)

        cluster_ctrl_group = QGroupBox("Clustering Controls:")
        c_layout = QVBoxLayout(cluster_ctrl_group)

        self.combo_embed = QComboBox()
        self.combo_parent_filter = QComboBox()
        self.combo_parent_filter.addItem("None (Run on Entire Map)")

        self.combo_algo = QComboBox()
        self.combo_algo.addItems([
            "Hierarchical", "K-Means", "Gaussian Mixture", "DBSCAN", "HDBSCAN",
        ])
        self.combo_algo.currentTextChanged.connect(self.toggle_cluster_params)

        self.combo_metric = QComboBox()
        self.combo_metric.addItems(["euclidean", "cosine", "correlation"])

        h_k = QHBoxLayout()
        self.spin_k = QSpinBox()
        self.spin_k.setRange(2, 50)
        self.spin_k.setValue(5)
        h_k.addWidget(QLabel("Clusters (k):"))
        h_k.addWidget(self.spin_k)

        h_eps = QHBoxLayout()
        self.spin_eps = QDoubleSpinBox()
        self.spin_eps.setRange(0.01, 10.0)
        self.spin_eps.setSingleStep(0.1)
        self.spin_eps.setValue(1.50)
        h_eps.addWidget(QLabel("DBSCAN eps:"))
        h_eps.addWidget(self.spin_eps)

        self.chk_umap_first = QCheckBox("Run UMAP First (HDBSCAN / Hierarchical)")
        self.chk_umap_first.setChecked(False)

        self.chk_normalize = QCheckBox("Normalize EDC Intensity (Max = 1.0)")
        self.chk_normalize.setChecked(True)

        c_layout.addWidget(QLabel("Target Embedding:"))
        c_layout.addWidget(self.combo_embed)
        c_layout.addWidget(QLabel("Parent Domain Filter:"))
        c_layout.addWidget(self.combo_parent_filter)
        c_layout.addWidget(QLabel("Algorithm:"))
        c_layout.addWidget(self.combo_algo)
        c_layout.addWidget(QLabel("Distance Metric:"))
        c_layout.addWidget(self.combo_metric)
        c_layout.addLayout(h_k)
        c_layout.addLayout(h_eps)
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
        self.combo_umap_plot_type.addItems([
            "Full Dispersion 2D", "Integrated EDC 1D", "Integrated MDC 1D",
        ])
        umap_ctrl_layout.addWidget(self.combo_umap_plot_type)
        umap_ctrl_layout.addStretch()

        self.umap_canvas = MplCanvas(self, width=4, height=4, is_3d=False)
        self.ax_umap, self.ax_band = self.umap_canvas.axes
        self.umap_canvas.mpl_connect("pick_event", self.on_umap_pick)

        cluster_layout.addWidget(cluster_ctrl_group)
        cluster_layout.addWidget(self.btn_cluster)
        cluster_layout.addWidget(self.btn_dendro)
        cluster_layout.addWidget(self.btn_save_labels)
        cluster_layout.addLayout(umap_ctrl_layout)
        cluster_layout.addStretch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(split_panel(cluster_controls, self.umap_canvas, sizes=(400, 400)))

    def set_embedding_keys(self, keys):
        """Repopulate the embedding combo; replaces writes from activate_data."""
        self.combo_embed.blockSignals(True)
        self.combo_embed.clear()
        self.combo_embed.addItems(keys)
        self.combo_embed.addItems(self._INTEGRATED_ITEMS)
        self.combo_embed.blockSignals(False)

    def _refresh_viewer_ml_layers(self, select_layer: str | None = None):
        """Push ML domain/label arrays into the shared DataViewer."""
        if not self.session.current_view_data or not self.session.viewer:
            return
        self.session.viewer.sync_ml_layers(self.session.current_view_data)
        if select_layer:
            self.session.viewer.focus_spatial_layer(select_layer)

    def show_cluster_guide(self):
        ClusterGuideDialog(self).exec()

    def show_dendrogram(self):
        if not self.session.current_view_data:
            QMessageBox.warning(self, "No Data", "Please load a file first.")
            return
        algo = self.combo_algo.currentText()
        if algo != "Hierarchical":
            QMessageBox.information(
                self, "Wrong Algorithm",
                "The Dendrogram is only mathematically possible when the "
                "'Hierarchical' algorithm is selected in the dropdown!",
            )
            return
        embed_key = self.combo_embed.currentText()
        k = self.spin_k.value()
        if k > 15:
            QMessageBox.warning(
                self, "Too Many Branches",
                "Please lower k to 15 or less to visually render the spectra.",
            )
            return
        if embed_key in self._INTEGRATED_ITEMS:
            QMessageBox.information(
                self, "Use ML Embeddings",
                "The Dendrogram preview works best with Neural Network embeddings. "
                "Please select an embedding like SimCLR or MAE.",
            )
            return
        embeds = self.session.current_view_data[embed_key]

        val = self.session.current_view_data["value"]
        E_arr = self.session.current_view_data["E"]
        A_arr = self.session.current_view_data["angle"]

        contrast_scale = 1.0
        gamma_scale = 1.0
        if self.session.viewer:
            contrast_scale = self.session.viewer.get_dispersion_contrast()

        self.session.set_status(1, "Calculating Hierarchical Tree (This may take a few seconds)...")
        QApplication.processEvents()
        dialog = DendrogramDialog(
            embeds, val, k, E_arr, A_arr, contrast_scale, gamma_scale, self,
        )
        self.session.set_status(100, "Ready.")
        dialog.exec()

    def toggle_cluster_params(self, algo_name):
        if algo_name == "DBSCAN":
            self.spin_k.setEnabled(False)
            self.spin_eps.setEnabled(True)
            self.chk_umap_first.setEnabled(False)
        else:
            self.spin_k.setEnabled(True)
            self.spin_eps.setEnabled(False)
            self.chk_umap_first.setEnabled(True)

    def run_clustering(self):
        if not self.session.current_view_data:
            return
        embed_key = self.combo_embed.currentText()
        parent_filter = self.combo_parent_filter.currentText()
        algo = self.combo_algo.currentText()
        k = self.spin_k.value()
        eps = self.spin_eps.value()

        if embed_key in self._INTEGRATED_ITEMS:
            val = self.session.current_view_data["value"]
            dim_E, dim_A, nY, nX = val.shape

            if self.session.viewer:
                e_c, de, a_c, da = self.session.viewer.get_slider_values()
            else:
                e_c, de = dim_E // 2, dim_E // 2
                a_c, da = dim_A // 2, dim_A // 2

            e1, e2 = max(0, e_c - de), min(dim_E, e_c + de + 1)
            a1, a2 = max(0, a_c - da), min(dim_A, a_c + da + 1)

            if "EDC" in embed_key:
                sliced = np.sum(val[e1:e2, a1:a2, :, :], axis=1)
                embeds = sliced.transpose(1, 2, 0).reshape(nY * nX, e2 - e1)
            else:
                sliced = np.sum(val[e1:e2, a1:a2, :, :], axis=0)
                embeds = sliced.transpose(1, 2, 0).reshape(nY * nX, a2 - a1)
            row_max = embeds.max(axis=1, keepdims=True) + 1e-8
            embeds = embeds / row_max
        else:
            embeds = self.session.current_view_data[embed_key]

        self.active_mask_indices = None
        if parent_filter != "None (Run on Entire Map)":
            domain_key, cluster_val = parent_filter.split(" -> Cluster ")
            cluster_val = int(cluster_val)
            labels = self.session.current_view_data[domain_key]
            self.active_mask_indices = np.where(labels == cluster_val)[0]
            if len(self.active_mask_indices) < 5:
                QMessageBox.warning(
                    self, "Too Small",
                    "Selected cluster has too few pixels to sub-cluster.",
                )
                return
            embeds = embeds[self.active_mask_indices]

        metric = self.combo_metric.currentText()
        do_umap_first = self.chk_umap_first.isChecked()
        do_normalize = self.chk_normalize.isChecked()

        self.btn_cluster.setEnabled(False)
        self.session.set_status(1, f"Running {algo}...")

        self.active_cluster_target = self.session.current_view_data

        self.cluster_worker = ClusterWorker(
            embeds, algo, k, eps,
            metric=metric, use_umap_first=do_umap_first, normalize_edcs=do_normalize,
        )
        self.cluster_worker.progress.connect(self.session.set_status)
        self.cluster_worker.finished.connect(
            lambda l, u: self.on_cluster_finish(l, u, embed_key, algo)
        )
        self.cluster_worker.error.connect(
            lambda e: self.session.set_status(0, f"Error: {e}")
        )
        self.cluster_worker.start()

    def on_cluster_finish(self, labels, umap_res, embed_key, algo):
        domain_key = f"domains_{embed_key}"

        if self.active_mask_indices is not None:
            dim_E, dim_A, nY, nX = self.active_cluster_target["value"].shape
            total_pixels = nY * nX
            full_labels = np.full(total_pixels, -1, dtype=int)
            full_labels[self.active_mask_indices] = labels
            parent_txt = self.combo_parent_filter.currentText().replace(" -> ", "_sub")
            domain_key = f"{domain_key}_{parent_txt}"
            self.active_cluster_target[domain_key] = full_labels
        else:
            self.active_cluster_target[domain_key] = labels

        if self.session.current_view_data is self.active_cluster_target:
            self._refresh_viewer_ml_layers(domain_key)
            self.session.notify_domains()

            self.combo_parent_filter.clear()
            self.combo_parent_filter.addItem("None (Run on Entire Map)")
            for k in self.session.current_view_data.keys():
                if k.startswith("domains_"):
                    unique_clusters = np.unique(self.session.current_view_data[k])
                    for c in unique_clusters:
                        if c != -1:
                            self.combo_parent_filter.addItem(f"{k} -> Cluster {c}")

            self.ax_umap.clear()
            self.ax_band.clear()
            self.scatter = self.ax_umap.scatter(
                umap_res[:, 0], umap_res[:, 1], c=labels, cmap="tab20",
                vmin=-0.5, vmax=19.5, s=15, alpha=0.8, picker=5,
            )
            self.ax_umap.set_title(f"{algo}: {embed_key}\n(Click a point!)")
            self.ax_umap.set_xlabel("UMAP 1")
            self.ax_umap.set_ylabel("UMAP 2")
            self.ax_band.axis("off")
            self.umap_canvas.draw_idle()

        self.btn_cluster.setEnabled(True)
        self.session.set_status(100, "Clustering complete.")

    def on_umap_pick(self, event):
        if event.mouseevent.inaxes != self.ax_umap or not self.session.current_view_data:
            return
        ind = event.ind[0]
        if self.active_mask_indices is not None:
            global_ind = self.active_mask_indices[ind]
        else:
            global_ind = ind

        data = self.session.current_view_data
        val = data["value"]
        dim_E, dim_A, nY, nX = val.shape
        flat_bands = val.transpose(2, 3, 0, 1).reshape((nY * nX, dim_E, dim_A))
        band = flat_bands[global_ind]

        E_arr, A_arr = data["E"], data["angle"]
        px_A = (A_arr[-1] - A_arr[0]) / max(1, dim_A - 1)
        px_E = (E_arr[-1] - E_arr[0]) / max(1, dim_E - 1)
        extent = (
            A_arr[0] - px_A / 2, A_arr[-1] + px_A / 2,
            E_arr[0] - px_E / 2, E_arr[-1] + px_E / 2,
        )

        self.ax_band.clear()
        self.ax_band.axis("on")
        plot_type = self.combo_umap_plot_type.currentText()
        if "EDC" in plot_type:
            edc = np.sum(band, axis=1)
            self.ax_band.plot(edc, E_arr, "r-", linewidth=2)
            self.ax_band.set_xlabel("Intensity")
            self.ax_band.set_ylabel("Energy (eV)")
        elif "MDC" in plot_type:
            mdc = np.sum(band, axis=0)
            self.ax_band.plot(A_arr, mdc, "b-", linewidth=2)
            self.ax_band.set_xlabel("Angle (Degrees)")
            self.ax_band.set_ylabel("Intensity")
        else:
            self.ax_band.imshow(
                band, aspect="auto", cmap="magma", origin="lower", extent=extent,
            )
            self.ax_band.set_xlabel("Angle (Degrees)")
            self.ax_band.set_ylabel("Energy (eV)")

        embed_key = self.combo_embed.currentText()
        domain_key = f"domains_{embed_key}"
        if self.active_mask_indices is not None:
            parent_txt = self.combo_parent_filter.currentText().replace(" -> ", "_sub")
            domain_key = f"{domain_key}_{parent_txt}"

        labels = data.get(domain_key, [])
        cluster_id = labels[global_ind] if len(labels) > global_ind else "Unknown"
        self.ax_band.set_title(f"Global Index: {global_ind} | Cluster: {cluster_id}")
        self.umap_canvas.draw_idle()

    def save_labels(self):
        if not self.session.current_view_data:
            QMessageBox.warning(self, "No Data", "Please load a file first.")
            return
        domain_keys = [
            k for k in self.session.current_view_data.keys() if k.startswith("domains_")
        ]
        if not domain_keys:
            QMessageBox.warning(
                self, "No Labels",
                "No clustering labels found. Run clustering first.",
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Labels", "MAESTRO_Cluster_Labels.csv",
            "CSV Files (*.csv);;Text Files (*.txt)",
        )
        if not path:
            return
        try:
            X_arr, Y_arr = self.session.current_view_data["x"], self.session.current_view_data["y"]
            X_grid, Y_grid = np.meshgrid(X_arr, Y_arr)
            x_flat, y_flat = X_grid.flatten(), Y_grid.flatten()
            header = "X,Y"
            cols = [x_flat, y_flat]
            for k in domain_keys:
                clean_name = k.replace("domains_", "")
                header += f",label_{clean_name}"
                cols.append(self.session.current_view_data[k])
            out_matrix = np.column_stack(cols)
            np.savetxt(
                path, out_matrix, delimiter=",", header=header, comments="", fmt="%g",
            )
            self.session.set_status(100, f"✅ Saved labels to {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save file:\n{str(e)}")
