import numpy as np
from PySide6.QtCore import QThread, Signal
import umap
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
import hdbscan

class ClusterWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(np.ndarray, np.ndarray) 
    error = Signal(str)

    def __init__(self, embeds, algo, k, eps, metric="euclidean", use_umap_first=False, normalize_edcs=False):
        super().__init__()
        self.embeds = embeds.copy() # Copy to avoid altering original workspace data
        self.algo = algo
        self.k = k
        self.eps = eps
        self.metric = metric 
        self.use_umap_first = use_umap_first 
        self.normalize_edcs = normalize_edcs # <-- NEW PARAMETER

    def run(self):
        try:
            self.progress.emit(5, "Preprocessing data...")
            
            # --- OPTIONAL INTENSITY NORMALIZATION ---
            if self.normalize_edcs:
                self.progress.emit(10, "Normalizing EDC intensities...")
                # Avoid division by zero for completely flat/empty noise channels
                max_vals = np.max(self.embeds, axis=1, keepdims=True)
                max_vals[max_vals == 0] = 1.0
                norm_embeds = self.embeds / max_vals
            else:
                norm_embeds = self.embeds
            
            # --- THE COSINE MATH TRICK ---
            if self.metric == "cosine" and not self.use_umap_first:
                from sklearn.preprocessing import normalize
                norm_embeds = normalize(norm_embeds, norm='l2', axis=1)
                safe_metric = "euclidean"
            else:
                safe_metric = self.metric
            
            # --- THE ROUTER: RAW vs UMAP SPACE ---
            if self.use_umap_first and self.algo != "DBSCAN":
                self.progress.emit(30, "Running pre-clustering UMAP...")
                cluster_data = umap.UMAP(n_components=3, metric=safe_metric, random_state=42).fit_transform(norm_embeds)
                cluster_metric = "euclidean" 
            else:
                cluster_data = norm_embeds
                cluster_metric = safe_metric

            # --- RUN ALGORITHM ---
            self.progress.emit(50, f"Running {self.algo}...")
            if self.algo == "K-Means":
                labels = KMeans(n_clusters=self.k, random_state=42, n_init='auto').fit_predict(cluster_data)
            elif self.algo == "Gaussian Mixture":
                labels = GaussianMixture(n_components=self.k, random_state=42, reg_covar=1e-3).fit_predict(cluster_data)
            elif self.algo == "Hierarchical":
                linkage_type = 'ward' if cluster_metric == 'euclidean' else 'average'
                labels = AgglomerativeClustering(n_clusters=self.k, metric=cluster_metric, linkage=linkage_type).fit_predict(cluster_data)
            elif self.algo == "DBSCAN":
                self.progress.emit(40, "Running UMAP for DBSCAN...")
                umap_res = umap.UMAP(n_components=2, metric=self.metric, random_state=42).fit_transform(norm_embeds)
                self.progress.emit(80, "Running DBSCAN...")
                labels = DBSCAN(eps=self.eps, min_samples=5).fit_predict(umap_res)
            elif self.algo == "HDBSCAN":
                labels = hdbscan.HDBSCAN(min_cluster_size=5, metric=cluster_metric).fit_predict(cluster_data)
            
            # --- RUN UMAP REDUCTION FOR 2D VISUALIZATION ---
            if self.algo != "DBSCAN":
                self.progress.emit(80, "Running UMAP Reduction for Visuals...")
                umap_res = umap.UMAP(n_components=2, metric=self.metric, random_state=42).fit_transform(norm_embeds)
            
            self.progress.emit(100, "Done!")
            self.finished.emit(labels, umap_res)
            
        except Exception as e:
            self.error.emit(str(e))