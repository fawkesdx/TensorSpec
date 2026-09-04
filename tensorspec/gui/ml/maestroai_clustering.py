import numpy as np
from PySide6.QtCore import QThread, Signal

from tensorspec.core.ml.clustering import run_clustering


class ClusterWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(np.ndarray, np.ndarray)
    error = Signal(str)

    def __init__(
        self,
        embeds,
        algo,
        k,
        eps,
        metric="euclidean",
        use_umap_first=False,
        normalize_edcs=False,
    ):
        super().__init__()
        self.embeds = embeds.copy()
        self.algo = algo
        self.k = k
        self.eps = eps
        self.metric = metric
        self.use_umap_first = use_umap_first
        self.normalize_edcs = normalize_edcs

    def run(self):
        try:
            labels, umap_res = run_clustering(
                self.embeds,
                self.algo,
                self.k,
                self.eps,
                metric=self.metric,
                use_umap_first=self.use_umap_first,
                normalize_edcs=self.normalize_edcs,
                on_progress=lambda v, m: self.progress.emit(v, m),
            )
            self.finished.emit(labels, umap_res)
        except Exception as e:
            self.error.emit(str(e))
