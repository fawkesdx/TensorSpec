import numpy as np
from PySide6.QtCore import QThread, Signal

from tensorspec.core.ml.training_sup import run_sup_test, run_sup_train


class SupTrainWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(object)

    def __init__(self, X, Y, num_classes):
        super().__init__()
        self.X, self.Y, self.num_classes = X, Y, num_classes

    def run(self):
        model = run_sup_train(
            self.X,
            self.Y,
            self.num_classes,
            on_progress=lambda v, m: self.progress.emit(v, m),
        )
        self.finished.emit(model)


class SupTestWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(np.ndarray)

    def __init__(self, model, val_array):
        super().__init__()
        self.model = model
        self.val_array = val_array

    def run(self):
        prob_map = run_sup_test(
            self.model,
            self.val_array,
            on_progress=lambda v, m: self.progress.emit(v, m),
        )
        self.finished.emit(prob_map)
