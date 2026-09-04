import traceback

import numpy as np
from PySide6.QtCore import QThread, Signal

from tensorspec.core.ml.active_learning import run_active_learning, run_simulate_al


class ActiveLearningWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, list)
    error = Signal(str)

    def __init__(self, x_arr, y_arr, labels_2d, algo):
        super().__init__()
        self.x_arr, self.y_arr = x_arr, y_arr
        self.labels_2d = labels_2d
        self.algo = algo

    def run(self):
        try:
            result = run_active_learning(
                self.x_arr, self.y_arr, self.labels_2d, self.algo,
                on_progress=self.progress.emit,
            )
            self.finished.emit(*result)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))


class SimulateALWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(np.ndarray, np.ndarray, int)
    error = Signal(str)

    def __init__(self, x_arr, y_arr, labels_2d, algo, measured_mask):
        super().__init__()
        self.x_arr, self.y_arr = x_arr, y_arr
        self.labels_2d = labels_2d
        self.algo = algo
        self.measured_mask = measured_mask

    def run(self):
        try:
            result = run_simulate_al(
                self.x_arr, self.y_arr, self.labels_2d, self.algo, self.measured_mask,
                on_progress=self.progress.emit,
            )
            self.finished.emit(*result)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))
