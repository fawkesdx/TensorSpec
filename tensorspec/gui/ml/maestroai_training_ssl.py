from PySide6.QtCore import QThread, Signal

from tensorspec.core.ml.training_ssl import run_ssl_train


class TrainWorker(QThread):
    progress = Signal(int, float)
    model_changed = Signal(str)
    finished = Signal(dict)

    def __init__(self, data_array, epochs, lr, selected_models):
        super().__init__()
        self.data_array = data_array
        self.epochs = epochs
        self.lr = lr
        self.selected_models = selected_models

    def run(self):
        results = run_ssl_train(
            self.data_array,
            self.epochs,
            self.lr,
            self.selected_models,
            on_progress=lambda e, loss: self.progress.emit(e, loss),
            on_model_changed=lambda n: self.model_changed.emit(n),
        )
        self.finished.emit(results)
