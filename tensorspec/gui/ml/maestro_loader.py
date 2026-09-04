import os
from PySide6.QtCore import QThread, Signal


class LoadWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(str, object)
    error = Signal(str)

    def __init__(self, path, var_name):
        super().__init__()
        self.path = path
        self.var_name = var_name

    def run(self):
        try:
            os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
            self.progress.emit(10, "Loading via ARPESLoader...")
            from tensorspec.core.io.arpes_loader import ARPESLoader

            tensor_data = ARPESLoader.load(self.path)
            self.progress.emit(100, "Done!")
            self.finished.emit(self.var_name, tensor_data)
        except Exception as e:
            import traceback

            traceback.print_exc()
            self.error.emit(str(e))
            self.progress.emit(0, "Load Failed.")
