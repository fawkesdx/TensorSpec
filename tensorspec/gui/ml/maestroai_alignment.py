import numpy as np
from PySide6.QtCore import QThread, Signal

from tensorspec.core.ml.alignment import (
    run_azimuthal_twist,
    run_coupled_azimuth_tilt,
    run_normal_tilt,
)


class AzimuthalTwistWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(np.ndarray, np.ndarray, np.ndarray, str)
    error = Signal(str)

    def __init__(self, xy_data, ref_data, gamma_s, gamma_d, max_shift=30):
        super().__init__()
        self.xy_data, self.ref_data = xy_data, ref_data
        self.gamma_slit, self.gamma_defl = gamma_s, gamma_d
        self.max_shift = max_shift

    def run(self):
        try:
            result = run_azimuthal_twist(
                self.xy_data, self.ref_data,
                self.gamma_slit, self.gamma_defl,
                max_shift=self.max_shift,
                on_progress=lambda v, m: self.progress.emit(v, m),
            )
            self.finished.emit(*result)
        except Exception as e:
            self.error.emit(str(e))


class CoupledAzimuthTiltWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(np.ndarray, np.ndarray, np.ndarray, str)
    error = Signal(str)

    def __init__(self, xy_data, ref_data, gamma_s, gamma_d, max_shift=20, max_tilt=15):
        super().__init__()
        self.xy_data, self.ref_data = xy_data, ref_data
        self.gamma_slit, self.gamma_defl = gamma_s, gamma_d
        self.max_shift, self.max_tilt = max_shift, max_tilt

    def run(self):
        try:
            result = run_coupled_azimuth_tilt(
                self.xy_data, self.ref_data,
                self.gamma_slit, self.gamma_defl,
                max_shift=self.max_shift,
                max_tilt=self.max_tilt,
                on_progress=lambda v, m: self.progress.emit(v, m),
            )
            self.finished.emit(*result)
        except Exception as e:
            self.error.emit(str(e))


class NormalTiltWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(np.ndarray, np.ndarray, np.ndarray, str)
    error = Signal(str)

    def __init__(self, xy_data, ref_data, gamma_s, gamma_d, max_shift=30):
        super().__init__()
        self.xy_data, self.ref_data = xy_data, ref_data
        self.gamma_slit, self.gamma_defl = gamma_s, gamma_d
        self.max_shift = max_shift

    def run(self):
        try:
            result = run_normal_tilt(
                self.xy_data, self.ref_data,
                self.gamma_slit, self.gamma_defl,
                max_shift=self.max_shift,
                on_progress=lambda v, m: self.progress.emit(v, m),
            )
            self.finished.emit(*result)
        except Exception as e:
            self.error.emit(str(e))
