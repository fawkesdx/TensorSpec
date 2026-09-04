import sys
from PySide6.QtWidgets import QApplication, QMainWindow

from tensorspec.gui.suites.ml_suite import MLSuite


def main():
    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("TensorSpec - Machine Learning Suite")
    window.setCentralWidget(MLSuite())
    window.resize(1500, 900)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
