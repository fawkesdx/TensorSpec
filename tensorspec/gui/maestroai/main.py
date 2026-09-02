import sys
from PySide6.QtWidgets import QApplication
from .maestroai_gui import MaestroAIApp

def main():
    app = QApplication(sys.argv)
    window = MaestroAIApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()