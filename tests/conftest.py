"""Shared pytest setup.

The Qt platform and matplotlib backend have to be chosen before anything
imports PySide6 or pyplot, so they are set here at collection time rather than
being required on every command line.
"""
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "QtAgg")
# The default ~/.matplotlib is not always writable; a throwaway dir avoids the
# "not a writable directory" warning and the font-cache rebuild it triggers.
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="tensorspec-mpl-"))

import pytest


@pytest.fixture(scope="session")
def qapp():
    """One QApplication for the whole test session; Qt allows only one."""
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])
