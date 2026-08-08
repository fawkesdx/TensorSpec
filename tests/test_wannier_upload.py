"""Wannier upload path helpers for DFT Suite."""
import tempfile
import unittest
from pathlib import Path

from tensorspec.web.server.routers import dft as dft_router
from tensorspec.web.server.schemas import BandRequest
from tensorspec.web.server.session import Session
from tensorspec.core.workspace import WorkspaceManager


class TestWannierUploadPaths(unittest.TestCase):
    def test_band_request_has_use_wannier(self):
        req = BandRequest()
        self.assertFalse(req.use_wannier)
        req2 = BandRequest(use_wannier=True)
        self.assertTrue(req2.use_wannier)

    def test_wannier_dir_and_hr_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Session(
                session_id="test",
                workspace=WorkspaceManager(project_dir=Path(tmp)),
            )
            dest = dft_router._wannier_dir(session, "MoS2")
            self.assertTrue(dest.is_dir())
            self.assertIsNone(dft_router._wannier_hr_path(session, "MoS2"))
            hr = dest / "wannier90_hr.dat"
            hr.write_text("comment\n1\n1\n1\n")
            found = dft_router._wannier_hr_path(session, "MoS2")
            self.assertEqual(found, hr)


if __name__ == "__main__":
    unittest.main()
