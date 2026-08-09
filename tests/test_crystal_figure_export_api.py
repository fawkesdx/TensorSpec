"""Crystal figure export endpoint (no browser)."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient
from pymatgen.core import Lattice, Structure

from tensorspec.core.workspace import WorkspaceManager
from tensorspec.web.server.app import create_app
from tensorspec.web.server.routers import crystal as crystal_router
from tensorspec.web.server.schemas import CrystalFigureExportRequest
from tensorspec.web.server.session import Session, current_session


class TestCrystalFigureExportSchema(unittest.TestCase):
    def test_cell_count_property(self):
        req = CrystalFigureExportRequest(nx=2, ny=3, nz=1)
        self.assertEqual(req.cell_count, 6)

    def test_defaults(self):
        req = CrystalFigureExportRequest()
        self.assertEqual(req.fmt, "png")
        self.assertTrue(req.show_bonds)
        self.assertFalse(req.show_polyhedra)


class TestCrystalFigureExportRoute(unittest.TestCase):
    def _session_with_si(self, tmp: str) -> Session:
        session = Session(
            session_id="t",
            workspace=WorkspaceManager(project_dir=Path(tmp)),
        )
        structure = Structure(
            Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]]
        )
        session.workspace.push_crystal_structure(
            "Si", structure.lattice.matrix, structure=structure
        )
        return session

    def test_export_figure_png(self):
        with TemporaryDirectory() as tmp:
            session = self._session_with_si(tmp)
            req = CrystalFigureExportRequest(fmt="png", show_bonds=True)
            resp = crystal_router.export_figure_route("Si", req, session=session)
            self.assertEqual(resp.media_type, "image/png")
            self.assertGreater(len(resp.body), 100)
            self.assertTrue(resp.body[:8] == b"\x89PNG\r\n\x1a\n")
            self.assertIn('filename="Si_figure.png"', resp.headers["Content-Disposition"])

    def test_export_figure_svg(self):
        with TemporaryDirectory() as tmp:
            session = self._session_with_si(tmp)
            req = CrystalFigureExportRequest(fmt="svg")
            resp = crystal_router.export_figure_route("Si", req, session=session)
            self.assertEqual(resp.media_type, "image/svg+xml")
            self.assertIn(b"<svg", resp.body.lower())

    def test_export_figure_omit_changes_output(self):
        with TemporaryDirectory() as tmp:
            session = self._session_with_si(tmp)
            full = crystal_router.export_figure_route(
                "Si",
                CrystalFigureExportRequest(fmt="png"),
                session=session,
            )
            trimmed = crystal_router.export_figure_route(
                "Si",
                CrystalFigureExportRequest(fmt="png", omit_atom_indices=[1]),
                session=session,
            )
            self.assertNotEqual(full.body, trimmed.body)

    def test_missing_structure_404(self):
        with TemporaryDirectory() as tmp:
            session = self._session_with_si(tmp)
            from fastapi import HTTPException

            with self.assertRaises(HTTPException) as ctx:
                crystal_router.export_figure_route(
                    "Missing",
                    CrystalFigureExportRequest(),
                    session=session,
                )
            self.assertEqual(ctx.exception.status_code, 404)


class TestCrystalFigureExportHttp(unittest.TestCase):
    def _client_with_si(self, tmp: str) -> tuple[TestClient, Session]:
        session = Session(
            session_id="t",
            workspace=WorkspaceManager(project_dir=Path(tmp)),
        )
        structure = Structure(
            Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]]
        )
        session.workspace.push_crystal_structure(
            "Si", structure.lattice.matrix, structure=structure
        )

        app = create_app()
        app.dependency_overrides[current_session] = lambda: session
        return TestClient(app), session

    def test_export_figure_http_png(self):
        with TemporaryDirectory() as tmp:
            client, _ = self._client_with_si(tmp)
            resp = client.post(
                "/api/crystal/Si/export/figure",
                json={"fmt": "png", "show_bonds": True},
            )
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.headers["content-type"], "image/png")
            self.assertGreater(len(resp.content), 100)
            self.assertTrue(resp.content[:8] == b"\x89PNG\r\n\x1a\n")

    def test_export_figure_route_registered_before_fmt_param(self):
        figure_idx = None
        fmt_idx = None
        for i, route in enumerate(crystal_router.router.routes):
            path = getattr(route, "path", "")
            if path.endswith("/export/figure"):
                figure_idx = i
            if path.endswith("/export/{fmt}"):
                fmt_idx = i
        self.assertIsNotNone(figure_idx)
        self.assertIsNotNone(fmt_idx)
        self.assertLess(figure_idx, fmt_idx)


if __name__ == "__main__":
    unittest.main()
