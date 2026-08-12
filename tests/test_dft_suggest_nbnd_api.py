"""DFT structures API includes suggest_nbnd."""

from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient
from pymatgen.core import Lattice, Structure

from tensorspec.core.workspace import WorkspaceManager
from tensorspec.web.server.app import create_app
from tensorspec.web.server.session import Session, current_session


class TestDftSuggestNbndApi(unittest.TestCase):
    def test_structures_include_suggest_nbnd(self):
        with TemporaryDirectory() as tmp:
            session = Session(
                session_id="nbnd-api",
                workspace=WorkspaceManager(project_dir=tmp),
            )
            graphene = Structure(
                Lattice.hexagonal(2.46, 20.0),
                ["C", "C"],
                [[0, 0, 0], [1 / 3, 2 / 3, 0]],
            )
            session.workspace.push_crystal_structure(
                "graphene", graphene.lattice.matrix, structure=graphene
            )
            app = create_app()
            app.dependency_overrides[current_session] = lambda: session
            client = TestClient(app)
            try:
                resp = client.get("/api/dft/structures")
                self.assertEqual(resp.status_code, 200)
                rows = resp.json()
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["name"], "graphene")
                self.assertEqual(rows[0]["suggest_nbnd"], 8)
            finally:
                app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()
