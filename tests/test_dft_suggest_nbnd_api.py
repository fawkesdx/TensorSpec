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

    def test_params_replaces_default_nbnd_12(self):
        from pathlib import Path

        from tensorspec.web.server.routers.dft import _params_from_request
        from tensorspec.web.server.schemas import QERequest

        cdw = Structure.from_file(
            Path(__file__).resolve().parents[1]
            / "tensorspec"
            / "cif_file"
            / "VTe2_CDW_XRD.cif"
        )
        request = QERequest(
            run_name="nbnd_check",
            ecutwfc=60,
            nbnd=12,
            kx=2,
            ky=2,
            kz=1,
            use_soc=True,
            mlwf_mode=False,
            use_mpi=False,
            mpi_ranks=1,
            slab_mode=False,
            functional="PBE",
            backend="local",
        )
        params = _params_from_request(request, 8, structure=cdw)
        self.assertEqual(params.nbnd, 324)

        request_no_soc = request.model_copy(update={"use_soc": False})
        params2 = _params_from_request(request_no_soc, 8, structure=cdw)
        self.assertEqual(params2.nbnd, 162)


if __name__ == "__main__":
    unittest.main()
