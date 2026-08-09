"""ARPES Queue backend einstein_ssh (no live SSH)."""
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from pymatgen.core import Lattice, Structure

from tensorspec.core.workspace import WorkspaceManager
from tensorspec.web.server.schemas import ArpesSimRequest, AxisBound
from tensorspec.web.server.routers import arpes as arpes_router
from tensorspec.web.server.session import Session


def _tiny_request(**kwargs):
    base = dict(
        crystal_name="Si",
        model="A",
        backend="local",
        kx=AxisBound(min=-0.3, max=0.3, steps=4),
        ky=AxisBound(min=-0.3, max=0.3, steps=4),
        energy=AxisBound(min=-1.0, max=0.5, steps=4),
        mesh_resolution=4,
    )
    base.update(kwargs)
    return ArpesSimRequest(**base)


class TestArpesEinsteinBackend(unittest.TestCase):
    def test_backend_default_local(self):
        self.assertEqual(ArpesSimRequest(crystal_name="Si").backend, "local")

    def test_einstein_ssh_accepted(self):
        self.assertEqual(_tiny_request(backend="einstein_ssh").backend, "einstein_ssh")

    def test_argv_contains_script(self):
        run_dir = Path("/tmp/fake_arpes_job").resolve()
        argv = arpes_router._einstein_arpes_argv(run_dir, "einstein")
        self.assertEqual(argv[0], "bash")
        self.assertTrue(str(argv[1]).endswith("remote_arpes_me.sh"))
        self.assertEqual(argv[2], str(run_dir))
        self.assertIn("--host", argv)
        self.assertIn("einstein", argv)

    def test_script_path_exists(self):
        self.assertTrue(arpes_router._remote_arpes_me_script_path().is_file())

    def test_b1_einstein_returns_422(self):
        with self.assertRaises(HTTPException) as ctx:
            arpes_router._refuse_b1_on_einstein(_tiny_request(backend="einstein_ssh", model="B1"))
        self.assertEqual(ctx.exception.status_code, 422)

    def test_missing_script_returns_503(self):
        with patch.object(
            arpes_router,
            "_remote_arpes_me_script_path",
            return_value=Path("/tmp/missing_remote_arpes_me.sh"),
        ):
            with self.assertRaises(HTTPException) as ctx:
                arpes_router._require_remote_arpes_script()
        self.assertEqual(ctx.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
