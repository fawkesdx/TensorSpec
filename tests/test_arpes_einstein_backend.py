"""ARPES Queue backend einstein_ssh (no live SSH)."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException
from pymatgen.core import Lattice, Structure

from tensorspec.core.workspace import WorkspaceManager
from tensorspec.web.server.jobs import Job, JobStatus
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

    def test_write_job_dir_and_load_npz(self):
        with TemporaryDirectory() as tmp:
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
            run_dir = Path(tmp) / "arpes_jobs" / "sim"
            run_dir.mkdir(parents=True)
            req = _tiny_request(backend="einstein_ssh")

            def fake_popen(argv, **kwargs):
                cube = np.ones((4, 4, 4), dtype=float)
                np.savez_compressed(
                    run_dir / "intensity.npz",
                    intensity=cube,
                    E=np.linspace(-1, 0.5, 4),
                    kx=np.linspace(-0.3, 0.3, 4),
                    ky=np.linspace(-0.3, 0.3, 4),
                )

                class P:
                    stdout = iter(["[fake] ok\n"])
                    pid = 1

                    def wait(self):
                        return 0

                return P()

            job = Job(
                job_id="j",
                session_id="t",
                run_name="sim",
                run_dir=run_dir,
                commands=[],
                total_steps=2,
            )
            job.status = JobStatus.RUNNING
            worker = arpes_router._build_einstein_sim_worker(session.session_id, req)
            with patch(
                "tensorspec.web.server.routers.arpes.subprocess.Popen",
                side_effect=fake_popen,
            ):
                with patch.object(arpes_router, "session_store") as store:
                    store._sessions = {session.session_id: session}
                    worker(job)
            self.assertIsInstance(job.result, dict)
            self.assertEqual(job.result["model"], "A")
            self.assertEqual(tuple(job.result["intensity"].shape), (4, 4, 4))
            self.assertTrue((run_dir / "request.json").is_file())
            self.assertTrue((run_dir / "structure.cif").is_file())
            dumped = json.loads((run_dir / "request.json").read_text())
            self.assertEqual(dumped["model"], "A")


if __name__ == "__main__":
    unittest.main()
