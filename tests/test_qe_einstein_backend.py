"""QE Queue backend einstein_ssh builds remote_qe.sh argv."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from pymatgen.core import Lattice, Structure

from tensorspec.core.workspace import WorkspaceManager
from tensorspec.web.server.config import load_solver_config
from tensorspec.web.server.schemas import QERequest
from tensorspec.web.server.routers import dft as dft_router
from tensorspec.web.server.session import Session


class TestQEEinsteinBackend(unittest.TestCase):
    def test_backend_default_local(self):
        self.assertEqual(QERequest().backend, "local")

    def test_einstein_ssh_backend_accepted(self):
        req = QERequest(backend="einstein_ssh")
        self.assertEqual(req.backend, "einstein_ssh")

    def test_einstein_commands_contain_script(self):
        run_dir = Path("/tmp/fake_run").resolve()
        cmds = dft_router._einstein_ssh_commands(run_dir, mpi_ranks=4, host="einstein")
        self.assertEqual(len(cmds), 1)
        argv = cmds[0]
        self.assertEqual(argv[0], "bash")
        self.assertTrue(str(argv[1]).endswith("remote_qe.sh"))
        self.assertEqual(argv[2], str(run_dir))
        self.assertIn("--np", argv)
        self.assertIn("4", argv)
        self.assertIn("--host", argv)
        self.assertIn("einstein", argv)

    def test_script_path_exists_in_repo(self):
        script = dft_router._remote_qe_script_path()
        self.assertTrue(script.is_file(), msg=str(script))

    def test_load_solver_config_soft_without_binaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            pseudo_dir = Path(tmp) / "pseudo"
            pseudo_dir.mkdir()
            env = {
                "PATH": "",
                "TENSORSPEC_PSEUDO_DIR": str(pseudo_dir),
            }
            for key in (
                "TENSORSPEC_PW",
                "TENSORSPEC_WANNIER90",
                "TENSORSPEC_PW2WANNIER90",
                "TENSORSPEC_MPIRUN",
            ):
                env.pop(key, None)
            with patch.dict(os.environ, env, clear=True):
                cfg = load_solver_config(require_binaries=False)
            self.assertEqual(cfg.pseudo_dir.resolve(), pseudo_dir.resolve())
            self.assertFalse(cfg.pw.exists())
            self.assertFalse(cfg.wannier90.exists())
            self.assertFalse(cfg.pw2wannier90.exists())

    def test_prepare_run_without_local_pw(self):
        with tempfile.TemporaryDirectory() as tmp:
            pseudo_dir = Path(tmp) / "pseudo"
            pseudo_dir.mkdir()
            (pseudo_dir / "Si.pbe.upf").write_text("stub\n")
            session = Session(
                session_id="test",
                workspace=WorkspaceManager(project_dir=Path(tmp)),
            )
            structure = Structure(Lattice.cubic(4.0), ["Si"], [[0, 0, 0]])
            session.workspace.push_crystal_structure(
                "Si", structure.lattice.matrix, structure=structure
            )
            env = {
                "PATH": "",
                "TENSORSPEC_PSEUDO_DIR": str(pseudo_dir),
            }
            with patch.dict(os.environ, env, clear=True):
                cfg, run_dir, params, files = dft_router._prepare_run(
                    session,
                    "Si",
                    QERequest(backend="einstein_ssh"),
                    relative_outdir=True,
                )
            self.assertEqual(cfg.pseudo_dir.resolve(), pseudo_dir.resolve())
            self.assertTrue(run_dir.is_dir())
            self.assertIn("scf.in", files)
            scf = (run_dir / "scf.in").read_text()
            self.assertIn("outdir = './out/'", scf)
            self.assertNotIn(str(run_dir), scf)

    def test_einstein_queue_rewrites_relative_outdir(self):
        """Regression: absolute Mac outdir breaks pw.x on Einstein scratch."""
        with tempfile.TemporaryDirectory() as tmp:
            pseudo_dir = Path(tmp) / "pseudo"
            pseudo_dir.mkdir()
            (pseudo_dir / "Si.pbe.upf").write_text("stub\n")
            session = Session(
                session_id="test",
                workspace=WorkspaceManager(project_dir=Path(tmp)),
            )
            structure = Structure(Lattice.cubic(4.0), ["Si"], [[0, 0, 0]])
            session.workspace.push_crystal_structure(
                "Si", structure.lattice.matrix, structure=structure
            )
            env = {
                "PATH": "",
                "TENSORSPEC_PSEUDO_DIR": str(pseudo_dir),
            }
            fake_job = type(
                "Job",
                (),
                {
                    "to_dict": lambda self: {
                        "job_id": "j1",
                        "run_name": "run_rel",
                        "status": "queued",
                        "current_step": 0,
                        "total_steps": 1,
                        "exit_code": None,
                        "error": None,
                        "created_at": 0.0,
                        "started_at": None,
                        "finished_at": None,
                    },
                },
            )()
            with patch.dict(os.environ, env, clear=True):
                with patch.object(dft_router, "get_job_queue") as mock_queue:
                    mock_queue.return_value.submit.return_value = fake_job
                    dft_router.queue_qe_run(
                        "Si",
                        QERequest(backend="einstein_ssh", run_name="run_rel"),
                        session=session,
                    )
            run_dir = Path(tmp) / "qe_runs" / "run_rel"
            scf = (run_dir / "scf.in").read_text()
            nscf = (run_dir / "nscf.in").read_text()
            self.assertIn("outdir = './out/'", scf)
            self.assertIn("outdir = './out/'", nscf)
            self.assertNotIn("/Users/", scf)

    def test_missing_remote_qe_script_returns_503(self):
        with tempfile.TemporaryDirectory() as tmp:
            pseudo_dir = Path(tmp) / "pseudo"
            pseudo_dir.mkdir()
            (pseudo_dir / "Si.pbe.upf").write_text("stub\n")
            session = Session(
                session_id="test",
                workspace=WorkspaceManager(project_dir=Path(tmp)),
            )
            structure = Structure(Lattice.cubic(4.0), ["Si"], [[0, 0, 0]])
            session.workspace.push_crystal_structure(
                "Si", structure.lattice.matrix, structure=structure
            )
            env = {
                "PATH": "",
                "TENSORSPEC_PSEUDO_DIR": str(pseudo_dir),
            }
            missing = Path(tmp) / "missing_remote_qe.sh"
            with patch.dict(os.environ, env, clear=True):
                with patch.object(
                    dft_router, "_remote_qe_script_path", return_value=missing
                ):
                    with self.assertRaises(HTTPException) as ctx:
                        dft_router.queue_qe_run(
                            "Si",
                            QERequest(backend="einstein_ssh"),
                            session=session,
                        )
            self.assertEqual(ctx.exception.status_code, 503)
            self.assertIn("remote_qe.sh", ctx.exception.detail)


if __name__ == "__main__":
    unittest.main()
