"""QE Queue backend einstein_ssh builds remote_qe.sh argv."""
import unittest
from pathlib import Path

from tensorspec.web.server.schemas import QERequest
from tensorspec.web.server.routers import dft as dft_router


class TestQEEinsteinBackend(unittest.TestCase):
    def test_backend_default_local(self):
        self.assertEqual(QERequest().backend, "local")

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


if __name__ == "__main__":
    unittest.main()
