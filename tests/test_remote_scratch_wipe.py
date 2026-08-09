"""Remote scratch sidecar parse + wipe argv (no live SSH)."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

from tensorspec.web.server import remote_scratch as rs


class TestRemoteScratchWipe(unittest.TestCase):
    def test_remote_qe_script_writes_sidecar_contract(self):
        repo_root = Path(__file__).resolve().parents[1]
        text = (repo_root / "scripts" / "remote_qe.sh").read_text(encoding="utf-8")
        self.assertIn(".tensorspec_remote_scratch", text)
        self.assertIn("printf", text)
        # macOS BSD date rejects GNU `date -Is`
        self.assertNotIn("date -Is", text)
        self.assertIn("date '+%Y-%m-%dT%H:%M:%S%z'", text)

    def test_parse_ok(self):
        self.assertEqual(
            rs.parse_remote_scratch_sidecar("einstein\t/home/sandy/qe_scratch/job1"),
            ("einstein", "/home/sandy/qe_scratch/job1"),
        )

    def test_parse_rejects_relative(self):
        self.assertIsNone(rs.parse_remote_scratch_sidecar("einstein\trelative/path"))

    def test_parse_rejects_dotdot(self):
        self.assertIsNone(
            rs.parse_remote_scratch_sidecar("einstein\t/home/sandy/../etc")
        )

    def test_wipe_argv(self):
        argv = rs.wipe_remote_scratch_argv("einstein", "/home/sandy/qe_scratch/j")
        self.assertEqual(
            argv,
            [
                "ssh",
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=15",
                "einstein",
                "--",
                "rm", "-rf", "--",
                "/home/sandy/qe_scratch/j",
            ],
        )

    def test_best_effort_missing_sidecar(self):
        with TemporaryDirectory() as tmp:
            runner = MagicMock()
            ok = rs.best_effort_wipe_remote_scratch(
                Path(tmp), log=lambda _m: None, runner=runner
            )
            self.assertFalse(ok)
            runner.assert_not_called()

    def test_best_effort_calls_runner(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / rs.SIDECAR_NAME
            p.write_text("einstein\t/home/sandy/qe_scratch/j\n", encoding="utf-8")
            runner = MagicMock(return_value=0)
            logs: list[str] = []
            ok = rs.best_effort_wipe_remote_scratch(
                Path(tmp), log=logs.append, runner=runner
            )
            self.assertTrue(ok)
            runner.assert_called_once()
            self.assertEqual(
                runner.call_args[0][0],
                rs.wipe_remote_scratch_argv("einstein", "/home/sandy/qe_scratch/j"),
            )

    def test_best_effort_invalid_sidecar(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / rs.SIDECAR_NAME
            p.write_text("   \n  ", encoding="utf-8")
            runner = MagicMock()
            ok = rs.best_effort_wipe_remote_scratch(
                Path(tmp), log=lambda _m: None, runner=runner
            )
            self.assertFalse(ok)
            runner.assert_not_called()

    def test_best_effort_nonzero_exit(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / rs.SIDECAR_NAME
            p.write_text("einstein\t/home/sandy/qe_scratch/j\n", encoding="utf-8")
            runner = MagicMock(return_value=1)
            ok = rs.best_effort_wipe_remote_scratch(
                Path(tmp), log=lambda _m: None, runner=runner
            )
            self.assertFalse(ok)
            runner.assert_called_once()

    def test_best_effort_runner_raises(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / rs.SIDECAR_NAME
            p.write_text("einstein\t/home/sandy/qe_scratch/j\n", encoding="utf-8")
            runner = MagicMock(side_effect=OSError("ssh failed"))
            ok = rs.best_effort_wipe_remote_scratch(
                Path(tmp), log=lambda _m: None, runner=runner
            )
            self.assertFalse(ok)
            runner.assert_called_once()


if __name__ == "__main__":
    unittest.main()
