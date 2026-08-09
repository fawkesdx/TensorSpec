"""Contract tests for scripts/remote_arpes_me.sh (no live SSH)."""
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "remote_arpes_me.sh"


class TestRemoteArpesMeScript(unittest.TestCase):
    def test_script_contract_strings(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("arpes_me_scratch", text)
        self.assertIn(".tensorspec_remote_scratch", text)
        self.assertIn("intensity.npz", text)
        self.assertIn("meta.json", text)
        self.assertIn("remote_arpes_me.log", text)
        self.assertIn("run_arpes_me_a.py", text)
        self.assertIn("--dry-run", text)
        self.assertIn("ssh", text)  # live path uses ssh; dry-run exits before invoke
        # macOS BSD date rejects GNU `date -Is`; keep portable format
        self.assertNotIn("date -Is", text)
        self.assertIn("date '+%Y-%m-%dT%H:%M:%S%z'", text)

    def test_script_writes_sidecar_contract(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("printf", text)
        self.assertIn(".tensorspec_remote_scratch", text)

    def test_dry_run_zero_network(self):
        with TemporaryDirectory() as tmp:
            job = Path(tmp)
            (job / "request.json").write_text('{"model":"A"}', encoding="utf-8")
            (job / "structure.cif").write_text("data_dummy\n", encoding="utf-8")
            r = subprocess.run(
                ["bash", str(SCRIPT), str(job), "--dry-run"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, msg=r.stderr + r.stdout)
            out = r.stdout + r.stderr
            self.assertIn("dry-run", out.lower())
            self.assertIn("arpes_me_scratch", out)

    def test_dry_run_rejects_missing_request(self):
        with TemporaryDirectory() as tmp:
            job = Path(tmp)
            (job / "structure.cif").write_text("data_dummy\n", encoding="utf-8")
            r = subprocess.run(
                ["bash", str(SCRIPT), str(job), "--dry-run"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 2)


if __name__ == "__main__":
    unittest.main()
