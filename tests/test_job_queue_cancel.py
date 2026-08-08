"""JobQueue cancel: process group kill + remote scratch wipe."""
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tensorspec.web.server import remote_scratch as rs
from tensorspec.web.server.jobs import JobQueue, JobStatus


class TestJobQueueCancel(unittest.TestCase):
    def test_cancel_invokes_wipe_when_sidecar_present(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            queue = JobQueue(max_global_jobs=1, max_jobs_per_session=1)
            job = queue.submit(
                session_id="sess1",
                run_name="test",
                run_dir=run_dir,
                commands=[["sleep", "30"]],
            )

            deadline = time.time() + 5
            while time.time() < deadline:
                current = queue.get(job.job_id)
                if current and current.status == JobStatus.RUNNING:
                    break
                time.sleep(0.05)
            self.assertEqual(queue.get(job.job_id).status, JobStatus.RUNNING)

            (run_dir / rs.SIDECAR_NAME).write_text(
                "einstein\t/home/sandy/qe_scratch/j\n", encoding="utf-8"
            )

            with patch(
                "tensorspec.web.server.jobs.best_effort_wipe_remote_scratch"
            ) as mock_wipe:
                queue.cancel(job.job_id, session_id="sess1")
                mock_wipe.assert_called_once()
                self.assertEqual(mock_wipe.call_args[0][0], run_dir)
                self.assertIn("log", mock_wipe.call_args[1])


if __name__ == "__main__":
    unittest.main()
