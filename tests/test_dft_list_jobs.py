"""API: GET /api/dft/jobs lists session jobs."""
from __future__ import annotations

import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient

from tensorspec.core.workspace import WorkspaceManager
from tensorspec.web.server.app import create_app
from tensorspec.web.server.jobs import JobQueue, JobStatus
from tensorspec.web.server.session import Session, current_session


class TestDftListJobs(unittest.TestCase):
    def test_list_jobs_session_scoped(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            session = Session(
                session_id="dft-list-jobs",
                workspace=WorkspaceManager(project_dir=tmp),
            )
            app = create_app()
            app.dependency_overrides[current_session] = lambda: session
            client = TestClient(app)
            queue = JobQueue(max_global_jobs=2, max_jobs_per_session=2)
            try:
                with patch(
                    "tensorspec.web.server.routers.dft.get_job_queue",
                    return_value=queue,
                ):
                    empty = client.get("/api/dft/jobs")
                    self.assertEqual(empty.status_code, 200)
                    self.assertEqual(empty.json(), [])

                    job = queue.submit(
                        session_id=session.session_id,
                        run_name="mine",
                        run_dir=run_dir,
                        commands=[["sleep", "20"]],
                    )
                    deadline = time.time() + 5
                    while time.time() < deadline:
                        if queue.get(job.job_id).status == JobStatus.RUNNING:
                            break
                        time.sleep(0.05)

                    listed = client.get("/api/dft/jobs")
                    self.assertEqual(listed.status_code, 200)
                    body = listed.json()
                    self.assertEqual(len(body), 1)
                    self.assertEqual(body[0]["job_id"], job.job_id)
                    self.assertEqual(body[0]["status"], "running")
                    queue.cancel(job.job_id, session.session_id)
            finally:
                app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()
