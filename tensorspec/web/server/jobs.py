"""In-process job queue for long Quantum ESPRESSO and Python simulations.

QE jobs run allowlisted argv lists with ``shell=False``. Callable jobs run a
server-owned Python worker (e.g. ARPES matrix-element sims). Both stream log
lines into a ring buffer for WebSocket clients and share the same per-session /
global concurrency caps. No physics lives here.
"""
from __future__ import annotations

import collections
import os
import secrets
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from tensorspec.web.server.remote_scratch import best_effort_wipe_remote_scratch


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    job_id: str
    session_id: str
    run_name: str
    run_dir: Path
    commands: list[list[str]] = field(default_factory=list)
    worker: Callable[["Job"], None] | None = field(default=None, repr=False)
    result: Any = field(default=None, repr=False)
    status: JobStatus = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    exit_code: int | None = None
    error: str | None = None
    current_step: int = 0
    total_steps: int = 0
    log_lines: collections.deque = field(default_factory=lambda: collections.deque(maxlen=5000))
    _process: subprocess.Popen | None = field(default=None, repr=False)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _subscribers: list[Callable[[str], None]] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def append_log(self, line: str) -> None:
        with self._lock:
            self.log_lines.append(line)
            subscribers = list(self._subscribers)
        for callback in subscribers:
            try:
                callback(line)
            except Exception:
                pass

    def subscribe(self, callback: Callable[[str], None]) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(callback)
            snapshot = list(self.log_lines)

        for line in snapshot:
            try:
                callback(line)
            except Exception:
                pass

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "run_name": self.run_name,
            "status": self.status.value,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "exit_code": self.exit_code,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class JobQueue:
    """Serialises long solver runs behind a small worker pool."""

    def __init__(self, max_global_jobs: int = 4, max_jobs_per_session: int = 1):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self.max_global_jobs = max_global_jobs
        self.max_jobs_per_session = max_jobs_per_session
        self._worker = threading.Thread(target=self._loop, name="qe-job-queue", daemon=True)
        self._worker.start()

    def _assert_session_capacity(self, session_id: str) -> None:
        active = [
            j for j in self._jobs.values()
            if j.session_id == session_id
            and j.status in (JobStatus.QUEUED, JobStatus.RUNNING)
        ]
        if len(active) >= self.max_jobs_per_session:
            raise RuntimeError(
                f"This session already has {len(active)} active job(s); "
                "wait for it to finish or cancel it."
            )

    def submit(
        self,
        *,
        session_id: str,
        run_name: str,
        run_dir: Path,
        commands: list[list[str]],
    ) -> Job:
        with self._condition:
            self._assert_session_capacity(session_id)
            job = Job(
                job_id=secrets.token_urlsafe(12),
                session_id=session_id,
                run_name=run_name,
                run_dir=Path(run_dir),
                commands=commands,
                total_steps=len(commands),
            )
            self._jobs[job.job_id] = job
            self._condition.notify()
            return job

    def submit_callable(
        self,
        *,
        session_id: str,
        run_name: str,
        run_dir: Path,
        worker: Callable[[Job], None],
        total_steps: int = 1,
    ) -> Job:
        """Queue a Python worker (ARPES sims, etc.) under the same caps as QE."""
        with self._condition:
            self._assert_session_capacity(session_id)
            job = Job(
                job_id=secrets.token_urlsafe(12),
                session_id=session_id,
                run_name=run_name,
                run_dir=Path(run_dir),
                commands=[],
                worker=worker,
                total_steps=max(1, int(total_steps)),
            )
            self._jobs[job.job_id] = job
            self._condition.notify()
            return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str, session_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.session_id != session_id:
                raise PermissionError("That job belongs to another session.")
            if job.status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED):
                return job
            job._cancel.set()
            process = job._process
            if job.status == JobStatus.QUEUED:
                job.status = JobStatus.CANCELLED
                job.finished_at = time.time()
                job.append_log("[queue] cancelled before start")
            elif process is not None:
                job.append_log("[queue] cancelling…")
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError, OSError):
                    try:
                        process.terminate()
                    except Exception:
                        pass
            best_effort_wipe_remote_scratch(job.run_dir, log=job.append_log)
            return job

    def _loop(self) -> None:
        while True:
            job = self._next_runnable()
            if job is None:
                with self._condition:
                    self._condition.wait(timeout=1.0)
                continue
            self._execute(job)

    def _next_runnable(self) -> Job | None:
        with self._lock:
            running = sum(1 for j in self._jobs.values() if j.status == JobStatus.RUNNING)
            if running >= self.max_global_jobs:
                return None
            for job in self._jobs.values():
                if job.status == JobStatus.QUEUED and not job._cancel.is_set():
                    job.status = JobStatus.RUNNING
                    job.started_at = time.time()
                    return job
            return None

    def _execute(self, job: Job) -> None:
        job.append_log(f"[queue] starting {job.run_name} ({job.total_steps} steps)")
        try:
            if job.worker is not None:
                self._execute_callable(job)
            else:
                self._execute_commands(job)
        except Exception as exc:
            job.status = JobStatus.FAILED
            job.error = str(exc)
            job.append_log(f"[queue] error: {exc}")
        finally:
            job.finished_at = time.time()
            job._process = None

    def _execute_callable(self, job: Job) -> None:
        if job._cancel.is_set():
            job.status = JobStatus.CANCELLED
            job.append_log("[queue] cancelled before start")
            return
        assert job.worker is not None
        job.current_step = 1
        job.worker(job)
        if job._cancel.is_set():
            job.status = JobStatus.CANCELLED
            job.append_log("[queue] cancelled")
            return
        if job.status == JobStatus.RUNNING:
            job.status = JobStatus.SUCCEEDED
            job.exit_code = 0
            job.append_log("[queue] finished successfully")

    def _execute_commands(self, job: Job) -> None:
        for index, command in enumerate(job.commands, start=1):
            if job._cancel.is_set():
                job.status = JobStatus.CANCELLED
                job.append_log("[queue] cancelled")
                break

            job.current_step = index
            rendered = " ".join(command)
            job.append_log(f"[step {index}/{job.total_steps}] {rendered}")

            # argv list only — never shell=True, never interpolate user text.
            process = subprocess.Popen(
                command,
                cwd=str(job.run_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            job._process = process

            assert process.stdout is not None
            for line in process.stdout:
                job.append_log(line.rstrip("\n"))

            code = process.wait()
            job._process = None
            if job._cancel.is_set():
                job.status = JobStatus.CANCELLED
                job.exit_code = code
                job.append_log("[queue] cancelled")
                break
            if code != 0:
                job.status = JobStatus.FAILED
                job.exit_code = code
                job.error = f"Step {index} exited with code {code}"
                job.append_log(f"[queue] failed: {job.error}")
                break
        else:
            job.status = JobStatus.SUCCEEDED
            job.exit_code = 0
            job.append_log("[queue] finished successfully")


# Lazily constructed so import does not require solvers to be present.
_job_queue: JobQueue | None = None
_job_queue_lock = threading.Lock()


def get_job_queue(max_global_jobs: int = 4, max_jobs_per_session: int = 1) -> JobQueue:
    global _job_queue
    with _job_queue_lock:
        if _job_queue is None:
            _job_queue = JobQueue(
                max_global_jobs=max_global_jobs,
                max_jobs_per_session=max_jobs_per_session,
            )
        return _job_queue
