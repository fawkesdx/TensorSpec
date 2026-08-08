# Einstein Queue null guard + cancel wipe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Null-safe Queue enable UI; Einstein Cancel kills process group and best-effort wipes remote scratch via sidecar.

**Architecture:** `remote_qe.sh` writes `.tensorspec_remote_scratch` (`host\tpath`); `JobQueue` uses `start_new_session` + `killpg` on cancel; shared wipe helper parses sidecar and runs allowlisted `ssh … rm -rf`; UI guards null solvers info.

**Tech Stack:** bash, FastAPI JobQueue, dft_suite.js, unittest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-08-einstein-queue-cancel-wipe-design.md`
- Branch: `HTML_einstein_app`
- Sidecar: `$run_dir/.tensorspec_remote_scratch` = `host<TAB>abs_scratch_path`
- Dry-run: **no** sidecar write
- Wipe: best-effort; Cancel API never fails due to wipe
- No live SSH in CI
- Tests: `./TensorSpec_env/bin/python -m unittest …`

## File map

| File | Role |
|------|------|
| `scripts/remote_qe.sh` | Write sidecar after SCRATCH known |
| `tensorspec/web/server/remote_scratch.py` | Parse sidecar + wipe argv + best-effort wipe |
| `tensorspec/web/server/jobs.py` | Process group + cancel wipe hook |
| `tensorspec/web/static/js/dft_suite.js` | `applyQueueEnable` null guard |
| `tensorspec/web/templates/dft_suite.html` (or wherever hint lives) | Hint text |
| `tests/test_remote_scratch_wipe.py` | Parse + wipe argv + cancel mock |

---

### Task 1: Sidecar parse + wipe helper + tests

**Files:**
- Create: `tensorspec/web/server/remote_scratch.py`
- Create: `tests/test_remote_scratch_wipe.py`

**Interfaces:**
- `SIDECAR_NAME = ".tensorspec_remote_scratch"`
- `parse_remote_scratch_sidecar(text: str) -> tuple[str, str] | None`
- `wipe_remote_scratch_argv(host: str, path: str) -> list[str]`
- `best_effort_wipe_remote_scratch(run_dir: Path, *, log: Callable[[str], None] | None = None, runner: Callable[[list[str]], int] | None = None) -> bool`  
  Returns True if wipe command ran and exit 0; False if no sidecar / bad parse / non-zero. Never raises.

- [ ] **Step 1: Failing tests**

```python
"""Remote scratch sidecar parse + wipe argv (no live SSH)."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

from tensorspec.web.server import remote_scratch as rs


class TestRemoteScratchWipe(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
```

Fix the duplicate/broken `test_best_effort_missing_sidecar` to a single clean version (no placeholder comment).

- [ ] **Step 2: Run — expect FAIL**

```bash
./TensorSpec_env/bin/python -m unittest tests.test_remote_scratch_wipe -v
```

- [ ] **Step 3: Implement `remote_scratch.py`**

```python
SIDECAR_NAME = ".tensorspec_remote_scratch"

def parse_remote_scratch_sidecar(text: str) -> tuple[str, str] | None:
    line = (text or "").strip().splitlines()[0] if text else ""
    if "\t" not in line:
        return None
    host, path = line.split("\t", 1)
    host, path = host.strip(), path.strip()
    if not host or not path.startswith("/") or ".." in path:
        return None
    return host, path

def wipe_remote_scratch_argv(host: str, path: str) -> list[str]:
    return [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        host, "--", "rm", "-rf", "--", path,
    ]

def best_effort_wipe_remote_scratch(run_dir, *, log=None, runner=None) -> bool:
    # read sidecar; parse; log; runner or subprocess.run(argv, …); never raise
```

Default `runner`: `subprocess.run(argv, check=False, capture_output=True, timeout=60).returncode`.

- [ ] **Step 4: Tests PASS + commit**

```bash
./TensorSpec_env/bin/python -m unittest tests.test_remote_scratch_wipe -v
git add tensorspec/web/server/remote_scratch.py tests/test_remote_scratch_wipe.py
git commit -m "feat(dft): remote scratch sidecar parse and wipe helper"
```

---

### Task 2: JobQueue process group + cancel wipe

**Files:**
- Modify: `tensorspec/web/server/jobs.py`
- Modify: `tests/test_remote_scratch_wipe.py` (add cancel integration test) **or** `tests/test_job_queue_cancel.py`

**Interfaces:**
- Consumes: `best_effort_wipe_remote_scratch(run_dir, log=job.append_log)`
- `Popen(..., start_new_session=True)`
- Cancel: `os.killpg(process.pid, signal.SIGTERM)` then wipe

- [ ] **Step 1: Failing test** — construct a Job with temp `run_dir` + sidecar; mock wipe; call `queue.cancel`; assert wipe invoked (inject wipe via optional hook on JobQueue or patch `best_effort_wipe_remote_scratch`).

Preferred: patch `tensorspec.web.server.jobs.best_effort_wipe_remote_scratch` (import it in jobs.py).

```python
def test_cancel_invokes_wipe_when_sidecar_present(self):
    # submit a long-sleep command job OR manually set job running with sidecar
    # cancel -> mock wipe called once with run_dir
```

Keep test deterministic: create `Job` via `submit` with `commands=[["sleep", "30"]]`, wait until RUNNING, write sidecar, cancel, assert patch called.

- [ ] **Step 2: Implement jobs.py changes**

Imports: `os`, `signal`; `from tensorspec.web.server.remote_scratch import best_effort_wipe_remote_scratch`

In `_execute_commands` Popen: add `start_new_session=True`.

In `cancel`, when process is not None:

```python
try:
    os.killpg(process.pid, signal.SIGTERM)
except (ProcessLookupError, PermissionError, OSError):
    try:
        process.terminate()
    except Exception:
        pass
# after kill attempt (and for QUEUED→CANCELLED with no process, still try wipe):
best_effort_wipe_remote_scratch(job.run_dir, log=job.append_log)
```

Always attempt wipe on cancel (sidecar missing → no-op). Safe for local jobs.

- [ ] **Step 3: Tests PASS + commit**

```bash
./TensorSpec_env/bin/python -m unittest tests.test_remote_scratch_wipe tests.test_job_queue_cancel -v
# (use whichever test module you created)
git commit -m "feat(jobs): process-group cancel and remote scratch wipe"
```

---

### Task 3: `remote_qe.sh` writes sidecar

**Files:**
- Modify: `scripts/remote_qe.sh`
- Modify: `scripts/README-remote-qe.md` (one line on sidecar)
- Optional test: assert script contains write of `.tensorspec_remote_scratch` after SCRATCH set (grep-style unittest)

- [ ] **Step 1:** After `SCRATCH=...` and `log "scratch=$SCRATCH"`, **before** preflight disk block ends / before rsync:

```bash
printf '%s\t%s\n' "$HOST" "$SCRATCH" >"$RUN_DIR/.tensorspec_remote_scratch"
log "sidecar: $RUN_DIR/.tensorspec_remote_scratch -> $HOST $SCRATCH"
```

Only on live path (not dry-run).

- [ ] **Step 2:** Unit test that script source contains the printf line (contract test) **or** run a tiny excerpt — prefer:

```python
def test_remote_qe_script_writes_sidecar_contract(self):
    text = Path("scripts/remote_qe.sh").read_text()
    self.assertIn(".tensorspec_remote_scratch", text)
    self.assertIn('printf', text)
```

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(scripts): write remote scratch sidecar for cancel wipe"
```

---

### Task 4: UI null guard + hint

**Files:**
- Modify: `tensorspec/web/static/js/dft_suite.js` (`applyQueueEnable`)
- Modify: template/HTML with Queue Einstein hint (search `qe-backend` / remote_qe hint)

- [ ] **Step 1:** Prepend null branch exactly as spec §1.
- [ ] **Step 2:** Hint: mention Cancel best-effort wipe.
- [ ] **Step 3: Commit**

```bash
git commit -m "fix(dft-ui): null-safe Queue enable; cancel wipe hint"
```

---

### Task 5: Push + smoke note

- [ ] **Step 1:**

```bash
./TensorSpec_env/bin/python -m unittest tests.test_remote_scratch_wipe tests.test_qe_einstein_backend -v
git push -u origin HEAD
```

- [ ] **Step 2:** Report optional live smoke: Queue Einstein → Cancel mid-run → confirm scratch gone on Einstein.

---

## Spec coverage

| Spec | Task |
|------|------|
| Null guard + hint | 4 |
| Sidecar write (live only) | 3 |
| Parse + wipe helper | 1 |
| Process group + cancel wipe | 2 |
| Tests / push | 1–2, 5 |
