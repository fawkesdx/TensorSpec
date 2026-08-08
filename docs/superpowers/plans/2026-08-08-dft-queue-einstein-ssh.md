# DFT Queue Einstein SSH Backend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add DFT Queue backend select Local | Einstein (SSH); Einstein path prepares run dir on Mac and runs `scripts/remote_qe.sh` via the existing job queue.

**Architecture:** `QERequest.backend`; `queue_qe_run` branches; local unchanged; einstein_ssh → single subprocess argv to `remote_qe.sh`. UI dropdown + enable Queue when Einstein selected even without local pw.x.

**Tech Stack:** FastAPI, existing JobQueue, bash `remote_qe.sh`, dft_suite HTML/JS.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-08-dft-queue-einstein-ssh-design.md`
- Branch: `HTML_einstein_app`
- Reuse `scripts/remote_qe.sh` as-is (no behavior change required)
- No live SSH in CI; Mac-targeted feature
- Tests: `./TensorSpec_env/bin/python -m unittest …`

## File map

| File | Role |
|------|------|
| `schemas.py` | `QERequest.backend` |
| `dft.py` router | Branch queue commands |
| `dft_suite.html` / `dft_suite.js` | `#qe-backend`, params, Queue enable logic |
| `tests/test_qe_einstein_backend.py` | Schema + argv builder tests |

---

### Task 1: Schema + queue argv helper + tests

**Files:**
- Modify: `tensorspec/web/server/schemas.py`
- Modify: `tensorspec/web/server/routers/dft.py`
- Create: `tests/test_qe_einstein_backend.py`

**Interfaces:**
- `QERequest.backend: Literal["local", "einstein_ssh"] = "local"`
- Helper e.g. `_einstein_ssh_commands(run_dir: Path, mpi_ranks: int, host: str) -> list[list[str]]` returns `[["bash", str(script), str(run_dir), "--np", str(n), "--host", host]]`

- [ ] **Step 1: Failing tests**

```python
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
```

- [ ] **Step 2: Run — expect FAIL**

```bash
./TensorSpec_env/bin/python -m unittest tests.test_qe_einstein_backend -v
```

- [ ] **Step 3: Implement schema + helpers + `queue_qe_run` branch**

In `queue_qe_run` after `_prepare_run`:

```python
if request.backend == "einstein_ssh":
    script = _remote_qe_script_path()
    if not script.is_file():
        raise HTTPException(503, detail=f"remote_qe.sh not found at {script}")
    host = os.environ.get("TENSORSPEC_QE_SSH_HOST", "einstein").strip() or "einstein"
    np_ranks = min(request.mpi_ranks, cfg.max_mpi_ranks)
    commands = _einstein_ssh_commands(run_dir.resolve(), np_ranks, host)
else:
    try:
        cfg.require_exists()
    except FileNotFoundError as exc:
        raise HTTPException(503, detail=str(exc))
    # existing SolverPaths + build_pipeline_commands
```

Import `os` if needed. Do not call `require_exists` for einstein_ssh.

- [ ] **Step 4: Tests PASS + commit**

```bash
./TensorSpec_env/bin/python -m unittest tests.test_qe_einstein_backend -v
git commit -m "feat(dft): Queue backend einstein_ssh via remote_qe.sh"
```

---

### Task 2: DFT Suite UI

**Files:**
- Modify: `dft_suite.html`, `dft_suite.js`

- [ ] **Step 1: HTML** — before Queue button, add:

```html
<div class="form-row">
  <label for="qe-backend">Queue backend:</label>
  <select class="field" id="qe-backend">
    <option value="local">Local</option>
    <option value="einstein_ssh">Einstein (SSH)</option>
  </select>
</div>
<p class="hint">Einstein (SSH): Mac must reach host <code>einstein</code>; runs <code>scripts/remote_qe.sh</code> (minimal pull). Cancel may leave remote scratch.</p>
```

- [ ] **Step 2: JS** — `qeBackend: el("qe-backend")`; `readQeParameters` add `backend: dom.qeBackend?.value || "local"`.

In `refreshSolvers`: if Einstein selected, keep Queue enabled even when `!info.available`; if Local, keep today’s disable-when-unavailable. Listen to `qeBackend` change to re-apply enable logic (cache last solvers info).

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(dft-ui): Queue backend Local vs Einstein SSH"
```

---

### Task 3: Push

- [ ] **Step 1:** Run both new + `tests.test_qe_functional`  
- [ ] **Step 2:** `git push -u origin HEAD`  
- [ ] **Step 3:** Report Mac smoke: start local uvicorn, select Einstein, Queue (optional live)

---

## Spec coverage

| Spec | Task |
|------|------|
| backend field + queue branch | 1 |
| UI select + Queue enable | 2 |
| Push | 3 |
