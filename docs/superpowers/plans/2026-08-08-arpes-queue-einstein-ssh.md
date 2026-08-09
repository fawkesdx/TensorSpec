# ARPES Queue Einstein SSH Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Local | Einstein (SSH) backend to ARPES ME Queue so Option A can run via `remote_arpes_me.sh` and still populate `job.result` for Push/Preview.

**Architecture:** Extend `ArpesSimRequest` with `backend`. Local keeps `_build_sim_worker`. Einstein uses a callable that writes a CLI job dir, subprocesses `remote_arpes_me.sh` (with `job._process` + `start_new_session` for cancel), then loads `intensity.npz` into the same `job.result` shape as local.

**Tech Stack:** FastAPI, JobQueue `submit_callable`, bash CLI, numpy, existing ARPES suite JS/HTML.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-08-arpes-queue-einstein-ssh-design.md`
- Branch: `HTML_einstein_app` only — never merge to `main`
- Reuse `scripts/remote_arpes_me.sh` as-is
- Einstein + `model=B1` → HTTP 422; UI disable/hide B1 when Einstein selected
- Cancel wipe: existing sidecar + JobQueue helper (no new wipe path)
- Host: `TENSORSPEC_ARPES_SSH_HOST` else `TENSORSPEC_QE_SSH_HOST` else `einstein`
- No live SSH in CI
- ME panel IDs use `ar-*` prefix → control id is `#ar-backend` (not `ap-backend`)

## File map

| File | Role |
|------|------|
| `tensorspec/web/server/schemas.py` | `ArpesSimRequest.backend` |
| `tensorspec/web/server/routers/arpes.py` | Gates, helpers, Einstein worker, route branch |
| `tensorspec/web/templates/suites/arpes_suite.html` | Backend select + hint |
| `tensorspec/web/static/js/arpes_suite.js` | Payload + B1 disable when Einstein |
| `tests/test_arpes_einstein_backend.py` | Schema, 422/503, argv, mocked worker |

---

### Task 1: Schema + queue gates + helpers

**Files:**
- Modify: `tensorspec/web/server/schemas.py` (`ArpesSimRequest`)
- Modify: `tensorspec/web/server/routers/arpes.py` (helpers + `queue_simulation` gates only; worker stub OK if Task 2 fills body)
- Test: `tests/test_arpes_einstein_backend.py`

**Interfaces:**
- Produces:
  - `ArpesSimRequest.backend: Literal["local","einstein_ssh"] = "local"`
  - `_remote_arpes_me_script_path() -> Path` → `REPO_ROOT / "scripts" / "remote_arpes_me.sh"`
  - `_arpes_ssh_host() -> str`
  - `_einstein_arpes_argv(run_dir: Path, host: str) -> list[str]` → `["bash", script, str(run_dir), "--host", host]`
- Consumes: `REPO_ROOT` from `tensorspec.web.server.config`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_arpes_einstein_backend.py
"""ARPES Queue backend einstein_ssh (no live SSH)."""
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from pymatgen.core import Lattice, Structure

from tensorspec.core.workspace import WorkspaceManager
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
            # Call the gate helper or queue_simulation with a session — prefer
            # invoking the same check used by queue_simulation.
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
```

If prefer not to add `_refuse_b1_on_einstein` / `_require_remote_arpes_script` as named helpers, inline the same logic in `queue_simulation` and test via calling `queue_simulation` with a temp Session + structure (mirror `test_qe_einstein_backend.test_missing_remote_qe_script_returns_503`). Prefer **named helpers** for unit tests without full queue.

- [ ] **Step 2: Run tests — expect FAIL**

```bash
./TensorSpec_env/bin/python -m unittest tests.test_arpes_einstein_backend -v
```

- [ ] **Step 3: Implement schema + helpers + gates**

In `schemas.py` on `ArpesSimRequest` (after `model` / near top fields):

```python
backend: Literal["local", "einstein_ssh"] = "local"
```

In `arpes.py`:

```python
import os
from tensorspec.web.server.config import REPO_ROOT

def _remote_arpes_me_script_path() -> Path:
    return REPO_ROOT / "scripts" / "remote_arpes_me.sh"

def _arpes_ssh_host() -> str:
    return (
        os.environ.get("TENSORSPEC_ARPES_SSH_HOST")
        or os.environ.get("TENSORSPEC_QE_SSH_HOST")
        or "einstein"
    ).strip() or "einstein"

def _einstein_arpes_argv(run_dir: Path, host: str) -> list[str]:
    return [
        "bash",
        str(_remote_arpes_me_script_path()),
        str(run_dir),
        "--host",
        host,
    ]

def _refuse_b1_on_einstein(request: ArpesSimRequest) -> None:
    if request.backend == "einstein_ssh" and request.model != "A":
        raise HTTPException(
            status_code=422,
            detail="Einstein (SSH) supports Option A only until chinook is installed on Einstein.",
        )

def _require_remote_arpes_script() -> Path:
    script = _remote_arpes_me_script_path()
    if not script.is_file():
        raise HTTPException(status_code=503, detail=f"remote_arpes_me.sh not found at {script}")
    return script
```

In `queue_simulation`, after voxel cap checks:

```python
_refuse_b1_on_einstein(request)
if request.backend == "einstein_ssh":
    _require_remote_arpes_script()
    worker = _build_einstein_sim_worker(session.session_id, request)  # Task 2
else:
    worker = _build_sim_worker(session.session_id, request)
```

For Task 1 only: if `_build_einstein_sim_worker` not ready, define a stub that raises `NotImplementedError` **or** implement Task 2 in the same commit if splitting is awkward. Prefer Task 1 ends with gates + helpers green; stub worker is OK until Task 2.

- [ ] **Step 4: Run tests — expect PASS** (stub worker not invoked by these tests)

- [ ] **Step 5: Commit**

```bash
git add tensorspec/web/server/schemas.py tensorspec/web/server/routers/arpes.py tests/test_arpes_einstein_backend.py
git commit -m "$(cat <<'EOF'
feat: add ArpesSimRequest.backend and Einstein queue gates

Refuse B1 on einstein_ssh; require remote_arpes_me.sh before queue.
EOF
)"
```

---

### Task 2: Einstein callable worker

**Files:**
- Modify: `tensorspec/web/server/routers/arpes.py`
- Modify: `tests/test_arpes_einstein_backend.py`

**Interfaces:**
- Consumes: `_einstein_arpes_argv`, `_arpes_ssh_host`, session workspace structure, `request.model_dump()`
- Produces: `_build_einstein_sim_worker(session_id, request) -> Callable[[Job], None]`
- Worker must:
  1. Write `structure.cif` + `request.json` (`model` forced `"A"`) under `job.run_dir`
  2. `Popen(argv, stdout=PIPE, stderr=STDOUT, text=True, start_new_session=True)`; assign `job._process`; stream lines via `job.append_log`
  3. `wait()`; clear `job._process`; if cancel set → return; if rc≠0 → raise `RuntimeError`
  4. Load `intensity.npz` → set `job.result` identical keys to local worker

- [ ] **Step 1: Write failing tests**

```python
def test_write_job_dir_and_load_npz(self):
    import json
    import numpy as np
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        session = Session(
            session_id="t",
            workspace=WorkspaceManager(project_dir=Path(tmp)),
        )
        structure = Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
        session.workspace.push_crystal_structure(
            "Si", structure.lattice.matrix, structure=structure
        )
        run_dir = Path(tmp) / "arpes_jobs" / "sim"
        run_dir.mkdir(parents=True)
        req = _tiny_request(backend="einstein_ssh")

        # Fake successful remote script: write intensity.npz then exit 0
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

        from tensorspec.web.server.jobs import Job, JobStatus
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
        with patch("tensorspec.web.server.routers.arpes.subprocess.Popen", side_effect=fake_popen):
            # Ensure session is findable — patch session_store lookup used by worker
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
```

Adjust session_store import path to match how `_build_sim_worker` resolves sessions (read existing code; reuse same pattern).

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement `_build_einstein_sim_worker`**

```python
import subprocess
import json

def _request_json_for_remote(request: ArpesSimRequest) -> dict:
    data = request.model_dump()
    data["model"] = "A"
    # AxisBound objects already plain dicts via model_dump
    return data

def _build_einstein_sim_worker(session_id: str, request: ArpesSimRequest):
    def worker(job: Job) -> None:
        session = session_store._sessions.get(session_id)
        if session is None:
            raise RuntimeError("Session expired before the simulation started.")
        structure = session.workspace.pull_structure_object(request.crystal_name)
        if structure is None:
            raise ValueError(f"Crystal '{request.crystal_name}' is missing from the workspace.")

        job.run_dir.mkdir(parents=True, exist_ok=True)
        cif_path = job.run_dir / "structure.cif"
        structure.to(filename=str(cif_path))
        (job.run_dir / "request.json").write_text(
            json.dumps(_request_json_for_remote(request), indent=2),
            encoding="utf-8",
        )
        # Remove stale outputs
        for name in ("intensity.npz", "meta.json"):
            p = job.run_dir / name
            if p.exists():
                p.unlink()

        argv = _einstein_arpes_argv(job.run_dir.resolve(), _arpes_ssh_host())
        job.append_log(f"[arpes] einstein_ssh: {' '.join(argv)}")
        if job._cancel.is_set():
            return

        process = subprocess.Popen(
            argv,
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
            return
        if code != 0:
            raise RuntimeError(f"remote_arpes_me.sh exited {code}")

        npz_path = job.run_dir / "intensity.npz"
        if not npz_path.is_file():
            raise RuntimeError("intensity.npz missing after remote run")
        data = np.load(npz_path)
        cube = np.asarray(data["intensity"], dtype=float)
        job.result = {
            "store_as": request.store_as,
            "crystal_name": request.crystal_name,
            "model": "A",
            "intensity": cube,
            "axes": {
                "E": np.asarray(data["E"], dtype=float),
                "kx": np.asarray(data["kx"], dtype=float),
                "ky": np.asarray(data["ky"], dtype=float),
            },
            "shape": list(cube.shape),
        }
        job.append_log(f"[arpes] cube shape {list(cube.shape)} (E, kx, ky)")

    return worker
```

Wire `queue_simulation` to use this worker when `backend == "einstein_ssh"` (remove Task 1 stub).

Verify `session_store` import already used by `_build_sim_worker`.

- [ ] **Step 4: Run tests — PASS**

```bash
./TensorSpec_env/bin/python -m unittest tests.test_arpes_einstein_backend -v
```

- [ ] **Step 5: Commit**

```bash
git add tensorspec/web/server/routers/arpes.py tests/test_arpes_einstein_backend.py
git commit -m "$(cat <<'EOF'
feat: Einstein ARPES sim worker via remote_arpes_me.sh

Write job dir, subprocess CLI, load intensity.npz into job.result.
EOF
)"
```

---

### Task 3: UI backend select + B1 disable

**Files:**
- Modify: `tensorspec/web/templates/suites/arpes_suite.html` (fieldset “1. Simulation Engine Selector” or new row before Run)
- Modify: `tensorspec/web/static/js/arpes_suite.js` (`simPayload`, change listener)

**Interfaces:**
- `#ar-backend` select: `local` | `einstein_ssh`
- `simPayload()` includes `backend: el("ar-backend")?.value || "local"`
- On Einstein: set `ar-model` to `A`; disable B1 `<option>` (or hide); restore enable on Local

- [ ] **Step 1: HTML**

Inside engine selector fieldset (after model select):

```html
<div class="form-row">
    <label for="ar-backend">Queue backend:</label>
    <select class="field" id="ar-backend">
        <option value="local">Local</option>
        <option value="einstein_ssh">Einstein (SSH)</option>
    </select>
</div>
<p class="hint">Einstein (SSH) runs Option A on Einstein via remote_arpes_me.sh (needs working <code>ssh einstein</code> from this Mac). Option B1 on Einstein is not available yet.</p>
```

- [ ] **Step 2: JS**

```javascript
function syncArpesBackendUi() {
    const einstein = el("ar-backend")?.value === "einstein_ssh";
    const model = el("ar-model");
    if (!model) return;
    [...model.options].forEach((opt) => {
        if (opt.value === "B1") {
            opt.disabled = einstein;
            opt.hidden = einstein;
        }
    });
    if (einstein && model.value === "B1") model.value = "A";
}

// in simPayload():
backend: el("ar-backend")?.value || "local",

// on DOM ready / init:
el("ar-backend")?.addEventListener("change", syncArpesBackendUi);
syncArpesBackendUi();
```

Find existing init / `ar-run` click handler; call `syncArpesBackendUi` once at startup near other ME listeners.

- [ ] **Step 3: Manual sanity** — open suite HTML, confirm ids unique; no JS syntax error.

- [ ] **Step 4: Commit**

```bash
git add tensorspec/web/templates/suites/arpes_suite.html tensorspec/web/static/js/arpes_suite.js
git commit -m "$(cat <<'EOF'
feat: ARPES ME UI Local|Einstein backend select

Disable Option B1 when Einstein SSH backend is selected.
EOF
)"
```

---

### Task 4: Push + Einstein pull (controller)

- [ ] Push `HTML_einstein_app`
- [ ] `ssh einstein 'cd ~/TensorSpec && git pull'` (Mac uvicorn hosts Queue; Einstein needs updated `remote_arpes_me.sh` only if CLI changed — still pull for consistency)
- [ ] Optional Mac smoke: UI or API Einstein + tiny grid → Push

Do not merge to `main`.

---

## Spec coverage

| Spec item | Task |
|-----------|------|
| `backend` field | 1 |
| B1 → 422 | 1 |
| Script 503 | 1 |
| Callable + CLI + npz → result | 2 |
| Cancel via `_process` + sidecar | 2 |
| UI dropdown + B1 hide | 3 |
| Host env cascade | 1–2 |
| No live SSH CI | 1–2 |

## Self-review

- Corrected control id to `#ar-backend` (ME panel `ar-*`).
- No TBD placeholders.
- `job.result` keys match local worker / Push endpoint.
