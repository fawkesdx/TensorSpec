# DFT Queue — Einstein SSH Backend — Design Spec

Date: 2026-08-08  
Status: approved for planning  
Branch: `HTML_einstein_app`  
Related: `docs/superpowers/specs/2026-08-08-remote-qe-einstein-design.md`, `scripts/remote_qe.sh`

## Problem

Mac Studio holds disk and the TensorSpec web UI for QE input generation, but heavy `pw.x` should run on Einstein. A CLI (`scripts/remote_qe.sh`) already does rsync/SSH. DFT Suite Queue only runs solvers on the machine hosting uvicorn (local today on Mac if QE present; Einstein deploy already has conda QE).

User wants Mac UI → Einstein compute via Queue, with an explicit backend choice.

## Goals

- Add Queue backend select: `Local` | `Einstein (SSH)`.
- `einstein_ssh`: prepare run dir on Mac as today; job worker invokes `scripts/remote_qe.sh` (subprocess); stream stdout into existing job logs; pullback/wipe handled by the script.
- `local`: unchanged pipeline (`pw.x` on Mac).

## Non-goals

- Changing `remote_qe.sh` behavior (reuse as-is).
- ARPES remote runner.
- Guaranteed cleanup of Einstein scratch on Cancel.
- Deploying this path on Einstein-hosted UI (there SSH-to-self is pointless; feature targets Mac uvicorn).

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| Where UI runs | Mac |
| UX | One Queue + backend dropdown |
| Remote invoke | Subprocess `scripts/remote_qe.sh` |
| Wiring | Thin: `QERequest.backend` on existing `POST …/qe/queue` |

## Architecture

```
Browser (Mac) → POST /qe/queue { backend, … }
                    │
                    ├─ backend=local → build_pipeline_commands → JobQueue (pw.x …)
                    └─ backend=einstein_ssh → _prepare_run only
                         → JobQueue one step:
                            bash scripts/remote_qe.sh <abs_run_dir> --np N --host einstein
                         → script rsync/SSH/pull/wipe
                         → outs appear in Mac run_dir; logs in JobInfo
```

---

## §1 — API + job + UI

### Schema

```python
# QERequest
backend: Literal["local", "einstein_ssh"] = "local"
```

### Router `queue_qe_run`

1. `_prepare_run(...)` always (writes `scf.in` etc. under session workspace).
2. If `backend == "local"`:
   - `cfg.require_exists()`; build multi-step pipeline as today.
3. If `backend == "einstein_ssh"`:
   - Do **not** require local `pw.x`.
   - Resolve script path: `REPO_ROOT / "scripts" / "remote_qe.sh"`; missing → HTTP 503.
   - Host: `os.environ.get("TENSORSPEC_QE_SSH_HOST", "einstein")`.
   - `np = min(request.mpi_ranks, cfg.max_mpi_ranks)` (same caps).
   - Commands: single list  
     `["bash", str(script), str(run_dir.resolve()), "--np", str(np), "--host", host]`
4. `queue.submit(...)` unchanged.

### Job worker

- Existing `Popen(command, cwd=job.run_dir)` is fine; prefer `cwd=REPO_ROOT` or keep `run_dir` since script receives absolute `run_dir`. Absolute path in argv → `cwd` irrelevant for the script’s first arg.
- Cancel = kill local bash/rsync/ssh child; Einstein scratch may remain (document).

### UI

- `#qe-backend` `<select>`: Local | Einstein (SSH).
- Hint under Queue: Einstein needs working `ssh einstein` from the Mac; uses `remote_qe.sh` (minimal pull).
- `readQeParameters()` → `backend: dom.qeBackend.value`.
- Enable Queue for Einstein backend even when `/api/dft/solvers` reports unavailable (local pw missing), **or** show solvers status but don’t disable Queue when Einstein selected — implement: if backend is Einstein, do not disable Queue solely due to local solvers flag.

---

## §2 — Errors, tests, ship

### Errors

| Case | Behavior |
|------|----------|
| Script missing | 503 |
| Local backend, no solvers | 503 (existing) |
| Script exit ≠ 0 | Job FAILED; logs contain script output |
| Cancel | Best-effort; remote scratch maybe left |

### Tests

- Schema: default `local`; accepts `einstein_ssh`.
- Unit: given mock prepare, `einstein_ssh` command list contains `remote_qe.sh`, `--np`, `--host`.
- No live SSH in CI.

### Success criteria

1. Mac: backend Einstein → Queue → job log shows remote steps; `scf.out` (and hr.dat if Wannier) land in run dir.
2. Backend Local still queues when Mac QE is installed.
3. Script missing → clear 503.

## Out of scope (later)

- Backend auto-detect
- Cancel → remote scratch cleanup
- ARPES Einstein backend
