# Option B1 + Chinook on Einstein — Design Spec

Date: 2026-08-08  
Status: approved for planning  
Branch: `HTML_einstein_app`  
Related: `docs/superpowers/specs/2026-08-08-remote-arpes-me-design.md`, `docs/superpowers/specs/2026-08-08-arpes-queue-einstein-ssh-design.md`, `scripts/run_arpes_me_a.py`

## Problem

Einstein `TensorSpec_env` lacks chinook. Option A remote ME works via chinook-free Simple Scalar mesh + three-step. Option B1 (Chinook one-step ME) and Slater-Koster mesh need chinook. Web Queue currently **422**s Einstein+B1 and UI hides B1.

## Goals

- Install chinook into Einstein `~/TensorSpec/TensorSpec_env` via pip.
- Unlock Option **B1** on remote CLI and web Queue Einstein backend.
- Keep Option A Simple Scalar path working without requiring chinook for A+Scalar.
- Document chinook requirement for B1 / SK in remote ARPES README.

## Non-goals

- Renaming `run_arpes_me_a.py` → `run_arpes_me.py` (Approach 1: extend in place).
- New conda env or pinned chinook version file unless already present.
- Changing `remote_arpes_me.sh` allowlist / scratch policy.
- Mac-local chinook install (Mac may already have it for Local B1).

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| Install | `pip install chinook` into Einstein `TensorSpec_env` (agent runs via SSH) |
| Surface | CLI + web Queue both unlock B1 |
| Script | Extend `run_arpes_me_a.py` (keep filename) |
| Missing chinook | Exit **6** when B1 or SK path needs it |
| Web gate | Remove Einstein+B1 HTTP 422; UI re-enable B1 on Einstein |

## Architecture

```
Einstein TensorSpec_env
  pip install chinook
        │
Mac CLI / Queue einstein_ssh
  request.json model=A|B1
        │ remote_arpes_me.sh (unchanged)
        ▼
run_arpes_me_a.py
  mesh (Scalar→numpy OK; SK→chinook)
  ARPESEngineRouter.run_simulation(model, …)
  → intensity.npz
```

---

## §1 — Einstein env

```bash
ssh einstein 'cd ~/TensorSpec && ./TensorSpec_env/bin/pip install chinook'
ssh einstein 'cd ~/TensorSpec && PYTHONPATH=. ./TensorSpec_env/bin/python -c "import chinook; print(chinook.__file__)"'
```

README note: chinook required for Option B1 and Slater-Koster mesh; Option A + Simple Scalar still works without chinook.

---

## §2 — CLI (`run_arpes_me_a.py`)

1. Accept `model` `"A"` or `"B1"`; other values → exit **2**.
2. Same mesh + caps as today; call `ARPESEngineRouter().run_simulation(model, band_data, kwargs)`.
3. `ImportError` / chinook unavailable when required → exit **6**.
4. `intensity.npz` / `meta.json` include actual `model`.
5. `remote_arpes_me.sh` unchanged (still invokes this script).

Tests: reject unknown model; B1 no longer treated as “A only” validation error; keep tiny Option A test; B1 integration may skip if chinook absent in CI Mac env.

---

## §3 — Web Queue + UI

1. Remove `_refuse_b1_on_einstein` enforcement (delete helper or make no-op; drop/replace 422 unit test).
2. Einstein worker: serialize requested `model` into `request.json` (do **not** force `"A"`).
3. `job.result["model"]` = request model.
4. UI: remove B1 disable/hide when Einstein selected; update hint (B1 needs chinook on Einstein).

---

## §4 — Success

1. Einstein `import chinook` succeeds.  
2. CLI tiny B1 job (or documented smoke) produces `intensity.npz`.  
3. Mac UI: Einstein + B1 queues without 422; Push works when remote succeeds.  
4. Local Option A/B1 and Einstein Option A unchanged.

## Out of scope (later)

- Script rename  
- Chinook version pin in repo  
- Auto-detect backend
