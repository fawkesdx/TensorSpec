# Remote ARPES ME on Einstein (Mac job dir + Einstein CPU) — Design Spec

Date: 2026-08-08  
Status: approved for planning  
Branch: `HTML_einstein_app`  
Related: `docs/superpowers/specs/2026-08-08-remote-qe-einstein-design.md`, `scripts/remote_qe.sh`, ARPES `ArpesSimRequest` / Option A three-step

## Problem

ARPES matrix-element (ME) Option A rebuilds a 2D TB mesh and runs the three-step simulator in-process on the Mac uvicorn host. Heavy grids burn Mac CPU/time. Einstein already has `~/TensorSpec` + `TensorSpec_env` (import via `PYTHONPATH`). Chinook is **not** installed → Option B1 blocked. User wants a **CLI** remote runner (same compromise as remote QE): prepare inputs on Mac, run Option A on Einstein, pull a small intensity cube.

## Goals

- Bash CLI: sync a prepared job directory → Einstein scratch → run Option A → pull allowlist → wipe scratch on success.
- Python remote entrypoint that only needs TensorSpec_env + repo on Einstein (no chinook).
- Document usage, scratch policy, and B1-out-of-scope.

## Non-goals

- Option B1 / installing chinook on Einstein.
- Web ARPES Suite “Einstein (SSH)” Queue backend.
- Live SSH in CI (dry-run only).
- SSHFS / mounting `/data` (sysop).

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| Delivery | CLI only |
| Model | Option **A** only |
| Architecture | Approach A — job dir + rsync + remote Python module |
| Host | `einstein` (overrideable) |
| Scratch | Prefer `/data/sandy/arpes_me_scratch` if writable, else `$HOME/arpes_me_scratch` |
| Fail | Keep scratch; still pull log |

## Architecture

```
Mac job_dir/
  structure.cif | structure.json
  request.json     (model must be A)
        │ rsync
        ▼
Einstein $SCRATCH/<job_id>/
        │ TensorSpec_env/bin/python -m … (or scripts/run_arpes_me_a.py)
        │   TB mesh + Option A three-step
        │ → intensity.npz + meta.json + remote_arpes_me.log
        │
        ├─ success: rsync allowlist → Mac; rm -rf scratch
        └─ failure: pull log; keep scratch
```

---

## §1 — Job directory contract

### Required

| File | Content |
|------|---------|
| `request.json` | JSON object compatible with `ArpesSimRequest` fields (crystal_name ignored on remote if structure file present; **`model` must be `"A"`**) |
| Structure | Exactly one of: `structure.cif` **or** `structure.json` (pymatgen-serializable / CIF) |

### Optional Mac helper

`scripts/prepare_arpes_me_job.py` — write `job_dir` from a CIF path + JSON overrides (or dump from a template). Not required if user hand-authors files.

### `intensity.npz` (pull product)

Keys at minimum:

- `intensity` — array shape `(nE, nkx, nky)` (viewer convention)
- `E`, `kx`, `ky` — 1D axis arrays

Optional `meta.json`: shape, model `"A"`, crystal formula, git-less timestamp.

---

## §2 — `scripts/run_arpes_me_a.py` (runs on Einstein)

### CLI

```bash
python scripts/run_arpes_me_a.py <job_dir>
```

### Behavior

1. Load structure from `structure.cif` or `structure.json`.
2. Load `request.json`; if `model != "A"` → exit nonzero with message (B1 not supported).
3. Cap voxels / mesh like web (`MAX_SIM_VOXELS` / `MAX_MESH_POINTS` or same constants imported from router/shared module — prefer shared constants to avoid drift).
4. Build TB mesh via existing `band_service.calculate_2d_mesh` + chinook hoppings path used by web worker (Option A still uses DFTEngineRouter/chinook **hopping table** for mesh — verify: web worker uses `DFTEngineRouter` + chinook for mesh even for Option A).

**Important:** Web Option A worker uses `DFTEngineRouter` / chinook for **TB mesh defaults**, then `ARPESEngineRouter` Option A. If chinook missing on Einstein, mesh build may fail.

**Locked mitigation for v1:**  
- Entry script uses the same code path as `_build_sim_worker` where possible.  
- If chinook import fails: exit with clear code **6** “chinook required for TB mesh defaults; install or use Mac”.  
- **OR** provide a chinook-free mesh path for Simple Scalar isotropic (preferred if feasible in-plan).

**Reality check for this install:** Einstein lacks chinook today. Spec success requires either:

1. Document “install chinook in TensorSpec_env” as prerequisite for CLI (even for A), **or**  
2. Implement Minimal Scalar mesh without chinook for `tb_mode` Simple Scalar.

**Locked for planning:** Prefer **(2)** when `tb_mode` starts with Simple Scalar — pure numpy/pymatgen path already in band_service if available; else document chinook install as prereq and fail with exit 6. Implementer verifies `calculate_2d_mesh` dependencies during plan execution and picks (2) if one-file feasible, else (1) with README install line.

5. Run `ARPESEngineRouter().run_simulation("A", band_data, kwargs)`.
6. Write `intensity.npz` + `meta.json`; append `remote_arpes_me.log`.

Exit codes:

| Code | Meaning |
|------|---------|
| 0 | Success |
| 2 | Local validation (missing files, bad JSON) |
| 4 | Simulation failed |
| 6 | Missing dependency (chinook) when required |

---

## §3 — `scripts/remote_arpes_me.sh`

### Usage

```bash
./scripts/remote_arpes_me.sh <local_job_dir> [--host einstein] [--keep-scratch] [--dry-run]
```

### Behavior (mirror remote_qe)

1. Validate `request.json` + structure file present.
2. SSH preflight; resolve scratch root (`/data/sandy/arpes_me_scratch` else `$HOME/arpes_me_scratch`).
3. Write sidecar `.tensorspec_remote_scratch` (`host\tpath`) for future cancel wipe consistency (optional but recommended).
4. rsync job dir → remote scratch.
5. Remote: `cd scratch && PYTHONPATH=$TENSORSPEC_ROOT $PYTHON scripts/run_arpes_me_a.py .`  
   Prefer absolute paths: `TENSORSPEC_ROOT` default `$HOME/TensorSpec`, python `$TENSORSPEC_ROOT/TensorSpec_env/bin/python`.
6. Pull allowlist: `intensity.npz`, `meta.json`, `remote_arpes_me.log`.
7. Success + pull OK → wipe scratch unless `--keep-scratch`.
8. Fail → keep scratch; pull log if present.

`--dry-run`: print plan; **no network**.

---

## §4 — Docs + tests

### README

`scripts/README-remote-arpes-me.md` — prerequisites, job dir layout, examples, exit codes, B1 out of scope.

### Tests

- Unit: `run_arpes_me_a` rejects `model=B1` without network.
- Dry-run / contract: script contains allowlist + scratch names (like remote_qe sidecar test).
- Optional: tiny in-process Option A on a 1-atom cell if deps allow in CI env (skip if heavy).

### Success criteria

1. Prepared Mac job dir → Einstein Option A → local `intensity.npz` with axes.  
2. `--dry-run` zero network.  
3. `model=B1` refused cleanly (exit 2 or 6).  

## Out of scope (later)

- Web ARPES Queue `backend=einstein_ssh`
- Option B1 + chinook install track
- Cancel wipe wired to ARPES (sidecar ready if added)
