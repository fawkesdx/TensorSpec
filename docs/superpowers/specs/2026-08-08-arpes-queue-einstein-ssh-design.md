# ARPES Queue — Einstein SSH Backend — Design Spec

Date: 2026-08-08  
Status: approved for planning  
Branch: `HTML_einstein_app`  
Related: `docs/superpowers/specs/2026-08-08-remote-arpes-me-design.md`, `scripts/remote_arpes_me.sh`, `docs/superpowers/specs/2026-08-08-dft-queue-einstein-ssh-design.md`

## Problem

ARPES Option A ME runs in-process on the Mac uvicorn host via `POST …/simulate`. Heavy grids burn Mac CPU. CLI `scripts/remote_arpes_me.sh` already runs Option A on Einstein and pulls `intensity.npz`. User wants the same backend choice in the ARPES Suite Queue UI as DFT: Local | Einstein (SSH), with Push/Preview still working from `job.result`.

## Goals

- Add ME backend select: `Local` | `Einstein (SSH)`.
- `einstein_ssh`: prepare job dir on Mac from session structure + `ArpesSimRequest`; worker runs `remote_arpes_me.sh`; load pulled `intensity.npz` into `job.result` (same contract as local).
- `local`: unchanged in-process `_build_sim_worker`.
- Einstein + Option B1 refused at queue time (HTTP 422) until chinook track; UI disables/hides B1 when Einstein selected.

## Non-goals

- Changing `remote_arpes_me.sh` behavior (reuse as-is).
- Option B1 on Einstein (separate chinook install track).
- Live SSH in CI.
- Einstein-hosted UI (SSH-to-self pointless; feature targets Mac uvicorn).
- New cancel-wipe path (rely on existing sidecar + JobQueue `best_effort_wipe_remote_scratch`).

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| UX | Dropdown Local \| Einstein (SSH); default Local |
| Architecture | Approach 1 — `submit_callable` + subprocess CLI + npz → `job.result` |
| B1 + Einstein | HTTP 422 + UI disable/hide B1 |
| Cancel wipe | Existing sidecar (script already writes `.tensorspec_remote_scratch`) |
| Host | `TENSORSPEC_ARPES_SSH_HOST` else `TENSORSPEC_QE_SSH_HOST` else `einstein` |

## Architecture

```
Browser (Mac) → POST /arpes/simulate { backend, model, … }
                    │
                    ├─ backend=local → submit_callable(_build_sim_worker)  # today
                    └─ backend=einstein_ssh
                         → model must be A (else 422)
                         → write run_dir/{structure.cif, request.json}
                         → submit_callable:
                              bash scripts/remote_arpes_me.sh <abs_run_dir> [--host …]
                              load intensity.npz → job.result (E,kx,ky cube)
                         → Push / Preview unchanged
```

---

## §1 — API / schema

### Schema

```python
# ArpesSimRequest
backend: Literal["local", "einstein_ssh"] = "local"
```

### `queue_simulation`

1. Caps as today (`MAX_SIM_VOXELS`, mesh check as today).
2. If `backend == "einstein_ssh"` and `model != "A"` → HTTP **422** with clear message (B1 not supported on Einstein yet).
3. If `backend == "einstein_ssh"`:
   - Resolve `scripts/remote_arpes_me.sh`; missing → HTTP **503**.
   - Worker = Einstein callable (see §2); do not require chinook on Mac for Option A mesh.
4. If `backend == "local"`: today’s `_build_sim_worker`.
5. `run_dir = …/arpes_jobs/<store_as>` as today; `submit_callable` unchanged entry.

---

## §2 — Einstein worker

1. Resolve structure from session workspace (`crystal_name`); write `structure.cif` (and/or `structure.json` — prefer CIF for parity with CLI).
2. Serialize request fields to `request.json` with `"model": "A"`.
3. Run:

```bash
bash <repo>/scripts/remote_arpes_me.sh <abs_run_dir> --host <host>
```

Stream stdout/stderr into job logs (same pattern as other callable workers that subprocess).

4. On exit 0: `np.load(run_dir / "intensity.npz")` → set

```python
job.result = {
    "store_as": request.store_as,
    "crystal_name": request.crystal_name,
    "model": "A",
    "intensity": cube,  # (E, kx, ky)
    "axes": {"E": …, "kx": …, "ky": …},
    "shape": list(cube.shape),
}
```

5. Nonzero exit → raise / fail job; leave script logs; scratch policy owned by CLI.
6. Cancel: JobQueue kills local process group; wipe uses sidecar if present.

---

## §3 — UI

- `#ap-backend` `<select>`: Local | Einstein (SSH).
- Hint: Einstein needs working `ssh einstein` from the Mac running uvicorn; uses `remote_arpes_me.sh`.
- When Einstein selected: disable or hide Option B1; keep/force Option A.
- Simulate payload includes `backend: dom.apBackend.value` (or equivalent).

---

## §4 — Tests / ship

### Tests

- Schema: default `local`; accepts `einstein_ssh`.
- Queue: `backend=einstein_ssh` + `model=B1` → 422.
- Unit: Einstein worker command/argv builder contains `remote_arpes_me.sh` and `--host` (mock subprocess; no live SSH).
- Optional: after mocked successful script, npz → `job.result` keys present.

### Success criteria

1. Mac: Einstein + Option A → Queue succeeds → Push loads cube into workspace.
2. Local Option A/B1 unchanged.
3. Einstein + B1 → 422; UI does not offer B1 when Einstein selected.
4. Missing script → 503.

## Out of scope (later)

- Option B1 + chinook on Einstein
- Backend auto-detect
- Changing CLI allowlist / scratch roots
