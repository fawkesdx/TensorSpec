# Einstein Queue polish — null guard + cancel wipe — Design Spec

Date: 2026-08-08  
Status: approved for planning  
Branch: `HTML_einstein_app`  
Related: `docs/superpowers/specs/2026-08-08-dft-queue-einstein-ssh-design.md`, `scripts/remote_qe.sh`

## Problem

1. `applyQueueEnable(lastSolversInfo)` in `queueRun` finally can throw if solvers never loaded (`info` null).
2. Cancel on `einstein_ssh` only terminates the local subprocess; Einstein scratch often remains (prior non-goal; now in scope).

## Goals

- Null-safe Queue enable logic (Einstein still usable before/without solvers info).
- Best-effort remote scratch wipe on Cancel using Approach **A**: sidecar path file + process-group kill + `ssh rm -rf`.

## Non-goals

- Guaranteed wipe if SSH is already dead.
- Changing success-path wipe / allowlist / `--keep-scratch` semantics.
- Track C/D/E or remote ARPES.
- Live SSH in CI.

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| Null guard | Einstein → enable; Local → disable when `!info` |
| Scratch identity | Sidecar `$run_dir/.tensorspec_remote_scratch` |
| Sidecar format | One line: `host<TAB>abs_scratch_path` |
| Cancel kill | Process group (`start_new_session` + `killpg` SIGTERM) |
| Wipe | Best-effort after kill; log failures; Cancel API still succeeds |
| Approach | A (sidecar), not B (--job-id) or C (trap-only) |

---

## §1 — UI null guard

### `applyQueueEnable(info)` in `dft_suite.js`

```javascript
function applyQueueEnable(info) {
    const einstein = dom.qeBackend?.value === "einstein_ssh";
    if (!info) {
        dom.qeQueue.disabled = !einstein;
        if (einstein) {
            setQeStatus("Einstein (SSH) queue enabled (solvers status pending)");
        }
        return;
    }
    // existing available / einstein / unavailable branches unchanged
}
```

Update Queue hint text: Cancel kills local SSH worker and **best-effort** wipes remote scratch.

---

## §2 — Sidecar + cancel wipe

### `remote_qe.sh`

After absolute `SCRATCH` is known (live path, post scratch-root resolve):

1. Write `$RUN_DIR/.tensorspec_remote_scratch` with contents:  
   `"$HOST\t$SCRATCH"` (no trailing junk).
2. Overwrite each run start.
3. Do not delete the sidecar on success wipe (stale path OK; cancel wipe becomes no-op if remote already gone).
4. Dry-run: may skip write (no live scratch) or write a dry marker — prefer **no write** on dry-run so CI/local dry-run stays network-free and side-effect free.

### Job process group

In `JobQueue._execute_commands`, start each step with:

```python
subprocess.Popen(..., start_new_session=True)
```

On cancel when `process` is set:

```python
os.killpg(process.pid, signal.SIGTERM)
```

Fall back to `process.terminate()` if killpg fails (e.g. process already exited).

Applies to all command jobs (local QE + Einstein); local multi-step pipelines benefit from cleaner child cleanup too.

### Cancel wipe hook

After signaling cancel / killpg:

1. Read `job.run_dir / ".tensorspec_remote_scratch"` if present.
2. Parse `host`, `path` (tab-separated; reject empty / path not absolute / path containing `..`).
3. Best-effort: `ssh -o BatchMode=yes -o ConnectTimeout=15 host -- rm -rf -- path`  
   (argv list only; never shell interpolate).
4. Append log lines: wipe attempted / success / failure. Do not raise out of `cancel()`.
5. Missing sidecar → no wipe attempt (queued-before-start or local backend).

Optional thin helper e.g. `_best_effort_wipe_remote_scratch(run_dir) -> None` in `dft` router or `jobs` module — keep allowlisted and testable.

### UI

No Cancel API change. Hint only.

---

## §3 — Tests + success

### Tests (no live SSH)

- Sidecar format helper / parse+reject `..` and relative paths.
- Wipe helper builds expected `ssh … rm -rf --` argv (mock subprocess).
- Cancel with sidecar present invokes wipe helper once (mock).
- `remote_qe.sh` live path writes sidecar — cover via small bash unit or Python invoking dry-run **skip** + documented live smoke; prefer extracting write into a bash function tested by sourcing, or assert in a scripted fragment. Minimum: Python tests for parse/wipe argv; script write verified by reading script contract in review + optional smoke.

JS null guard: manual / no dedicated harness required if none exists.

### Success criteria

1. `queueRun` finally with `lastSolversInfo === null` does not throw; Einstein Queue stays enabled.
2. Mid-run Cancel on Einstein job: local process group dies; when SSH works, remote scratch dir removed (or already absent); job status cancelled.
3. Local backend Cancel unchanged aside from process-group kill.
4. CI: no live SSH.

## Out of scope (later)

- SIGKILL escalation after SIGTERM grace period
- Wipe of orphaned scratches without sidecar
- Track C / Remote ARPES
