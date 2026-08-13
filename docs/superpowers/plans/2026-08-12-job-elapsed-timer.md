# Job Elapsed Timer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show elapsed wall time plus a labeled rough ETA on DFT Queue and ARPES Simulate after the user presses the button.

**Architecture:** Browser-only. Shared `job_timer.js` owns the 1 s clock. Heuristic + `localStorage` last-success live in the same module. DFT/ARPES suites start/stop the timer; no Python API change. Heuristic formulas are mirrored in a tiny Python test helper so pytest can lock the numbers.

**Tech Stack:** Vanilla JS (classic script, `window.JobTimer`), existing suite HTML/JS, unittest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-12-job-elapsed-timer-design.md`
- Branch: `HTML_einstein_app` only — never merge to `main`
- Copy must say `heuristic` or `last run` — never a factual remaining countdown
- Format: `mm:ss` under 1 h, else `h:mm:ss`
- Start = button press; stop = succeeded / failed / cancelled
- DFT Queue + ARPES Simulate only (not Generate/Bundle, not TB Calculate)

## File map

| File | Role |
|------|------|
| `tensorspec/web/static/js/job_timer.js` | format, heuristic, localStorage, start/stop |
| `tensorspec/core/jobs/eta_heuristic.py` | same DFT/ARPES formulas for pytest |
| `tests/test_eta_heuristic.py` | lock VTe2 vs small-mesh ordering |
| `dft_suite.html` / `dft_suite.js` | `#qe-elapsed`, queue/watch/cancel |
| `arpes_suite.html` / `arpes_suite.js` | `#ar-elapsed`, simulate/watch/cancel |

---

### Task 1: Heuristic formulas + tests

**Files:**
- Create: `tensorspec/core/jobs/__init__.py` (empty ok)
- Create: `tensorspec/core/jobs/eta_heuristic.py`
- Create: `tests/test_eta_heuristic.py`

**Interfaces:**
- Produces: `estimate_dft_seconds(backend, nbnd, kx, ky, kz, soc, ranks) -> int`
- Produces: `estimate_arpes_seconds(n_energy, n_kx, n_ky) -> int`
- Produces: `format_elapsed(seconds) -> str`
- Produces: `format_estimate(seconds) -> str`

```python
def estimate_dft_seconds(backend: str, nbnd: int, kx: int, ky: int, kz: int, soc: bool, ranks: int) -> int:
    base = 20 * 60 if backend == "einstein_ssh" else 60 * 60
    seconds = base * (nbnd / 162) * ((kx * ky * kz) / 216) * (2 if soc else 1) * (8 / max(ranks, 1))
    return int(max(120, min(12 * 3600, seconds)))

def estimate_arpes_seconds(n_energy: int, n_kx: int, n_ky: int) -> int:
    voxels = max(1, n_energy * n_kx * n_ky)
    seconds = 180 * (voxels / (48 * 64 * 64))
    return int(max(30, min(2 * 3600, seconds)))
```

- [x] **Step 1:** Write `tests/test_eta_heuristic.py`
  - VTe2-ish: einstein, nbnd=324, 6×6×6, soc=True, ranks=20 → between 120 and 12*3600; **greater than** same but soc=False and 4×4×1
  - `format_elapsed(75) == "01:15"`; `format_elapsed(3661) == "1:01:01"`
  - `format_estimate(45)` mentions seconds; `format_estimate(180)` mentions min
- [ ] **Step 2:** Implement `eta_heuristic.py` + pass
- [ ] **Step 3:** Commit `test(jobs): lock DFT/ARPES ETA heuristic`

---

### Task 2: `job_timer.js` clock + storage

**Files:**
- Create: `tensorspec/web/static/js/job_timer.js`
- Modify: `dft_suite.html` and `arpes_suite.html` — classic `<script src=".../job_timer.js?v=eta1">` **before** the suite module

**Interfaces:**
- `window.JobTimer.formatElapsed(seconds)`
- `window.JobTimer.estimateDftSeconds({backend, nbnd, kx, ky, kz, soc, ranks})`
- `window.JobTimer.estimateArpesSeconds({nEnergy, nKx, nKy})`
- `window.JobTimer.dftKey({backend, soc, nbnd, kx, ky, kz})`
- `window.JobTimer.arpesKey({model, nEnergy, nKx, nKy})`
- `window.JobTimer.lookupLast(key) -> number|null`
- `window.JobTimer.remember(key, seconds)`
- `window.JobTimer.start(el, {estimateSeconds, estimateSource})`
- `window.JobTimer.stop(el, terminalStatus)`  // `succeeded`|`failed`|`cancelled`
- `window.JobTimer.elapsedSeconds(el) -> number`  // for remember-on-success

Bins (spec §4):
- nbnd: 12 if ≤20; 100 if ≤150; 200 if ≤300; else 400
- kprod = kx*ky*kz: 8 / 64 / 216 / 512 thresholds
- voxels: 1e4 / 1e5 / 2e5 / 5e5 bins as integers

localStorage key: `tensorspec.jobTimes.v1`

Copy while running: `elapsed MM:SS · est. ~N min (heuristic|last run)`

Stop copy:
- succeeded: `finished in MM:SS`
- failed: `finished in MM:SS (failed)`
- cancelled: `cancelled at MM:SS`

Keep `hidden=false` after stop so the freeze line stays visible.

- [ ] **Step 1:** Implement module; formulas must match Python (same numbers as Task 1 cases)
- [ ] **Step 2:** Commit `feat(web): add JobTimer elapsed + localStorage ETA`

---

### Task 3: Wire DFT Queue + ARPES Simulate

**Files:**
- Modify: `tensorspec/web/templates/suites/dft_suite.html` — `#qe-elapsed` after `#qe-status`
- Modify: `tensorspec/web/static/js/dft_suite.js` — `queueRun`, `watchJob`, `cancelRun`
- Modify: `tensorspec/web/templates/suites/arpes_suite.html` — `#ar-elapsed` after `#ar-sim-status`
- Modify: `tensorspec/web/static/js/arpes_suite.js` — simulate click, `watchSimJob`, cancel
- Cache-bust script `?v=` on both suite HTML files

DFT `queueRun` after successful `qeQueue`:

```javascript
const p = readQeParameters();
const key = JobTimer.dftKey({
    backend: p.backend, soc: p.use_soc, nbnd: p.nbnd, kx: p.kx, ky: p.ky, kz: p.kz,
});
const last = JobTimer.lookupLast(key);
const estimateSeconds = last ?? JobTimer.estimateDftSeconds({
    backend: p.backend, nbnd: p.nbnd, kx: p.kx, ky: p.ky, kz: p.kz,
    soc: p.use_soc, ranks: p.mpi_ranks,
});
JobTimer.start(document.getElementById("qe-elapsed"), {
    estimateSeconds,
    estimateSource: last != null ? "last run" : "heuristic",
});
```

On watchJob terminal `succeeded`: `JobTimer.remember(key, JobTimer.elapsedSeconds(el))` then `stop`. Keep last key on a module-level `let qeEtaKey`.

ARPES: same with `simPayload()` fields `n_energy` / `n_kx` / `n_ky` / `model`.

- [ ] **Step 1:** Wire both suites
- [ ] **Step 2:** Commit `feat(web): show elapsed timer on DFT Queue and ARPES Simulate`
- [ ] **Step 3:** Push `HTML_einstein_app` (Einstein pull optional — Mac serves JS)

---

## Spec coverage

| Spec | Task |
|------|------|
| §1 UI second line | 3 |
| §2 start/stop/interval | 2–3 |
| §3 heuristics | 1–2 |
| §4 localStorage last success | 2–3 |
| §5 DFT+ARPES wire | 3 |
| No Generate/Bundle/TB | 3 (do not touch those buttons) |
