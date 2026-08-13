# Suite job elapsed timer + rough ETA — Design Spec

Date: 2026-08-12  
Status: approved for planning (user: yes; predict-first; heuristic + last-run)  
Branch: `HTML_einstein_app`

## Problem

Long jobs (DFT Queue on Einstein, ARPES Simulate) show log spam and status words but **no clock**. User cannot tell how long the current run has been going, or even a rough ballpark for “should I wait.” QE/ARPES wall times vary a lot; a fake precise countdown would lie.

## Goals

- Start an **elapsed** timer when the user presses the long-job button.
- Show a **rough estimate** labeled as **heuristic** or **last run**.
- Cover DFT Queue and ARPES Simulate in this pass.
- Stop the clock on succeeded / failed / cancelled.

## Non-goals

- Server-side ETA API or cross-device history.
- Percent-complete from QE SCF iterations.
- Killing leftover remote jobs.
- Generate Input / Download Bundle timers (short).
- TB Calculate Band Structure (can add later if it stays slow).

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| Approach | **A** — browser-only clock + heuristic + `localStorage` last success |
| Predict | Both: heuristic until a matching last job exists, then prefer last run |
| Format | `mm:ss` under 1 h, then `h:mm:ss` |
| Copy | Always say `heuristic` or `last run` — never “remaining 3:12” as fact |
| Start | Button press (Queue / Simulate), not first WS log line |

## §1 — UI

DFT Suite, near `#qe-status`:

```html
<p class="status-line" id="qe-elapsed" hidden>elapsed 00:00 · est. ~20 min (heuristic)</p>
```

ARPES Simulate, near `#ar-sim-status`:

```html
<p class="status-line" id="ar-elapsed" hidden>elapsed 00:00 · est. ~3 min (heuristic)</p>
```

Update every 1 s. Hide (or freeze + keep last line) when job terminals.

Status line can still show `run_01: running`. Elapsed is a **second** line so logs/status do not overwrite the clock.

## §2 — Clock helper

Shared tiny module e.g. `tensorspec/web/static/js/job_timer.js` (classic or ES — match suite load style):

- `startJobTimer(el, { estimateSeconds, estimateSource })`
- `stopJobTimer(el, terminalStatus)` — freeze text, e.g. `finished in 12:34 (failed)` / `cancelled at 03:02`
- Internal `setInterval`; one timer per element; clear previous if user re-queues

Elapsed origin = `Date.now()` at start.

## §3 — Heuristic (order of magnitude)

**DFT Queue**

```text
base = backend==einstein_ssh ? 20*60 : 60*60   # seconds, VTe2-ish reference
seconds = base * (nbnd/162) * (kx*ky*kz / 216) * (soc ? 2 : 1) * (8 / max(ranks,1))
clamp 2 min … 12 h
```

**ARPES Simulate**

```text
voxels = nE * nkx * nky   # from current sim payload
seconds = 180 * (voxels / (48*64*64))
clamp 30 s … 2 h
```

Show rounded minutes if ≥ 90 s, else seconds.

## §4 — Last-run memory

`localStorage` key `tensorspec.jobTimes.v1` JSON map:

```text
dft:qe:{backend}:{soc}:{nbndBin}:{kprodBin} -> seconds
arpes:sim:{model}:{voxelBin} -> seconds
```

Bins keep “similar” coarse (e.g. nbnd 12 / 100 / 200 / 400; kprod 8 / 64 / 216 / 512).

On **succeeded** only: write wall seconds. Failed/cancelled do not overwrite.

Next start: if key exists → `est. ~N min (last run)`; else heuristic.

## §5 — Wire-up

- DFT: `queueRun` → start; `watchJob` terminal statuses → stop; cancel → stop.
- ARPES: Simulate click → start; `watchSimJob` terminal → stop; cancel → stop.

No Python/schema change required.

## Test

- Manual: Queue DFT (or dry cancel) → elapsed ticks; cancel freezes.
- Unit (optional small JS or Python port of heuristic): VTe2 SOC 6×6×6 20 ranks Einstein → finite minutes, larger than non-SOC 4×4×1.

## Ship

`HTML_einstein_app` only; push; Einstein pull not required for Mac Queue (static JS). Never merge to main from agent.
