# TensorSpec GUI folder cleanup — design

**Date:** 2026-09-04  
**Branch:** `TensorSpec_GUI`  
**Decision:** Option **4** — do Phase B (GUI normalize) now; park Phase C (core engine split) as a follow-up.

## Problem

`tensorspec/` grew suite-by-suite. Live ML shell is correct (`gui/suites/ml_suite.py` + `gui/components/ml_tabs/`), but:

- Workers still live under the historical name `gui/maestroai/`
- Dead copy at repo root `/maestroai/` (gitignored)
- Loose helpers at `gui/` top level (`cluster_utils`, `compute_mode`, `nersc_auth`, `ml_session`)
- Empty stub `gui/components/crystal_tabs/`
- Engines for PEEM/ML are not yet under `core/<domain>/` (deferred)

## Goals

1. One readable rule for where new code goes.
2. No behavior change — import path renames only.
3. Small, reviewable commits; ML + DFT/compute smoke stay green.

## Non-goals (Phase B)

- Moving ML workers into `core/ml/`
- Reshaping PEEM into `core/peem/`
- Splitting the fat Crystal suite into panels
- Merging to `main`

## Target layout (Phase B end state)

```
tensorspec/
  core/                    # unchanged in Phase B
  gui/
    main_browser.py
    suites/                # thin shells
    components/            # panels (+ ml_tabs/)
    ml/                    # was maestroai/: workers, guides, viewers, model stubs
    services/              # shared GUI helpers (auth, compute mode, cluster display, …)
  plotting/                # unchanged
```

## Phase B work

| Step | Action |
|---|---|
| B0 | Inventory imports (`maestroai`, loose `gui/*.py`, `crystal_tabs`) |
| B1 | Delete root `/maestroai/`; remove `/maestroai/` from `.gitignore` |
| B2 | `git mv gui/maestroai → gui/ml`; rewrite imports `tensorspec.gui.maestroai` → `tensorspec.gui.ml` |
| B3 | Move `ml_session.py` → `gui/ml/session.py` (or keep re-export shim at old path for one release — prefer direct move + rewrite) |
| B4 | Move `nersc_auth.py`, `compute_mode.py` → `gui/services/`; move `cluster_utils.py` → `gui/services/cluster_utils.py` (callers: DFT suite, panels) |
| B5 | Delete empty `components/crystal_tabs/` if unused |
| B6 | Add short `docs/FOLDER_LAYOUT.md` describing the rule |

## Phase C (parked — do not start without a new approval)

| Step | Action |
|---|---|
| C1 | `gui/ml/` workers (training, clustering, AL, alignment, models) → `core/ml/` |
| C2 | Keep Qt-only viewers/guides under `gui/ml/` or `gui/components/ml_tabs/` |
| C3 | Flatten PEEM: `core/peem_*.py` → `core/peem/` |
| C4 | Optional: Crystal suite → `components/crystal_tabs/` |

Phase C gets its own spec + plan when ready.

## Compatibility

- Prefer hard cut on `TensorSpec_GUI` (no long-lived shims) unless a public import is documented outside the repo.
- Entry `gui/ml/main.py` (was `maestroai/main.py`) still launches `MLSuite`.

## Success criteria

- `rg maestroai` under `tensorspec/` and `tests/` is empty (except historical docs if any).
- Root `/maestroai/` gone.
- ML panel/layout/session tests pass; `launch_ml_suite` smoke OK.
- DFT suite still imports cluster helper from new path.
- `docs/FOLDER_LAYOUT.md` exists.

## Risks

- Missed import string in scripts/docs → fix with `rg`.
- `model_warehouse` package path under `gui/ml/` must move with the package.
- Compute/DFT tests must be run after B4.
