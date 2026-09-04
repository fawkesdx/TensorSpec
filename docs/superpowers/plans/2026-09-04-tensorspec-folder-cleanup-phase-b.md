# TensorSpec GUI folder cleanup (Phase B) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or subagent-driven-development) to implement this plan task-by-task.

**Goal:** Normalize `tensorspec/gui/` layout: rename `maestroai` → `ml`, delete dead root copy, park loose helpers under `gui/services/`, document the rule. No behavior change.

**Architecture:** Mechanical moves + import rewrites on branch `TensorSpec_GUI`. Phase C (`core/ml`, `core/peem/`) is explicitly out of scope — see `docs/superpowers/specs/2026-09-04-tensorspec-folder-cleanup-design.md`.

**Tech stack:** Python package layout, `git mv`, pytest offscreen Qt, existing `TensorSpec_env`.

---

## File map (end of Phase B)

| Path | Action |
|---|---|
| `maestroai/` (repo root) | Delete |
| `.gitignore` `/maestroai/` | Remove rule |
| `tensorspec/gui/maestroai/` | `git mv` → `tensorspec/gui/ml/` |
| `tensorspec/gui/ml_session.py` | Move → `tensorspec/gui/ml/session.py` |
| `tensorspec/gui/nersc_auth.py` | Move → `tensorspec/gui/services/nersc_auth.py` |
| `tensorspec/gui/compute_mode.py` | Move → `tensorspec/gui/services/compute_mode.py` |
| `tensorspec/gui/cluster_utils.py` | Move → `tensorspec/gui/services/cluster_utils.py` |
| `tensorspec/gui/components/crystal_tabs/` | Delete if unused |
| `docs/FOLDER_LAYOUT.md` | Create |
| All `from tensorspec.gui.maestroai…` / `ml_session` / `cluster_utils` / `compute_mode` / `nersc_auth` imports | Rewrite |

---

### Task 1: Delete dead root `/maestroai/`

**Files:**
- Delete: repo-root `maestroai/` (entire tree)
- Modify: `.gitignore` (drop `/maestroai/`)

- [ ] **Step 1:** Confirm nothing imports root package

```bash
rg -n "from maestroai|import maestroai" --glob '*.py' . || true
```

Expected: no hits (or only docs).

- [ ] **Step 2:** Delete and fix ignore

```bash
rm -rf maestroai
# edit .gitignore: remove the /maestroai/ line
```

- [ ] **Step 3:** Commit

```bash
git add -A maestroai .gitignore
git commit -m "chore: remove dead root maestroai copy"
```

---

### Task 2: Rename `gui/maestroai` → `gui/ml`

**Files:**
- Move: `tensorspec/gui/maestroai/` → `tensorspec/gui/ml/`
- Rewrite imports in `tensorspec/gui/components/ml_tabs/*.py`, `tensorspec/gui/suites/ml_suite.py`, any tests/docs that reference the package path

- [ ] **Step 1:** Move package

```bash
git mv tensorspec/gui/maestroai tensorspec/gui/ml
```

- [ ] **Step 2:** Bulk rewrite imports

```bash
# Prefer a small Python/sed pass over the repo for:
#   tensorspec.gui.maestroai  →  tensorspec.gui.ml
rg -n "tensorspec\.gui\.maestroai" --glob '*.py' tensorspec tests
```

Update every hit. Historical plans under `docs/superpowers/plans/` may keep old paths as history, or get a one-line note — do not break executable code.

- [ ] **Step 3:** Verify

```bash
PYTHONPATH="$PWD" TensorSpec_env/bin/python -c "from tensorspec.gui.ml.maestroai_training_ssl import TrainWorker; print('ok')"
PYTHONPATH="$PWD" TensorSpec_env/bin/python -m pytest tests/test_ssl_panel.py tests/test_ml_suite_layout.py -q
```

Expected: pass.

- [ ] **Step 4:** Commit

```bash
git commit -m "refactor(gui): rename maestroai package to ml"
```

---

### Task 3: Move `MLSession` under `gui/ml/`

**Files:**
- Move: `tensorspec/gui/ml_session.py` → `tensorspec/gui/ml/session.py`
- Rewrite: all `from tensorspec.gui.ml_session import MLSession` → `from tensorspec.gui.ml.session import MLSession`

- [ ] **Step 1:** `git mv` + rewrite imports (`rg ml_session`)

- [ ] **Step 2:** Test

```bash
PYTHONPATH="$PWD" TensorSpec_env/bin/python -m pytest tests/test_ml_session.py tests/test_ml_panels.py -q
```

- [ ] **Step 3:** Commit

```bash
git commit -m "refactor(gui): move MLSession into gui.ml.session"
```

---

### Task 4: Park loose helpers in `gui/services/`

**Files:**
- Move: `nersc_auth.py`, `compute_mode.py`, `cluster_utils.py` → `tensorspec/gui/services/`
- Callers to update (non-exhaustive — re-`rg` after move):
  - `components/compute_panel.py`, `qe_generator_panel.py` → `nersc_auth`
  - `components/arpes_panel.py`, `dft_panels.py`, `sprkkr_panels.py`, `suites/dft_suite.py`, `cluster_utils`/`compute_mode` mutual imports → new paths
  - `tests/test_compute_mode.py`

- [ ] **Step 1:** Move the three modules into `services/` (ensure `services/__init__.py` exists)

- [ ] **Step 2:** Fix circular imports carefully: today `cluster_utils` ↔ `compute_mode` import each other lazily in functions — preserve that pattern after the move.

- [ ] **Step 3:** Test

```bash
PYTHONPATH="$PWD" TensorSpec_env/bin/python -m pytest tests/test_compute_mode.py tests/test_ml_suite_layout.py -q
QT_QPA_PLATFORM=offscreen PYTHONPATH="$PWD" TensorSpec_env/bin/python -c "
from tensorspec.gui.suites.dft_suite import DFTSuite
from tensorspec.gui.suites.ml_suite import MLSuite
print('suite imports ok')
"
```

- [ ] **Step 4:** Commit

```bash
git commit -m "refactor(gui): move cluster/compute/nersc helpers into gui.services"
```

---

### Task 5: Remove empty `crystal_tabs` if unused

**Files:**
- Delete: `tensorspec/gui/components/crystal_tabs/` only if `rg crystal_tabs` shows no real imports (ignore stale comments)

- [ ] **Step 1:**

```bash
rg -n "crystal_tabs" --glob '*.py' tensorspec tests
```

If only comments (e.g. in `crystal_panel.py`), delete the empty package and optionally fix the stale comment path.

- [ ] **Step 2:** Commit if deleted

```bash
git commit -m "chore(gui): remove unused empty crystal_tabs package"
```

---

### Task 6: Document layout rule

**Files:**
- Create: `docs/FOLDER_LAYOUT.md`

Content (short):

```markdown
# TensorSpec folder layout

- `core/<domain>/` — engines, IO, math (no Qt)
- `gui/suites/` — thin suite shells
- `gui/components/` — panels; multi-tab domains may use `<domain>_tabs/`
- `gui/ml/` — ML workers, guides, viewers (Qt-adjacent)
- `gui/services/` — shared GUI helpers (auth, compute mode, …)
- `gui/main_browser.py` — app launcher

Parked: move ML workers to `core/ml/` and PEEM to `core/peem/` (Phase C).
```

- [ ] **Step 1:** Write file
- [ ] **Step 2:** Commit

```bash
git commit -m "docs: add FOLDER_LAYOUT guide"
```

---

### Task 7: Final verification + push

- [ ] **Step 1:**

```bash
rg -n "tensorspec\.gui\.maestroai|gui/maestroai|from tensorspec\.gui\.ml_session|from tensorspec\.gui\.cluster_utils|from tensorspec\.gui\.compute_mode|from tensorspec\.gui\.nersc_auth" --glob '*.py' tensorspec tests
```

Expected: no hits.

- [ ] **Step 2:**

```bash
QT_QPA_PLATFORM=offscreen MPLCONFIGDIR=/tmp/mplcache PYTHONPATH="$PWD" \
  TensorSpec_env/bin/python -m pytest \
  tests/test_ml_session.py tests/test_ml_panels.py tests/test_ml_suite_layout.py \
  tests/test_ssl_panel.py tests/test_cluster_panel.py tests/test_supervised_panel.py \
  tests/test_data_browser_panel.py tests/test_simulate_al_panel.py tests/test_alignment_panel.py \
  tests/test_compute_mode.py -q
```

- [ ] **Step 3:** Smoke `launch_ml_suite` reuse (same as prior ML plan).

- [ ] **Step 4:** `git push origin TensorSpec_GUI`

---

## Phase C reminder (do not execute in this plan)

See design doc section “Phase C (parked)”. Requires a new user approval before any `core/ml` or `core/peem` moves.
