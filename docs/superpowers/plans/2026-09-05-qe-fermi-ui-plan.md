# QE Fermi UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dedicated editable QE Fermi (eV) control, auto-filled from nscf/scf/FERMI_ENERGY.txt, folded into H once with no double-subtract.

**Architecture:** Shared `detect_qe_fermi_eV(work_dir)` in core; TB panel spinbox + label; suite/pass-through uses panel value; Wannier parse folds EF; band/ARPES paths do not subtract again.

**Tech Stack:** Python, PyQt, pytest, existing `chinook_tb` / DFT suite.

## Global Constraints

- Branch: `TensorSpec_GUI` only (no merge to main)
- On-site E remains a separate knob
- After EF fold: `arpes_e_fermi_shift = 0.0`; store QE EF for display only
- Do not commit unless user asks

---

### Task 1: detect_qe_fermi_eV helper + tests

**Files:**
- Create: `tensorspec/core/dft/qe_fermi.py`
- Create: `tests/test_qe_fermi.py`

**Interfaces:**
- Produces: `detect_qe_fermi_eV(work_dir: str) -> tuple[float, str]`  
  sources: `"nscf.out" | "scf.out" | "FERMI_ENERGY.txt" | "none"`

- [ ] Implement detection order + tests; pytest green

### Task 2: TB panel QE Fermi spinbox

**Files:**
- Modify: `tensorspec/gui/components/dft_panels.py`

**Interfaces:**
- Produces: `spin_qe_fermi`, `lbl_qe_fermi_src`, `qe_fermi_eV() -> float`, autofill on `load_w90_file`

- [ ] Add spinbox (±100 eV), status label, autofill via Task 1 helper; manual edit updates label to `(manual)`

### Task 3: Wire suite + fix double-shift

**Files:**
- Modify: `tensorspec/gui/suites/dft_suite.py`
- Modify: `tensorspec/core/dft/chinook_tb.py` (optional `qe_fermi` into parse / cache key)
- Modify: `tensorspec/core/dft/tb_remote_client.py`, `tb_remote_runner.py`

**Interfaces:**
- Consumes: `tb_panel.qe_fermi_eV()`
- Parse folds UI EF into H; remote/local **do not** `eigenvalues -= qe_fermi` after fold
- Metadata: `fermi_energy`/`qe_fermi_eV` for display; `h_includes_qe_fermi_shift=True`; `arpes_e_fermi_shift=0.0`

- [ ] Replace `_fermi_energy_from_w90` call sites with panel value (keep helper as thin wrapper around detect for tests if useful)
- [ ] Pass `qe_fermi` into prepare/solve/cache key
- [ ] Remove post-diag subtract of QE EF when H already folded

### Task 4: Verify

- [ ] `pytest tests/test_qe_fermi.py -v`
- [ ] Manual check notes in plan result (load qe_workspace → 12.4202)
