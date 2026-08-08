# Suite Gaps Batch 1–4 — Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or executing-plans. Checkboxes for tracking.

**Goal:** Ship DFT Wannier hr.dat load; honest ARPES slit/defl disable; crystal AA+thresh polish; DFT pts/kgrid/isoe polish. Push + Einstein deploy. No main merge.

**Order:** Phase 1 (Wannier) → 3 (crystal) → 4 (DFT polish) → 2 (ARPES disable).

**Spec decisions (approved):**
- Phase 2 = disable UI only (no new physics)
- Phase 4 = no QE XC picker (no backend field)
- Align camera skipped

## Global Constraints

- Branch `HTML_einstein_app` only
- Use `TensorSpec_env/bin/python` for tests
- After push: `ssh einstein` pull + restart uvicorn

---

### Task 1: Wannier hr.dat upload + bands flag

**Files:** schemas.py, dft.py router, api.js, dft_suite.html/js, workspace if needed, tests

- [ ] Add `use_uploaded_wannier: bool = False` to `BandRequest` (or resolve path from session)
- [ ] `POST /api/dft/{name}/wannier` multipart: save `hr.dat` (+ optional scf.out) under workspace uploads; store path on session
- [ ] `compute_bands` passes `w90_filepath` when flag/path set
- [ ] Enable Load button + file input; JS upload then Calculate uses flag
- [ ] Test: upload tiny fixture or mock path → engine gets filepath
- [ ] Commit

### Task 2: Crystal polish

- [ ] Add `Graphene (AA Bilayer)` to `#st-template`
- [ ] `#cr-thresh` min=0.5 max=3.0
- [ ] Commit

### Task 3: DFT polish

- [ ] `#tb-pts` max=500
- [ ] Disable `#tb-kgrid` and `#tb-isoe` + hint
- [ ] Commit

### Task 4: ARPES disable slit/defl

- [ ] Disable `#ar-slitsize` and `#ar-defl` + title/hint not yet
- [ ] Commit

### Task 5: Verify + push + Einstein

- [ ] Run relevant tests
- [ ] Push origin HEAD
- [ ] ssh einstein pull + restart uvicorn
