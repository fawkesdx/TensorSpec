# GrizzlyME full-GPU remote path — Implementation Plan

> **For agentic workers:** Implement task-by-task. Do not kill remote chinook jobs.

**Goal:** Single-process Grizzly CUDA full-cube path in the remote runner so GPU maps beat max-safe-CPU chinook wall time.

**Architecture:** Add `--layout {slices,full}` to `chinook_remote_runner_template.py`. `full` = one `GrizzlyExperiment` over the whole angle cube. GUI passes `--layout full` for spinless remote + CUDA. Phase 0 timing script for remote A/B after the current job finishes.

**Tech Stack:** Python 3.10+, chinook, grizzlyme, PyTorch CUDA, Paramiko GUI upload of template.

**Global Constraints:**
- Branch work stays in TensorSpec_GUI / GrizzlyME; never merge HTML TensorSpec to main for this.
- Do not disturb running remote chinook processes.
- SARPES → always `slices` + chinook.
- No private hostnames in public GrizzlyME docs.

---

### Task 1: Phase 0 timing helper

**Files:**
- Create: `tensorspec/core/arpes/one_step/time_grizzly_vs_chinook_slice.py`
- Test: run `--help`; dry logic with tiny synthetic skip if no TB

**Interfaces:**
- Consumes: `tb_data.npz`, same CLI angle/energy flags as runner
- Produces: printed wall times for one θ-slice chinook vs Grizzly

- [ ] **Step 1:** Script times one fixed θ: chinook `experiment.spectral()` vs `GrizzlyExperiment.datacube()+spectral()` with `--device`.
- [ ] **Step 2:** Print seconds + speedup; exit 0.
- [ ] **Step 3:** Commit when user asks (do not commit unprompted).

---

### Task 2: Runner `--layout full`

**Files:**
- Modify: `tensorspec/core/arpes/one_step/chinook_remote_runner_template.py`
- Test: local `--help`; optional tiny graphene smoke if env has grizzly

**Interfaces:**
- `--layout {auto,slices,full}` (default `auto`)
- `auto` → `full` if USE_GRIZZLY and device is cuda; else `slices`
- `run_full_cube_grizzly(...)` returns Ig ndarray shape `(ntheta, nphi, ne)`

- [ ] **Step 1:** Add CLI + resolve layout.
- [ ] **Step 2:** Implement full path with stage timers; save npz same schema.
- [ ] **Step 3:** Keep existing ProcessPool path for `slices`.
- [ ] **Step 4:** On CUDA OOM, print clear message suggesting smaller grid or future chunk mode; exit non-zero.

---

### Task 3: GUI wire-up

**Files:**
- Modify: `tensorspec/gui/components/arpes_panel.py` (remote chinook `run_args`)

- [ ] **Step 1:** Append `--layout full` alongside `--engine auto --device cuda` for non-SARPES remote chinook launches.
- [ ] **Step 2:** If SARPES, do not pass `--layout full` (runner forces slices via chinook anyway).

---

### Task 4: Stage on remote host (no kill)

**Files:** upload template + timing script to `chinook_gui_run/`

- [ ] **Step 1:** scp updated `chinook_remote_runner.py` + timing script (does not affect running process).
- [ ] **Step 2:** Leave active chinook job alone.
- [ ] **Step 3:** After finish: user/agent renames cube, runs timing script and/or full Grizzly job, then `compare_arpes_cubes.py`.

---

## Spec coverage

| Spec item | Task |
|-----------|------|
| `--layout full` single-process GPU | Task 2 |
| GUI default full for Grizzly CUDA | Task 3 |
| Phase 0 baseline timing | Task 1 |
| Stage without killing job | Task 4 |
| OOM message | Task 2 |
| Chunked θ (Phase 2) | Deferred — only if OOM on real maps |
