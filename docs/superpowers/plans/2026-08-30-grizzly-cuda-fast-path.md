# Grizzly CUDA fast path (kmesh hybrid)

**Date:** 2026-08-30  
**Status:** implemented locally; stage to remote host after in-flight 200³ job.

## Bottleneck (before)

`run_grizzly_arpes` (used by remote `layout=full`):

1. **Diag** — chinook `TB.solve_H()` (CPU, many segments on large Nk)
2. **datacube** — chinook radint / peak setup (CPU)
3. **Mk** — Grizzly `compute_all_Mk` (CUDA) ✓ already fast
4. **spectral** — chinook `exp.spectral()` (CPU) ← main post-ME bottleneck

## Changes (this branch)

| Layer | File | What |
|-------|------|------|
| GPU spectral | `GrizzlyME/grizzly/spectral.py` | `build_raw_I_from_experiment(..., device)` + `spectral_maps_from_experiment` |
| GPU diag | `chinook_arpes_kmesh.py` | `_grizzly_diagonalize_tb` when `device=cuda` |
| Wire-up | `chinook_arpes_kmesh.py` | `run_grizzly_arpes` uses Grizzly spectral on cuda/mps by default |
| Profiling | `chinook_remote_runner_template.py` | `profile_stages=True` prints setup/datacube/mk/spectral |

## A/B toggle

```python
run_grizzly_arpes(..., use_grizzly_spectral=False)  # old Chinook CPU spectral
```

## Deploy to remote host (after current job)

```bash
scp tensorspec/core/arpes/one_step/chinook_arpes_kmesh.py \
    tensorspec/core/arpes/one_step/chinook_remote_runner_template.py \
    remote:$TENSORSPEC_RUN_DIR/

# GrizzlyME spectral.py if not installed via pip:
scp GrizzlyME/grizzly/spectral.py \
    remote:$TENSORSPEC_REPO/TensorSpec_env/lib/python3.10/site-packages/grizzly/spectral.py
```

Re-run a ladder rung (e.g. 80×40×40) with/without `use_grizzly_spectral` to measure win.

## Not in this patch

- Multi-GPU θ-chunks (2× V100)
- Radint cache across θ-chunks (still per-chunk datacube)
- Full `GrizzlyExperiment` replacing custom K_BULK kmesh
