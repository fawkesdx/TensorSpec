# GrizzlyME full-GPU remote path — design

**Date:** 2026-08-28  
**Repos:** TensorSpec_GUI remote ARPES runner + GrizzlyME  
**Status:** Approved (user: agree / go)

## Goal

Wall-clock ARPES intensity cubes on a remote GPU host **faster than chinook at max safe CPU workers** for the same TB + grid, by feeding the GPU a single (or chunked-in-one-process) Grizzly pipeline — not N process-pool θ-slices fighting CUDA.

Success bar: GPU path wall ≪ chinook-on-max-safe-CPUs; `nvidia-smi` shows sustained util during Mk/diag/spectral; cube parity via `compare_arpes_cubes.py` (rel L2).

## Non-goals (this iteration)

- 40 CPUs + GPU doing the same ME work
- SARPES / spin (stay on chinook θ-slice path)
- HTML TensorSpec / merge to main
- Vectorized Slater rewrite (parked until Phase 0 says so)
- Killing the in-flight chinook job

## Current failure mode

`chinook_remote_runner` submits one ProcessPool job per θ. CUDA mode caps workers at 2 → underfills GPU and loses CPU parallelism. Each slice re-pays setup. ME kernel is fast; wall time is not.

## Design

### Layout modes

| `--layout` | When | Behavior |
|------------|------|----------|
| `slices` | chinook, SARPES, or explicit | Existing ProcessPool over θ (CPU ME or per-slice Grizzly) |
| `full` | default when Grizzly + CUDA + spinless | **One** process: one `GrizzlyExperiment` for full `(Tx,Ty,E)` cube → `datacube()` → `spectral()` → npz |

### Full path (Phase 1)

1. Reconstruct TB (unchanged).
2. Build one `ARPES_dict` with `cube.Tx/Ty/E = (min, max, n)`.
3. `GrizzlyExperiment(TB, dict, device=cuda|cpu)`.
4. Print stage timers: radint/datacube, Mk, spectral, save.
5. Write `Ig` as `cube` in npz (same keys as today: cube, energy, theta, phi).

No ProcessPool in `full` mode.

### OOM (Phase 2, if needed)

If CUDA OOM on large maps: chunk along θ **inside the same process** (sequential GPU batches, assemble cube). Still one radint warm-up. Not multi-process CUDA.

### Baseline (Phase 0)

After current chinook job finishes:

1. Rename cube → `*_chinook.npz`.
2. Timed microbench: 1 θ-slice chinook vs Grizzly CUDA on same TB (script).
3. Optional: small full-cube Grizzly vs chinook wall + compare.

### GUI

Remote chinook launch appends `--layout full` when engine auto/grizzly and not SARPES (with `--device cuda`). SARPES / forced chinook keep `slices`.

### Prove

Same grid A/B → compare script; report wall times and GPU util notes.

## Out of scope details later

Hybrid CPU-prep + GPU-consumer queue only if Phase 0–1 profiling shows setup dominates after full layout.
