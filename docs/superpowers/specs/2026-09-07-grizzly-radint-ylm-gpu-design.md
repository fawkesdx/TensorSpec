# GrizzlyME: Chinook CPU → GPU (radint + Ylm) — design

**Date:** 2026-09-07  
**Repos:** GrizzlyME (primary) + TensorSpec thin wire  
**Status:** Approved (user: B then A; §§1–3 approved)  
**Related:** `2026-08-28-grizzly-full-gpu-path-design.md` (cube path; parked Slater rewrite)

## Goal

Move remaining Chinook **CPU setup** work on the Full-ME path into GrizzlyME so cold and multi-GPU jobs stop burning wall time on duplicated / serial NumPy radint (and related Ylm setup), while keeping Chinook TB + ARPES_dict workflow and intensity parity.

**Pain today (VTe₂ Hybrid Full ME):** multi-GPU workers each print `ME shell: building radial integrals…` for ~10+ min on **hydrogenic** (cache miss); parent shows `0/16` θ-blocks; GPUs mostly idle. Slater disk cache does not help hydrogenic keys.

## Non-goals

- Spin / SARPES (stay Chinook)
- Removing Chinook as a dependency
- Auto-pausing SSL / cluster job scheduler
- Fresnel / Approach C
- HTML TensorSpec merge to main
- Rewriting `grid` / `fixed` / `exec` rad_type in torch (Chinook fallback)

## Current split (as-is)

| Piece | Owner | Device |
|--------|--------|--------|
| `make_radint_pointer` | Chinook `radint_lib` | CPU |
| Radint reuse | Grizzly `radint_cache` | pickle mem+disk |
| Setup `all_Y(basis)` | Chinook | CPU |
| Angle Ylm in ME eval | Grizzly `ylm_torch` | CPU/CUDA |
| H(k) + ME + spectral cube | Grizzly | CPU/CUDA |

## Approach (chosen)

**B then A**, implemented primarily in **GrizzlyME**, with TensorSpec as a thin consumer (Approach 2 from brainstorm).

---

## Phase B — Quick wins (single build + Ylm where safe)

### Behavior

1. **Single radint build** for multi-GPU Full layout: parent/rank-0 builds once (or disk HIT); workers load shared artifact — never two parallel `make_radint_pointer` calls for the same key.
2. **API:** `prepare_me_shell(exp, …)` (name may refine) returns shell pieces used today (`Bfuncs`, `radint_pointers`, Ylm/Gaunt-related arrays as needed).
3. **Setup Ylm:** prefer `ylm_torch` for pieces already equivalent to Chinook; keep Chinook `all_Y` fallback if parity gap remains.
4. **TensorSpec:** `chinook_arpes_kmesh` + multi-GPU collect path consume shared shell; log **one** HIT/MISS line per job.
5. **Cache:** keep existing `make_radint_cache_key` (includes `rad_type`, so hydrogenic ≠ Slater).

### Success (B)

- Second GPU worker does **not** log `building radial integrals…`
- Cold hydrogenic still slow **once**; warm HIT ≪ 1 min
- No intensity regression on existing Slater benchmark maps

### Out of B

- GPU/torch radint kernels (Phase A)

---

## Phase A — Torch radint (Slater → hydrogenic)

### Behavior

1. New module e.g. `grizzly/radint_torch.py`: port Chinook `make_radint_pointer` physics for **`slater`**, then **`hydrogenic`**.
2. Unsupported types (`grid`, `fixed`, `exec`) → Chinook fallback.
3. **Device:** `cpu` or `cuda` (align with experiment device). Prefer CUDA when VRAM allows; vectorized CPU torch still acceptable if faster than Chinook loops.
4. **Parity tests:** vs Chinook on small basis + dig_range; tolerances on `Bfuncs` samples / pointers. **Slater green before hydrogenic.**
5. **Cache:** same key + disk dir; serialize so HIT path unchanged for TensorSpec.
6. **Wire:** `prepare_me_shell` tries torch radint → fallback Chinook. Flag/env: `GRIZZLY_RADINT=torch|chinook|auto` (default `auto`).

### Success (A)

- Cold hydrogenic ~324-orbital VTe₂ ≪ current ~10+ min (order-of-magnitude target)
- Warm HIT remains fast
- Tiny Full-ME map: intensity vs Chinook/Grizzly-with-Chinook-radint within agreed relative L2 / max abs tol

### Out of A

- Full Chinook removal from ME path
- Spin/SARPES
- Guaranteeing radint on GPU when SSL holds all VRAM (CPU torch fallback OK)

---

## Repos, versioning, deploy

| Item | Choice |
|------|--------|
| Primary code | [GrizzlyME](https://github.com/fawkesdx/GrizzlyME) (`TensorSpec_GUI/GrizzlyME`) |
| Version | **0.1.5** after Phase B; **0.2.0** after Phase A (or 0.1.5 + A behind flag if preferred at release time) |
| TensorSpec | Call new API; stop dual-worker radint; bump/pin `grizzlyme` on Einstein |
| Tests | GrizzlyME unit (parity + cache); TensorSpec smoke (single-build log) |
| Deploy | wheel/PyPI → Einstein `pip install -U grizzlyme` in TensorSpec_env |
| Docs | GrizzlyME README note + TensorSpec ARPES checklist (hydrogenic warm cache) |

## Risks

- Hydrogenic torch port subtle vs Slater (nodeless critic motivation) — gate on tests
- Multi-process pickle of large `Bfuncs` — size/time; prefer disk cache path workers already share
- CUDA OOM during radint while SSL occupies GPUs — fall back to CPU torch / Chinook without failing the job

## Implementation order

1. Phase B in GrizzlyME + TensorSpec wire + Einstein bump  
2. Phase A Slater torch + tests  
3. Phase A hydrogenic + VTe₂ cold-build timing note  
4. Release 0.2.0 / docs

## Open at plan time (not blockers)

- Exact public function names  
- Whether 0.1.5 includes torch behind `auto` off by default  
- Numeric parity tolerances (set in plan from first failing/passing test)
