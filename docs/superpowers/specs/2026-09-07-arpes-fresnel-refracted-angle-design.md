# Fresnel v2 — refracted-angle local-field rebuild

**Date:** 2026-09-07  
**Branch:** `TensorSpec_GUI` (merge to `main` after smoke if requested)  
**Status:** approved / implemented  
**Depends on:** Approach C v1 (`fresnel.py` vacuum-angle + photon-q)

## Problem

Approach C vacuum-angle Fresnel scales LH `|A|` by `t_p(θ)` only.  
`Ay/Ax = tan α` unchanged → **Ez/E∥ mix** does not change with `n` or incidence the way H2 needs.  
Einstein smoke (n=2, Fresnel ON, 55 vs 20, φ=−10 LH) still kills the **wrong** Dirac arm.

## Goal

Replace vector rebuild with **refracted-angle** basis so transmitted p-pol direction follows Snell `θ_t`, changing Ez/E∥ (lab `Ay`/`Ax`) when `n ≠ 1`.

## Non-goals

- Dual-mode vacuum/refracted GUI toggle (user chose **replace**)  
- Anisotropic / 3-layer dielectric stack  
- GrizzlyME CUDA Fresnel kernel (Fresnel is O(1) on `pol` before ME; already consumed on GPU)  
- Photon-q / H0 / H1 / SPR-KKR

## Locked decisions

| Decision | Choice |
|----------|--------|
| Replace vs toggle | **Replace** vacuum-angle rebuild |
| Complex `optical_k` | **Complex Snell continuation** (same as `t_s`/`t_p`) |
| Implementation locus | Approach **1**: rewrite `apply_fresnel_to_A_lab` only |
| GUI / physics keys | Unchanged (`fresnel_enabled`, `optical_n`, `optical_k`) |

## Context (Chinook / Grizzly)

- Fresnel is **not** native in Chinook. Chinook takes a fixed vacuum `pol` 3-vector.  
- TensorSpec builds `A_lab` (+ Fresnel) → manip/hkl → `pol` for Chinook **and** GrizzlyME.  
- No GrizzlyME version bump required for this change.

## Design

### Formula

Keep existing amplitude transmission (`fresnel_ts_tp`):

```
t_s = 2 cos θ_i / (cos θ_i + n cos θ_t)
t_p = 2 cos θ_i / (n cos θ_i + cos θ_t)
sin θ_t = sin θ_i / n
cos θ_t = √(1 − sin² θ_t)   # complex continuation
```

Lab frame (unchanged vs Approach C / `compute_A_lab`):

- Incidence plane: `x`–`y`; surface normal `+y`; s / LV = `+z`.  
- Vacuum LH: `A = [cos α, −sin α, 0]` with `α = θ_i`.

**Rebuild (v2):**

1. `A_s = A_z` along `ẑ`.  
2. In-plane vacuum p magnitude `A_p = hypot(A_x, A_y)` with phase taken from vacuum p direction (LH sign: positive `A_x` → `Ê_p` as below).  
3. Refracted p unit (same convention as vacuum LH, angle `θ_t`):

   `Ê_p = [cos θ_t, −sin θ_t, 0]`

4. Output:

   `A_out = t_p · A_p · Ê_p + t_s · A_s · ẑ`

5. `n = 1` ⇒ `θ_t = θ_i`, `t_s = t_p = 1` ⇒ **exact identity** with vacuum `A`.  
6. Complex `n`: `θ_t`, `Ê_p`, `A_out` may be complex (already allowed on CR/CL / Grizzly `pol` path).

### Code touch

| File | Change |
|------|--------|
| `tensorspec/core/arpes/one_step/fresnel.py` | Docstring + `apply_fresnel_to_A_lab` rebuild |
| `tests/test_arpes_approach_c.py` | Strengthen LH Ez/E∥ asserts; keep n=1 parity |
| Remote upload | Already ships `fresnel.py` (no panel change) |

No changes to `chinook_arpes_kmesh` call sites, GUI, or GrizzlyME.

### Tests

1. `n=1`: LH/LV `A` matches `compute_A_lab` (atol 1e-12).  
2. LH `n=2`, α=55°: `|A_y|/|A_x|` ≠ `|tan 55°|` (refracted mix).  
3. LH `n=2`: Ez/E∥ proxy `|A_y|/hypot(A_x,A_y)` differs between α=55° and α=20°.  
4. LV `n=2`: direction stays along `ẑ`; only `|A|` scaled by `|t_s|`.  
5. Existing photon-q / physics-dict tests remain green.

### Validation (science)

Einstein / local: φ=−10, Full ME, LH, hydrogenic, mfp=5, kz=0.2, Fresnel ON, **n≈2**, photon-q OFF, incidence **55 vs 20**.  
Judge outer/inner arm weights vs ARPES. If still wrong arm → escalate H0/H1 / LV sweep (out of this spec).

## Risks

| Risk | Mitigation |
|------|------------|
| Wrong Ê_p sign vs beam | `n=1` identity + compare vacuum LH unit vector |
| Absorbing `k` phases surprise ME | Document; real-n smoke first |
| Old npz bit-compare breaks for `n≠1` | Expected; `n=1` still matches |

## Success

1. Unit tests green as above.  
2. User can re-run 55 vs 20 with refracted Fresnel and see **direction mix** change in metadata/`A` (and hopefully arm weights).  
3. No GrizzlyME release required.
