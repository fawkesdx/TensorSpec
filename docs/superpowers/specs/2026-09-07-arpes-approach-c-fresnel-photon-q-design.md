# Approach C — Fresnel local field + photon momentum (design)

**Date:** 2026-09-07  
**Branch:** `TensorSpec_GUI`  
**Status:** approved / implemented (Tasks 1–5)  
**Depends on:** Approach B shipped (hydrogenic, mfp, kz broaden, incidence→`A_lab`)

## Problem

VTe₂ magneto-ARPES Setup 2 shows selective intensity change when beam-to-normal goes ~55° → ~20° at a **fixed crystal cut** (user-confirmed: deflector unchanged; intensity-only). Chinook/Grizzly Full ME today uses the **vacuum** polarization vector inside the solid and **ignores photon momentum**. That under-models the true light–surface geometry.

Approach B (radial + kz) did not fix wrong-arm / missing-branch intensity. Approach C adds the two incidence-dependent light corrections ranked in `Matrix Element Issue/01_Why_Chinook_Fails_Magneto_ARPES.md` §H2 / §5:

1. **Surface optical field (Fresnel)** — O(1) change in E_z/E∥ vs incidence  
2. **Photon momentum q** — soft X-ray relevant; ~0.02 Å⁻¹ q∥ shift between 55° and 20° at 84 eV

## Goals

1. Apply a **Fresnel-corrected effective A** (transmitted / inside-solid field) on the one-step Chinook/Grizzly path, driven by existing `incidence_angle` + polarization.  
2. Optionally apply **full photon momentum** `q` (in-plane + out-of-plane) when mapping photoelectron k, for soft X-ray later.  
3. Keep defaults safe for current “cut locked” VTe₂ work: Fresnel useful on; photon-q **off** unless enabled.  
4. Wire end-to-end: GUI → `experiment_kwargs` / `arpes_physics.json` → `build_k_bulk_mesh` / `compute_A_*` → metadata.

## Non-goals

- Full non-dipole operator expansion `e^{iq·r}` beyond kinematic k-shift (no multipoles / retardation in M).  
- Footprint / domain averaging (H1).  
- Detection-map Lorentz / B coupling.  
- SPR-KKR final state.  
- Changing GrizzlyME radint/Ylm (engine already consumes `pol` / k from TensorSpec).  
- Replacing dielectric with a full material-specific optical constant library (v1 uses a simple, documented model).

## User decisions (locked)

| Decision | Choice |
|----------|--------|
| Cut behavior | Fixed cut; intensity from beam↔surface geometry |
| Fresnel | **In scope** (primary for 55↔20 intensity) |
| Photon momentum | **In scope**, full **q∥ + q_z** (option B) |
| Photon-q default | **Off** for locked-cut VTe₂; **on** for soft X-ray later |
| First implementation slice | Fresnel + photon-q flag in one design; ship together with defaults above |
| **`fresnel_enabled` default** | **On** (user 2026-09-07: Fresnel is the leading beam–surface field correction) |

## Design

### 1. Fresnel → local `A`

**Note:** v2 (2026-09-07): refracted-angle rebuild replaced vacuum-angle. See `2026-09-07-arpes-fresnel-refracted-angle-design.md`.

**Where:** extend `compute_A_lab` (or a thin wrapper `compute_A_lab_fresnel`) in `chinook_arpes_kmesh.py` so all callers of `build_k_bulk_mesh` get corrected `A_bulk`.

**Model (v1 — simple, explicit; superseded by v2 refracted-angle):**

- Build vacuum `A_vac` exactly as today (LH / LV / Arbitrary / CR / CL from incidence + `lin_pol_angle`).  
- Decompose into p-components (in incidence plane) and s-component (⊥).  
- Apply scalar Fresnel transmission amplitudes for an interface vacuum → solid with complex refractive index `n` (or dielectric `ε = n²`):

  - s: `t_s(θ, n)`  
  - p: `t_p(θ, n)`  

  Standard Snell + Fresnel formulas; document the exact convention (field E, not intensity; phase of `t` kept for complex `n`).

- Reassemble `A_in = t_s A_s + t_p A_p` (with p-part’s surface-normal component scaled consistently with transmitted geometry — document whether we use vacuum-angle or refracted-angle basis for the vector rebuild).  
- Pass `A_in` through existing `sample_to_bulk_frame` + hkl map unchanged.

**Dielectric input (v1):**

- Physics keys: `fresnel_enabled` (bool, default **True**), `optical_n` (complex or real; default e.g. `n = 1.0` ⇒ identity / no change until user sets material n, or a documented VTe₂ placeholder).  
- GUI: checkbox “Fresnel local field” (default checked) + optional `n` (real) / `k` (imag) spins.  
- If `fresnel_enabled` and `n ≈ 1`, behavior ≈ current vacuum A (parity test). Metadata always records the flag so old/new compares stay honest.

**Parity / tests:**

- `n = 1` → `A` matches current `compute_A_lab` within 1e-12.  
- LH at 55° vs 20° with `n > 1`: \|A_z\|/\|A_∥\| ratio changes vs vacuum (unit test with fixed n).  
- LV (pure s): only `t_s` scales magnitude; direction stays ⊥ incidence plane.

### 2. Photon momentum (full q)

**Where:** `build_k_bulk_mesh` after vacuum/photoelectron `K_LAB` (or equivalent sample-frame step), before or after refraction with `V0` — document order:

Recommended order (common SX-ARPES practice):

1. Build photoelectron `k_vac` from angles + E_kin (existing).  
2. Refract with inner potential → `k` inside (existing `k_z` sqrt).  
3. If `include_photon_momentum`: subtract photon wavevector projected into the same frame:

   `q = (hv / ħc) * â_beam` with `ħc ≈ 1973.27 eV·Å`,  
   `â_beam` from lab incidence (same convention as `compute_A_lab` beam direction: toward sample).  

   Apply **full 3-vector** `q` (q∥ in incidence plane + q_z).  
   Crystal/bulk frame: rotate `q` with the same manip + hkl maps used for `K` and `A`.

4. `K_final = K_photoelectron - q` (sign convention: crystal momentum conservation for photon in; document and fix one convention; unit-test that |q| = hv/ħc and that 55° vs 20° changes q∥ by ~0.02 Å⁻¹ at 84 eV).

**Reuse:** fold/extend `ARPESKinematics` helpers so one-step path and kinematics module share `HBAR_C` and beam-direction projection (today `include_photon_momentum` exists only on unused `calculate_kz` and defaults False).

**Flag:** `include_photon_momentum: bool` default **False**. GUI checkbox “Photon momentum (soft X-ray)”. Soft X-ray users turn on; VTe₂ locked-cut runs leave off.

**Tests:**

- Flag off → bit-identical k mesh to current (same grid).  
- Flag on, hv=84, θ_inc=55 vs 20: Δ|q∥| ≈ 0.0203 Å⁻¹ (tolerance ~1e-4).  
- Flag on shifts both in-plane and kz components (assert q_z ≠ 0 for θ_inc ≠ 0).

### 3. Physics / GUI / remote JSON

Extend `physics_from_experiment_kwargs` + ARPES panel + remote `arpes_physics.json`:

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `fresnel_enabled` | bool | `True` | Apply Fresnel to A |
| `optical_n` | float | `1.0` | Real refractive index (`n=1` ⇒ vacuum-A parity) |
| `optical_k` | float | `0.0` | Extinction (Imag n) |
| `include_photon_momentum` | bool | `False` | Subtract full photon q from k |

Metadata must record these so 55/20 maps are auditable (fix Approach B metadata gap).

**Shipped defaults (locked-cut VTe₂):** Fresnel **ON** with `optical_n=1` / `optical_k=0` (numerically old vacuum A; metadata still records `fresnel_enabled=True`); photon-q **OFF**. Set material `n` (and optional `k`) when comparing 55° vs 20° intensity. Enable photon-q later for soft X-ray.

### 4. Interaction with existing features

- **kz broadening / mfp / hydrogenic:** unchanged; apply after k mesh built (photon-q first if enabled).  
- **Bare ME Off:** Fresnel irrelevant to intensity (no A·…); photon-q still shifts k if enabled.  
- **GrizzlyME:** no package change; receives corrected `pol` / peaks.  
- **Slit / azi:** unchanged; photon q and Fresnel use **lab incidence**, then existing manip maps.

## Einstein validation recipe (science smoke — user runs)

Not a CI gate. Single-cut intensity check after deploy on Einstein (`HTML_einstein_app`):

```text
φ = −10 single cut, Full ME, LH, hydrogenic, mfp=5, kz=0.2
Fresnel ON, n=… (user sets; n=1 = vacuum parity)
Photon-q OFF
Compare incidence 55 vs 20 (same slit/azi pairing as experiment)
```

Inspect whether outer/inner branch weights move toward Setup 2 experiment. Checklist: `Matrix Element Issue/02_TensorSpec_ApproachB_Rerun_Checklist.md` (Overleaf VTe₂ project).

## Implementation status

1. Unit tests green: Fresnel n=1 identity; LH 55/20 ratio; photon-q magnitude + Δq∥; flag-off k identity.  
2. `fresnel.py` + `photon_momentum.py` beside one-step path.  
3. Wired into `build_k_bulk_mesh` + physics dict.  
4. GUI knobs + remote JSON + metadata.  
5. Docs: this spec + Matrix Element Issue checklist.

## Risks

| Risk | Mitigation |
|------|------------|
| Wrong Fresnel vector rebuild convention | n=1 parity + published Fresnel limit tests; document formula |
| Wrong photon-q sign | Document conservation; one golden number from critic (Δq∥ 55↔20) |
| Default Fresnel on vs old npz | Metadata always records flag; `n=1` recovers vacuum A; uncheck Fresnel for bit-compare to pre-C maps |
| Complex n without material data | Start real-n only; k optional; ship with `n=1` identity until user sets n |

## Open questions

None — Fresnel default **On** (2026-09-07). Photon-q default **Off**.

## Success criteria

1. Tests green for Fresnel n=1 and photon-q kinematics.  
2. User can run VTe₂ 55 vs 20 with Fresnel on, photon-q off, fixed cut, and inspect whether inner/outer intensity trend moves toward experiment.  
3. Soft X-ray later: same pipeline with photon-q on.  
4. No GrizzlyME version bump required unless pol dtype/path breaks (not expected).
