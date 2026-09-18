# HANDOFF — GPU one-step KKR ARPES (post–Chinook/Grizzly ME)

**Date:** 2026-09-08  
**For:** Fresh agent session (read this file first; scope = new package + TensorSpec_GUI glue later)  
**Repo context:** `/Users/sandyai/Documents/GitHub/TensorSpec_GUI` (branch `TensorSpec_GUI` / also on `main`)  
**Related science docs (Overleaf, optional):**  
`/Users/sandyai/Dropbox/Apps/Overleaf/VTe2 Project/Matrix Element Issue/`  
especially `00_HANDOFF_READ_ME_FIRST.md`, `01_Why_Chinook_Fails_Magneto_ARPES.md`

---

## Goal for the new session

**Build a GPU-capable one-step ARPES code in the SPR‑KKR / multiple-scattering final-state spirit** (time-reversed LEED–style final state + relativistic / SOC-capable ME), **not** another Chinook dipole plane-wave engine.

User will start by **brainstorming the package name**. Product must:

- Be **SPR‑KKR–method related** (physics lineage: KKR-GF / one-step photoemission), **without** copying Munich SPR‑KKR source or violating its license.
- **May use GrizzlyME** for GPU primitives where useful (radint/Ylm/etc.), but this is **not** “Chinook-on-GPU” — that niche is already **GrizzlyME**.
- Prefer a **separate package** from GrizzlyME (user leaning yes).

---

## Where TensorSpec / VTe₂ left off (do not reopen unless asked)

### Science problem
VTe₂ magneto-ARPES Setup‑2: tip changes beam↔surface (~55° vs ~20°) at **claimed fixed crystal cut**; experiment shows **selective branch intensity** change. Chinook/Grizzly Full ME (even with Approach B+C) still kills the **wrong** arm / fails to match.

### What was implemented in TensorSpec_GUI
| Piece | Status |
|-------|--------|
| Approach B | hydrogenic rad, mfp, kz Lorentz broaden, incidence→`A_lab` |
| Approach C | Fresnel local A + optional photon-q |
| Fresnel v2 | **refracted-angle** rebuild (`fresnel.py`); vacuum-angle superseded |
| Remote Einstein | uploads `fresnel.py` / `photon_momentum.py` with runner |
| GrizzlyME | separate PyPI package; Chinook ME on CUDA |

Key paths:

- `tensorspec/core/arpes/one_step/fresnel.py`
- `tensorspec/core/arpes/one_step/photon_momentum.py`
- `tensorspec/core/arpes/one_step/chinook_arpes_kmesh.py`
- Specs/plans under `docs/superpowers/specs/` and `docs/superpowers/plans/` (Approach C + Fresnel v2)

### User conclusions (locked for next work)
1. **Fresnel n≈2, k=0 (v2) still wrong branch** — ME dipole path exhausted for this puzzle.
2. Pol / photon-q / similar Chinook knobs — user already tried; **do not re-suggest as main fix**.
3. **H1 (footprint / domain average)** — user rejects as explanation for kill/revive of the branch.
4. **H0 (stray-B detection map)** — user **rejects** for now. Clarification they want on record:

   > H0 = Lorentz deflection of photoelectrons **between sample and analyzer** (instrumental trajectory / warped θ–Φ map), **not** intrinsic magnetic ME in the solid.  
   > User: they have a **full Fermi map with B on**; deflection exists but data are captured. They do **not** believe the missing branch is simply “fell outside Fermi-map coverage.” So H0 will **not** be the next engineering focus.

5. **Next direction = own GPU one-step KKR / SPR‑KKR–class ARPES** (clean-room / open ecosystem; collaborate or reimplement from published methods — **no Munich source redistribution**).

### Existing GPU KKR landscape (for naming + design)
- **GrizzlyME** = GPU Chinook-style ME (dipole / radial / Ylm). Different product.
- **tfQMRgpu** (Jülich, open) = GPU solver for block-sparse KKR Green’s functions (KKRnano lineage) — **not** a full SPR one-step ARPES app.
- **Munich SPR‑KKR** = license-gated; ASE2SPRKKR = Python wrapper, not a free GPU rewrite of the engine.
- Publishing path: clean-room / open KKR + own ARPES layer, or collaboration — **not** “CUDA port of SPR Fortran on GitHub.”

---

## First task for the new agent (user request)

**Brainstorm package naming** only (use brainstorming skill / design gate before code).

Constraints from user:
- Something that signals **SPR‑KKR / one-step / multiple-scattering ARPES** method family.
- Can **depend on** or **interop with** GrizzlyME, but **should be a different package** (GrizzlyME stays “Chinook on GPU”).
- GPU-first for ARPES simulation cost.

Deliverable of that brainstorm: shortlist of names + one recommended name + one-sentence positioning vs GrizzlyME / vs Munich SPR‑KKR branding risk.

**Do not** start implementing GPU KKR until name + scope design are approved.

---

## What not to do in the new session

- Do not merge TensorSpec HTML work into `main` unless user asks (HTML stays on its branch; this ME work already landed on `TensorSpec_GUI` / `main`).
- Do not propose more Fresnel / Chinook pol roulette as the primary path.
- Do not copy or translate Munich SPR‑KKR source into a public repo.
- Do not require the new agent to re-litigate H0/H1 unless user reopens them.

---

## One-line status

**Chinook/Grizzly+Fresnel closed for VTe₂ wrong-arm kill → next product: separate GPU one-step KKR ARPES package; start with naming brainstorm.**
