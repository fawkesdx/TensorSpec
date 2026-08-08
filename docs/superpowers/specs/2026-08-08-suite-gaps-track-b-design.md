# Suite Gaps Track B — Design Spec

Date: 2026-08-08  
Status: approved for planning  
Branch: `HTML_einstein_app` (local → push → Einstein; no merge to `main`)  
Prior: Track A `docs/superpowers/specs/2026-08-08-suite-gaps-track-a-design.md`

## Problem

Parked physics gaps after Track A:

1. QE XC functional picker missing (roadmap claimed done; HTML has no `input_dft`).
2. ARPES `#ar-slitsize` / `#ar-defl` are dead UI; real meaning is analyzer slit opening (→ ΔE with pass energy) and deflector angle (→ ky).
3. DFT `#tb-kgrid` / `#tb-isoe` disabled; core already has `calculate_2d_mesh` for ARPES but DFT suite is 1D-only.

## Goals

- **B1:** QE `input_dft` from UI (PBE / LDA / PBEsol); emit in scf/nscf; no pseudo filter.
- **B2:** Wire deflector (° → ky shift); slit size + pass energy → analyzer ΔE; combine with beam/extra in quadrature; auto `res_E` with manual override.
- **B3:** 2D isoenergy cut on kx–ky via existing mesh solver + Gaussian spectral density heatmap.

## Non-goals

- HSE / hybrids; pseudo filename filtering by XC.
- Scienta discrete slit-plate catalog (use continuous mm + formula).
- Angular acceptance blur on kx from slit opening.
- Full 2D band-index mesh UI (isoenergy only).
- Tracks C–E (polyhedra, shells, ARPES history, …).

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| Delivery | Thin slices B1 → B2 → B3 → deploy |
| QE XC | Emit `input_dft` only; warn about PBE-heavy `Pseudo/` |
| Deflector | Shift ky window center; keep width |
| Slit size | Analyzer ΔE = `(w_mm/400)×PE` (R₀=200 mm class) |
| Total ΔE | `√(ana² + beam² + extra²)`; not analyzer alone |
| `#ar-de` | Auto from total + checkbox manual override |
| DFT 2D | Isoenergy map only |

## Architecture

```
B1: QERequest.functional → PipelineParams → write_scf/nscf input_dft

B2: UI
      slit_mm, PE, beam, extra → ΔE_ana, ΔE_total → #ar-de (unless manual)
      deflector° → Δky = 0.5123√(hν−φ)·sin(θ) → shift ky min/max
    → simPayload.res_E + ky bounds (+ optional metadata fields)

B3: POST /dft/{name}/isoenergy
      → calculate_2d_mesh → I=Σ_n exp(−(E_n−E)²/(2σ²))
      → heatmap in DFT panel
```

---

## §1 — QE XC picker

### API / core

- `QERequest.functional: Literal["PBE","LDA","PBEsol"] = "PBE"`
- `PipelineParams.functional: str = "PBE"`
- Map to QE keywords: PBE→`pbe`, LDA→`lda`, PBEsol→`pbesol`
- `write_scf_input` / `write_nscf_input`: add `input_dft = '…'` inside `&SYSTEM`
- `_params_from_request` / `readQeParameters` pass `functional`
- Pseudopotential selection unchanged

### UI

- `#qe-xc` select in QE panel
- Hint: sets QE `input_dft`; local pseudos are mostly PBE-tagged — XC/pseudo mismatch may be wrong physics

### Tests

- Default scf contains `input_dft = 'pbe'`
- LDA request → `input_dft = 'lda'`
- Schema rejects unknown functional

---

## §2 — ARPES slit / deflector / resolution

### Deflector

- Enable `#ar-defl` (−15…15 °)
- `Δk_y = 0.5123 * sqrt(max(hν − φ, 0)) * sin(radians(defl))` (Å⁻¹)
- Shift ky min/max by `Δk_y` relative to a stored base window (or recompute from a base each time so repeated edits don’t stack)
- Prefer: keep “base” ky range in data attributes / JS state; apply deflector offset when building payload and when syncing displayed min/max
- Send `deflector_angle` in payload for metadata; simulation uses shifted `ky` bounds

### Analyzer ΔE from slit

- Enable `#ar-slitsize` (mm)
- Add `#ar-pe` pass energy (eV, default 20)
- Add `#ar-de-beam` (eV, default 0.01), `#ar-de-extra` (eV, default 0)
- `ΔE_ana = (slit_mm / 400.0) * pass_energy_eV`
- `ΔE_total = sqrt(ana² + beam² + extra²)`
- Status line: ana / beam / extra / total
- Document: Scienta-like estimate (R₀≈200 mm), not a calibrated beamline table

### Auto + override `res_E`

- `#ar-de-manual` checkbox (default off)
- Off: `#ar-de` read-only, synced to ΔE_total
- On: user edits `#ar-de`
- `simPayload().res_E` always = current `#ar-de` value
- Optional schema fields for logging: `slit_size_mm`, `pass_energy`, `res_E_beam`, `res_E_extra`, `res_E_manual`, `deflector_angle`

### Pure helper

- `tensorspec/core/arpes/resolution.py` (or similar): `analyzer_delta_e`, `total_delta_e`, `deflector_dk` — unit tested, no Chinook

### Hints

- Total resolution ≠ analyzer alone
- Deflector moves ky; kx = along-slit domain

---

## §3 — DFT isoenergy cut

### API

- `POST /api/dft/{name}/isoenergy`
- Body: TB parameters (hoppings, cutoffs, SOC, tb_mode, …) + `energy` + `kx_min/max`, `ky_min/max`, `resolution` (≤48) + `smear` (default 0.05 eV)
- Implementation: `calculate_2d_mesh` then  
  `I[i,j] = Σ_n exp(−(E_n(i,j) − energy)² / (2 smear²))`
- Response: `{kx, ky, intensity, energy, smear, n_bands, elapsed_seconds, …}`
- Diagonalisation budget check analogous to 1D bands

### UI

- `#tb-kgrid`: `1D High-Symmetry Path` | `2D Isoenergy (kx, ky)`
- Enable `#tb-isoe` in isoenergy mode
- Calculate branches: 1D bands vs isoenergy endpoint
- Display: heatmap in DFT results panel (ImageViewer-style or dedicated small canvas); not BandPlot polylines
- Hint: TB isoenergy density on mesh — not QE Fermi surface

### Tests

- Helper or endpoint: intensity higher near a band energy than far away on a tiny structure
- Resolution cap enforced

---

## Errors, tests, ship

| Case | Behavior |
|------|----------|
| Unknown XC | schema 422 |
| Iso mesh too heavy | 422 |
| No crystal | 422 |
| Manual dE off | `#ar-de` tracks total |

**Ship order:** B1 → B2 → B3 → push `HTML_einstein_app` → Einstein pull/restart.

**Success criteria**

1. Generated `scf.in` contains chosen `input_dft`.
2. Changing slit/PE/beam updates total dE; manual override sticks for sim.
3. Deflector nonzero shifts ky bounds used in sim.
4. Isoenergy mode shows kx–ky heatmap at chosen E.

## Out of scope (later)

- Track C: polyhedra / eraser / PBR / Matplotlib backends  
- Track D: ML / XAS / PEEM / Transport shells  
- Track E: ARPES history browse  
