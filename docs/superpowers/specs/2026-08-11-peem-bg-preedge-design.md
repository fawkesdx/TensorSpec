# PEEM Suite — Linear Pre-edge Background (ROI / Picture-wide) — Design Spec

Date: 2026-08-11  
Status: approved for planning  
Branch: `HTML_einstein_app`  
Related: `roadmap.md` PEEM BG bullets, `sandy_rule.md` `/analysis` + `/processed`, prior PEEM pair/drift/separate specs

## Problem

PEEM energy stacks encode spectra along the frame axis (L₂/L₃ etc. at each pixel). Users need a trustworthy **spectral** background before sum rule. Full pixel-to-pixel BG with shared params and paper formulas is large. The suite BG fieldset is still disabled. Roadmap asks for apply-to-all-frames, clear functions (Co₃Sn₂S₂ as starter), and spectra UI with overlays — this slice ships a **focused** first cut: linear pre-edge on a mean spectrum, plot + window interaction, write analysis + BG-subtracted stack, light window-ensemble uncertainty.

## Goals

- Extract a 1D spectrum from the current viewer stack (picture-wide mean or ROI mean).
- Fit **linear pre-edge** on a user window (`e0`–`e1`), with plot click/drag synced to fields.
- Light **ensemble**: jitter pre-edge endpoints → band on BG / subtracted curve (choice-driven uncertainty, not fit-σ alone).
- Preview without writing; Apply writes `/analysis/background` and a BG-subtracted image cube under `/processed/bg` (or `/processed/{tag}_bg` when source was a separated channel).
- Subtract the same `bg(E)` from **every pixel** of the source stack (ROI used only to estimate `bg`).
- Energy axis: beamline CSV energy when present and length-aligned; else frame index.
- Enable BG controls in PEEM suite; leave sum-rule disabled.

## Non-goals

- Pixel-to-pixel spectral BG with per-pixel / shared-param maps.
- Spatial 2D background map (detector/illumination) as a separate stage.
- Full Co₃Sn₂S₂ paper formula beyond linear pre-edge; L₂/L₃ window integrals; sum rule; I0 application.
- Multi-spectrum library / many independent spectra toggled together (roadmap follow-up).
- Overwriting flat paired `/processed` or `/raw`.

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| Product | ROI / picture-wide spectral BG (not pixel-to-pixel) |
| Architecture | Engine-first (`peem_bg` / reusable later by XAS) + thin API + suite UI |
| Input | Current viewer node: raw 3D, separated `processed/{tag}`, or paired 4D + channel |
| Energy | CSV energy if present and aligned; else frame index |
| Formula | Linear pre-edge |
| Spatial estimate | Picture-wide default; optional ROI mean (reuse drift ROI) |
| Apply pixels | Whole image: `I'(x,y,E)=I(x,y,E)−bg(E)` |
| Outputs | `/analysis/background` **and** `/processed/bg` (or `{tag}_bg`) |
| Pre-edge UI | Numeric `e0`/`e1` + click/drag on spectrum plot |
| Uncertainty | Light pre-edge window ensemble → std band on curves |

## Clarification: “art” and uncertainty

Window placement dominates error versus least-squares fit σ. V1 ships a **small ensemble** over plausible pre-edge windows. Full L₂/L₃ window variation and sum-rule spread remain for the sum-rule slice.

---

## §1 — Core + DataTree

**Engine** (new `tensorspec/core/peem_bg.py`; keep pure NumPy; XAS may import later):

```text
extract_spectrum(stack, mask|None, energy) -> spectrum
fit_linear_preedge(energy, spectrum, e0, e1) -> slope, intercept, bg_full
ensemble_preedge(..., delta, n) -> bg_mean, bg_std, sub_mean, sub_std
apply_bg_to_stack(stack, bg_curve) -> stack_out
```

- Mask `None` → mean over all `y,x`; else mean inside `roi_to_mask` (existing).
- Pre-edge window must contain ≥2 valid points; else `ValueError`.
- Ensemble: sample `e0,e1` within ±delta (clamp to energy range); skip invalid draws; require ≥1 valid sample.

**DataTree**

| Node | Content |
|------|---------|
| `/analysis/background` | energy, raw_spectrum, bg, bg_std, subtracted, subtracted_std, fit params, windows, ensemble meta, source `node`/`channel`/ROI summary |
| `/processed/bg` or `/processed/{tag}_bg` | 3D `(frame,y,x)` BG-subtracted cube; metadata links to analysis + source |
| `/history` | Log preview/apply lines as appropriate (apply required) |

Do **not** replace flat paired `/processed` when writing BG stack.

---

## §2 — API + UI

**Router**

| Endpoint | Role |
|----------|------|
| `POST /api/peem/{name}/bg/preview` | Fit + ensemble; return curves; no tree write |
| `POST /api/peem/{name}/bg/apply` | Write analysis + processed BG child; return summary |
| `GET /api/peem/{name}/meta` | `has_background`, `has_processed_bg`, `energy_source` |
| Frame GET | `node=processed/bg` or `processed/{tag}_bg` via existing nested pull |
| Optional `GET .../bg/spectrum` | Reload stored analysis curves |

**Request fields (minimum):** `node`, `channel` (paired), optional `roi`, `e0`, `e1`, `ensemble_delta`, `ensemble_n` (capped).

**UI:** Enable BG subsection (sum-rule stays disabled). Picture-wide / use-ROI; plot with raw / BG±band / subtracted toggles; window drag ↔ fields; Preview + Apply; after apply, viewer node for BG stack.

**Errors:** 422 for bad node, empty ROI, invalid window, ensemble failure.

---

## §3 — Success criteria & tests

**Done when:**

- Preview does not mutate tree; apply writes analysis + `/processed/bg` (or tag variant); sources intact.
- Meta/frame expose BG stack; picture-wide and ROI paths tested.
- Energy CSV vs index behavior tested.
- Ensemble returns finite std when `n>1` and delta>0.
- Unit + API tests green; suite BG usable; roadmap BG partially marked (multi-spectrum / sum-rule / pixel-to-pixel left open).
- Stay on `HTML_einstein_app`; push + Einstein; never merge to main.

## Later (explicit backlog)

- Pixel-to-pixel BG with shared parameters.
- Spatial 2D subtract stage before spectral BG.
- Co₃Sn₂S₂ (or other) models beyond linear; L₂/L₃ windows; sum-rule + I0; multi-spectrum UI; richer window ensembles.

## Spec self-review

- Locked decisions have no TBD.
- Scope = linear pre-edge mean spectrum + apply whole-image + light ensemble; heavier paths deferred.
- Aligns with DataTree `/analysis` + `/processed` children pattern from Separate.
