# PEEM Suite — Two-Step Background Model — Design Spec

Date: 2026-08-12  
Status: approved for planning  
Branch: `HTML_einstein_app`  
Related: `docs/superpowers/specs/2026-08-11-peem-bg-preedge-design.md`, sum-rule BG ensemble leg, `roadmap.md` BG documentation bullet

## Problem

Linear pre-edge BG is shipped and wired into sum-rule ensembles, but many XMCD pipelines need a **post-edge** continuum level as well. Roadmap asks to document / offer clearer BG function choices (Co₃Sn₂S₂-style starter). Multi-spectra and batch remain separate. Users need a method picker with a second model that still applies to whole stacks and feeds sum-rule uncertainty.

## Goals

- Keep **linear** pre-edge unchanged.
- Add **two-step**: fit linear on pre window + linear on post window; **linear ramp connect** between `pre_e1` and `post_e0` (require `pre_e1 < post_e0`).
- Method picker in API + suite UI; two-step shows pre + post fields and plot drag handles/bands.
- Ensemble jitters all four endpoints (±delta); skip invalid draws.
- Store `method` + windows in `/analysis/background`; sum-rule BG leg re-fits the stored method.
- Apply still writes `/analysis/background` and `/processed/bg` (or `{tag}_bg`).

## Non-goals

- Polynomial, arctan/logistic, or full Co₃Sn₂S₂ paper encoding beyond this two-step.
- Multi-spectra overlay UI; batch apply across stacks.
- Pixel-to-pixel BG; spatial 2D map.
- New DataTree node names (reuse existing background / bg children).

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| Scope | New BG models only (not multi-spectra / batch) |
| Methods | `linear` + `two_step` |
| Two-step geometry | Two linear segments + connect between windows |
| Architecture | Extend `peem_bg` with method dispatch |
| Ensemble | Jitter pre+post endpoints; sum-rule uses stored method |
| Outputs | Same `/analysis/background` + `/processed/bg` |

## Clarification: connect region

Between `pre_e1` and `post_e0`, `bg(E)` is the straight line joining `line_pre(pre_e1)` to `line_post(post_e0)`. Outside: extrapolate/evaluate the respective fitted lines on the full energy axis as specified in implementation (pre line for E≤pre_e1, post line for E≥post_e0).

---

## §1 — Core + DataTree

**Extend** `tensorspec/core/peem_bg.py`:

```text
fit_two_step_pre_post(energy, spectrum, pre=(e0,e1), post=(p0,p1)) -> slopes/intercepts + bg
fit_background(method, ...)  # "linear" | "two_step"
ensemble_background(...)     # method-aware jitter
```

- Reuse `extract_spectrum`, `apply_bg_to_stack`, `resolve_energy`.
- Analysis attrs: `method`, windows (`e0`/`e1` for linear; pre+post for two-step).
- Sum-rule: read method+windows from `/analysis/background`; call matching fit inside BG ensemble samples.

---

## §2 — API + UI

- Extend `PeemBgRequest` with `method` and post-edge fields (linear keeps current e0/e1).
- Preview/Apply/response echo method + windows.
- Suite: method select; when two-step, show post window + four drag affordances on BG plot.
- Short UI hint naming the model.
- 422 on invalid two-step windows.

---

## §3 — Success criteria & tests

**Done when:**

- Linear regression-safe; two-step fit/apply/ensemble tested.
- Invalid windows 422; `pre_e1 < post_e0` enforced.
- Sum-rule BG leg works with `method=two_step` stored analysis.
- Suite method picker + post-edge UI; roadmap note two-step shipped; multi-spectra/batch left open.
- `HTML_einstein_app` only; push + Einstein; never merge to main.

## Later

- Polynomial / smooth step models; paper-exact variants.
- Multi-spectra overlays; batch BG.
- Pixel-to-pixel BG.

## Spec self-review

- Locked decisions have no TBD.
- Scope = method dispatch + two-step + ensemble/sum-rule wire; heavier UI/models deferred.
- Reuses existing BG tree paths.
