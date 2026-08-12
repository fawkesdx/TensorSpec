# PEEM Suite — XMCD Sum Rule (ROI / Picture-wide) — Design Spec

Date: 2026-08-12  
Status: approved for planning  
Branch: `HTML_einstein_app`  
Related: `docs/superpowers/specs/2026-08-11-peem-bg-preedge-design.md`, `roadmap.md` sum-rule / I0 bullets, `sandy_rule.md` `/analysis`

## Problem

After pair / separate / optional BG, users need XMCD sum-rule numbers from CP vs CM (or LH vs LV) mean spectra. Roadmap requires I0 normalization when available, L₂/L₃ analysis, and uncertainty dominated by window/BG choices — not fit-σ alone. No sum-rule engine exists yet; the suite button stays disabled.

## Goals

- Resolve a CP/CM (or LH/LV) stack pair: prefer `*_bg` children, else separated channels, else paired 4D.
- Build picture-wide or ROI mean spectra; apply I0 when CSV length matches; otherwise warn and proceed unnormalized.
- Classic two-edge integrals **p, q, r** → **m_orb** and **m_spin + m_dipole** with user **nₕ**.
- L₃ / L₂ / r energy windows via numeric fields + spectrum plot drag (same spirit as BG pre-edge UI).
- Dual light ensemble: BG pre-edge window jitter (when applicable) **and** L-window jitter → mean±std on integrals and moments.
- Preview without writing; Apply writes `/analysis/sumrule`.
- Enable sum-rule controls in PEEM suite; leave pixel maps and domain tools disabled.

## Non-goals

- Pixel-to-pixel sum-rule maps.
- Spatial 2D background stage; new BG formulas beyond existing linear pre-edge reuse.
- Forcing I0 (hard require).
- Overwriting `/raw` or flat paired `/processed`.
- Multi-azimuth / domain-area tools; full Co₃Sn₂S₂ paper encoding beyond classic Thole/Carra p,q,r form.

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| Scope | ROI / picture-wide XMCD sum rule |
| Architecture | Engine-first `peem_sumrule` + thin API + suite panel |
| I0 | Use when present and length-aligned; else warn + unnormalized |
| Stacks | Prefer `*_bg` pair → separated → paired 4D |
| Math | p, q, r → m_orb, m_spin+m_dipole; user nₕ |
| Windows | L₃ / L₂ / r fields + plot drag |
| Uncertainty | BG pre-edge ensemble **and** L-window jitter |
| Outputs | `/analysis/sumrule` only (no image write in v1) |

## Clarification: formulas

Document exact integral limits and moment prefactors in code docstrings (standard XMCD / Thole–Carra style). v1 uses one published classic form with explicit constants; later slices may add alternate paper variants.

---

## §1 — Core + DataTree

**Engine** (`tensorspec/core/peem_sumrule.py`, pure NumPy):

```text
resolve_sumrule_sources(...) -> (stack_plus, stack_minus, tags, source_kind)
mean_spectrum(stack, mask|None) -> spectrum
apply_i0(spectrum, I0|None) -> (spectrum, i0_applied)
integrate_windows(energy, mu_plus, mu_minus, windows) -> p, q, r
moments(p, q, r, nh) -> m_orb, m_spin_plus_dipole
ensemble_sumrule(...) -> means + stds for integrals and moments
```

- ROI via existing `roi_to_mask`; mask `None` = picture-wide.
- Known channel pairs: CP/CM or LH/LV (from `channel_tags` / resolved sources).
- Ensemble caps analogous to BG (n, delta); seedable for tests.

**DataTree**

| Node | Content |
|------|---------|
| `/analysis/sumrule` | energy; μ± (and I0-normalized if applied); dichroism; windows; p,q,r ±std; moments ±std; nh; i0_applied; source nodes/tags; ensemble meta |
| `/history` | Log on apply |

No processed image child in v1.

---

## §2 — API + UI

**Router**

| Endpoint | Role |
|----------|------|
| `POST /api/peem/{name}/sumrule/preview` | Full compute; no tree write |
| `POST /api/peem/{name}/sumrule/apply` | Write `/analysis/sumrule` |
| `GET /api/peem/{name}/sumrule` | Reload stored analysis |
| `GET .../meta` | `has_sumrule`, last `i0_applied`, tags used |

**Request (minimum):** `use_roi`, optional `roi`, `nh`, L₃/L₂/r bounds, ensemble params; optional explicit node overrides.

**UI:** Enable Run Sum Rule. nh; L₃/L₂/r fields ↔ drag; μ+/μ−/dichroism toggles; show p/q/r and moments with ±ensemble; I0-missing warning. Picture-wide / use-ROI like BG.

**Errors:** 422 if pair unresolved, empty ROI, invalid windows, nh≤0.

---

## §3 — Success criteria & tests

**Done when:**

- Preview does not mutate tree; apply writes `/analysis/sumrule`; meta flags sum-rule.
- Source resolution order tested; I0 on/off paths tested.
- Picture-wide + ROI; dual ensemble finite std when configured.
- Unit + API tests green; suite sum-rule usable; pixel / domain left open on roadmap.
- Stay on `HTML_einstein_app`; push + Einstein; never merge to main.

## Later (explicit backlog)

- Pixel-to-pixel sum-rule maps.
- Alternate paper formula variants; C dipole correction UI.
- Richer multi-spectrum library UI; batch across stacks.
- Shared XAS suite front-end over same core.

## Spec self-review

- Locked decisions have no TBD.
- Scope = ROI/picture-wide classic XMCD + dual ensemble + optional I0; heavier paths deferred.
- Aligns with `/analysis` hierarchy and existing BG/ROI patterns.
