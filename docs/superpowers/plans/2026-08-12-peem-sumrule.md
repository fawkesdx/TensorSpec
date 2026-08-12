# PEEM XMCD Sum Rule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship ROI/picture-wide XMCD sum rule on CP/CM (or LH/LV) mean spectra: optional I0, p/q/r → m_orb and m_spin+m_dipole, dual ensemble, `/analysis/sumrule`, suite UI with L-window drag.

**Architecture:** Pure NumPy `peem_sumrule` (integrals/moments/ensemble + analysis Dataset). Router resolves stack pair (prefer `*_bg` → separated → paired), reuses `resolve_energy` / `extract_spectrum` / `roi_to_mask` / optional BG analysis attrs for BG jitter. Thin preview/apply API + suite panel.

**Tech Stack:** NumPy, xarray, FastAPI, existing PEEM BG plot patterns.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-12-peem-sumrule-design.md`
- Branch: `HTML_einstein_app` only — never merge to `main`
- No pixel maps; no forced I0; no overwrite `/raw` or flat paired `/processed`
- Classic Thole/Carra form only (locked below)
- Tests: `PYTHONPATH=. TensorSpec_env/bin/pytest …`
- After ship: push + Einstein; update roadmap/README; never merge to main

## File map

| File | Role |
|------|------|
| `tensorspec/core/peem_sumrule.py` | I0, windows, integrals, moments, ensemble, analysis Dataset |
| `tests/test_peem_sumrule.py` | Unit tests |
| `tensorspec/web/server/schemas.py` | Sum-rule request/response/meta |
| `tensorspec/web/server/routers/peem.py` | resolve sources + preview/apply/get + meta |
| `tests/test_peem_api.py` | API sum-rule tests |
| `tensorspec/web/static/js/api.js` | peemSumrulePreview / Apply / Get |
| `tensorspec/web/static/js/peem_suite.js` | Sum-rule UI + L-window drag on plot |
| `tensorspec/web/templates/suites/peem_suite.html` | Enable sum-rule controls |
| `roadmap.md` / `README.md` | Mark sum-rule (partial) shipped |

## Locked formulas (v1)

Dichroism `dμ = μ+ − μ−`, sum `sμ = μ+ + μ−`.

Trapezoid integrate on energy axis:

- **p** = ∫_{L3} dμ dE  
- **q** = ∫_{L3∪L2} dμ dE  
- **r** = ∫_{r_window} sμ dE  

Moments (⟨T_z⟩=0 form; document in module docstring):

- **m_orb** = −(4/3) · nₕ · q / r  
- **m_spin_plus_dipole** = nₕ · (6p − 4q) / r  

Require `|r| > eps` else ValueError. `nh > 0`.

## Locked details

- **Pair tags:** `("CP","CM")` or `("LH","LV")` — plus = first tag, minus = second.
- **Source order:** both `processed/{tag}_bg` → both `processed/{tag}` → paired 4D `/processed` channels 0/1.
- **I0:** from raw metadata `I0` list length `n_frames`; apply to both spectra; else `i0_applied=False`.
- **Energy:** `peem_bg.resolve_energy` on plus stack metadata (or raw).
- **BG ensemble leg:** if `/analysis/background` exists with e0/e1, jitter those and re-subtract linear BG from mean spectra before integrals; else skip BG leg (n_valid_bg=0).
- **L-window ensemble:** jitter each window endpoint ±delta independently; default delta=5% energy span (1.0 if index); n default 21; caps 1..101.
- **Analysis node:** `sumrule`.
- **RNG:** `default_rng(seed)`.

---

### Task 1: `peem_sumrule` engine + unit tests

**Files:**
- Create: `tensorspec/core/peem_sumrule.py`
- Create: `tests/test_peem_sumrule.py`

**Interfaces:**

```python
def apply_i0(spectrum: np.ndarray, i0: np.ndarray | float | None) -> tuple[np.ndarray, bool]: ...

def integrate_windows(
    energy, mu_plus, mu_minus, *,
    l3: tuple[float, float],
    l2: tuple[float, float],
    r_win: tuple[float, float],
) -> dict:  # p, q, r

def moments(p: float, q: float, r: float, nh: float) -> dict:
    # m_orb, m_spin_plus_dipole

def ensemble_sumrule(
    energy, mu_plus, mu_minus, *,
    l3, l2, r_win, nh,
    window_delta: float, window_n: int,
    bg_e0: float | None = None, bg_e1: float | None = None,
    bg_delta: float = 0.0, bg_n: int = 1,
    seed: int = 0,
) -> dict:
    # means/stds for p,q,r,m_orb,m_spin_plus_dipole; n_valid

def analysis_dataset(...) -> xr.Dataset: ...
```

Note: `resolve_sumrule_sources` may live in router (needs workspace); engine stays pure arrays. Optionally add a pure helper `pick_source_kind(available_nodes, tags) -> str` for unit tests.

- [ ] **Step 1: Failing tests** — I0 on/off; known analytic p,q,r on square pulses; moments formula; r≈0 raises; ensemble std>0; invalid nh.

- [ ] **Step 2:** `pytest tests/test_peem_sumrule.py -v` → FAIL

- [ ] **Step 3: Implement engine**

- [ ] **Step 4:** PASS

- [ ] **Step 5: Commit** `feat(peem): XMCD sum-rule engine (p,q,r moments)`

---

### Task 2: Schemas + API resolve/preview/apply

**Files:** `schemas.py`, `peem.py`, `tests/test_peem_api.py`

**Schemas:** `PeemSumruleRequest`, `PeemSumrulePreviewResponse`, `PeemSumruleApplySummary`; meta `has_sumrule`, `sumrule_i0_applied`, `sumrule_tags`.

**Router helpers:**
- `_resolve_sumrule_stacks(session, name, request)` → plus/minus TensorData or ndarray + tags + source_kind
- Shared `_run_sumrule` for preview/apply (DRY like BG `_run_bg_fit`)
- `POST .../sumrule/preview`, `.../apply`, `GET .../sumrule`
- Apply: `write_analysis_data(name, "sumrule", ds)` only

**Tests (mirror TestPeemApi):** load CP/CM → pair → separate → (optional bg) → sumrule preview/apply → meta; I0 path; 422 without pair; preview no write.

- [ ] Implement TDD + commit `feat(peem): sum-rule preview/apply API`

---

### Task 3: Suite UI

**Files:** `api.js`, `peem_suite.js`, `peem_suite.html`

- Enable sum-rule button / fieldset section (keep pixel stuff disabled)
- nh, L3/L2/r fields, ensemble details, use-ROI, Preview/Apply
- Extend or add spectrum plot: μ+, μ−, dichroism toggles; drag L3/L2/r bands (may reuse peem-bg-plot or `#peem-sumrule-plot`)
- Busy/name guards like BG/Separate; I0 warning from response flag
- Results readout: p,q,r,m_orb,m_spin ± std

- [ ] Commit `feat(peem): enable XMCD sum-rule in suite UI`

---

### Task 4: Docs + push + Einstein

- Roadmap: check sum-rule / I0-apply-when-present partial; leave pixel-to-pixel open
- README PEEM blurb
- Full `pytest tests/test_peem_*.py`
- Push + Einstein pull/health

---

## Spec coverage

| Spec | Task |
|------|------|
| Integrals/moments/I0/ensemble | 1 |
| Source resolve + API + analysis write | 2 |
| Suite UI + window drag | 3 |
| Docs/Einstein | 4 |

## Placeholder scan

Formulas locked. Source order locked. No TBD.
