# PEEM Two-Step BG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `two_step` BG (pre+post linear segments + connect) beside existing linear; method picker; ensemble on four endpoints; wire sum-rule BG leg to stored method.

**Architecture:** Extend `peem_bg.py` with `fit_two_step_pre_post`, `fit_background`, `ensemble_background`. Router/request gain `method` + post windows. Suite method select + post-edge drag. Sum-rule ensemble calls method-aware fit.

**Tech Stack:** NumPy, existing BG API/UI, FastAPI.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-12-peem-bg-two-step-design.md`
- Branch: `HTML_einstein_app` only — never merge to `main`
- Keep linear path regression-safe
- Require `pre_e1 < post_e0` for two-step; ≥2 points each window
- No multi-spectra / batch / pixel BG
- Tests: `PYTHONPATH=. TensorSpec_env/bin/pytest …`
- Push + Einstein after ship

## File map

| File | Role |
|------|------|
| `tensorspec/core/peem_bg.py` | two-step fit, dispatch, ensemble |
| `tests/test_peem_bg.py` | unit tests |
| `tensorspec/core/peem_sumrule.py` | BG leg uses method-aware fit |
| `tests/test_peem_sumrule.py` | two-step BG leg sample |
| `schemas.py` / `peem.py` | method + post fields on BG request/response |
| `tests/test_peem_api.py` | two-step preview/apply + sumrule BG leg |
| `api.js` / `peem_suite.js` / `peem_suite.html` | method picker + post UI |
| `roadmap.md` / `README.md` | note two-step |

## Locked connect geometry

Given pre fit `(s0,i0)` on `[pre_e0, pre_e1]` and post `(s1,i1)` on `[post_e0, post_e1]` with `pre_e1 < post_e0`:

```
y_pre(E) = s0*E + i0
y_post(E) = s1*E + i1
bg(E) =
  y_pre(E)                           if E <= pre_e1
  lerp(y_pre(pre_e1), y_post(post_e0), t)  if pre_e1 < E < post_e0
  y_post(E)                          if E >= post_e0
```

where `t = (E - pre_e1) / (post_e0 - pre_e1)`.

Field mapping: linear keeps `e0`,`e1`. Two-step uses `e0`,`e1` as **pre** and `post_e0`,`post_e1` as **post** (minimal schema churn).

---

### Task 1: Engine two-step + dispatch + ensemble

**Files:** `peem_bg.py`, `tests/test_peem_bg.py`

```python
def fit_two_step_pre_post(energy, spectrum, pre_e0, pre_e1, post_e0, post_e1) -> dict:
    # slopes/intercepts pre+post, bg full axis
    # ValueError if pre_e1 >= post_e0 or <2 pts each window

def fit_background(method: str, energy, spectrum, *, e0, e1, post_e0=None, post_e1=None) -> dict:
    # linear | two_step; include method in return

def ensemble_background(method, energy, spectrum, *, e0, e1, post_e0=None, post_e1=None, delta, n, seed=0) -> dict:
    # same mean/std keys as ensemble_preedge
```

Keep `fit_linear_preedge` / `ensemble_preedge` as wrappers or call through dispatch.

- [ ] Failing tests: known two-step geometry; invalid order; ensemble std>0; linear still passes
- [ ] Implement + pass
- [ ] Commit `feat(peem): two-step pre/post BG fit and ensemble`

---

### Task 2: API + analysis attrs + sum-rule BG leg

**Files:** `schemas.py`, `peem.py`, `peem_sumrule.py`, tests

- `PeemBgRequest.method: Literal["linear","two_step"] = "linear"`
- `post_e0`, `post_e1` optional floats
- Preview/Apply use `fit_background` / `ensemble_background`
- `analysis_dataset` stores `method`, `post_e0`, `post_e1` when two-step
- `_sumrule_bg_params` returns method + windows; `ensemble_sumrule` BG samples call `fit_background` for that method
- API tests: two-step apply stores method; sumrule with two-step analysis; 422 bad windows; linear regression

- [ ] Commit `feat(peem): two-step BG API and sum-rule BG leg`

---

### Task 3: Suite UI method picker + post-edge drag

**Files:** `peem_suite.html`, `peem_suite.js`, `api.js` if needed

- Select Linear | Two-step
- When two-step: show post_e0/post_e1 fields; draw pre+post bands; drag four endpoints (or two bands)
- Payload includes method + post fields
- Restore method/windows from spectrum GET / analysis attrs
- Hint: “two linear segments + connect”

- [ ] Commit `feat(peem): two-step BG method picker in suite`

---

### Task 4: Docs + push + Einstein

- Roadmap: document two-step; leave multi-spectra/batch open
- README blurb
- Full `pytest tests/test_peem_*.py`
- Push + Einstein

---

## Spec coverage

| Spec | Task |
|------|------|
| two-step fit + connect | 1 |
| ensemble four endpoints | 1 |
| API method + sum-rule wire | 2 |
| Suite picker + drag | 3 |
| Docs/Einstein | 4 |
