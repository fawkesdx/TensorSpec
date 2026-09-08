# Fresnel v2 Refracted-Angle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace vacuum-angle Fresnel vector rebuild with refracted-angle `Ê_p(θ_t)` so LH Ez/E∥ changes when `n ≠ 1`.

**Architecture:** Single-function change in `apply_fresnel_to_A_lab`. Keep `fresnel_ts_tp` amplitudes. Complex Snell for `θ_t`. No GUI / physics-key / GrizzlyME changes. Remote already ships `fresnel.py`.

**Tech Stack:** Python 3, NumPy, pytest, TensorSpec_GUI branch

## Global Constraints

- Branch: `TensorSpec_GUI` (do not merge to `main` unless user asks)
- Replace vacuum-angle (no dual-mode toggle)
- `n = 1` must remain exact identity with vacuum `A`
- Complex `optical_k` via complex Snell continuation
- Spec: `docs/superpowers/specs/2026-09-07-arpes-fresnel-refracted-angle-design.md`

## File map

| File | Role |
|------|------|
| `tensorspec/core/arpes/one_step/fresnel.py` | Rewrite `apply_fresnel_to_A_lab` |
| `tests/test_arpes_approach_c.py` | Strengthen Fresnel tests |
| Spec status line (optional in Task 2) | Mark v2 implemented |

---

### Task 1: Refracted-angle rebuild (TDD)

**Files:**
- Modify: `tests/test_arpes_approach_c.py`
- Modify: `tensorspec/core/arpes/one_step/fresnel.py`
- Test: `tests/test_arpes_approach_c.py`

**Interfaces:**
- Consumes: `fresnel_ts_tp(incidence_deg, n) -> (t_s, t_p)` (unchanged)
- Produces: `apply_fresnel_to_A_lab(A_vac, incidence_deg, n) -> ndarray` with refracted rebuild

- [ ] **Step 1: Replace / strengthen failing tests**

In `tests/test_arpes_approach_c.py`, keep existing `test_fresnel_n1_*` parity tests. Replace weak `test_fresnel_n_gt1_changes_LH_Ez_ratio` and add:

```python
def test_fresnel_n_gt1_changes_LH_direction_mix():
    """Refracted rebuild: |Ay|/|Ax| != |tan α| when n>1."""
    alpha = 55.0
    A = compute_A_lab("Linear Horizontal (p-pol)", alpha)
    Af = apply_fresnel_to_A_lab(A, alpha, n=2.0)
    vac_ratio = abs(A[1] / A[0])
    new_ratio = abs(Af[1] / Af[0])
    assert abs(new_ratio - vac_ratio) > 1e-6


def test_fresnel_LH_Ez_mix_differs_55_vs_20():
    """Fixed n>1: |Ay|/|A_p| differs between 55° and 20°."""
    n = 2.0
    ratios = []
    for alpha in (55.0, 20.0):
        A = compute_A_lab("Linear Horizontal (p-pol)", alpha)
        Af = apply_fresnel_to_A_lab(A, alpha, n=n)
        ap = np.hypot(np.real(Af[0]), np.real(Af[1]))  # or abs complex hypot
        # Prefer: ap = abs(complex hypot of Af[0], Af[1])
        ap = float(np.sqrt(abs(Af[0]) ** 2 + abs(Af[1]) ** 2))
        ratios.append(abs(Af[1]) / ap)
    assert abs(ratios[0] - ratios[1]) > 1e-6


def test_fresnel_LV_stays_along_z():
    A = compute_A_lab("Linear Vertical (s-pol)", 55.0)
    Af = apply_fresnel_to_A_lab(A, 55.0, n=2.0)
    assert abs(Af[0]) < 1e-12 and abs(Af[1]) < 1e-12
    assert abs(Af[2]) > 1e-6
```

- [ ] **Step 2: Run tests — expect NEW asserts to FAIL on vacuum-angle code**

```bash
cd /Users/sandyai/Documents/GitHub/TensorSpec_GUI
PYTHONPATH=. pytest tests/test_arpes_approach_c.py::test_fresnel_n_gt1_changes_LH_direction_mix tests/test_arpes_approach_c.py::test_fresnel_LH_Ez_mix_differs_55_vs_20 -v
```

Expected: FAIL (vacuum-angle keeps `|Ay|/|Ax| = |tan α|`).

- [ ] **Step 3: Implement refracted rebuild in `fresnel.py`**

Update module docstring: vector rebuild is **refracted-angle** basis.

Replace `apply_fresnel_to_A_lab` body with:

```python
def apply_fresnel_to_A_lab(
    A_vac: np.ndarray,
    incidence_deg: float,
    n: complex,
) -> np.ndarray:
    """Scale + remap vacuum lab A onto refracted p-basis (Approach C v2)."""
    A = np.asarray(A_vac)
    n_c = complex(n)
    t_s, t_p = fresnel_ts_tp(incidence_deg, n_c)

    theta_i = np.radians(float(incidence_deg))
    sin_i = np.sin(theta_i)
    sin_t = sin_i / n_c
    cos_t = np.sqrt(1.0 - sin_t * sin_t)

    # Vacuum p amplitude (in-plane); preserve complex phase via Ax + i*0 path:
    # Use signed magnitude along vacuum Ê_p when A is real LH.
    A_p = np.hypot(A[0], A[1]) if np.isrealobj(A) else np.sqrt(A[0] * A[0] + A[1] * A[1])
    # Prefer phase-safe: project onto vacuum Ê_p_i = [cos θ_i, -sin θ_i, 0]
    cos_i = np.cos(theta_i)
    e_px_i = cos_i
    e_py_i = -sin_i
    # Complex-safe projection of in-plane A onto vacuum p-hat
    A_p = A[0] * e_px_i + A[1] * e_py_i

    e_px_t = cos_t
    e_py_t = -sin_t

    out = np.array(
        [t_p * A_p * e_px_t, t_p * A_p * e_py_t, t_s * A[2]],
        dtype=np.result_type(A.dtype, complex),
    )
    if np.isrealobj(A) and np.isreal(n) and np.isreal(out).all():
        return np.real(out).astype(A.dtype, copy=False)
    return out
```

Notes for implementer:
- Projection `A_p = A·Ê_p(θ_i)` keeps LH sign and `n=1` identity.
- Do **not** use unsigned `hypot` alone (loses sign / breaks identity for LH).
- Keep `fresnel_ts_tp` unchanged.

- [ ] **Step 4: Run full Approach C suite**

```bash
PYTHONPATH=. pytest tests/test_arpes_approach_c.py tests/test_arpes_critic_gaps.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_arpes_approach_c.py tensorspec/core/arpes/one_step/fresnel.py
git commit -m "feat(arpes): refracted-angle Fresnel local-field rebuild"
```

---

### Task 2: Spec status + checklist smoke note

**Files:**
- Modify: `docs/superpowers/specs/2026-09-07-arpes-fresnel-refracted-angle-design.md` (status → implemented)
- Modify if present: `/Users/sandyai/Dropbox/Apps/Overleaf/VTe2 Project/Matrix Element Issue/02_TensorSpec_ApproachB_Rerun_Checklist.md` (note v2 refracted; vacuum-angle superseded) — outside TensorSpec git; edit file but do not force into TensorSpec commit
- Modify: `docs/superpowers/specs/2026-09-07-arpes-approach-c-fresnel-photon-q-design.md` — one-line note that A-rebuild is now refracted (v2)

**Interfaces:**
- Consumes: Task 1 green tests
- Produces: docs aligned with shipped behavior

- [ ] **Step 1: Update design statuses**

Fresnel v2 spec header: `Status: approved / implemented`.  
Approach C design: add note under Fresnel section: “v2 (2026-09-07): refracted-angle rebuild replaced vacuum-angle.”

Overleaf checklist (writable path above): note Fresnel rebuild = refracted; re-run 55 vs 20 with n≠1.

- [ ] **Step 2: Commit TensorSpec docs only**

```bash
git add docs/superpowers/specs/2026-09-07-arpes-fresnel-refracted-angle-design.md docs/superpowers/specs/2026-09-07-arpes-approach-c-fresnel-photon-q-design.md
git commit -m "docs(arpes): mark refracted Fresnel v2 implemented"
```

---

## Spec coverage (self-review)

| Spec requirement | Task |
|------------------|------|
| Replace vacuum-angle | 1 |
| Complex Snell | 1 (`cos_t` sqrt) |
| n=1 identity | 1 tests |
| LH Ez/E∥ changes | 1 tests |
| LV along z | 1 tests |
| No GUI/GrizzlyME | (none — constraint) |
| Docs / checklist | 2 |
