# Approach C (Fresnel + Photon Momentum) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Fresnel-corrected local light field (default on) and optional full photon-momentum `q` (default off) to the TensorSpec one-step Chinook/Grizzly ARPES path so beam↔surface geometry can change ME intensities at fixed crystal cut, with soft X-ray k-shift available later.

**Architecture:** Keep existing vacuum `compute_A_lab` as the base vector. Add a small Fresnel module that scales s/p components by transmission amplitudes for refractive index `n`. Optionally subtract full photon wavevector `q` (∥+z) inside `build_k_bulk_mesh` after photoelectron refraction. Wire flags through `physics_from_experiment_kwargs`, ARPES GUI, remote JSON, and simulation metadata. No GrizzlyME release required.

**Tech Stack:** Python 3.11, NumPy, PyQt ARPES panel, existing `chinook_arpes_kmesh.py`, pytest, `ARPESKinematics.HBAR_C`.

## Global Constraints

- Branch: `TensorSpec_GUI` only (never merge to `main` unless user asks)
- Spec: `docs/superpowers/specs/2026-09-07-arpes-approach-c-fresnel-photon-q-design.md`
- Defaults: `fresnel_enabled=True`, `optical_n=1.0`, `optical_k=0.0`, `include_photon_momentum=False`
- With `optical_n=1` (+k=0), Fresnel must reproduce vacuum `A` (bit-level / 1e-12)
- Photon-q off must leave k mesh identical to pre-C
- No footprint/H0/H1, no B-field, no SPR-KKR, no full `e^{iq·r}` multipole operator
- Do not bump GrizzlyME unless pol path breaks (not expected)
- Commits only when user asks; docs/commits = normal prose
- Caveman chat OK

## File map

| File | Role |
|------|------|
| `tensorspec/core/arpes/one_step/fresnel.py` | **Create:** Fresnel `t_s`/`t_p` + apply to vacuum `A` |
| `tensorspec/core/arpes/one_step/photon_momentum.py` | **Create:** build lab/sample/bulk `q`, `|q|=hv/ħc` |
| `tensorspec/core/arpes/one_step/chinook_arpes_kmesh.py` | Wire Fresnel into A path; photon-q into k mesh; physics keys |
| `tensorspec/core/kinematics.py` | Share `HBAR_C`; optionally thin wrappers (no duplicate constants) |
| `tensorspec/gui/components/arpes_panel.py` | GUI knobs + kwargs + metadata |
| `tests/test_arpes_approach_c.py` | **Create:** unit tests (no full TB cube) |
| `Matrix Element Issue/02_…Checklist.md` or handoff | Mark Approach C in progress / done when validating |

Lab-frame convention (must match `compute_A_lab` today):

- Incidence angle `α` from surface normal toward beam.
- LH vacuum: `A = [cos α, −sin α, 0]` (p in x–y lab).
- LV vacuum: `A = [0, 0, 1]` (s along z).
- Photon beam direction for `q`: unit vector along photon propagation toward the sample in the same lab frame as A (document in `photon_momentum.py` docstring; golden test: Δ|q∥| 55°↔20° at 84 eV ≈ 0.0203 Å⁻¹).

---

### Task 1: Fresnel unit tests + `fresnel.py`

**Files:**
- Create: `tensorspec/core/arpes/one_step/fresnel.py`
- Create: `tests/test_arpes_approach_c.py`
- Test: `tests/test_arpes_approach_c.py`

**Interfaces:**
- Produces:
  - `fresnel_ts_tp(incidence_deg: float, n: complex) -> tuple[complex, complex]`
  - `apply_fresnel_to_A_lab(A_vac: np.ndarray, incidence_deg: float, n: complex) -> np.ndarray`
- Consumes: NumPy only

- [ ] **Step 1: Write failing tests**

```python
# tests/test_arpes_approach_c.py
import numpy as np
from tensorspec.core.arpes.one_step.chinook_arpes_kmesh import compute_A_lab
from tensorspec.core.arpes.one_step.fresnel import apply_fresnel_to_A_lab


def test_fresnel_n1_matches_vacuum_LH():
    for alpha in (20.0, 55.0):
        A = compute_A_lab("Linear Horizontal (p-pol)", alpha)
        Af = apply_fresnel_to_A_lab(A, alpha, n=1.0 + 0.0j)
        np.testing.assert_allclose(Af, A, atol=1e-12)


def test_fresnel_n1_matches_vacuum_LV():
    A = compute_A_lab("Linear Vertical (s-pol)", 55.0)
    Af = apply_fresnel_to_A_lab(A, 55.0, n=1.0)
    np.testing.assert_allclose(Af, A, atol=1e-12)


def test_fresnel_n_gt1_changes_LH_Ez_ratio():
    """With n>1, transmitted p-field mix must differ from vacuum at fixed α."""
    alpha = 55.0
    A = compute_A_lab("Linear Horizontal (p-pol)", alpha)
    Af = apply_fresnel_to_A_lab(A, alpha, n=2.0)
    assert not np.allclose(Af, A, atol=1e-6)
```

- [ ] **Step 2: Run tests — expect FAIL (import / missing module)**

Run: `pytest tests/test_arpes_approach_c.py -v --tb=short`  
Expected: FAIL on missing `fresnel`

- [ ] **Step 3: Implement `fresnel.py`**

Use interface vacuum → isotropic solid, complex `n`, Snell `sin θ_t = sin θ_i / n` (handle TIR by returning zeros or finite complex continuation — pick one, document, and test n=1 only for identity). Standard amplitude transmission:

```text
t_s = 2 n_i cos θ_i / (n_i cos θ_i + n_t cos θ_t)
t_p = 2 n_i cos θ_i / (n_t cos θ_i + n_i cos θ_t)
```

with `n_i=1`, `n_t=n`. Decompose `A_vac` into s (lab z) and p (in x–y plane) for this codebase’s LH/LV convention; scale; reassemble. Keep complex dtype if input complex.

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/test_arpes_approach_c.py -v`  
Expected: PASS

- [ ] **Step 5: Commit** (only if user asked)

```bash
git add tensorspec/core/arpes/one_step/fresnel.py tests/test_arpes_approach_c.py
git commit -m "feat(arpes): Fresnel local-field helper for Approach C"
```

---

### Task 2: Photon-momentum helpers + golden Δq∥ test

**Files:**
- Create: `tensorspec/core/arpes/one_step/photon_momentum.py`
- Modify: `tensorspec/core/kinematics.py` (export / reuse `HBAR_C` only; do not change call sites yet)
- Modify: `tests/test_arpes_approach_c.py`

**Interfaces:**
- Consumes: `ARPESKinematics.HBAR_C` (1973.269804 eV·Å)
- Produces:
  - `photon_q_lab(hv_eV: float, incidence_deg: float) -> np.ndarray` shape `(3,)`  
    `|q| = hv / HBAR_C`; direction = photon propagation in lab (same incidence plane as `compute_A_lab`)
  - Document sign: later `K_crystal = K_pe - R(q_lab)` (photoelectron crystal momentum)

- [ ] **Step 1: Write failing tests**

```python
from tensorspec.core.arpes.one_step.photon_momentum import photon_q_lab
from tensorspec.core.kinematics import ARPESKinematics


def test_photon_q_magnitude():
    q = photon_q_lab(84.0, 55.0)
    assert abs(np.linalg.norm(q) - 84.0 / ARPESKinematics.HBAR_C) < 1e-10


def test_photon_q_parallel_delta_55_vs_20():
    """Critic golden number at 84 eV: Δ|q∥| ≈ 0.0203 Å⁻¹ between 55° and 20°."""
    q55 = photon_q_lab(84.0, 55.0)
    q20 = photon_q_lab(84.0, 20.0)
    # q∥ = components in surface plane (lab x,z if y is surface normal toward vacuum)
    # Implement consistent with chosen lab frame; assert:
    dq = abs(_q_parallel_mag(q55) - _q_parallel_mag(q20))
    assert abs(dq - 0.0203) < 5e-4
```

Define `_q_parallel_mag` in the test to match the module’s documented frame (surface normal = lab +y in current kmesh refraction).

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `photon_q_lab`**

Match beam geometry used for A: if LH uses `[cos α, −sin α, 0]` as E-field for p-pol, photon wavevector direction is along the beam. Use the same `α` and lab axes as `build_k_bulk_mesh` comments / `compute_A_lab`.

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit** (if asked)

```bash
git commit -m "feat(arpes): photon-q lab vector helper for soft X-ray"
```

---

### Task 3: Physics dict keys + `compute_A` / `build_k_bulk_mesh` wire

**Files:**
- Modify: `tensorspec/core/arpes/one_step/chinook_arpes_kmesh.py`
- Modify: `tests/test_arpes_approach_c.py`
- Modify: `tests/test_arpes_critic_gaps.py` only if physics-key assertions need extending

**Interfaces:**
- Extends `physics_from_experiment_kwargs` with:
  - `fresnel_enabled: bool` default `True`
  - `optical_n: float` default `1.0`
  - `optical_k: float` default `0.0`
  - `include_photon_momentum: bool` default `False`
- `build_k_bulk_mesh` (or its callers) must:
  1. `A_vac = compute_A_lab(...)`
  2. if fresnel: `A_lab = apply_fresnel_to_A_lab(A_vac, incidence, n=optical_n+1j*optical_k)` else `A_lab = A_vac`
  3. existing manip/hkl maps → `A_bulk`
  4. build `K` as today; if photon-q: subtract rotated `q` from `K_BULK` (full 3-vector)

- [ ] **Step 1: Failing tests for physics defaults + mesh identity**

```python
def test_physics_approach_c_defaults():
    phys = physics_from_experiment_kwargs({"photon_energy": 84.0})
    assert phys["fresnel_enabled"] is True
    assert phys["optical_n"] == 1.0
    assert phys["optical_k"] == 0.0
    assert phys["include_photon_momentum"] is False


def test_build_k_mesh_photon_q_off_matches_baseline(monkeypatch):
    """Tiny synthetic B_matrix + k_bounds: flag off == legacy path."""
    # Compare K_BULK with include_photon_momentum False vs calling
    # internal path that skips q (or snapshot from current main before wire).
    ...
```

Use a minimal orthonormal `B_matrix = np.eye(3)` and small angle grid (`ntheta=3,nphi=1,ne=2`) so the test is fast. For Fresnel: with defaults n=1, `A_bulk` must match pre-Fresnel vacuum path.

- [ ] **Step 2: Run — FAIL on missing keys / behavior**

- [ ] **Step 3: Implement physics keys + wire A and K**

Keep function signatures stable; read new fields from `physics` inside `run_*` / `build_k_bulk_mesh` via optional kwargs or physics dict already passed at call sites. Prefer extending `build_k_bulk_mesh` with optional keyword args that `run_chinook_arpes` / `prepare` fill from physics — avoid breaking positional callers.

Order for K (document in docstring):

1. vacuum photoelectron k from angles  
2. inner-potential refraction (existing)  
3. manip + hkl → `K_BULK`  
4. if photon-q: `K_BULK = K_BULK - q_bulk`

- [ ] **Step 4: Run Approach C + critic-gap tests — PASS**

Run:  
`pytest tests/test_arpes_approach_c.py tests/test_arpes_critic_gaps.py -q`

- [ ] **Step 5: Commit** (if asked)

```bash
git commit -m "feat(arpes): wire Fresnel A and optional photon-q into kmesh"
```

---

### Task 4: GUI + remote JSON + metadata

**Files:**
- Modify: `tensorspec/gui/components/arpes_panel.py`
- Modify: `tests/test_arpes_approach_c.py` (kwargs round-trip via `physics_from_experiment_kwargs` only; no Qt required)

**Interfaces:**
- Extend `_critic_gap_physics_kwargs()` (or sibling `_approach_c_physics_kwargs()`) to include the four keys
- Widgets near incidence / beam group:
  - `QCheckBox` “Fresnel local field” default **checked**
  - `optical_n` / `optical_k` spins (enable when Fresnel on)
  - `QCheckBox` “Photon momentum (soft X-ray)” default **unchecked**
- Ensure local simulate, remote physics JSON, and `get_simulation_metadata()` all record the four keys

- [ ] **Step 1: Test physics kwargs mapping from a fake dict** (already in Task 3); add explicit remote-key presence test if a pure function builds remote JSON — otherwise manual checklist in Step 4

- [ ] **Step 2: Add widgets + wire `_critic_gap_physics_kwargs` / metadata**

- [ ] **Step 3: Grep that remote `arpes_physics` dump includes new keys**

Run: `rg "fresnel_enabled|include_photon_momentum" tensorspec/gui/components/arpes_panel.py`

- [ ] **Step 4: Smoke** — launch not required in CI; agent verifies defaults in code

- [ ] **Step 5: Commit** (if asked)

```bash
git commit -m "feat(arpes): GUI knobs for Fresnel and photon momentum"
```

---

### Task 5: Docs + checklist + Einstein smoke note

**Files:**
- Modify: `docs/superpowers/specs/2026-09-07-arpes-approach-c-fresnel-photon-q-design.md` (status → approved/implemented when done)
- Modify: `Matrix Element Issue/02_TensorSpec_ApproachB_Rerun_Checklist.md` or add Approach C note in handoff
- Optional: one paragraph in TensorSpec ARPES docs if such a user-facing doc exists

- [x] **Step 1: Update checklist** — Approach C no longer “deferred”; defaults documented
  (Overleaf path outside TensorSpec git: `VTe2 Project/Matrix Element Issue/02_TensorSpec_ApproachB_Rerun_Checklist.md`)

- [x] **Step 2: Record Einstein validation recipe** (in design spec + checklist)

```text
φ = −10 single cut, Full ME, LH, hydrogenic, mfp=5, kz=0.2
Fresnel ON, n=… (user sets; n=1 = vacuum parity)
Photon-q OFF
Compare incidence 55 vs 20 (same slit/azi pairing as experiment)
```

- [x] **Step 3: Full unit suite for touched tests** — 20 passed

Run: `pytest tests/test_arpes_approach_c.py tests/test_arpes_critic_gaps.py -q`

- [x] **Step 4: Commit**

```bash
git commit -m "docs(arpes): Approach C Fresnel/photon-q checklist"
```

---

## Spec coverage checklist

| Spec item | Task |
|-----------|------|
| Fresnel local A | 1, 3 |
| `fresnel_enabled` default True | 3, 4 |
| `optical_n` / `optical_k` | 3, 4 |
| n=1 vacuum parity | 1, 3 |
| Full photon q (∥+z) | 2, 3 |
| `include_photon_momentum` default False | 3, 4 |
| Δq∥ golden ~0.0203 @ 84 eV | 2 |
| GUI + remote JSON + metadata | 4 |
| No GrizzlyME bump | Global |
| Non-goals respected | Global |

## Placeholder / consistency scan

- Lab frame for `q` must match `compute_A_lab` / surface normal used in `build_k_bulk_mesh` (lab +y toward vacuum in refraction block).
- Do not invent a second `HBAR_C`; import from `ARPESKinematics`.
- Default Fresnel on + `n=1` ⇒ numerically old A; metadata still records `fresnel_enabled=True`.

---

## Execution handoff

Plan saved to `docs/superpowers/plans/2026-09-07-arpes-approach-c-fresnel-photon-q.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — this session with executing-plans checkpoints  

Which approach?
