# ARPES Critic Gaps (Approach B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close Matrix Element Issue critic simulation gaps that TensorSpec already almost owns: wire `rad_type`/`mfp`, fix `Vo` key + metadata audit trail, add optional incoherent k_z broadening — so VTe₂ 84 eV beam55/beam20 maps can be re-run as a real geometric ME test.

**Architecture:** Keep incidence→`compute_A_lab` path. Extend `physics` dict end-to-end (GUI → `physics_from_experiment_kwargs` → `arpes_dict` / Grizzly ME shell → remote JSON). Fix chinook key `Vo` (TensorSpec currently writes unused `V0`). Add optional k_z Lorentzian sum as a thin wrapper around existing `run_chinook_arpes` / `run_grizzly_arpes` by shifting `K_BULK[2]` and weight-averaging intensity cubes. Defer Fresnel + q∥ (Approach C) until B is green.

**Tech Stack:** Python 3.11, PyQt ARPES panel, chinook 1.1.3, existing `chinook_arpes_kmesh.py` / remote runner, pytest.

## Global Constraints

- Branch: TensorSpec_GUI work only (no merge to main)
- Do not claim B-field / footprint physics (H0/H1) — out of scope
- Do not change default behavior for old jobs unless knobs left at defaults (`rad_type=slater`, `mfp=10`, `kz_halfwidth=0`)
- Caveman chat OK; commits/PR/docs = normal prose
- Do not commit unless user asks
- Context docs: `VTe2 Project/Matrix Element Issue/00_HANDOFF_READ_ME_FIRST.md`, `01_Why_Chinook_Fails_Magneto_ARPES.md`; graphify at `Matrix Element Issue/tensorspec_gui_graphify/graphify-out/graph.json`

## Confirmed facts (do not re-derive)

1. chinook 1.1.3 defaults `rad_type='slater'` when key missing (`ARPES_lib.py` ~186–188).
2. TensorSpec `arpes_dict` never sets `rad_type` / `mfp` / `phase_shifts` today.
3. TensorSpec writes `"V0"`; chinook reads `"Vo"` → `self.Vo=-1`. Own `build_k_bulk_mesh` still applies `inner_potential` refraction on `K_SAMPLE[2]`.
4. VTe₂ `beam55_*` vs `beam20_*` intensities differ; metadata identical and **omits** `incidence_angle` — incidence path almost certainly used, not recorded.
5. `get_simulation_metadata()` currently omits incidence, slit, manip, ME mode, rad_type, mfp, V0.

## File map

| File | Role |
| --- | --- |
| `tensorspec/core/arpes/one_step/chinook_arpes_kmesh.py` | physics map, `arpes_dict`, k_z broaden helper |
| `tensorspec/gui/components/arpes_panel.py` | UI knobs + kwargs + metadata |
| `tensorspec/core/arpes/one_step/chinook_remote_runner_template.py` | remote physics JSON / CLI if needed |
| `tensorspec/core/arpes/one_step/chinook_wrapper.py` | pass-through only if kwargs shape changes |
| `tests/test_arpes_critic_gaps.py` | unit tests (no full TB cube required) |

---

### Task 1: Unit tests for physics mapping + A_lab + Vo key

**Files:**
- Create: `tests/test_arpes_critic_gaps.py`
- Modify (later tasks): `tensorspec/core/arpes/one_step/chinook_arpes_kmesh.py`

**Interfaces:**
- Consumes: `compute_A_lab`, `physics_from_experiment_kwargs`
- Produces: failing tests that lock Approach B contracts

- [ ] **Step 1: Write failing tests**

```python
# tests/test_arpes_critic_gaps.py
import numpy as np
from tensorspec.core.arpes.one_step.chinook_arpes_kmesh import (
    compute_A_lab,
    physics_from_experiment_kwargs,
)


def test_A_lab_lh_differs_55_vs_20():
    a55 = compute_A_lab("Linear Horizontal (p-pol)", 55.0)
    a20 = compute_A_lab("Linear Horizontal (p-pol)", 20.0)
    assert not np.allclose(a55, a20)
    assert np.allclose(a55, [np.cos(np.radians(55)), -np.sin(np.radians(55)), 0.0])


def test_physics_maps_rad_type_mfp_kz():
    phys = physics_from_experiment_kwargs(
        {
            "photon_energy": 84.0,
            "rad_type": "hydrogenic",
            "mfp": 5.0,
            "kz_halfwidth": 0.2,
            "kz_npoints": 7,
        }
    )
    assert phys["rad_type"] == "hydrogenic"
    assert phys["mfp"] == 5.0
    assert phys["kz_halfwidth"] == 0.2
    assert phys["kz_npoints"] == 7


def test_physics_defaults_preserve_legacy():
    phys = physics_from_experiment_kwargs({"photon_energy": 84.0})
    assert phys["rad_type"] == "slater"
    assert phys["mfp"] == 10.0
    assert phys["kz_halfwidth"] == 0.0
```

- [ ] **Step 2: Run tests — expect FAIL** (new keys missing from `physics_from_experiment_kwargs`)

```bash
cd /Users/sandyai/Documents/GitHub/TensorSpec_GUI
TensorSpec_env/bin/pytest tests/test_arpes_critic_gaps.py -v
```

Expected: `test_A_lab_lh_differs_55_vs_20` PASS; mapping tests FAIL on KeyError / assert.

- [ ] **Step 3: Minimal `physics_from_experiment_kwargs` extension**

In `chinook_arpes_kmesh.py`, add to the returned dict:

```python
"rad_type": str(experiment_kwargs.get("rad_type", "slater")),
"mfp": float(experiment_kwargs.get("mfp", 10.0)),
"kz_halfwidth": float(experiment_kwargs.get("kz_halfwidth", 0.0)),
"kz_npoints": int(experiment_kwargs.get("kz_npoints", 1)),
```

- [ ] **Step 4: Re-run tests — expect PASS** for Task 1 tests

- [ ] **Step 5: Commit only if user asks**

---

### Task 2: Pass `rad_type` / `mfp` / `Vo` into chinook `arpes_dict`

**Files:**
- Modify: `tensorspec/core/arpes/one_step/chinook_arpes_kmesh.py` (both `arpes_dict` builders ~473 and ~699)
- Modify: `tests/test_arpes_critic_gaps.py`

**Interfaces:**
- Consumes: `physics["rad_type"]`, `physics["mfp"]`, `physics["inner_potential"]`
- Produces: chinook experiment with non-default radial type when requested; `Vo` set so chinook-internal Vo path matches mesh refraction if ever used

- [ ] **Step 1: Add test that builds `arpes_dict` shape helper**

Prefer a small pure helper to avoid needing a TB model:

```python
def build_arpes_dict_from_physics(physics, *, cube, energy_axis, A_bulk, is_full: bool):
    """Single place for chinook ARPES_dict keys (local + Grizzly shell)."""
    return {
        "cube": cube,
        "ang": 0.0,
        "E": energy_axis,
        "hv": float(physics["hv"]),
        "W": float(physics["work_function"]),
        "Vo": float(physics["inner_potential"]),  # chinook key is Vo, not V0
        "T": float(physics.get("temperature", 10.0)),
        "pol": np.asarray(A_bulk, dtype=float),
        "ME": bool(is_full),
        "SE": ["constant", float(physics.get("se_width", 0.01))],
        "resolution": {
            "E": float(physics.get("res_E", 0.02)),
            "k": float(physics.get("res_k", 0.02)),
        },
        "rad_type": str(physics.get("rad_type", "slater")),
        "mfp": float(physics.get("mfp", 10.0)),
    }
```

Test:

```python
def test_arpes_dict_uses_Vo_and_rad_type():
    from tensorspec.core.arpes.one_step.chinook_arpes_kmesh import build_arpes_dict_from_physics
    d = build_arpes_dict_from_physics(
        {
            "hv": 84.0,
            "work_function": 4.5,
            "inner_potential": 12.0,
            "rad_type": "hydrogenic",
            "mfp": 5.0,
        },
        cube={"X": [0, 0, 1], "Y": [0, 0, 1], "E": [-1, 0, 2]},
        energy_axis=np.linspace(-1, 0, 2),
        A_bulk=np.array([1.0, 0.0, 0.0]),
        is_full=True,
    )
    assert "Vo" in d and "V0" not in d
    assert d["Vo"] == 12.0
    assert d["rad_type"] == "hydrogenic"
    assert d["mfp"] == 5.0
```

- [ ] **Step 2: Refactor both `arpes_dict = {...}` sites to call helper**; delete dead `"V0"` key

- [ ] **Step 3: pytest Task 1+2 green**

```bash
TensorSpec_env/bin/pytest tests/test_arpes_critic_gaps.py -v
```

---

### Task 3: Incoherent k_z broadening wrapper

**Files:**
- Modify: `tensorspec/core/arpes/one_step/chinook_arpes_kmesh.py`
- Modify: `tests/test_arpes_critic_gaps.py`

**Interfaces:**
- Produces: `lorentzian_kz_weights(halfwidth: float, npoints: int) -> tuple[np.ndarray, np.ndarray]`  
  returns `(delta_kz_angstrom_inv, weights)` with `weights.sum()==1`, and if `halfwidth<=0` or `npoints<=1` → `delta=[0], weights=[1]`
- Produces: `run_chinook_arpes_with_kz_broaden(... same as run_chinook_arpes ..., physics=...)` that:
  1. Builds base context / intensity once per Δk_z by copying `K_BULK` and adding `Δk_z` to component `2` in **bulk** frame (document this choice in docstring: approximates escape-depth broadening as incoherent stack of fixed-geometry cuts)
  2. Returns `sum_i w_i I_i`

**Physics note (must stay in docstring):** Critic wants Δk_z ≈ 1/λ_mfp ≈ 0.2 Å⁻¹ HWHM. Default off (`kz_halfwidth=0`) preserves legacy. This is **not** coherent LEED final-state interference.

- [ ] **Step 1: Failing weight tests**

```python
def test_kz_weights_legacy_off():
    from tensorspec.core.arpes.one_step.chinook_arpes_kmesh import lorentzian_kz_weights
    dk, w = lorentzian_kz_weights(0.0, 7)
    assert list(dk) == [0.0]
    assert list(w) == [1.0]


def test_kz_weights_normalized():
    from tensorspec.core.arpes.one_step.chinook_arpes_kmesh import lorentzian_kz_weights
    dk, w = lorentzian_kz_weights(0.2, 7)
    assert len(dk) == 7
    assert np.isclose(w.sum(), 1.0)
    assert dk[0] < 0 < dk[-1]
```

- [ ] **Step 2: Implement `lorentzian_kz_weights`**

```python
def lorentzian_kz_weights(halfwidth: float, npoints: int) -> tuple[np.ndarray, np.ndarray]:
    hw = float(halfwidth)
    n = int(npoints)
    if hw <= 0.0 or n <= 1:
        return np.array([0.0]), np.array([1.0])
    n = n if n % 2 == 1 else n + 1  # odd grid, include 0
    span = 3.0 * hw  # ±3 HWHM
    dk = np.linspace(-span, span, n)
    w = (hw / np.pi) / (dk**2 + hw**2)
    w = w / w.sum()
    return dk, w
```

- [ ] **Step 3: Wire broaden into `run_chinook_arpes` / `run_grizzly_arpes` entry points**

Prefer: after reading physics, if `kz_halfwidth > 0`, loop:

```python
dk, w = lorentzian_kz_weights(physics["kz_halfwidth"], physics.get("kz_npoints", 7))
acc = None
for dki, wi in zip(dk, w):
    phys_i = dict(physics)
    # cheapest correct approach: shift K after build_k_bulk_mesh
    I = _run_once_with_kz_shift(phys_i, ..., kz_shift=float(dki))
    acc = I * wi if acc is None else acc + I * wi
return acc
```

Implement `_run_once_with_kz_shift` by factoring existing body so mesh build stays once where possible; if refactor risk high, accept N full runs for N k_z points (N≈7).

- [ ] **Step 4: pytest weights + existing chinook tests still green**

```bash
TensorSpec_env/bin/pytest tests/test_arpes_critic_gaps.py tests/test_bare_arpes_no_fermi_cutoff.py -v
```

---

### Task 4: GUI knobs + metadata fix

**Files:**
- Modify: `tensorspec/gui/components/arpes_panel.py`
  - `_setup_ui` (~param / beam groups)
  - `experiment_kwargs` local launch (~994)
  - remote `physics_from_experiment_kwargs` / JSON block (~875)
  - `get_simulation_metadata` (~1268)

**Interfaces:**
- Produces UI:
  - `rad_type_combo`: `slater | hydrogenic | grid` (grid may stay disabled/tooltip “needs rad_args” unless already supported)
  - `mfp_spin`: Å, default 10.0, range 0.5–50
  - `kz_halfwidth_spin`: Å⁻¹, default 0.0
  - `kz_npoints_spin`: odd int, default 7, enabled only if halfwidth > 0
- Metadata must include at least:

```python
{
  "crystal": ...,
  "engine": ...,
  "photon_energy": ...,
  "work_function": ...,
  "inner_potential": ...,
  "temperature": ...,
  "polarization": ...,
  "lin_pol_angle": ...,
  "incidence_angle": ...,
  "matrix_element_mode": ...,
  "manip_theta": ...,
  "manip_azimuth": ...,  # keep key name consistent with existing files: manip_azi OR manip_azimuth — prefer writing BOTH during transition
  "manip_tilt": ...,
  "slit_angle": ...,
  "hkl": ...,
  "rad_type": ...,
  "mfp": ...,
  "kz_halfwidth": ...,
  "kz_npoints": ...,
}
```

- [ ] **Step 1: Add widgets next to Inner Pot / Beam Incidence**

- [ ] **Step 2: Thread values into local `experiment_kwargs` and remote physics JSON**

- [ ] **Step 3: Expand `get_simulation_metadata()`** (this alone would have made beam55/beam20 auditable)

- [ ] **Step 4: Manual smoke** — change incidence 55→20, save npz, confirm metadata has both angles and differing intensities (optional if no TB at hand)

---

### Task 5: Remote runner / physics JSON parity

**Files:**
- Modify: `tensorspec/core/arpes/one_step/chinook_remote_runner_template.py` (read new physics keys; no CLI flags required if JSON already carries them)
- Modify: `scratch/chinook_gui_run/arpes_physics.json` only as example if present in docs — do not overwrite user scratch blindly

**Interfaces:**
- Remote `global_physics` / `physics` must forward `rad_type`, `mfp`, `kz_halfwidth`, `kz_npoints` into the same run entry points as local

- [ ] **Step 1: Grep runner for `physics[` / `global_physics` and ensure no whitelist drops new keys**

- [ ] **Step 2: If runner rebuilds `arpes_dict` separately, switch it to `build_arpes_dict_from_physics`**

- [ ] **Step 3: Add a tiny JSON round-trip test**

```python
def test_physics_json_roundtrip_keys():
    raw = {
        "photon_energy": 84.0,
        "rad_type": "hydrogenic",
        "mfp": 5.0,
        "kz_halfwidth": 0.2,
        "kz_npoints": 7,
        "incidence_angle": 20.0,
    }
    phys = physics_from_experiment_kwargs(raw)
    assert phys["incidence_angle"] == 20.0
    assert phys["rad_type"] == "hydrogenic"
```

---

### Task 6: VTe₂ re-run checklist (no code — verification doc in plan result)

After Tasks 1–5 land, user/agent re-runs four maps with:

| Knob | Value |
| --- | --- |
| hv | 84 eV |
| polarization | Linear Horizontal |
| incidence | 55° and 20° |
| slit | 0° and 90° |
| rad_type | `hydrogenic` (diff vs existing Slater maps) |
| mfp | 5.0 Å |
| kz_halfwidth | 0.2 Å⁻¹ |
| kz_npoints | 7 |
| Vo / inner_potential | measured or 10–12 eV (record in metadata) |

- [ ] Diff new vs old `beam{20,55}_slit{0,90}_*.npz`
- [ ] If hydrogenic+k_z still cannot suppress central branch → Approach C (Fresnel + q∥) justified; update Matrix Element Issue handoff

---

## Spec coverage self-check

| Critic / confirm item | Task |
| --- | --- |
| rad_type hydrogenic/grid | 1, 2, 4 |
| mfp not silent 10 Å | 1, 2, 4 |
| Vo vs V0 mismatch | 2 |
| metadata missing incidence | 4 |
| incoherent k_z sum | 3, 4, 5 |
| incidence path already exists | verified; tests in 1 |
| Fresnel / q∥ | **deferred (Approach C)** |
| H0/H1 / B remap | **out of scope** |

## Placeholder scan

No TBD steps. Approach C explicitly deferred, not stubbed as empty tasks.

---

## Execution handoff

Plan saved to `docs/superpowers/plans/2026-09-06-arpes-critic-gaps.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — this session with executing-plans checkpoints  

Which do you want?
