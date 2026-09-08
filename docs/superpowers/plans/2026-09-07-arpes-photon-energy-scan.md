# ARPES Photon-Energy Scan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Single|Range photon-energy input to ARPES Suite (Chinook/Grizzly only), outer-loop stack into 3D (kz-style dispersion) or 4D (Fermi×hv), with one Einstein GPU job for Range.

**Architecture:** Pure helpers build `hv_list` and stack single-hv cubes along axis 0 labeled `Photon Energy`. Local Chinook loops existing single-hv sim. Remote extends `chinook_remote_runner` to loop hv inside one SSH job (keep TB warm, rebuild ME shell per hv, same `--ngpus`/CUDA). `DataViewerPanel` unchanged; panel preview shows one hv slice.

**Tech Stack:** Python 3.11, NumPy, PyQt5/6 ARPES panel, Chinook/Grizzly remote runner, pytest, TensorData/workspace.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-07-arpes-photon-energy-scan-design.md`
- Branch: `TensorSpec_GUI` only for feature commits; do not touch `HTML_web_app`
- `main` already fast-forwarded to TensorSpec_GUI tip at plan start (`96e55bc`); do not force-push main unless user asks again
- hv-dependent scan = Chinook (Grizzly) only; SPR-KKR later (Range UI disabled for non-Chinook)
- Never reuse hoisted `me_shell` across different hv values
- Einstein Range = **one** SSH job (not N submits)
- Axis label for hv: `Photon Energy` (unit `eV`)
- Finish inclusive on step grid: `np.arange(start, finish + step/2, step)`
- Warn/confirm if `len(hv_list) > 64`
- Caveman chat OK; commits/docs = normal prose
- Do not commit unless user asks (except when a plan step says Commit and user already authorized plan execution)

## File map

| File | Role |
| --- | --- |
| `tensorspec/core/arpes/photon_energy_scan.py` | **Create:** `build_hv_list`, `stack_hv_cubes`, shape helpers |
| `tests/test_photon_energy_scan.py` | **Create:** unit tests for helpers + CLI parse |
| `tensorspec/gui/components/arpes_panel.py` | UI Single\|Range; gate; local loop; remote CLI; push/save/preview |
| `tensorspec/core/arpes/one_step/chinook_remote_runner_template.py` | `--hv_start/finish/step`; in-job loop; stacked npz |
| `tensorspec/core/workspace.py` | save/load N-D simulated ARPES with optional `Photon Energy` |
| `tensorspec/core/io/simulated_loader.py` | load 3D/4D stacked sim files |

## Confirmed facts (do not re-derive)

1. Today `photon_energy_spin` is scalar only (`arpes_panel.py` ~181, row ~207).
2. Dispersion vs cube: `ny==1 or nx==1 or _axis_degenerate` (`arpes_panel.py` ~1154–1183, `_axis_degenerate` ~1310).
3. Push today: transpose `(Θ,Φ,E)→(E,Θ,Φ)` labels `Energy`, `Θ (Slit)`, `Φ (Deflect)` (~1458–1467).
4. Remote npz keys: `cube, energy, theta, phi, …` (`chinook_remote_runner_template.py` ~911–926); single `--hv` (~582).
5. Workspace save keys: `intensity, kx, ky, E, metadata` (`workspace.py` ~115–121); load assumes 3D transpose `(2,0,1)`.
6. ME shell is hv-tied (`build_grizzly_me_shell`); must rebuild per hv.

---

### Task 1: Pure hv list + stack helpers (TDD)

**Files:**
- Create: `tensorspec/core/arpes/photon_energy_scan.py`
- Create: `tests/test_photon_energy_scan.py`

**Interfaces:**
- Consumes: NumPy only
- Produces:
  - `build_hv_list(start: float, finish: float, step: float) -> np.ndarray`
  - `stack_hv_cubes(cubes: list[np.ndarray], hv_list: np.ndarray) -> tuple[np.ndarray, np.ndarray]`
  - `is_dispersion_axes(theta: np.ndarray, phi: np.ndarray) -> bool`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_photon_energy_scan.py
import numpy as np
import pytest
from tensorspec.core.arpes.photon_energy_scan import (
    build_hv_list,
    stack_hv_cubes,
    is_dispersion_axes,
)


def test_build_hv_list_inclusive_finish():
    hv = build_hv_list(80.0, 90.0, 5.0)
    assert np.allclose(hv, [80.0, 85.0, 90.0])


def test_build_hv_list_rejects_bad_step():
    with pytest.raises(ValueError):
        build_hv_list(80.0, 90.0, 0.0)
    with pytest.raises(ValueError):
        build_hv_list(90.0, 80.0, 5.0)


def test_stack_dispersion_shapes_3d():
    # single-hv cube (ntheta, nphi=1, ne)
    cubes = [np.ones((10, 1, 5)) * i for i in range(3)]
    hv = np.array([80.0, 85.0, 90.0])
    stacked, hv_out = stack_hv_cubes(cubes, hv)
    assert stacked.shape == (3, 10, 1, 5)
    assert np.allclose(hv_out, hv)
    assert stacked[2, 0, 0, 0] == 2.0


def test_stack_fermi_shapes_4d_raw():
    cubes = [np.ones((8, 9, 4)) * i for i in range(2)]
    hv = np.array([84.0, 90.0])
    stacked, _ = stack_hv_cubes(cubes, hv)
    assert stacked.shape == (2, 8, 9, 4)


def test_is_dispersion_degenerate_phi():
    theta = np.linspace(-15, 15, 31)
    phi = np.array([0.0])
    assert is_dispersion_axes(theta, phi) is True
    phi2 = np.linspace(-10, 10, 21)
    assert is_dispersion_axes(theta, phi2) is False
```

- [ ] **Step 2: Run tests — expect FAIL** (module missing)

```bash
cd /Users/sandyai/Documents/GitHub/TensorSpec_GUI
TensorSpec_env/bin/pytest tests/test_photon_energy_scan.py -v
```

Expected: `ModuleNotFoundError` or import error.

- [ ] **Step 3: Implement helpers**

```python
# tensorspec/core/arpes/photon_energy_scan.py
"""Photon-energy scan helpers for Chinook/Grizzly ARPES stacking."""
from __future__ import annotations

import numpy as np


def build_hv_list(start: float, finish: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    if finish < start:
        raise ValueError(f"finish ({finish}) < start ({start})")
    hv = np.arange(float(start), float(finish) + float(step) / 2.0, float(step))
    if hv.size == 0:
        raise ValueError("hv list empty")
    return hv.astype(float)


def stack_hv_cubes(
    cubes: list[np.ndarray], hv_list: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if len(cubes) == 0:
        raise ValueError("no cubes to stack")
    if len(cubes) != len(hv_list):
        raise ValueError("cubes and hv_list length mismatch")
    stacked = np.stack([np.asarray(c) for c in cubes], axis=0)
    return stacked, np.asarray(hv_list, dtype=float)


def is_dispersion_axes(theta: np.ndarray, phi: np.ndarray, tol: float = 1e-9) -> bool:
    def _deg(a: np.ndarray) -> bool:
        a = np.asarray(a)
        return a.size <= 1 or float(np.ptp(a)) < tol

    return _deg(theta) or _deg(phi)
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
TensorSpec_env/bin/pytest tests/test_photon_energy_scan.py -v
```

- [ ] **Step 5: Commit** (when user authorizes commits)

```bash
git add tensorspec/core/arpes/photon_energy_scan.py tests/test_photon_energy_scan.py
git commit -m "$(cat <<'EOF'
feat(arpes): add photon-energy list and cube stack helpers

EOF
)"
```

---

### Task 2: ARPES panel UI — Single | Range + Chinook gate

**Files:**
- Modify: `tensorspec/gui/components/arpes_panel.py` (~181–207 UI; ~1553 `_sync_remote_ui` / engine change hooks)
- Test: extend `tests/test_photon_energy_scan.py` with pure helpers used by panel (prefer extracting getters so Qt not required)

**Interfaces:**
- Consumes: `build_hv_list`
- Produces: `get_photon_energies() -> list[float]` (len 1 in Single); Range widgets; `update_hv_mode_ui()` disables Range unless Chinook

- [ ] **Step 1: Write failing tests for resolution helper** (no Qt)

```python
# add to tests/test_photon_energy_scan.py
from tensorspec.core.arpes.photon_energy_scan import resolve_photon_energies


def test_resolve_single():
    assert resolve_photon_energies(mode="single", single=90.0) == [90.0]


def test_resolve_range():
    out = resolve_photon_energies(
        mode="range", start=80.0, finish=90.0, step=5.0
    )
    assert out == [80.0, 85.0, 90.0]
```

Add to `photon_energy_scan.py`:

```python
def resolve_photon_energies(
    *,
    mode: str,
    single: float | None = None,
    start: float | None = None,
    finish: float | None = None,
    step: float | None = None,
) -> list[float]:
    if mode == "single":
        if single is None:
            raise ValueError("single hv required")
        return [float(single)]
    if mode == "range":
        return build_hv_list(start, finish, step).tolist()
    raise ValueError(f"unknown hv mode: {mode}")
```

- [ ] **Step 2: Run test — FAIL then implement `resolve_photon_energies` — PASS**

- [ ] **Step 3: Wire UI in `arpes_panel.py`**

Replace single row `"Photon (hv):"` with:

1. `self.hv_mode_combo = QComboBox()` items `Single`, `Range` (userData `"single"` / `"range"`).
2. Keep `self.photon_energy_spin` for Single.
3. Add `self.hv_start_spin`, `self.hv_finish_spin`, `self.hv_step_spin` (defaults e.g. 80 / 100 / 5).
4. Put Range spins in a small `QHBoxLayout` widget `self.hv_range_widget`; hide by default.
5. `hv_mode_combo.currentIndexChanged` → toggle spin vs range widget visibility.
6. In `on_engine_changed` / `_sync_remote_ui`: if model is not Chinook path (B1 / B2 as used for Chinook TB — **Range only when `engine_dropdown` data is B1 or the Chinook-capable selection used today for ME sims**), disable Range item and force Single. Tooltip: `"hv-dependent scan = Chinook (Grizzly) only. SPR-KKR later."`

Exact Chinook gate (match existing router):

```python
def _hv_scan_allowed(self) -> bool:
    # B1 Chinook TB; B2 bare still uses chinook_wrapper path in panel — allow only
    # engines that call Chinook/Grizzly ME (B1). Exclude A (three-step) and B3 (SPR-KKR).
    return self.engine_dropdown.currentData() == "B1"
```

If product intent was also B2, still gate to **B1 only** per spec “Chinook (Grizzly) only”.

7. Add method:

```python
def get_photon_energies(self) -> list[float]:
    mode = self.hv_mode_combo.currentData()
    if mode == "range":
        return resolve_photon_energies(
            mode="range",
            start=self.hv_start_spin.value(),
            finish=self.hv_finish_spin.value(),
            step=self.hv_step_spin.value(),
        )
    return resolve_photon_energies(
        mode="single", single=self.photon_energy_spin.value()
    )
```

8. Before run: if `len(hv_list) > 64`, `QMessageBox.question` confirm; abort on No.

- [ ] **Step 4: Manual smoke** — launch GUI, toggle Single/Range; switch engine to B3 → Range disabled.

- [ ] **Step 5: Commit** (when authorized)

```bash
git commit -m "$(cat <<'EOF'
feat(arpes): add Single|Range photon-energy UI with Chinook gate

EOF
)"
```

---

### Task 3: Local Chinook Range loop + workspace push/save

**Files:**
- Modify: `tensorspec/gui/components/arpes_panel.py` (`experiment_kwargs` ~1078; `ARPESRunnerThread` / finish ~1111; `push_arpes_to_workspace` ~1452; `save_arpes_to_disk` ~1475; `get_simulation_metadata` ~1354)
- Modify: `tensorspec/core/workspace.py` (`save_simulated_arpes`, `load_simulated_arpes_to_tensor`)
- Modify: `tensorspec/core/io/simulated_loader.py`
- Test: `tests/test_photon_energy_scan.py` (stack→TensorData label contract without full TB)

**Interfaces:**
- Consumes: `get_photon_energies`, `stack_hv_cubes`, existing `engine_router.run_simulation`
- Produces: `sim_intensity` shape `(Nhv, nθ, nφ, nE)` for Range; metadata `photon_energies`; TensorData with leading `Photon Energy` axis

- [ ] **Step 1: Failing test — TensorData axis contract helper**

```python
# tests/test_photon_energy_scan.py
from tensorspec.core.arpes.photon_energy_scan import tensor_from_stacked_sim


def test_tensor_from_stacked_fermi():
    # stacked raw (Nhv, nθ, nφ, nE)
    stacked = np.zeros((2, 3, 4, 5))
    hv = np.array([80.0, 90.0])
    theta = np.linspace(-1, 1, 3)
    phi = np.linspace(-2, 2, 4)
    energy = np.linspace(-1, 0, 5)
    td = tensor_from_stacked_sim(stacked, hv, theta, phi, energy)
    assert td.value.shape == (2, 5, 3, 4)  # (hv, E, Θ, Φ)
    assert td.labels[0] == "Photon Energy"
    assert td.labels[1:] == ["Energy", "Θ (Slit)", "Φ (Deflect)"]
```

Implement:

```python
def tensor_from_stacked_sim(stacked, hv, theta, phi, energy, metadata=None):
    from tensorspec.core.data_models import TensorData

    # stacked (Nhv, nθ, nφ, nE) -> (Nhv, nE, nθ, nφ)
    value = np.transpose(stacked, (0, 3, 1, 2))
    return TensorData(
        value=value,
        axes=[hv, energy, theta, phi],
        labels=["Photon Energy", "Energy", "Θ (Slit)", "Φ (Deflect)"],
        units=["eV", "eV", "deg", "deg"],
        data_type="Simulated ARPES Matrix Elements",
        metadata=metadata or {},
    )
```

For **single** hv keep legacy transpose `(nθ,nφ,nE)→(E,Θ,Φ)` (no Photon Energy axis) for backward compatibility.

- [ ] **Step 2: Local run path**

In `trigger_simulation` local Chinook branch:

```python
hv_list = self.get_photon_energies()
if len(hv_list) > 1:
    # confirm >64 already done
    # Run sequential simulations; prefer one thread that loops internally
    ...
```

Preferred: extend `ARPESRunnerThread` to accept `photon_energies: list[float] | None`. If multiple:

```python
cubes = []
for i, hv in enumerate(photon_energies):
    kw = dict(experiment_kwargs)
    kw["photon_energy"] = float(hv)
    kw["photon_energies"] = photon_energies
    result = engine_router.run_simulation(..., experiment_kwargs=kw)
    cubes.append(result["intensity_broadened"])
    self.progress.emit(f"hv {i+1}/{len(photon_energies)} ({hv} eV)")
stacked, hv_arr = stack_hv_cubes(cubes, np.asarray(photon_energies))
# emit stacked result dict
```

`on_simulation_finished`:

- If intensity ndim == 4 and leading dim = Nhv: store `sim_hv`, `sim_intensity` as stacked raw `(Nhv,nθ,nφ,nE)`; panel preview uses mid or last hv index `sim_intensity[ihv]` with existing dispersion/cube plot logic.
- Else: legacy 3D path unchanged.

`experiment_kwargs` / metadata:

```python
"photon_energy": float(hv_list[0]),
"photon_energies": hv_list,
```

- [ ] **Step 3: `push_arpes_to_workspace` / save**

```python
if getattr(self, "sim_hv", None) is not None and self.sim_intensity.ndim == 4:
    sim_tensor = tensor_from_stacked_sim(
        self.sim_intensity, self.sim_hv, self.sim_kx, self.sim_ky, self.sim_E_axis,
        metadata=self.get_simulation_metadata(),
    )
else:
    # existing transpose path
    ...
```

Extend `save_simulated_arpes` to accept optional `hv=None`. If `hv` given, save key `hv=` and intensity as stacked; update `load_simulated_arpes_to_tensor` and `SimulatedARPESLoader.load` to detect `hv` key → 4D TensorData with `Photon Energy` first.

```python
# workspace save when hv present
np.savez_compressed(
    file_path,
    intensity=intensity,  # (Nhv, nθ, nφ, nE) or legacy (nθ, nφ, nE)
    kx=kx, ky=ky, E=E,
    hv=hv,
    metadata=metadata or {},
)
```

Loader:

```python
if "hv" in data:
    stacked = data["intensity"]
    value = np.transpose(stacked, (0, 3, 1, 2))
    return TensorData(
        value=value,
        axes=[data["hv"], data["E"], data["kx"], data["ky"]],
        labels=["Photon Energy", "Energy", "Θ (Slit)", "Φ (Deflect)"],
        units=["eV", "eV", "deg", "deg"],
        ...
    )
# else legacy transpose
```

- [ ] **Step 4: Run unit tests**

```bash
TensorSpec_env/bin/pytest tests/test_photon_energy_scan.py -v
```

- [ ] **Step 5: Commit** (when authorized)

```bash
git commit -m "$(cat <<'EOF'
feat(arpes): local Chinook hv scan stack and N-D workspace IO

EOF
)"
```

---

### Task 4: Remote runner — one-job hv loop + stacked npz (GPU)

**Files:**
- Modify: `tensorspec/core/arpes/one_step/chinook_remote_runner_template.py` (argparse ~582; physics hv ~718; compute+save ~745–926)
- Test: `tests/test_photon_energy_scan.py` (argparse + loop orchestration with mocks)

**Interfaces:**
- Consumes: existing `run_grizzly_arpes` / `run_chinook_arpes` per hv
- Produces: stacked `cube` shape `(Nhv, ntheta, nphi, ne)` + `hv` axis in npz; keep GPU/`--ngpus` path

- [ ] **Step 1: Failing tests for CLI hv resolution**

```python
# tests/test_photon_energy_scan.py
from tensorspec.core.arpes.photon_energy_scan import hv_list_from_cli_args


class _Args:
    def __init__(self, hv=90.0, hv_start=None, hv_finish=None, hv_step=None):
        self.hv = hv
        self.hv_start = hv_start
        self.hv_finish = hv_finish
        self.hv_step = hv_step


def test_cli_single_hv():
    assert hv_list_from_cli_args(_Args(hv=84.0)) == [84.0]


def test_cli_range_overrides_single():
    out = hv_list_from_cli_args(
        _Args(hv=90.0, hv_start=80.0, hv_finish=90.0, hv_step=5.0)
    )
    assert out == [80.0, 85.0, 90.0]
```

```python
def hv_list_from_cli_args(args) -> list[float]:
    if (
        getattr(args, "hv_start", None) is not None
        and getattr(args, "hv_finish", None) is not None
        and getattr(args, "hv_step", None) is not None
    ):
        return build_hv_list(args.hv_start, args.hv_finish, args.hv_step).tolist()
    return [float(args.hv)]
```

- [ ] **Step 2: Add argparse flags**

```python
parser.add_argument("--hv", type=float, default=90.0)
parser.add_argument("--hv_start", type=float, default=None)
parser.add_argument("--hv_finish", type=float, default=None)
parser.add_argument("--hv_step", type=float, default=None)
```

- [ ] **Step 3: Wrap compute in hv loop**

Pseudo-structure (preserve existing full/slices GPU code **inside** each iteration):

```python
from tensorspec.core.arpes.photon_energy_scan import (
    build_hv_list,
    hv_list_from_cli_args,
    stack_hv_cubes,
)

hv_values = hv_list_from_cli_args(args)
cubes = []
for i, hv in enumerate(hv_values):
    print(f"=== Photon energy {i+1}/{len(hv_values)}: {hv} eV ===", flush=True)
    args.hv = float(hv)  # or local variable
    physics["hv"] = float(hv)
    # rebuild e_kin / k mesh inputs that depend on hv
    # IMPORTANT: do not pass previous me_shell into next hv
    cube_i = <existing single-hv compute path returning cube>
    cubes.append(cube_i)

stacked, hv_arr = stack_hv_cubes(cubes, np.asarray(hv_values))
np.savez_compressed(
    args.out_file,
    cube=stacked,           # (Nhv, nθ, nφ, nE) if Nhv>1 else squeeze? Prefer ALWAYS include hv axis when Nhv>=1 for simplicity when Nhv>1; when Nhv==1 keep legacy 3D cube shape for back-compat
    hv=hv_arr if len(hv_values) > 1 else None,
    energy=e_axis,
    theta=thetas,
    phi=phis,
    ...
)
```

**Back-compat rule:** If `len(hv_values)==1`, save `cube` as 3D exactly as today (no `hv` key). If `len>1`, save 4D `cube` + `hv` array.

Multi-GPU: keep parent prewarm **per hv** (or clear shell between hv). Do not hoist one shell across the hv loop.

- [ ] **Step 4: Unit-test argparse registration** (import parser builder if extracted; else test `hv_list_from_cli_args` only + a thin `_run_hv_loop` with mocked compute).

```bash
TensorSpec_env/bin/pytest tests/test_photon_energy_scan.py -v
```

- [ ] **Step 5: Commit** (when authorized)

```bash
git commit -m "$(cat <<'EOF'
feat(arpes): remote Chinook runner hv loop in one GPU job

EOF
)"
```

---

### Task 5: GUI remote submit + fetch stacked cube

**Files:**
- Modify: `tensorspec/gui/components/arpes_panel.py` (remote `run_args` ~996–1020; fetch ~1382–1431)

**Interfaces:**
- Consumes: `get_photon_energies`, remote npz with optional `hv`
- Produces: same `on_simulation_finished` path as local stacked

- [ ] **Step 1: Extend remote CLI construction**

```python
hv_list = self.get_photon_energies()
if len(hv_list) == 1:
    hv_args = f"--hv {hv_list[0]}"
else:
    hv_args = (
        f"--hv_start {self.hv_start_spin.value()} "
        f"--hv_finish {self.hv_finish_spin.value()} "
        f"--hv_step {self.hv_step_spin.value()}"
    )
run_args = f"... {hv_args} ... --engine {me_engine} ... --ngpus {me_ngpus} ..."
```

Still **one** sbatch/nohup. Do not loop SSH submits.

- [ ] **Step 2: Fetch path**

When loading `chinook_arpes_cube.npz`:

```python
cube = data["cube"]
if "hv" in data and cube.ndim == 4:
    self.sim_hv = np.asarray(data["hv"], dtype=float)
    # intensity_broadened = cube  # (Nhv, nθ, nφ, nE)
else:
    self.sim_hv = None
# pass into on_simulation_finished
```

- [ ] **Step 3: Manual checklist (Einstein)**

1. Small Range: start=84, finish=90, step=6 → 2 points; tiny θ/φ/E grid.
2. Hybrid + Grizzly + CUDA + ngpus≥1.
3. Confirm log shows two `=== Photon energy ... ===` lines in **one** job.
4. Fetch → DataViewer sees `Photon Energy` axis.

- [ ] **Step 4: Commit** (when authorized)

```bash
git commit -m "$(cat <<'EOF'
feat(arpes): wire remote hv-range submit and stacked fetch

EOF
)"
```

---

### Task 6: Spec coverage smoke + docs note

**Files:**
- Modify: `docs/superpowers/specs/2026-09-07-arpes-photon-energy-scan-design.md` (status → Implemented) only if user wants; otherwise skip
- Test: full `tests/test_photon_energy_scan.py`

- [ ] **Step 1: Run full related tests**

```bash
cd /Users/sandyai/Documents/GitHub/TensorSpec_GUI
TensorSpec_env/bin/pytest tests/test_photon_energy_scan.py tests/test_arpes_critic_gaps.py tests/test_grizzly_shell_prewarm.py -v
```

Expected: all PASS.

- [ ] **Step 2: Verify constraints checklist**

- [ ] Single mode unchanged (3D cube, no hv axis)
- [ ] Range + degenerate φ → stacked with Photon Energy; viewer OK
- [ ] Range + Fermi → 4D
- [ ] B3 Range disabled + tooltip mentions SPR-KKR later
- [ ] Remote one job only
- [ ] HTML branch untouched

- [ ] **Step 3: Commit** only if doc status update requested

---

## Self-review (plan vs spec)

| Spec requirement | Task |
| --- | --- |
| Single \| Range UI | Task 2 |
| Inclusive finish hv list | Task 1 |
| Chinook-only gate + SPR-KKR later note | Task 2 |
| Outer loop local stack | Task 3 |
| Dispersion 3D / Fermi 4D with Photon Energy | Task 1+3 |
| One Einstein GPU job loop | Task 4+5 |
| DataViewer via TensorData push | Task 3 |
| Panel preview one hv slice | Task 3 |
| IO save/load stacked | Task 3 |
| Tests listed in spec | Tasks 1–4 |

No TBD placeholders. Types consistent: stacked raw `(Nhv,nθ,nφ,nE)` → viewer `(Nhv,nE,nθ,nφ)`.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-07-arpes-photon-energy-scan.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — execute in this session with executing-plans checkpoints  

Which approach?
