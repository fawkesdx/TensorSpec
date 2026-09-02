# Maestro ARPES Loader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Plugin-based Maestro HDF5 loader that parses `Low_Level_Scan`, packs `TensorData`, and is shared by ARPES suite, main browser, and ML suite Load buttons.

**Architecture:** Keep `ARPESLoader` facade. Replace monolithic `MaestroLoader` with `tensorspec/core/io/loaders/maestro/` package: `ScanPlan` parser, detector helpers, ordered kind registry (`xy_fine_4d`, `focus_xy_fine_5d`, `fermi_defl_3d`, `process000_generic`). GUI entry points call the facade only.

**Tech Stack:** Python 3.11, h5py, numpy, PySide6, pytest, TensorSpec `TensorData` / `global_workspace`.

**Spec:** `docs/superpowers/specs/2026-09-02-maestro-arpes-loader-design.md`

## Global Constraints

- Dim source of truth: `Headers/Low_Level_Scan` loops only (not high-level UI Scan tree).
- Support datasets `Fixed_Spectra1` and `Process_000`.
- Axis order for spatial kinds: greeting mesh on dims 0,1 (`Y`, `X`) so `DataViewerPanel.spawn_view(x=1,y=0)` shows XY map.
- Open SMB HDF5 with locking disabled (`os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")` before `h5py.File`).
- Do not wire crystal/DFT/XAS/PEEM loaders.
- Commits only when the user asks, or as part of an explicit commit step the user already approved for this plan run; never push.
- Prefer cheap/fast implementer models for surgical tasks; main thread reviews.

---

## File structure (locked)

```
tensorspec/core/io/loaders/maestro/
  __init__.py                 # export MaestroLoader facade class
  types.py                    # ScanMotor, ScanLoop, ScanPlan dataclasses
  low_level_scan.py           # parse Headers/Low_Level_Scan → ScanPlan
  detect.py                   # is_maestro, pick 2D dataset, is_fixed
  detector.py                 # energy/angle axes from dataset attrs
  reshape.py                  # shared reshape/abort helpers
  registry.py                 # ordered kind match + load
  kinds/
    __init__.py
    xy_fine_4d.py
    focus_xy_fine_5d.py
    fermi_defl_3d.py
    process000_generic.py
tensorspec/core/io/loaders/maestro_loader.py   # thin re-export of MaestroLoader
tensorspec/core/io/arpes_loader.py             # pack units from labels/units lists
tests/test_maestro_low_level_scan.py
tests/test_maestro_kinds.py
tests/test_arpes_loader_maestro.py
```

---

### Task 1: `ScanPlan` types + `Low_Level_Scan` parser

**Files:**
- Create: `tensorspec/core/io/loaders/maestro/types.py`
- Create: `tensorspec/core/io/loaders/maestro/low_level_scan.py`
- Create: `tensorspec/core/io/loaders/maestro/__init__.py` (minimal exports)
- Test: `tests/test_maestro_low_level_scan.py`

**Interfaces:**
- Produces:
  - `@dataclass ScanMotor`: `name: str`, `units: str`, `start: float`, `end: float`, `n: int`
  - `@dataclass ScanLoop`: `motors: list[ScanMotor]`
  - `@dataclass ScanPlan`: `mode_name: str`, `loops: list[ScanLoop]`, `parallel: bool`
  - `ScanPlan.expected_cycles: int` property = product of all motor `n`
  - `parse_low_level_scan(header_table) -> ScanPlan` where `header_table` is iterable of rows `(longname, name, value, comment)` (bytes or str OK)
  - Helpers used by kinds: `plan.has_xy_mesh() -> bool`, `plan.angle_motors() -> list[ScanMotor]`, `plan.xy_motors() -> tuple[ScanMotor, ScanMotor] | None`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_maestro_low_level_scan.py
from tensorspec.core.io.loaders.maestro.low_level_scan import parse_low_level_scan

def _row(tag, value, comment=""):
    return (tag, tag, value, comment)

def test_parse_xy_fine_one_loop():
    rows = [
        _row("lwlvnm", "'XY Scan Fine'"),
        _row("scanpar", "F"),
        _row("lwlvlpn", "1"),
        _row("scntyp0", "0"),
        _row("devnm_0", "'motors'"),
        _row("nmsbdv0", "2"),
        _row("nm_0_0", "'Scan X'"),
        _row("un_0_0", "'um'"),
        _row("nm_0_1", "'Scan Y'"),
        _row("un_0_1", "'um'"),
        _row("st_0_0", "-40"),
        _row("en_0_0", "40"),
        _row("n_0_0", "81"),
        _row("st_0_1", "-40"),
        _row("en_0_1", "40"),
        _row("n_0_1", "81"),
    ]
    plan = parse_low_level_scan(rows)
    assert plan.mode_name == "XY Scan Fine"
    assert len(plan.loops) == 1
    assert plan.expected_cycles == 81 * 81
    assert plan.has_xy_mesh()

def test_parse_focus_xy_fine_two_loops():
    rows = [
        _row("lwlvnm", "'Focus XY Fine'"),
        _row("scanpar", "F"),
        _row("lwlvlpn", "2"),
        _row("nmsbdv0", "1"),
        _row("nm_0_0", "'Slit Defl.'"),
        _row("un_0_0", "'Deg'"),
        _row("st_0_0", "-8.5"),
        _row("en_0_0", "7.5"),
        _row("n_0_0", "17"),
        _row("nmsbdv1", "2"),
        _row("nm_1_0", "'Scan X'"),
        _row("un_1_0", "'um'"),
        _row("nm_1_1", "'Scan Y'"),
        _row("un_1_1", "'um'"),
        _row("st_1_0", "-9.9"),
        _row("en_1_0", "6.0"),
        _row("n_1_0", "81"),
        _row("st_1_1", "-2.7"),
        _row("en_1_1", "13.3"),
        _row("n_1_1", "81"),
    ]
    plan = parse_low_level_scan(rows)
    assert plan.expected_cycles == 17 * 81 * 81
    assert len(plan.angle_motors()) == 1
    assert plan.angle_motors()[0].name == "Slit Defl."
```

- [ ] **Step 2: Run tests — expect FAIL (import/missing)**

```bash
TensorSpec_env/bin/python -m pytest tests/test_maestro_low_level_scan.py -v
```

- [ ] **Step 3: Implement `types.py` + `low_level_scan.py`**

Parse rules:
- Decode bytes; strip quotes from string values.
- Read `LWLVNM` / `lwlvnm` → `mode_name`.
- Read `LWLVLPN` → number of loops.
- For each loop `i`: `NMSBDVi` motor count; for each `j`: `NM_i_j`, `UN_i_j`, `ST_i_j`, `EN_i_j`, `N_i_j`.
- `has_xy_mesh`: any loop with motors named like `Scan X`/`Scan Y` (case-insensitive, allow `Sample X`/`Sample Y`).
- `angle_motors`: motors whose names match Defl/Deflection/Theta/Tilt/Phi (not X/Y).

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit only if user asked for commits this run; otherwise leave staged note in ledger**

---

### Task 2: Detect + detector axes helpers

**Files:**
- Create: `tensorspec/core/io/loaders/maestro/detect.py`
- Create: `tensorspec/core/io/loaders/maestro/detector.py`
- Modify: `tests/test_maestro_low_level_scan.py` (or new `tests/test_maestro_detect.py`)

**Interfaces:**
- Consumes: open `h5py.File`
- Produces:
  - `assert_maestro_signature(f)` → raises `ValueError` if missing `0D_Data`/`2D_Data`/`Headers`
  - `select_spectra_dataset(f) -> h5py.Dataset` — prefer `2D_Data/Process_000`, else first of `Fixed_Spectra*`, else first dataset in `2D_Data`
  - `is_fixed_mode(f) -> bool` — `DAQ_Fixed` present (and not only Swept)
  - `detector_axes(dataset, is_fixed: bool) -> tuple[np.ndarray, np.ndarray, str, str]` → `(energy, angle, energy_unit, angle_unit)` from attrs `unitNames`/`scaleOffset`/`scaleDelta`; if fixed and units are eV/pixels, map pixels→Angle with linspace centered (placeholder OK); if attrs missing, fall back to length-based linspace

- [ ] **Step 1: Failing tests with a tiny in-memory or tempfile H5**

```python
import h5py, numpy as np, tempfile, os
from tensorspec.core.io.loaders.maestro.detect import assert_maestro_signature, select_spectra_dataset
from tensorspec.core.io.loaders.maestro.detector import detector_axes

def test_select_fixed_spectra(tmp_path):
    p = tmp_path / "t.h5"
    with h5py.File(p, "w") as f:
        f.create_group("0D_Data")
        f.create_group("Headers")
        g = f.create_group("2D_Data")
        ds = g.create_dataset("Fixed_Spectra1", data=np.zeros((10, 20, 4), dtype=np.int32))
        ds.attrs["unitNames"] = [b"eV", b"pixels"]
        ds.attrs["scaleOffset"] = [0.0, 0.0]
        ds.attrs["scaleDelta"] = [0.1, 1.0]
    with h5py.File(p, "r") as f:
        assert_maestro_signature(f)
        ds = select_spectra_dataset(f)
        assert ds.name.endswith("Fixed_Spectra1")
        e, a, eu, au = detector_axes(ds, is_fixed=True)
        assert len(e) == 10 and len(a) == 20
```

- [ ] **Step 2: Implement detect.py + detector.py**

- [ ] **Step 3: pytest PASS**

---

### Task 3: Registry + `xy_fine_4d` kind

**Files:**
- Create: `tensorspec/core/io/loaders/maestro/reshape.py`
- Create: `tensorspec/core/io/loaders/maestro/registry.py`
- Create: `tensorspec/core/io/loaders/maestro/kinds/__init__.py`
- Create: `tensorspec/core/io/loaders/maestro/kinds/xy_fine_4d.py`
- Test: `tests/test_maestro_kinds.py`

**Interfaces:**
- Kind module exports:
  - `KIND_ID = "xy_fine_4d"`
  - `match(plan: ScanPlan, is_fixed: bool) -> bool` — True when `len(loops)==1` and `has_xy_mesh()` and not extra angle loop
  - `load(f, plan, dataset) -> dict` with keys:
    - `data`: ndarray shape `(nY, nX, nE, nA)`
    - `axes`: **list** aligned to dims OR dict with insertion-order `Y`,`X`,`Energy`,`Angle` (facade must accept both — prefer **lists** + `labels` + `units` in dict for new kinds)
    - Prefer return shape for new kinds:
      ```python
      {
        "data": value,
        "labels": ["Y", "X", "Energy", "Angle"],
        "axes": [y, x, e, a],
        "units": ["um", "um", "eV", "deg"],
        "mode": plan.mode_name,
        "is_fixed": True,
        "facility": "MAESTRO",
        "metadata": {"scan_plan": ..., "source_path": ...},
      }
      ```
- `reshape.py`: `points_axis(shape, n_points) -> int`; `load_spectra_buffer(dataset)`; abort if `actual < expected` → truncate warning string
- `registry.py`: `KIND_MODULES` ordered list; `match_kind(plan, is_fixed)`; `load_with_kind(f, path) -> dict`

**Reshape rule for Fixed `(*, *, n_points)` or `(n_points, *, *)`:** find points axis; reshape to `(nY, nX, nE, nA)` using plan `n` for X/Y; build Y/X axes via `np.linspace(start, end, n)`.

- [ ] **Step 1: Failing test builds minimal Fixed XY H5 (2×3 mesh, tiny detector) and asserts labels/order/shape**

- [ ] **Step 2: Implement xy_fine_4d + registry skeleton (only this kind registered for now)**

- [ ] **Step 3: pytest PASS**

---

### Task 4: `focus_xy_fine_5d` kind

**Files:**
- Create: `tensorspec/core/io/loaders/maestro/kinds/focus_xy_fine_5d.py`
- Modify: `registry.py` — register after checking more-specific first (5D before 4D)
- Modify: `tests/test_maestro_kinds.py`

**Interfaces:**
- `match`: `len(loops)==2`, one loop single angle motor, other loop XY mesh
- `load` → shape `(nY, nX, nDefl, nE, nA)` labels `["Y","X",<DeflName>,"Energy","Angle"]`
- Loop order in file may be Defl then XY (as in `_00742`); transpose so Y,X are dims 0,1

- [ ] **Step 1: Failing test with 2×2 XY × 3 Defl × small detector; expected_cycles=12**

- [ ] **Step 2: Implement**

- [ ] **Step 3: pytest PASS**

---

### Task 5: `fermi_defl_3d` kind

**Files:**
- Create: `tensorspec/core/io/loaders/maestro/kinds/fermi_defl_3d.py`
- Modify: `registry.py`
- Modify: `tests/test_maestro_kinds.py`

**Interfaces:**
- `match`: single loop, exactly one angle motor, no XY mesh
- `load` → `(nDefl, nE, nA)` labels `[DeflName, "Energy", "Angle"]`

- [ ] **Step 1–3:** TDD as above

---

### Task 6: `process000_generic` + `MaestroLoader` facade class

**Files:**
- Create: `tensorspec/core/io/loaders/maestro/kinds/process000_generic.py`
- Create/replace: `tensorspec/core/io/loaders/maestro/__init__.py` exporting `MaestroLoader`
- Replace body of `tensorspec/core/io/loaders/maestro_loader.py` with:

```python
from tensorspec.core.io.loaders.maestro import MaestroLoader
__all__ = ["MaestroLoader"]
```

**Interfaces:**
- `class MaestroLoader:`
  - `__init__(self, filepath: str)`
  - `load(self) -> dict` — sets HDF5 locking env; opens file; `assert_maestro_signature`; parse plan if `Low_Level_Scan` present; `registry.load_with_kind`; on Fixed+plan miss → raise `ValueError` with cycles detail; if `Process_000` and no kind match → `process000_generic` (port logic from old unique-motor reshape)
- Keep return keys compatible with current `ARPESLoader` pack **and** new list form:
  - If `labels`/`axes` lists present, facade uses those
  - Else legacy `axes` dict

- [ ] **Step 1: Port old Process_000 reshape into `process000_generic.py` (preserve aborted-scan + fixed transpose behavior)**

- [ ] **Step 2: Wire `MaestroLoader.load` end-to-end**

- [ ] **Step 3: Test `process000_generic` with tempfile Process_000 mock**

- [ ] **Step 4: pytest `tests/test_maestro_kinds.py tests/test_maestro_low_level_scan.py` PASS**

---

### Task 7: Upgrade `ARPESLoader` packing

**Files:**
- Modify: `tensorspec/core/io/arpes_loader.py`
- Test: `tests/test_arpes_loader_maestro.py`

**Interfaces:**
- Consumes loader dict with either:
  - New: `data`, `labels`, `axes` (list), `units` (list), `mode`, `metadata`
  - Legacy: `data`, `axes` (dict), `mode`
- Pack:
  ```python
  if "labels" in raw_dict:
      labels = raw_dict["labels"]
      axes = raw_dict["axes"]
      units = raw_dict.get("units") or heuristic(labels)
  else:
      axes_dict = raw_dict["axes"]
      labels = list(axes_dict.keys())
      axes = list(axes_dict.values())
      units = heuristic(labels)
  metadata = dict(raw_dict.get("metadata") or {})
  metadata.setdefault("facility", raw_dict.get("facility"))
  metadata.setdefault("is_fixed", raw_dict.get("is_fixed"))
  return TensorData(value=..., axes=..., labels=..., units=..., data_type=raw_dict.get("mode","ARPES"), metadata=metadata)
  ```
- Keep chain `[MockDataLoader, MaestroLoader]`

- [ ] **Step 1: Failing test: tempfile Fixed XY → `ARPESLoader.load` → `TensorData.ndim==4` and labels `["Y","X","Energy","Angle"]`**

- [ ] **Step 2: Implement pack upgrade**

- [ ] **Step 3: pytest PASS**

---

### Task 8: Main browser disk Load

**Files:**
- Modify: `tensorspec/gui/main_browser.py` (`init_central_layout` ~317–327; new method `load_workspace_files`)

**Interfaces:**
- Button label: `Load ARPES / Maestro Data...`
- Handler mirrors `arpes_suite.load_arpes_files`: QFileDialog multi; `.npz`→`SimulatedARPESLoader`; else `ARPESLoader`; `global_workspace.push_spectroscopy_data`; `refresh_workspace_tree`; QMessageBox on errors

- [ ] **Step 1: Add button + handler**

- [ ] **Step 2: Smoke import check**

```bash
TensorSpec_env/bin/python -c "from tensorspec.gui.main_browser import MainBrowser; print('ok')"
```

---

### Task 9: ML suite uses shared facade

**Files:**
- Modify: `tensorspec/gui/maestroai/maestro_loader.py` — `LoadWorker` only
- Modify: `tensorspec/gui/maestroai/maestroai_gui.py` — `request_load`, `on_load_finish`, `activate_data`, `_convert_to_tensor_data`

**Interfaces:**
- `LoadWorker(__init__(path, var_name))` — drop `process_mode`
- `run()`:
  ```python
  os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
  self.progress.emit(10, "Loading via ARPESLoader...")
  from tensorspec.core.io.arpes_loader import ARPESLoader
  td = ARPESLoader.load(self.path)
  self.progress.emit(100, "Done!")
  self.finished.emit(self.var_name, td)
  ```
- `finished` signal payload type becomes `TensorData` (not old dict)
- `request_load`: remove Low_Level_Scan sniff; always start worker
- `on_load_finish(var_name, tensor_data)`: store `TensorData` in session; also `global_workspace.push_spectroscopy_data(var_name, tensor_data)`
- `activate_data`: if value is `TensorData`, call `self.viewer.load_data(td)` directly; keep legacy dict path for old in-memory sessions if needed
- Delete unused `process_xy` / `process_fermi` / `recursively_load` from maestroai `maestro_loader.py`

- [ ] **Step 1: Rewrite LoadWorker**

- [ ] **Step 2: Update GUI load/activate paths**

- [ ] **Step 3: Import smoke**

```bash
TensorSpec_env/bin/python -c "from tensorspec.gui.maestroai.maestro_loader import LoadWorker; from tensorspec.gui.maestroai import maestroai_gui; print('ok')"
```

---

### Task 10: Optional live-mount integration + README note

**Files:**
- Create: `tests/test_maestro_live_mount.py` (skip if path missing)
- Modify: `README.md` only if there is an existing IO/loaders section; otherwise skip README

**Interfaces:**
- Set `TENSORESPEC_MAESTRO_LIVE_DIR` to a folder containing:
  - `20260629_00736.h5` → ndim 4, labels start with Y,X
  - `20260630_00742.h5` → ndim 5, shape product of first three spatial/defl == 17*81*81
- Mark `@pytest.mark.integration`; skip if env unset / files absent

- [ ] **Step 1: Write skipped-by-default live tests**

- [ ] **Step 2: If live dir present, run without skip and fix any reshape bugs**

```bash
HDF5_USE_FILE_LOCKING=FALSE TENSORESPEC_MAESTRO_LIVE_DIR=/path/to/samples \
  TensorSpec_env/bin/python -m pytest tests/test_maestro_live_mount.py -v -m integration
```

---

## Spec coverage

| Spec requirement | Task |
|------------------|------|
| Plugin package + registry | 3–6 |
| `Low_Level_Scan` source of truth | 1 |
| `Fixed_Spectra1` + `Process_000` | 2, 6 |
| kinds 4D / 5D / Fermi / generic | 3–6 |
| `TensorData` pack | 7 |
| ARPES suite (auto via facade) | 7 |
| Main browser Load | 8 |
| ML suite shared facade | 9 |
| SMB locking | 6, 9 |
| Unit tests fixtures | 1–7 |
| Live mount optional | 10 |
| XY-first axis order | 3–4 |
| Abort / ∏ mismatch errors | 3, 6 |

## Execution

User requested **multiagent**. After this plan is saved: use **subagent-driven-development** — fresh implementer per task (cheap/fast model), review between tasks, continue without pausing for “should I continue?”.
