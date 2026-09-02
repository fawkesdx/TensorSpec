# Maestro ARPES Data Loader — Design

**Date:** 2026-09-02  
**Status:** Draft for review  
**Branch context:** TensorSpec_GUI

## Goal

Build a Maestro beamline ARPES loader that packages results into TensorSpec’s established `TensorData` contract and runs from ARPES suite, main browser, and ML (maestroai) Load entry points. Architecture is N-D-ready via `Low_Level_Scan` + kind plugins; **v1 ships** 4D XY Fine, 5D Focus XY Fine (Defl×XY), Fermi-only Defl/Theta maps, and legacy `Process_000`. Pure 1D and other exotic kinds come later as additional plugins.

## Decisions (locked)

| ID | Choice |
|----|--------|
| Architecture | Keep `ARPESLoader` facade; replace monolithic `MaestroLoader` with plugin registry + kind scripts |
| Dim source of truth | Parse `Headers/Low_Level_Scan` loop table only (not high-level UI tree) |
| GUI wiring | ARPES suite + main browser disk Load + ML suite (shared facade) |
| v1 kinds | `xy_fine_4d`, `focus_xy_fine_5d`, `fermi_defl_3d`, plus legacy `process000_generic` (Swept / `Process_000`) |
| Approach | Registry + kind plugins |

## Problem evidence (sample files)

Live validation used private beamline mount samples (paths not published). Point optional tests at a local folder via `TENSORESPEC_MAESTRO_LIVE_DIR`.

| File | Mode (`LWLVNM`) | Loops | Cycles | Dataset |
|------|-----------------|-------|--------|---------|
| `20260630_00742.h5` | Focus XY Fine | Defl(17) + XY(81×81) | 111537 | `Fixed_Spectra1` |
| `20260629_00736.h5` | XY Scan Fine | XY(81×81) | 6561 | `Fixed_Spectra1` |
| `20260629_00737.h5` | XY Scan Fine | XY(81×81) | 6561 | `Fixed_Spectra1` |

Current core `MaestroLoader` fails: requires `Process_000`; these files use `Fixed_Spectra1` + `DAQ_Fixed`. Motor unique-count reshape does not recover the intended grid; `Low_Level_Scan` does (`∏ n == num_cycles`).

Old maestroai `LoadWorker` only sniffs mode name / motor presence and builds 4D XY or 3D Fermi — no nested 5D, no full loop parse.

## Canonical data contract

Loaders return / pack into `TensorData` (`tensorspec/core/data_models.py`):

- `value: np.ndarray`
- `axes: List[np.ndarray]` (one per dim)
- `labels: List[str]`, `units: List[str]` (len == ndim)
- `data_type: str`
- `metadata: dict`

Workspace path: `push_spectroscopy_data` → `xarray.DataTree` (`spectroscopy_tree`) → `pull_tensor_data` → `DataViewerPanel.load_data`.

Default viewer map uses dims `(0, 1)` as Y×X (`spawn_view(x_idx=1, y_idx=0)`). Loader must place the greeting spatial mesh on those dims.

## Architecture

```
GUI Load (ARPES | main browser | ML suite)
  → ARPESLoader.load(path)          # facade → TensorData
      → detect Maestro signature
      → parse Low_Level_Scan → ScanPlan
      → registry match → kind plugin
      → plugin → intermediate payload
      → pack_tensor_data → TensorData
  → workspace.push_spectroscopy_data
  → DataViewerPanel
```

### Package layout

```
tensorspec/core/io/arpes_loader.py          # facade + pack (keep/extend)
tensorspec/core/io/loaders/maestro/
  __init__.py
  detect.py                                 # Fixed vs Swept; dataset name
  low_level_scan.py                         # → ScanPlan
  registry.py                               # ordered kind matchers
  kinds/
    xy_fine_4d.py
    focus_xy_fine_5d.py
    fermi_defl_3d.py
    process000_generic.py                   # legacy Swept / Process_000
```

Replace monolithic `tensorspec/core/io/loaders/maestro_loader.py` with a thin re-export during migration, then remove.

Maestroai `LoadWorker` becomes a background wrapper around the same facade (no parallel XY/Fermi-only logic).

## ScanPlan

Parsed **only** from `Headers/Low_Level_Scan`:

- `mode_name` from `LWLVNM`
- `loops[]`: each loop has motors `[{name, units, start, end, n}]`
- Expected cycles = product of all motor `n` values
- Cross-check against `Headers/Main/num_cycles` and detector point-axis length

Detector axes from `2D_Data/<dataset>` attributes: `unitNames`, `scaleOffset`, `scaleDelta`. Map `eV` → Energy; angle/`pixels` → Slit Angle (refine calibration later if needed).

Supported datasets: `Fixed_Spectra1` and `Process_000`.

## Kind plugins (v1)

| Kind | Match | TensorData axis order (labels) |
|------|--------|--------------------------------|
| `focus_xy_fine_5d` | 2 loops: one angle motor (Defl/Theta…) + X+Y mesh | `Y, X, Defl, Energy, Angle` |
| `xy_fine_4d` | 1 loop: X+Y mesh | `Y, X, Energy, Angle` |
| `fermi_defl_3d` | 1 loop: single angle motor; no XY | `Defl, Energy, Angle` (fixed XY) |
| `process000_generic` | Swept / `Process_000` / unmatched legacy | best-effort reshape + names |

**Fermi-only meaning:** stay on fixed XY; scan deflector or theta map only.

**Abort handling:** if completed cycles < plan product, truncate motors / emit warning in `metadata`; do not invent a wrong rectangular grid.

**Metadata (minimum):** `facility=MAESTRO`, `mode_name`, `is_fixed`, `scan_plan`, `source_path`; optional raw motors.

## GUI wiring

1. **ARPES suite** — existing `load_arpes_files` stays; benefits automatically when `ARPESLoader` uses plugins.
2. **Main browser** — add disk Load for `.h5/.hdf5/.npz`; push workspace; refresh tree; Launch Viewer unchanged.
3. **ML suite** — `request_load` / `LoadWorker` call shared facade; remove exclusive XY-vs-Fermi sniff for reshape; inject `TensorData` into session/viewer.

Other suites (crystal CIF, DFT, XAS CSV, PEEM TIF) stay on their own loaders.

## Viewer UX (no redesign required for v1)

- Correct axis order → default panel is XY (or Defl×E map for Fermi-only).
- Remaining dims → existing per-dim sliders.
- Snap New Panel already supports alternate cuts (dispersion vs isoenergy).
- Example: gating-as-outer-dim (when encoded as a Low_Level_Scan loop) greets as XY with gate on a slider — same mechanism as Defl on 5D Focus XY Fine.

## Error handling

- Missing Maestro signature → raise; facade may try `MockDataLoader` then fail clearly.
- Missing/unparseable `Low_Level_Scan` when Fixed path needs it → fail with plan vs cycles vs data length (no silent wrong reshape).
- Unknown Fixed kind with no Swept fallback → fail explicitly (no opaque Raw blob).
- SMB mounts: open HDF5 with file locking disabled (`HDF5_USE_FILE_LOCKING=FALSE` / h5py-compatible open).

## Out of scope (v1)

- High-level Maestro UI tree labels (repeat / one-motor / two-motor / K-dosing) when not present in `Low_Level_Scan`
- Other beamline facilities
- Viewer redesign beyond correct default axis order
- Perfect angle calibration from `pixels` (placeholder mapping OK)

## Testing

Unit tests with fixture/mocked `Low_Level_Scan` tables (no mount required):

- 4D XY Fine → shape/labels
- 5D Focus XY Fine → `17×81×81` + detector; labels `Y,X,Defl,E,Angle`
- Fermi Defl 3D
- Process_000 legacy path
- Abort truncate behavior
- ∏ mismatch → error

Optional integration test marked for live paths when `TENSORESPEC_MAESTRO_LIVE_DIR` is set.

## Success criteria

1. Load `_00736` / `_00737` → 4D `TensorData`; viewer opens on XY map.
2. Load `_00742` → 5D; viewer opens on XY; Defl on slider; snap can show dispersion / isoenergy.
3. Fermi-only file → 3D Defl map path works.
4. Legacy `Process_000` / mock files still load.
5. Same loader path from ARPES, main browser, and ML suite Load buttons.
