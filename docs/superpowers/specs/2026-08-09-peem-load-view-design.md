# PEEM Suite — Load + View (first slice) — Design Spec

Date: 2026-08-09  
Status: approved for planning  
Branch: `HTML_einstein_app`  
Related: `roadmap.md` PEEM vision, `sandy_rule.md` (`peem_loaders.py`, DataTree), ARPES load (`/api/arpes/load`), `TensorData` / `push_spectroscopy_data`

## Problem

PEEM Suite HTML shell exists but all controls are disabled (“no engine yet”). Users need to load TIF stacks / folder sequences into the session workspace and inspect frames before drift, pairing, background, or sum-rule work.

## Goals

- Load multipage TIF **and** folder sequences of single-frame TIFs.
- Enter via **browser upload** or **server path** (Einstein/local disks).
- Load optional **accompanying beamline CSV** with the stack (I0 / beam current and other params) into metadata — required later for CP↔CM normalization in sum rule; attach on load in this slice even though sum-rule UI is deferred.
- Push into workspace as spectroscopy DataTree via `TensorData` → `push_spectroscopy_data`.
- Simple viewer: canvas + frame slider + clim; best-effort polarisation tags from filenames; show whether CSV / I0 was found.
- Unlock PEEM suite for load+view only; leave later fieldsets disabled.

## Non-goals

- Drift correction, CP/CM or LH/LV stack/separate, background, **sum-rule math / I0 normalization application**, ROI, domain tools.
- XAS suite UI / shared BG core (ribbon stays separate; share later).
- Plotly H/V line profiles (frame API must stay stable for a later drop-in).
- PyVista / fancy figure export.
- Transport suite.

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| XAS vs PEEM | Two ribbon buttons; shared `core/` later |
| First slice | Load + view only |
| Ingest formats | Multipage TIF + folder sequence |
| Beamline CSV | Optional companion CSV on load; store I0/current (+ raw columns) in attrs for later sum-rule norm |
| Entry | Upload + server path |
| Viewer | Simple canvas + slider + clim; hooks for profiles later |
| Filename tags | Best-effort `pol` / `pair_id`; no stacking yet |
| Architecture | Approach 1: TensorData stack + thin PEEM API |

---

## §1 — Data model & loaders

**Shape:** `(frame, y, x)` float array.  
**Axes:** `frame` = `0…N-1` (file order), `y`/`x` = pixel indices.  
**Labels / units:** `["frame","y","x"]` / `["","px","px"]`.  
**`data_type`:** `"Experimental PEEM"`.

**Workspace:** Loaders return `TensorData` → `session.workspace.push_spectroscopy_data(name, tensor)` → DataTree `/raw` (same contract as ARPES). Tags live in `metadata` / dataset attrs.

**New file:** `tensorspec/core/io/peem_loaders.py`

| Function | Behavior |
|----------|----------|
| `load_tif_stack(path)` | Multipage TIF via `tifffile` → stack |
| `load_tif_sequence(dir)` | Sorted `*.tif` / `*.tiff` in directory → stack |
| `load_beamline_csv(path)` | Parse companion CSV → dict / columnar attrs (I0, current, …) |
| Shared packaging | Build `TensorData`; parse filenames → `metadata`; merge CSV |

**Metadata (minimum):**

- `frame_names`: list[str]
- `pol`: list[str] — per-frame `CP` / `CM` / `LH` / `LV` / `unknown` (case-insensitive substring heuristics)
- `pair_id`: optional list or nulls when not detectable
- `source`: original path or upload filename
- `loader`: `"tif_stack"` | `"tif_sequence"`
- `beamline_csv`: source filename/path if provided
- `I0` / `beam_current` (or beamline column aliases): scalar and/or per-frame series when CSV has rows matching frames
- `beamline_table`: optional full parsed columns (preserve unknowns for later)

**CSV association (v1):** optional upload alongside TIF/zip **or** auto-discover sidecar next to `server_path` (same stem / common names in folder — exact heuristics in plan). Missing CSV → load images OK; meta flags `csv_attached=false`.

**Out of this module:** drift, pairing stack, BG, applying I0 normalization (`peem_engine.py` later).

---

## §2 — API + UI

**Router:** `tensorspec/web/server/routers/peem.py` (register in app).

| Endpoint | Role |
|----------|------|
| `POST /api/peem/load` | Upload one TIF **or** zip of a folder sequence; **or** Form `server_path` (mutually exclusive with file). Optional companion `csv` upload / `csv_path`. Optional `name`. → loader → push → summary JSON |
| `GET /api/peem/{name}/meta` | Shape, labels, `frame_names`, per-frame `pol`, n_frames, CSV/I0 attachment summary |
| `GET /api/peem/{name}/frame/{i}` | 2D float intensity + suggested clim (`vmin`/`vmax`, e.g. percentiles) as JSON for canvas |

**Summary JSON fields:** `name`, `shape`, `n_frames`, `data_type`, `pol_summary` (counts), `source`, `loader`, `csv_attached`, `I0_present` (bool).

**Server path safety:** Resolve path; reject escapes outside allowed roots (at least session `project_dir` and any configured data roots). Empty / missing → 400/404; parse errors → 422.

**Size limits:** Mirror ARPES-style caps for uploads (document concrete MB in plan/impl); zip unpack bounded.

**Dependency:** ensure `tifffile` in project requirements / Einstein env notes.

**UI:** `peem_suite.html` + new `peem_suite.js`

- Enable Load TIF Stack (file input) and Load folder/path (path field + optional zip upload); optional CSV file / path field (mirror ARPES measurement-log pattern).
- Status line: workspace name, N frames, tag hint if any non-unknown `pol`, CSV/I0 attached or missing.
- Center: canvas draw of current frame; slider `0…N-1`; clim min/max (default from frame endpoint percentiles).
- Pairing / drift / BG fieldsets stay **disabled** with hint “next slice”.
- Badge: e.g. `Load + view` (not “no engine yet”).
- Main browser ribbon: PEEM remains reachable; no XAS change.

---

## §3 — Success criteria & tests

**Done when:**

- Multipage TIF upload → named workspace entry → slider shows all frames.
- Folder sequence via server path **or** zip upload → same.
- Best-effort `pol` in meta (`unknown` OK).
- Companion CSV (or sidecar) attaches I0/current into attrs; load still works if CSV absent.
- Works on Einstein after pull when files are local server paths.
- Unit tests: synthetic multipage TIF + 2-frame folder for loaders; CSV parse → `I0` in metadata; TestClient happy path for load + frame.

**Roadmap / README:** On ship, check PEEM loader + load+view bullets; leave drift/BG/sum-rule open. Update suite-available summary if present. Stay on `HTML_einstein_app`; never merge to main from this workstream.

---

## Open for implementation plan (not blockers)

- Exact upload size / zip limits and allowed path roots list.
- Clim percentile defaults (e.g. 1–99).
- Whether zip is required for “folder upload” vs multi-file Form (zip preferred for v1 simplicity).
- Exact beamline CSV column names / layout (I0 aliases); sample file from beamline if available.

## Later (roadmap — not this slice)

- Apply I0 / beam-current normalization when stacking CP vs CM and running sum rule.

## Spec self-review

- No TBD placeholders for locked decisions (CSV column schema deferred to plan with sample).
- Scope matches “load + view” + **store** beamline CSV; sum-rule I0 **application** deferred.
- Aligns with DataTree via existing `push_spectroscopy_data` (Approach 1 ≠ bypassing tree).
- Frame JSON API is the hook for future profiles without redesigning load.
