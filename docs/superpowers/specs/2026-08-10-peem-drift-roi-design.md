# PEEM Suite — Drift Correction (ROI NCC) — Design Spec

Date: 2026-08-10  
Status: approved for planning  
Branch: `HTML_einstein_app`  
Related: `docs/superpowers/specs/2026-08-09-peem-load-view-design.md`, `docs/superpowers/specs/2026-08-09-peem-pair-stack-design.md`, `roadmap.md` drift bullet, `sandy_rule.md` (`peem_engine.py`)

## Problem

PEEM stacks (raw frames or CP/CM·LH/LV paired cubes) drift in the FOV over time. Fiji users typically pick an ROI on a stable feature, measure translational drift vs a reference frame, and apply the same shift to the whole image (and to both channels of a pair so Separate stays aligned). TensorSpec has only disabled Drift UI stubs — **no drift engine exists yet** (not a duplicate of an old function).

## Goals

- General stack drift correction: **translation only**, measured inside a user ROI, applied to the full plane.
- ROI shapes: **rectangle**, **ellipse**, **clicked polygon** (straight segments).
- Same core engine for **raw** `(frame, y, x)` and **paired** `(pair, channel, y, x)`.
- On paired data: estimate shift from **one track channel**; apply **identical (dx, dy)** to both channels so CP/CM (or LH/LV) remain registered after later Separate.
- Algorithm v1: **normalized cross-correlation** in ROI, **integer-pixel** shifts, **edge-clamp** fill.
- User-chosen **reference index** + **search radius**.
- Write result to `/processed` with `drift_*` attrs (including per-plane shifts).
- Enable Drift fieldset in PEEM suite with canvas ROI tools.

## Non-goals

- Phase correlation or manual per-frame offsets (UI may leave stubs disabled).
- Separate pairs, background, sum rule, I0 application.
- Curved/interpolated polygon edges.
- Subpixel shifts.
- Non-translational warps (rotation, scale, non-homogeneous).

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| Prior art in repo | None — greenfield (Fiji-like, not porting Fiji) |
| Stack axes | Raw frames **and** paired stacks (same engine) |
| Algorithm v1 | NCC in ROI, integer `(dx, dy)` |
| Storage | Always write drift result to `/processed`; shifts in attrs |
| ROI shapes | Rect + ellipse + clicked polygon (this slice) |
| Paired channels | One feature / one track channel → same shift on both channels |
| Reference | User-picked frame/pair index |
| Empty rim after shift | Edge-clamp |
| Architecture | `peem_engine.drift_correct` (+ ROI mask helper) + thin API + suite UI |

---

## §1 — Core (ROI + drift)

**Extend:** `tensorspec/core/peem_engine.py`  
Optional split: `tensorspec/core/peem_roi.py` if mask helpers clutter the engine file.

### ROI → mask

```python
def roi_to_mask(ny: int, nx: int, roi: dict) -> np.ndarray:
    """Return bool mask shape (ny, nx)."""
```

`roi` schemas:

| `kind` | Fields |
|--------|--------|
| `rect` | `x0, y0, x1, y1` (pixel coords; normalize so min/max ordered) |
| `ellipse` | `cx, cy, rx, ry` |
| `polygon` | `points: [[x, y], ...]` (≥3); straight edges; filled interior |

Invalid / empty mask → `ValueError`.

### Drift

```python
def drift_correct(
    tensor: TensorData,
    *,
    ref_index: int,
    roi: dict,
    search_radius: int,
    track_channel: int = 0,
) -> TensorData:
    ...
```

**Behavior:**

- Accept raw `(frame, y, x)` or paired `(pair, channel, y, x)`.
- Template = ROI region of reference plane (paired: plane `ref_index`, channel `track_channel`).
- For each stack index: NCC search within `±search_radius` → integer `(dx, dy)`; reference shift is `(0, 0)`.
- Apply the same `(dx, dy)` to every channel of that pair (or the single raw frame).
- Translate with **edge clamp** (no wrap-around).
- Output same shape and axis labels as input; `data_type` e.g. keep Experimental PEEM / paired, plus metadata marking drift.
- Metadata (minimum): `drift_method="ncc_roi"`, `drift_ref_index`, `drift_roi`, `drift_search_radius`, `drift_track_channel`, `drift_shifts` (list of `{index, dx, dy}`), pass through pair/CSV/I0 attrs when present.

**Errors:** bad shape, bad `ref_index` / `track_channel`, empty ROI, `search_radius < 1` → clear `ValueError`.

---

## §2 — API + UI

**Router** (`tensorspec/web/server/routers/peem.py`):

| Endpoint | Role |
|----------|------|
| `POST /api/peem/{name}/drift` | Body: `source` (`raw`\|`processed`), `ref_index`, `search_radius`, `track_channel`, `roi`. Pull → `drift_correct` → `write_processed_data` → summary |
| `GET /api/peem/{name}/meta` | Extend with `has_drift` (and light drift summary from attrs when present) |
| Frame GET | Existing `node` / pair / channel contract; after drift, UI views `/processed` |

**Summary JSON (minimum):** `name`, `source`, `n_planes`, `ref_index`, `search_radius`, `has_processed`, optional shift stats (`max_|dx|`, `max_|dy|`).

**Source selection:**

- UI default: `processed` when `has_processed` (paired or prior drift), else `raw`.
- Pairing continues to read `/raw` only — drifting a raw cube into `/processed` does not block later re-pair from raw.
- If `source=processed`, require valid paired **or** already drift-compatible cube (document: v1 expects paired 4D or prior 3D/4D PEEM processed with expected labels).

**UI** (`peem_suite.html` / `peem_suite.js` / `api.js`):

- Enable Drift fieldset: algorithm = **NCC (ROI)** only for v1; Phase / Manual disabled or hidden.
- Existing `#peem-ref`, `#peem-search`; add track-channel control when source is paired.
- Canvas ROI tools: Rect | Ellipse | Polygon (click vertices; close via button or double-click); Clear ROI; overlay on current view.
- **Apply Drift Correction** → POST; on success set viewer to processed; status shows applied / max shifts.
- Separate / BG / sum-rule remain disabled.

---

## §3 — Success criteria & tests

**Done when:**

- Rect / ellipse / polygon masks work; synthetic stack with known translation recovers integer `(dx, dy)` within search radius.
- Paired path: both channels receive identical shifts.
- Raw and processed sources both work end-to-end via API.
- UI can draw ROI and apply drift; corrected stack viewable.
- Unit tests for `roi_to_mask` + `drift_correct`; API TestClient for drift routes.
- Roadmap drift checkbox marked; Separate / BG / sum-rule left open.
- Stay on `HTML_einstein_app`; push + Einstein pull; never merge to main.

**Roadmap / README:** Update PEEM available bullets for ROI drift only.

---

## Open for implementation plan (not blockers)

- Exact inclusive/exclusive pixel bounds for rect.
- Polygon close UX details and max vertex count.
- Caps: max `search_radius`, max planes × pixels to avoid long sync requests (job queue later if needed).
- Whether re-drifting overwrites `/processed` without confirmation (v1: overwrite OK; attrs record last drift).

## Spec self-review

- No TBD on locked decisions.
- Scope = ROI NCC drift only; Separate/phase/manual/BG/sum-rule deferred.
- Aligns with `/processed` via `write_processed_data` and existing pair viewer.
- Clarifies greenfield (not duplicating an existing TensorSpec drift function).
