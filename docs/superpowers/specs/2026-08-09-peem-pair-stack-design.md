# PEEM Suite — Pair Stack — Design Spec

Date: 2026-08-09  
Status: approved for planning  
Branch: `HTML_einstein_app`  
Related: `docs/superpowers/specs/2026-08-09-peem-load-view-design.md`, `roadmap.md` PEEM pairing bullets, `sandy_rule.md` (`peem_engine.py`), `WorkspaceManager.write_processed_data`

## Problem

PEEM load+view stores a flat `(frame, y, x)` cube with best-effort `pol` tags (`CP`/`CM`/`LH`/`LV`/`unknown`). Contrast pairing UI is still disabled. Downstream drift and sum-rule need paired CP↔CM (or LH↔LV) data in a structured cube, without destroying `/raw`.

## Goals

- Pair frames into `(pair, channel, y, x)` and write to `/processed` on the same workspace name.
- User-selectable mode: Auto | CP/CM | LH/LV.
- Unequal channel counts: pair `min(n_a, n_b)` in file order; record leftovers as `unpaired`.
- Enable Stack Pairs in the PEEM suite; viewer supports Raw | Processed with pair + channel navigation.
- Keep Separate / drift / BG / sum-rule / I0 application out of this slice.

## Non-goals

- Separate pairs back to interleaved frames.
- Drift correction.
- Background subtraction, sum rule, I0 normalization apply.
- XAS suite / shared analysis core.
- Changing how `pol` is inferred on load (reuse existing tags).

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| Data layout | `(pair, channel, y, x)` |
| Storage | `/processed` via `write_processed_data`; `/raw` untouched |
| Mode UI | Auto \| CP/CM \| LH/LV |
| Unequal counts | `min` pairs + `unpaired` attrs |
| Slice scope | Stack only (Separate disabled) |
| Architecture | `peem_engine.pair_stack` + thin PEEM API + suite UI |

---

## §1 — Core pairing

**New:** `tensorspec/core/peem_engine.py`

```python
def pair_stack(
    tensor: TensorData,
    mode: Literal["auto", "CP_CM", "LH_LV"],
) -> TensorData: ...
```

**Input:** `/raw` `TensorData` with shape `(frame, y, x)` and `metadata["pol"]` (list length = n_frames).

**Mode resolution:**

- `CP_CM` → channel tags `["CP", "CM"]`
- `LH_LV` → `["LH", "LV"]`
- `auto` → if any CP/CM and no LH/LV → `CP_CM`; elif any LH/LV and no CP/CM → `LH_LV`; else raise `ValueError` (mixed or none)

**Pairing algorithm:** Walk frames in order. Maintain queues of unused indices per channel tag. While both queues non-empty, pop one from each to form a pair. Leftover indices → `unpaired`.

**Output `TensorData`:**

- `value`: `(n_pairs, 2, y, x)` float
- `labels`: `["pair", "channel", "y", "x"]`
- `units`: `["", "", "px", "px"]`
- `axes`: `arange(n_pairs)`, `arange(2)`, `y`, `x` (pixel axes from raw)
- `data_type`: `"Experimental PEEM (paired)"`
- Metadata (minimum):
  - `pair_mode`: resolved mode string
  - `channel_tags`: e.g. `["CP", "CM"]`
  - `pair_sources`: list of `{pair, channels: [{tag, frame_index, frame_name}, ...]}`
  - `unpaired`: list of `{frame_index, frame_name, pol}`
  - Pass through beamline CSV / I0 / `source` / `loader` from raw where present
  - `csv_attached` / `I0` unchanged (still not applied)

**Errors:** missing/short `pol`, zero pairs, unsupported raw shape → `ValueError` with clear message.

---

## §2 — API + UI

**Router** (`tensorspec/web/server/routers/peem.py`):

| Endpoint | Role |
|----------|------|
| `POST /api/peem/{name}/pair` | Body `{ "mode": "auto"\|"CP_CM"\|"LH_LV" }` → pull raw → `pair_stack` → `write_processed_data` → summary |
| `GET /api/peem/{name}/meta` | Extend with `has_processed`, pair summary fields when present |
| `GET /api/peem/{name}/frame/...` | Support `node=raw\|processed` (default `raw`). For processed: `pair` + `channel` query params → 2D plane + clim |

**Pair summary JSON (minimum):** `name`, `n_pairs`, `channel_tags`, `unpaired_count`, `mode`, `has_processed`.

**UI** (`peem_suite.html` / `peem_suite.js` / `api.js`):

- Enable Contrast Pairing: mode select + **Stack Pairs** button.
- Separate remains disabled.
- On success: status shows pairs + unpaired; set viewer to processed.
- Viewer: Raw \| Processed toggle; when processed — pair slider + channel select; clim behavior as load+view (defaults from percentiles; preserve user clim while scrubbing).
- Badge/footer reflect paired state when `/processed` exists.

---

## §3 — Success criteria & tests

**Done when:**

- Stack Pairs writes paired cube to `/processed` for Auto / CP_CM / LH_LV.
- Unequal channel counts produce `min` pairs and visible `unpaired` count.
- Viewer navigates raw frames and processed pair×channel planes.
- Unit tests for `pair_stack` (happy, auto, unequal, mixed fail).
- API tests: pair → meta `has_processed` → frame `node=processed`.
- Roadmap checkbox for stack CP/CM or LH/LV marked; Separate / drift remain open.
- Stay on `HTML_einstein_app`; push + Einstein pull after ship; never merge to main.

**Roadmap / README:** Update PEEM available bullets for pair stack only.

---

## Open for implementation plan (not blockers)

- Exact schema field names for `PeemPairRequest` / extended `PeemMeta`.
- Whether channel axis uses integer 0/1 only or also stores tag labels in coords attrs.
- History log entry text for `/history` when writing processed (follow existing `DataTreeBuilder` pattern if any).

## Spec self-review

- Locked decisions have no TBD.
- Scope = stack only; Separate/drift/BG/sum-rule/I0-apply deferred.
- Aligns with DataTree `/raw` + `/processed` and existing `write_processed_data`.
- Reuses load-time `pol` tags; no new loader work required.
