# PEEM Suite — Separate Channels — Design Spec

Date: 2026-08-11  
Status: approved for planning  
Branch: `HTML_einstein_app`  
Related: `docs/superpowers/specs/2026-08-09-peem-pair-stack-design.md`, `docs/superpowers/specs/2026-08-10-peem-drift-roi-design.md`, `roadmap.md` separate bullet, `sandy_rule.md` DataTree `/processed`

## Problem

After Stack (and optional Drift), PEEM data lives as a 4D paired cube `(pair, channel, y, x)` in `/processed`. Downstream BG / sum-rule want per-channel stacks (CP vs CM, or LH vs LV). The Separate button is still disabled. Roadmap: “separate those CP and CM or LH and LV.”

## Goals

- Split a **4D paired** `/processed` cube into two channel children under the **same** spectroscopy tree.
- Keep the paired 4D cube at flat `/processed`.
- Write `/processed/CP` and `/processed/CM` (or LH/LV from `channel_tags`).
- Enable Separate in the PEEM suite; viewer can show Raw | Paired | each channel.
- Leave energy-axis matching / interpolation / sum-rule gates for a later slice.

## Non-goals

- Energy-grid alignment, interpolate-to-common-step, or “sum rule impossible” warnings (documented for later).
- Background, sum rule, I0 application.
- Separating from `/raw` by `pol` without Stack.
- Re-merging channels back into 4D.
- New top-level workspace names (`{name}_CP`) — rejected in brainstorm.

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| Output | Two channel stacks under same tree |
| Paths | Flat `/processed` = paired 4D; children `/processed/{tag}` |
| Keep paired cube | Yes |
| Input | 4D paired `/processed` only |
| Unequal / unpaired leftovers | Ignore (channel stacks = paired indices only) |
| Energy-aware pairing | Later (not this slice) |
| Architecture | `separate_pairs` + `write_processed_child` + thin API + UI |

## Clarification: “unequal counts” today

Current Stack pairs by **frame count / file order** from `pol` tags (`min(n_CP, n_CM)`), not by energy. Energy-range / ΔE mismatch / interpolate-on-overlap is **not** implemented and is out of scope here. Separate v1 assumes a valid 4D paired cube and splits channels.

---

## §1 — Core + DataTree

**Engine** (`tensorspec/core/peem_engine.py`):

```python
def separate_pairs(tensor: TensorData) -> dict[str, TensorData]:
    """
    Input: (pair, channel, y, x), metadata['channel_tags'] length 2.
    Output: {tag: TensorData} each shape (n_pairs, y, x),
    labels ['frame','y','x'], data_type e.g. 'Experimental PEEM (CP)'.
    """
```

Metadata on each child (minimum): `channel_tag`, `separated_from="paired"`, pass through `pair_mode`, drift/CSV/I0 attrs where present.

**DataTree / workspace:**

- Add `DataTreeBuilder.write_processed_child(tree, child_name, tensor_data)` → `/processed/<child_name>` (mirror `write_analysis` child pattern).
- Do **not** replace flat `/processed` when writing children.
- Extend `WorkspaceManager.pull_tensor_data(name, node)` to accept `processed`, `processed/CP`, `processed/CM`, etc.
- Append `/history` log line on separate.

---

## §2 — API + UI

**Router** (`peem.py`):

| Endpoint | Role |
|----------|------|
| `POST /api/peem/{name}/separate` | Require valid 4D paired `/processed` → `separate_pairs` → write children → summary |
| `GET /api/peem/{name}/meta` | Add `separated_channels: list[str]` |
| `GET /api/peem/{name}/frame/{i}` | `node` supports `processed/CP`, `processed/CM`, … (plus existing raw / processed) |

**Summary JSON (minimum):** `name`, `channels`, `n_frames`, `has_separated=true`.

**UI** (`peem_suite`):

- Enable **Separate Pairs** when paired `/processed` exists.
- Viewer node options: Raw | Paired | `{tag}` for each separated channel (labels from `channel_tags` / `separated_channels`).
- Status line after success.
- Stack / Drift / BG / sum-rule behavior unchanged except Separate enabled.

---

## §3 — Success criteria & tests

**Done when:**

- Separate creates `/processed/{tag}` children; paired cube still pullable as `node=processed`.
- Meta lists separated channels; frame GET works per channel.
- 422 if no paired processed cube.
- Unit tests for `separate_pairs` + tree child write/pull.
- API TestClient: load → pair → separate → frame on `processed/CP`.
- Roadmap checkbox for separate marked; energy/sum-rule left open.
- Stay on `HTML_einstein_app`; push + Einstein; never merge to main.

**Roadmap / README:** Note separated channel children under `/processed`.

---

## Later (explicit backlog)

- Pair / analyze on shared energy grid when ΔE differs but ranges overlap (interpolate).
- Warn when energy overlap insufficient for sum rule.

## Open for implementation plan (not blockers)

- Whether flat `/processed` Dataset and child nodes coexist cleanly in current xarray DataTree version (may need empty parent + `paired` child rename — resolve in plan with a spike if assign fails).
- Max tag charset for path segments (reuse analysis node sanitizer).

## Spec self-review

- Locked decisions have no TBD.
- Scope = channel split only; energy logic deferred and documented.
- Aligns with DataTree hierarchy and existing pair/drift `/processed` usage.
