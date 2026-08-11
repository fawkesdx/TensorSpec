# PEEM Separate Channels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split a 4D paired PEEM `/processed` cube into per-channel children (`/processed/CP`, `/processed/CM` or LH/LV), keep the paired cube at flat `/processed`, and enable Separate + channel viewing in the PEEM suite.

**Architecture:** `peem_engine.separate_pairs` slices channel axis into 3D `(frame,y,x)` stacks. `DataTreeBuilder.write_processed_child` attaches children under `/processed` without replacing the parent Dataset (xarray 2026.7.0 allows this when child leading dim is `frame`, not a conflicting size on shared dims). Workspace `pull_tensor_data` must resolve nested paths (today `"processed/CP" in tree` is False even though `tree["processed/CP"]` works). Thin `POST /separate` + meta/frame/UI.

**Tech Stack:** NumPy, TensorData, xarray DataTree, FastAPI, existing PEEM suite JS.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-11-peem-separate-channels-design.md`
- Branch: `HTML_einstein_app` only — never merge to `main`
- Input: 4D paired `/processed` only (`Experimental PEEM (paired)`, labels `pair,channel,y,x`, `channel_tags` length 2)
- Keep flat `/processed` paired cube; write children `/processed/{tag}`
- No energy-grid align, interpolate, BG, sum-rule, or I0 application
- Re-pair / re-drift via `write_processed` **wipes** processed children — expected; user re-runs Separate
- After ship: push + Einstein pull/restart; update `roadmap.md` / README
- Tests: `PYTHONPATH=. TensorSpec_env/bin/pytest …`

## File map

| File | Role |
|------|------|
| `tensorspec/core/peem_engine.py` | `separate_pairs` |
| `tests/test_peem_engine.py` | Unit tests for separate |
| `tensorspec/core/data_tree.py` | `write_processed_child`, optional `list_processed_children` |
| `tensorspec/core/workspace.py` | Nested pull + `write_processed_child_data` |
| `tests/test_peem_data_tree.py` | Tree child write/pull (new) |
| `tensorspec/web/server/schemas.py` | `PeemSeparateSummary`; `PeemMeta.separated_channels` |
| `tensorspec/web/server/routers/peem.py` | `POST /separate`; meta + frame for `processed/{tag}` |
| `tests/test_peem_api.py` | API separate + frame child |
| `tensorspec/web/static/js/api.js` | `peemSeparate`; frame `node` passthrough |
| `tensorspec/web/static/js/peem_suite.js` | Separate button + channel view radios |
| `tensorspec/web/templates/suites/peem_suite.html` | Enable Separate; node radio host |
| `roadmap.md` / `README.md` | Mark separate shipped |

## Locked implementation details (spec open items resolved)

- **DataTree coexistence:** Spike confirms flat `/processed` 4D Dataset **plus** children with dims `(frame,y,x)` works. Do **not** rename paired cube to `/processed/paired`.
- **Nested pull:** `pull_tensor_data` must not use `if node not in tree` for nested paths. Use try/get on `tree[node.strip("/")]` (or walk segments).
- **Tag path safety:** Same rule as `write_analysis`: single segment, non-empty, no `/`. Tags from `channel_tags` are `CP`/`CM`/`LH`/`LV` — still validate.
- **Child metadata minimum:** `channel_tag`, `separated_from="paired"`, plus passthrough of `pair_mode`, `drift_method`, `drift_shifts` (if present), and `_PASSTHROUGH_KEYS` (csv/I0/source/loader/beamline_*).
- **Child TensorData:** `labels=["frame","y","x"]`, `data_type=f"Experimental PEEM ({tag})"`, axes = `[arange(n_pairs), y_axis, x_axis]`.
- **`write_processed` wipe:** Document only; no auto-reseparate. Meta returns `separated_channels=[]` after pair/drift until Separate again.
- **UI node model:** `state.node` is `"raw"` | `"processed"` | `"processed/CP"` | … Frame slider for raw and separated channels; pair+channel controls only when `node === "processed"` and paired.

---

### Task 1: `separate_pairs` engine + unit tests

**Files:**
- Modify: `tensorspec/core/peem_engine.py`
- Modify: `tests/test_peem_engine.py`

**Interfaces:**
- Consumes: `TensorData` 4D paired cube from `pair_stack` / drift
- Produces:

```python
def separate_pairs(tensor: TensorData) -> dict[str, TensorData]:
    """
    Input: (pair, channel, y, x), metadata['channel_tags'] length 2.
    Output: {tag: TensorData} each shape (n_pairs, y, x),
    labels ['frame','y','x'], data_type f'Experimental PEEM ({tag})'.
    Raises ValueError on invalid input.
    """
```

- [ ] **Step 1: Write failing tests**

Append to `tests/test_peem_engine.py`:

```python
class TestSeparatePairs(unittest.TestCase):
    def test_cp_cm_split(self):
        paired = eng.pair_stack(_raw(["CP", "CM", "CP", "CM"]), "CP_CM")
        out = eng.separate_pairs(paired)
        self.assertEqual(set(out), {"CP", "CM"})
        self.assertEqual(out["CP"].value.shape, (2, 2, 2))
        self.assertEqual(out["CP"].labels, ["frame", "y", "x"])
        self.assertEqual(out["CP"].data_type, "Experimental PEEM (CP)")
        self.assertEqual(out["CP"].metadata["channel_tag"], "CP")
        self.assertEqual(out["CP"].metadata["separated_from"], "paired")
        self.assertEqual(out["CP"].metadata["pair_mode"], "CP_CM")
        self.assertTrue(out["CP"].metadata["csv_attached"])
        np.testing.assert_array_equal(out["CP"].value[0], paired.value[0, 0])
        np.testing.assert_array_equal(out["CM"].value[0], paired.value[0, 1])

    def test_lh_lv_tags(self):
        paired = eng.pair_stack(_raw(["LH", "LV"]), "LH_LV")
        out = eng.separate_pairs(paired)
        self.assertEqual(set(out), {"LH", "LV"})

    def test_rejects_raw_3d(self):
        with self.assertRaises(ValueError):
            eng.separate_pairs(_raw(["CP", "CM"]))

    def test_rejects_bad_channel_tags(self):
        paired = eng.pair_stack(_raw(["CP", "CM"]), "CP_CM")
        paired.metadata["channel_tags"] = ["CP"]
        with self.assertRaises(ValueError):
            eng.separate_pairs(paired)
```

- [ ] **Step 2: Run — expect FAIL**

```bash
PYTHONPATH=. TensorSpec_env/bin/pytest tests/test_peem_engine.py::TestSeparatePairs -v
```

Expected: FAIL (`separate_pairs` missing or ImportError).

- [ ] **Step 3: Implement `separate_pairs`**

In `peem_engine.py`, after `pair_stack`:

```python
_SEPARATE_PASSTHROUGH_KEYS = _PASSTHROUGH_KEYS + (
    "pair_mode",
    "drift_method",
    "drift_shifts",
)


def separate_pairs(tensor: TensorData) -> dict[str, TensorData]:
    if (
        tensor.data_type != "Experimental PEEM (paired)"
        or list(tensor.labels) != ["pair", "channel", "y", "x"]
        or tensor.value.ndim != 4
        or tensor.value.shape[1] != 2
    ):
        raise ValueError(
            "separate_pairs requires a (pair, channel=2, y, x) paired PEEM cube"
        )
    tags = list(tensor.metadata.get("channel_tags") or [])
    if len(tags) != 2 or any(not str(t).strip() or "/" in str(t) for t in tags):
        raise ValueError("channel_tags must be two single-segment names")
    tags = [str(t).strip() for t in tags]
    if tags[0] == tags[1]:
        raise ValueError("channel_tags must be distinct")

    n_pairs, _, y_size, x_size = tensor.value.shape
    out: dict[str, TensorData] = {}
    for ch, tag in enumerate(tags):
        meta: dict[str, Any] = {
            "channel_tag": tag,
            "separated_from": "paired",
        }
        for key in _SEPARATE_PASSTHROUGH_KEYS:
            if key in tensor.metadata:
                meta[key] = tensor.metadata[key]
        out[tag] = TensorData(
            value=np.asarray(tensor.value[:, ch], dtype=float),
            axes=[
                np.arange(n_pairs),
                np.asarray(tensor.axes[2]),
                np.asarray(tensor.axes[3]),
            ],
            labels=["frame", "y", "x"],
            units=["", tensor.units[2], tensor.units[3]],
            data_type=f"Experimental PEEM ({tag})",
            metadata=meta,
        )
    return out
```

- [ ] **Step 4: Run — expect PASS**

```bash
PYTHONPATH=. TensorSpec_env/bin/pytest tests/test_peem_engine.py::TestSeparatePairs -v
```

- [ ] **Step 5: Commit**

```bash
git add tensorspec/core/peem_engine.py tests/test_peem_engine.py
git commit -m "$(cat <<'EOF'
feat(peem): add separate_pairs channel split

Split 4D paired cube into per-tag (frame,y,x) stacks for
/processed children.
EOF
)"
```

---

### Task 2: DataTree child write + nested pull

**Files:**
- Modify: `tensorspec/core/data_tree.py`
- Modify: `tensorspec/core/workspace.py`
- Create: `tests/test_peem_data_tree.py`

**Interfaces:**
- Consumes: `DataTreeBuilder.dataset_from_tensor`, existing trees
- Produces:

```python
# data_tree.py
@staticmethod
def write_processed_child(tree: DataTree, child_name: str, tensor_data: TensorData) -> DataTree:
    """Write /processed/<child_name> without replacing parent Dataset; append history."""

@staticmethod
def list_processed_children(tree: DataTree) -> list[str]:
    """Return child names under /processed that contain a 'data' variable."""

# workspace.py
def write_processed_child_data(self, name: str, child_name: str, tensor_data: TensorData) -> bool: ...
def list_processed_children(self, name: str) -> list[str]: ...
# pull_tensor_data(name, node) must resolve "processed/CP"
```

- [ ] **Step 1: Write failing tests**

```python
# tests/test_peem_data_tree.py
import unittest
import numpy as np
from tensorspec.core.data_models import TensorData
from tensorspec.core.data_tree import DataTreeBuilder
from tensorspec.core.workspace import WorkspaceManager
from tensorspec.core import peem_engine as eng


def _raw(pols):
    n = len(pols)
    frames = np.stack([np.full((2, 2), i + 1, dtype=float) for i in range(n)], axis=0)
    return TensorData(
        value=frames,
        axes=[np.arange(n), np.arange(2), np.arange(2)],
        labels=["frame", "y", "x"],
        units=["", "px", "px"],
        data_type="Experimental PEEM",
        metadata={"pol": list(pols), "frame_names": [f"f{i}" for i in range(n)]},
    )


class TestProcessedChildren(unittest.TestCase):
    def test_write_child_keeps_paired_parent(self):
        raw = _raw(["CP", "CM"])
        tree = DataTreeBuilder.build_from_tensor("t", raw)
        paired = eng.pair_stack(raw, "CP_CM")
        tree = DataTreeBuilder.write_processed(tree, paired)
        channels = eng.separate_pairs(paired)
        tree = DataTreeBuilder.write_processed_child(tree, "CP", channels["CP"])
        tree = DataTreeBuilder.write_processed_child(tree, "CM", channels["CM"])

        parent = tree["processed"].to_dataset()
        self.assertIn("data", parent)
        self.assertEqual(parent["data"].ndim, 4)
        self.assertEqual(set(DataTreeBuilder.list_processed_children(tree)), {"CP", "CM"})
        self.assertEqual(tree["processed/CP"].ds["data"].shape, (1, 2, 2))

    def test_workspace_pull_nested(self):
        ws = WorkspaceManager()
        raw = _raw(["CP", "CM", "CP", "CM"])
        ws.push_tensor_data("peem", raw)
        paired = eng.pair_stack(raw, "CP_CM")
        ws.write_processed_data("peem", paired)
        for tag, td in eng.separate_pairs(paired).items():
            self.assertTrue(ws.write_processed_child_data("peem", tag, td))
        child = ws.pull_tensor_data("peem", "processed/CP")
        self.assertIsNotNone(child)
        self.assertEqual(child.value.shape, (2, 2, 2))
        parent = ws.pull_tensor_data("peem", "processed")
        self.assertEqual(parent.value.ndim, 4)
        self.assertEqual(ws.list_processed_children("peem"), ["CM", "CP"])  # sorted

    def test_write_processed_wipes_children(self):
        ws = WorkspaceManager()
        raw = _raw(["CP", "CM"])
        ws.push_tensor_data("peem", raw)
        paired = eng.pair_stack(raw, "CP_CM")
        ws.write_processed_data("peem", paired)
        ws.write_processed_child_data("peem", "CP", eng.separate_pairs(paired)["CP"])
        ws.write_processed_data("peem", paired)
        self.assertEqual(ws.list_processed_children("peem"), [])
```

- [ ] **Step 2: Run — expect FAIL**

```bash
PYTHONPATH=. TensorSpec_env/bin/pytest tests/test_peem_data_tree.py -v
```

- [ ] **Step 3: Implement DataTree + workspace**

`data_tree.py` — add:

```python
@staticmethod
def write_processed_child(
    tree: DataTree, child_name: str, tensor_data: TensorData
) -> DataTree:
    safe = child_name.strip().strip("/")
    if not safe or "/" in safe:
        raise ValueError("Processed child name must be a single path segment.")
    ds = DataTreeBuilder.dataset_from_tensor(tensor_data)
    tree[f"processed/{safe}"] = ds

    history_node = tree["history"]
    history = history_node.to_dataset() if hasattr(history_node, "to_dataset") else history_node.ds
    log = list(history.attrs.get("log") or [])
    log.append(
        f"[{datetime.datetime.now().time()}] Wrote /processed/{safe} ({tensor_data.data_type})"
    )
    history = history.copy()
    history.attrs["log"] = log
    tree["history"] = history
    return tree

@staticmethod
def list_processed_children(tree: DataTree) -> list[str]:
    try:
        processed = tree["processed"]
    except Exception:
        return []
    names: list[str] = []
    for name, child in processed.children.items():
        ds = child.to_dataset() if hasattr(child, "to_dataset") else child.ds
        if ds is not None and "data" in ds:
            names.append(str(name))
    return sorted(names)
```

`workspace.py` — fix `pull_tensor_data` membership check:

```python
node = node.strip("/")
try:
    target = tree[node]
except Exception:
    print(f"Error: Node '{node}' does not exist in dataset '{name}'.")
    return None
ds = target.to_dataset() if hasattr(target, "to_dataset") else target
```

Add:

```python
def write_processed_child_data(self, name: str, child_name: str, tensor_data: TensorData) -> bool:
    item = self._data.get(name)
    if not item or item.get("type") != "spectroscopy_tree":
        return False
    item["tree"] = DataTreeBuilder.write_processed_child(
        item["tree"], child_name, tensor_data
    )
    return True

def list_processed_children(self, name: str) -> list[str]:
    item = self._data.get(name)
    if not item or item.get("type") != "spectroscopy_tree":
        return []
    return DataTreeBuilder.list_processed_children(item["tree"])
```

- [ ] **Step 4: Run — expect PASS**

```bash
PYTHONPATH=. TensorSpec_env/bin/pytest tests/test_peem_data_tree.py tests/test_peem_engine.py::TestSeparatePairs -v
```

- [ ] **Step 5: Commit**

```bash
git add tensorspec/core/data_tree.py tensorspec/core/workspace.py tests/test_peem_data_tree.py
git commit -m "$(cat <<'EOF'
feat(workspace): write/pull /processed children

Support nested PEEM channel nodes without wiping paired parent.
EOF
)"
```

---

### Task 3: API schemas + `/separate` + meta/frame

**Files:**
- Modify: `tensorspec/web/server/schemas.py`
- Modify: `tensorspec/web/server/routers/peem.py`
- Modify: `tests/test_peem_api.py`

**Interfaces:**
- Consumes: `separate_pairs`, `write_processed_child_data`, `_processed_pair_tensor`
- Produces: `POST /api/peem/{name}/separate` → `PeemSeparateSummary`; meta `separated_channels`; frame `node=processed/{tag}`

- [ ] **Step 1: Extend schemas**

```python
class PeemSeparateSummary(BaseModel):
    """Summary after splitting paired /processed into channel children."""

    name: str
    channels: list[str]
    n_frames: int
    has_separated: bool = True
    shape: list[int]  # per-channel (n_frames, y, x) — same for both


# PeemMeta additions:
separated_channels: list[str] = Field(default_factory=list)
```

- [ ] **Step 2: Write failing API tests**

In `tests/test_peem_api.py`, add (reuse existing load helpers / fixtures from that file — mirror pair+drift style):

```python
def test_separate_after_pair(client, peem_session_name):
    # Assume helper already loaded a CP/CM sequence as peem_session_name
    r = client.post(f"/api/peem/{peem_session_name}/pair", json={"mode": "CP_CM"})
    assert r.status_code == 200
    r = client.post(f"/api/peem/{peem_session_name}/separate")
    assert r.status_code == 200
    body = r.json()
    assert body["has_separated"] is True
    assert set(body["channels"]) == {"CP", "CM"}
    assert body["n_frames"] >= 1

    meta = client.get(f"/api/peem/{peem_session_name}/meta").json()
    assert set(meta["separated_channels"]) == {"CP", "CM"}
    assert meta["processed_is_paired"] is True

    frame = client.get(
        f"/api/peem/{peem_session_name}/frame/0",
        params={"node": "processed/CP"},
    )
    assert frame.status_code == 200
    assert frame.json()["node"] == "processed/CP"
    assert frame.json()["channel_tag"] == "CP"

    # paired still works
    paired = client.get(
        f"/api/peem/{peem_session_name}/frame/0",
        params={"node": "processed", "channel": 0},
    )
    assert paired.status_code == 200


def test_separate_without_pair_422(client, peem_session_name):
    # loaded raw only, no pair
    r = client.post(f"/api/peem/{peem_session_name}/separate")
    assert r.status_code == 422
```

Adapt fixture names to whatever `test_peem_api.py` already uses (copy the load+pair pattern from existing `test_pair_*` / `test_drift_*` tests — do not invent a new session harness).

- [ ] **Step 3: Run — expect FAIL**

```bash
PYTHONPATH=. TensorSpec_env/bin/pytest tests/test_peem_api.py -k separate -v
```

- [ ] **Step 4: Implement router**

Import `separate_pairs`. Add helpers:

```python
def _separated_tensor(session: Session, name: str, node: str):
    """Pull processed/<tag> channel stack; validate 3D PEEM channel cube."""
    rel = node.strip("/")
    if not rel.startswith("processed/") or rel.count("/") != 1:
        raise HTTPException(status_code=422, detail="Invalid separated node path.")
    tensor = session.workspace.pull_tensor_data(name, rel)
    if tensor is None:
        raise HTTPException(status_code=404, detail=f"No data at '{rel}'.")
    tag = rel.split("/", 1)[1]
    ok = (
        tensor.value.ndim == 3
        and list(tensor.labels) == ["frame", "y", "x"]
        and tensor.data_type.startswith("Experimental PEEM")
    )
    if not ok:
        raise HTTPException(status_code=422, detail=f"Invalid channel stack at '{rel}'.")
    return tensor, tag


@router.post("/{name}/separate", response_model=PeemSeparateSummary)
def separate_peem(name: str, session: Session = Depends(current_session)) -> PeemSeparateSummary:
    _require_tensor(session, name)
    paired = _processed_pair_tensor(session, name)
    if paired is None:
        raise HTTPException(
            status_code=422,
            detail="Separate requires a paired /processed cube. Run Stack Pairs first.",
        )
    try:
        channels = separate_pairs(paired)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    for tag, td in channels.items():
        if not session.workspace.write_processed_child_data(name, tag, td):
            raise HTTPException(status_code=404, detail=f"PEEM data '{name}' not found.")
    tags = sorted(channels)
    sample = channels[tags[0]]
    return PeemSeparateSummary(
        name=name,
        channels=tags,
        n_frames=int(sample.value.shape[0]),
        shape=[int(s) for s in sample.value.shape],
    )
```

Update `get_meta` to set:

```python
separated_channels=session.workspace.list_processed_children(name),
```

Update `get_frame`:

```python
elif node == "processed":
    ...  # existing
elif node.startswith("processed/"):
    tensor, tag = _separated_tensor(session, name, node)
    if not 0 <= i < tensor.value.shape[0]:
        raise HTTPException(status_code=404, detail=f"Frame index {i} is out of range.")
    frame = np.asarray(tensor.value[i], dtype=float)
    # respond with node=node, channel_tag=tag, pol=tag, pair=None, channel=None
else:
    raise HTTPException(
        status_code=422,
        detail="node must be 'raw', 'processed', or 'processed/<tag>'.",
    )
```

Wire response fields for the separated branch the same way raw frames are built (vmin/vmax percentiles, shape, intensity).

- [ ] **Step 5: Run — expect PASS**

```bash
PYTHONPATH=. TensorSpec_env/bin/pytest tests/test_peem_api.py -k "separate or pair or drift or meta or frame" -v
```

Also run full PEEM suite:

```bash
PYTHONPATH=. TensorSpec_env/bin/pytest tests/test_peem_api.py tests/test_peem_engine.py tests/test_peem_data_tree.py tests/test_peem_drift.py -v
```

- [ ] **Step 6: Commit**

```bash
git add tensorspec/web/server/schemas.py tensorspec/web/server/routers/peem.py tests/test_peem_api.py
git commit -m "$(cat <<'EOF'
feat(peem): POST /separate and channel frame nodes

Expose separated /processed/{tag} stacks in meta and frame GET.
EOF
)"
```

---

### Task 4: Suite UI — Separate button + channel view

**Files:**
- Modify: `tensorspec/web/static/js/api.js`
- Modify: `tensorspec/web/static/js/peem_suite.js`
- Modify: `tensorspec/web/templates/suites/peem_suite.html`

**Interfaces:**
- Consumes: `PeemSeparateSummary`, `PeemMeta.separated_channels`
- Produces: enabled Separate; viewer radios Raw | Processed/Paired | each tag

- [ ] **Step 1: HTML**

Replace disabled Separate button:

```html
<button type="button" class="btn btn--block" id="peem-separate-pairs" disabled>&#9878; Separate Pairs</button>
```

In View fieldset, keep Raw + Processed radios; add host for dynamic channel radios:

```html
<div class="inline" id="peem-separated-nodes"></div>
```

Update planned list item to note Separate is live (or remove that bullet).

- [ ] **Step 2: api.js**

```javascript
peemSeparate: (name) =>
    request(`/api/peem/${encodeURIComponent(name)}/separate`, {
        method: "POST",
        body: JSON.stringify({}),
    }),
peemFrame: (name, i, { node = "raw", channel = 0 } = {}) => {
    const query = new URLSearchParams({ node });
    if (node === "processed") query.set("channel", String(channel));
    return request(
        `/api/peem/${encodeURIComponent(name)}/frame/${i}?${query}`
    );
},
```

(Empty JSON body is fine even if endpoint takes no body — or omit body if router has no body model.)

Prefer **no body** if endpoint has no request model:

```javascript
peemSeparate: (name) =>
    request(`/api/peem/${encodeURIComponent(name)}/separate`, { method: "POST" }),
```

- [ ] **Step 3: peem_suite.js**

Add `dom.separatePairs`, `dom.separatedNodes`.

State:

```javascript
separatedChannels: [],
separatedFrameCounts: {}, // optional; default use nPairs after separate
```

Helpers:

```javascript
function isSeparatedNode(node = state.node) {
    return typeof node === "string" && node.startsWith("processed/");
}

function viewerUsesFrameNav(node = state.node) {
    return node === "raw" || (node === "processed" && !state.processedIsPaired) || isSeparatedNode(node);
}

function viewerFrameCount(node = state.node) {
    if (node === "raw") return state.nFrames;
    if (isSeparatedNode(node)) return state.nPairs || state.nProcessedFrames;
    return state.processedIsPaired ? state.nPairs : state.nProcessedFrames;
}
```

In `configureViewer(summary)`:

```javascript
state.separatedChannels = summary.separated_channels || [];
// rebuild #peem-separated-nodes radios: value=`processed/${tag}`, label=tag
// enable separate button when processedIsPaired
dom.separatePairs.disabled = !state.processedIsPaired;
// if current node is a separated channel no longer listed, fall back to processed or raw
// label processed radio text: "Paired" when processedIsPaired else "Processed"
```

Wire click:

```javascript
dom.separatePairs.addEventListener("click", async () => {
    if (!state.name || !state.processedIsPaired) return;
    setBusy(true, "Separating channels…");
    try {
        const summary = await TensorSpecAPI.peemSeparate(state.name);
        const meta = await TensorSpecAPI.peemMeta(state.name);
        configureViewer({ ...meta, n_frames: meta.n_frames });
        dom.status.textContent = `Separated ${summary.channels.join(", ")} (${summary.n_frames} frames)`;
        // optionally switch view to first channel
        if (summary.channels.length) {
            state.node = `processed/${summary.channels[0]}`;
            configureViewer(meta);
        }
        await showFrame(viewerFrameIndex());
    } catch (err) {
        dom.status.textContent = String(err.message || err);
    } finally {
        setBusy(false);
    }
});
```

Ensure `showFrame` / node radio listeners pass `state.node` through (already does for raw/processed).

After Stack Pairs / Apply Drift, call meta refresh so `separated_channels` clears if wipe happened.

- [ ] **Step 4: Manual smoke (local)**

```bash
# with uvicorn running on HTML_einstein_app
# Load CP/CM sequence → Stack → Separate → view CP → view CM → view Paired
```

- [ ] **Step 5: Commit**

```bash
git add tensorspec/web/static/js/api.js tensorspec/web/static/js/peem_suite.js tensorspec/web/templates/suites/peem_suite.html
git commit -m "$(cat <<'EOF'
feat(peem): enable Separate Pairs in suite UI

Channel radios for /processed/{tag}; keep paired view.
EOF
)"
```

---

### Task 5: Docs + verify + Einstein

**Files:**
- Modify: `roadmap.md`
- Modify: `README.md`

- [ ] **Step 1: Roadmap / README**

`roadmap.md`: mark separate checkbox `[x]`; leave energy/BG/sum-rule unchecked.

`README.md`: extend PEEM blurb — pair stack, ROI drift, **separate channels under `/processed/{tag}`**.

- [ ] **Step 2: Full PEEM test pass**

```bash
PYTHONPATH=. TensorSpec_env/bin/pytest tests/test_peem_*.py -v
```

Expected: all PASS.

- [ ] **Step 3: Commit docs**

```bash
git add roadmap.md README.md
git commit -m "$(cat <<'EOF'
docs: mark PEEM separate channels shipped
EOF
)"
```

- [ ] **Step 4: Push + Einstein**

```bash
git push -u origin HEAD
ssh einstein 'cd ~/TensorSpec && git fetch && git checkout HTML_einstein_app && git pull'
# restart uvicorn if needed; curl health
```

- [ ] **Step 5: Done checklist**

- Separate creates `/processed/{tag}`; paired still at `node=processed`
- Meta lists `separated_channels`; frame GET works per channel
- 422 without paired processed
- Unit + API tests green
- Suite Separate enabled
- Branch `HTML_einstein_app` only; Einstein updated

---

## Spec coverage self-review

| Spec requirement | Task |
|------------------|------|
| `separate_pairs` → dict of channel stacks | Task 1 |
| `write_processed_child`; keep flat `/processed` | Task 2 |
| Nested `pull_tensor_data` | Task 2 |
| History log on separate | Task 2 (`write_processed_child`) |
| `POST /separate` | Task 3 |
| Meta `separated_channels` | Task 3 |
| Frame `processed/{tag}` | Task 3 |
| 422 without paired cube | Task 3 |
| UI Separate + Raw\|Paired\|channels | Task 4 |
| Roadmap / README / Einstein | Task 5 |
| Energy align deferred | Explicit non-goal; no task |

## Placeholder scan

No TBD / “implement later” steps remain. DataTree coexistence resolved by spike (keep flat parent). Tag sanitizer specified.
