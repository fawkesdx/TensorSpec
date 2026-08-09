# PEEM Pair Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pair PEEM raw frames into `(pair, channel, y, x)` on `/processed` with Auto|CP_CM|LH_LV modes, enable Stack Pairs UI, and view Raw|Processed with pair+channel navigation.

**Architecture:** `peem_engine.pair_stack` builds the paired `TensorData` from raw + `pol` tags. PEEM router `POST /pair` writes via `write_processed_data`. Meta/frame gain `node` / pair / channel. Suite enables Contrast Pairing (Separate stays disabled).

**Tech Stack:** NumPy, existing DataTree/workspace, FastAPI, vanilla JS canvas viewer.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-09-peem-pair-stack-design.md`
- Branch: `HTML_einstein_app` only — never merge to `main`
- `/raw` untouched; paired cube only in `/processed`
- No Separate, drift, BG, sum-rule, or I0 application
- Reuse existing load-time `pol` tags — do not change loaders
- After ship: push + Einstein pull/restart if needed
- Update `roadmap.md` / README when pairing ships

## File map

| File | Role |
|------|------|
| `tensorspec/core/peem_engine.py` | `pair_stack` (+ mode resolve helpers) |
| `tests/test_peem_engine.py` | Unit tests for pairing |
| `tensorspec/web/server/schemas.py` | `PeemPairRequest`, `PeemPairSummary`, extend `PeemMeta` / `PeemFrame` |
| `tensorspec/web/server/routers/peem.py` | `/pair`, meta+frame node/pair/channel |
| `tests/test_peem_api.py` | API pair + processed frame tests |
| `tensorspec/web/static/js/api.js` | `peemPair`, extend `peemFrame` / `peemMeta` |
| `tensorspec/web/static/js/peem_suite.js` | Stack Pairs + Raw/Processed viewer |
| `tensorspec/web/templates/suites/peem_suite.html` | Enable pairing controls + viewer chrome |
| `roadmap.md` / `README.md` | Mark stack checkbox |

---

### Task 1: `peem_engine.pair_stack` + unit tests

**Files:**
- Create: `tensorspec/core/peem_engine.py`
- Create: `tests/test_peem_engine.py`

**Interfaces:**
- Produces:

```python
from typing import Literal
from tensorspec.core.data_models import TensorData

PairMode = Literal["auto", "CP_CM", "LH_LV"]

def resolve_pair_mode(pol: list[str], mode: PairMode) -> tuple[str, list[str]]:
    """
    Returns (resolved_mode, channel_tags).
    auto: CP/CM-only → ("CP_CM", ["CP","CM"]); LH/LV-only → ("LH_LV", ["LH","LV"]);
    mixed or none → ValueError.
    """
    ...

def pair_stack(tensor: TensorData, mode: PairMode) -> TensorData:
    """
    Input shape (frame,y,x) + metadata['pol'] (+ optional frame_names).
    Output shape (n_pairs, 2, y, x), data_type "Experimental PEEM (paired)".
    Metadata: pair_mode, channel_tags, pair_sources, unpaired, plus CSV/I0 passthrough.
    """
    ...
```

- Consumes: `TensorData` only (no FastAPI)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_peem_engine.py
import unittest
import numpy as np
from tensorspec.core.data_models import TensorData
from tensorspec.core import peem_engine as eng


def _raw(pols, frames=None):
    n = len(pols)
    if frames is None:
        frames = np.stack([np.full((2, 2), i + 1, dtype=float) for i in range(n)], axis=0)
    return TensorData(
        value=frames,
        axes=[np.arange(n), np.arange(2), np.arange(2)],
        labels=["frame", "y", "x"],
        units=["", "px", "px"],
        data_type="Experimental PEEM",
        metadata={
            "pol": list(pols),
            "frame_names": [f"f{i}_{p}" for i, p in enumerate(pols)],
            "csv_attached": True,
            "I0": [1.0] * n,
            "source": "test",
            "loader": "tif_sequence",
        },
    )


class TestPairStack(unittest.TestCase):
    def test_cp_cm_happy(self):
        out = eng.pair_stack(_raw(["CP", "CM", "CP", "CM"]), "CP_CM")
        self.assertEqual(out.value.shape, (2, 2, 2, 2))
        self.assertEqual(out.labels, ["pair", "channel", "y", "x"])
        self.assertEqual(out.metadata["channel_tags"], ["CP", "CM"])
        self.assertEqual(out.metadata["unpaired"], [])
        np.testing.assert_array_equal(out.value[0, 0], np.full((2, 2), 1.0))
        np.testing.assert_array_equal(out.value[0, 1], np.full((2, 2), 2.0))

    def test_auto_cp_cm(self):
        out = eng.pair_stack(_raw(["CM", "CP"]), "auto")
        self.assertEqual(out.metadata["pair_mode"], "CP_CM")
        # file order: first CM with first CP
        self.assertEqual(out.value.shape[0], 1)

    def test_unequal_leftovers(self):
        out = eng.pair_stack(_raw(["CP", "CP", "CM"]), "CP_CM")
        self.assertEqual(out.value.shape[0], 1)
        self.assertEqual(len(out.metadata["unpaired"]), 1)
        self.assertEqual(out.metadata["unpaired"][0]["pol"], "CP")

    def test_mixed_auto_fails(self):
        with self.assertRaises(ValueError):
            eng.pair_stack(_raw(["CP", "CM", "LH"]), "auto")

    def test_zero_pairs_fails(self):
        with self.assertRaises(ValueError):
            eng.pair_stack(_raw(["CP", "CP"]), "CP_CM")
```

- [ ] **Step 2: Run — expect FAIL**

```bash
PYTHONPATH=. TensorSpec_env/bin/pytest tests/test_peem_engine.py -v
```

- [ ] **Step 3: Implement `peem_engine.py`**

Pairing: queues per tag in file order; while both channel queues non-empty, pop left from each. Pass through `csv_attached`, `I0`, `source`, `loader`, `beamline_csv`, `beamline_table` if present. JSON-friendly metadata only (lists/dicts/str/float/None).

- [ ] **Step 4: Run — expect PASS**

```bash
PYTHONPATH=. TensorSpec_env/bin/pytest tests/test_peem_engine.py -v
```

- [ ] **Step 5: Commit**

```bash
git add tensorspec/core/peem_engine.py tests/test_peem_engine.py
git commit -m "feat(peem): pair_stack engine for CP/CM and LH/LV"
```

---

### Task 2: Schemas + PEEM pair API

**Files:**
- Modify: `tensorspec/web/server/schemas.py`
- Modify: `tensorspec/web/server/routers/peem.py`
- Modify: `tests/test_peem_api.py`

**Interfaces:**
- Produces:

```python
class PeemPairRequest(BaseModel):
    mode: Literal["auto", "CP_CM", "LH_LV"] = "auto"

class PeemPairSummary(BaseModel):
    name: str
    n_pairs: int
    channel_tags: list[str]
    unpaired_count: int
    mode: str
    has_processed: bool = True
    shape: list[int]

# Extend PeemMeta:
#   has_processed: bool = False
#   pair_mode: str | None = None
#   n_pairs: int | None = None
#   channel_tags: list[str] = []
#   unpaired_count: int = 0

# Extend PeemFrame (optional fields for processed):
#   node: str = "raw"
#   pair: int | None = None
#   channel: int | None = None
#   channel_tag: str | None = None
```

```python
# peem.py
@router.post("/{name}/pair", response_model=PeemPairSummary)
def pair_peem(name: str, request: PeemPairRequest, session: Session = Depends(...)): ...

# get_meta: if tree has processed child, fill has_processed + pair fields from processed attrs
# get_frame(name, i, node="raw", pair: int | None = None, channel: int | None = None):
#   raw: existing behavior (i = frame index)
#   processed: require pair + channel (or use i as pair and channel query); return plane [pair, channel, :, :]
```

**Processed frame contract (lock):**  
`GET /api/peem/{name}/frame/{pair}?node=processed&channel=0|1`  
Path `{pair}` is pair index; `channel` query required when `node=processed` (default `0`).

**Helper:** detect processed existence via workspace tree (`"processed" in tree` / pull_tensor_data(name, "processed") is not None).

- Consumes: Task 1 `pair_stack`; `pull_tensor_data`; `write_processed_data`

- [ ] **Step 1: Write failing API tests** (append to `tests/test_peem_api.py`)

```python
def test_pair_writes_processed_and_frame(self):
    with tempfile.TemporaryDirectory() as tmp:
        session = self._session(tmp)
        root = Path(tmp) / "run"
        root.mkdir()
        tifffile.imwrite(root / "a_CP.tif", np.ones((4, 4), dtype=np.uint16))
        tifffile.imwrite(root / "b_CM.tif", np.full((4, 4), 2, dtype=np.uint16))
        peem_router.load_peem(
            file=None, server_path=str(root), csv=None, csv_path=None,
            name="pair_me", session=session,
        )
        summary = peem_router.pair_peem(
            "pair_me",
            peem_router.PeemPairRequest(mode="CP_CM")
            if hasattr(peem_router, "PeemPairRequest")
            else __import__("tensorspec.web.server.schemas", fromlist=["PeemPairRequest"]).PeemPairRequest(mode="CP_CM"),
            session=session,
        )
        # Prefer importing PeemPairRequest from schemas in the real test file
        self.assertEqual(summary.n_pairs, 1)
        self.assertTrue(summary.has_processed)
        meta = peem_router.get_meta("pair_me", session=session)
        self.assertTrue(meta.has_processed)
        self.assertEqual(meta.n_pairs, 1)
        frame = peem_router.get_frame(
            "pair_me", 0, node="processed", channel=1, session=session
        )
        self.assertEqual(frame.shape, [4, 4])
        self.assertEqual(frame.channel_tag, "CM")
```

Import `PeemPairRequest` from `schemas` in the test (clean). Also add unequal unpaired_count test.

- [ ] **Step 2: Run — expect FAIL**

```bash
PYTHONPATH=. TensorSpec_env/bin/pytest tests/test_peem_api.py -k pair -v
```

- [ ] **Step 3: Implement schemas + router**

`get_meta` today pulls only raw — extend to also inspect processed without breaking raw fields (`n_frames` / `pol` stay raw-oriented; add processed fields alongside).

`get_frame` signature change: add query params `node: str = "raw"`, `channel: int = 0`. When `node=="processed"`, `i` is pair index.

- [ ] **Step 4: Run full PEEM tests — expect PASS**

```bash
PYTHONPATH=. TensorSpec_env/bin/pytest tests/test_peem_engine.py tests/test_peem_api.py -q
```

- [ ] **Step 5: Commit**

```bash
git add tensorspec/web/server/schemas.py tensorspec/web/server/routers/peem.py tests/test_peem_api.py
git commit -m "feat(peem): pair API and processed frame access"
```

---

### Task 3: Suite UI — Stack Pairs + Raw/Processed viewer

**Files:**
- Modify: `tensorspec/web/static/js/api.js`
- Modify: `tensorspec/web/static/js/peem_suite.js`
- Modify: `tensorspec/web/templates/suites/peem_suite.html`

**Interfaces:**
- Produces:

```javascript
peemPair: (name, mode = "auto") =>
  request(`/api/peem/${encodeURIComponent(name)}/pair`, {
    method: "POST",
    body: JSON.stringify({ mode }),
  }),
peemFrame: (name, i, { node = "raw", channel = 0 } = {}) => {
  const q = new URLSearchParams({ node });
  if (node === "processed") q.set("channel", String(channel));
  return request(`/api/peem/${encodeURIComponent(name)}/frame/${i}?${q}`);
},
```

**HTML:**
- Remove `disabled` from Contrast Pairing fieldset (keep Separate button disabled).
- Mode `<select id="peem-mode">` values: `auto`, `CP_CM`, `LH_LV` (not display-only text).
- Stack button `id="peem-stack-pairs"` enabled.
- Viewer: Raw | Processed radio/toggle; `#peem-pair` slider; `#peem-channel` select (labels from `channel_tags`).

**JS:**
- On Stack: call `peemPair`, update status (`n_pairs`, unpaired), set `state.node = "processed"`, refresh meta/sliders, show first pair/ch0.
- `showFrame`: pass `node` + `channel`; preserve clim scrubbing behavior from load+view.
- Processed disabled in toggle until `has_processed`.

- [ ] **Step 1: Wire `api.js`**

- [ ] **Step 2: Update `peem_suite.html` controls**

- [ ] **Step 3: Implement `peem_suite.js` pairing + viewer modes**

- [ ] **Step 4: Static sanity** — ids in HTML match JS `getElementById`; no syntax errors

```bash
node --check tensorspec/web/static/js/peem_suite.js
node --check tensorspec/web/static/js/api.js
```

- [ ] **Step 5: Commit**

```bash
git add tensorspec/web/static/js/api.js tensorspec/web/static/js/peem_suite.js \
  tensorspec/web/templates/suites/peem_suite.html
git commit -m "feat(peem): Stack Pairs UI and processed viewer"
```

---

### Task 4: Docs + Einstein deploy

**Files:**
- Modify: `roadmap.md` — check “stack the CP and CM together or LH and LV…”
- Modify: `README.md` — PEEM bullet mentions pair stack → `/processed` if Key Features lists PEEM

- [ ] **Step 1: Update roadmap/README** (do not check Separate / drift / sum-rule)

- [ ] **Step 2: Commit**

```bash
git add roadmap.md README.md
git commit -m "docs: mark PEEM pair stack shipped"
```

- [ ] **Step 3: Push + Einstein**

```bash
git push origin HEAD
ssh einstein 'cd ~/TensorSpec && git fetch && git checkout HTML_einstein_app && git pull && curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/api/health'
```

- [ ] **Step 4: Verify** health 200; OpenAPI or curl shows `/api/peem/{name}/pair`

---

## Spec coverage (self-review)

| Spec item | Task |
|-----------|------|
| `pair_stack` + modes + unpaired | 1 |
| `/processed` via write_processed | 2 |
| POST /pair, meta, frame node/channel | 2 |
| Stack UI + Raw/Processed viewer | 3 |
| Separate/drift/BG deferred | 3 (left disabled) |
| Roadmap + Einstein | 4 |
| Tests engine + API | 1, 2 |

## Placeholder scan

No TBD steps; frame URL contract locked; schema fields named.

## Type consistency

- Modes: `"auto" | "CP_CM" | "LH_LV"` everywhere (HTML option values match).
- `has_processed`, `n_pairs`, `channel_tags`, `unpaired_count` shared by summary/meta/UI.
- Processed frame: path pair index + `channel` query.
