# PEEM Linear Pre-edge BG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship linear pre-edge BG on a mean spectrum (picture-wide or ROI), preview + apply, write `/analysis/background` and `/processed/bg` (or `{tag}_bg`), light window ensemble, suite plot with window drag.

**Architecture:** Pure NumPy `peem_bg` engine → thin PEEM router preview/apply → reuse `write_analysis` + `write_processed_child` → suite BG panel with canvas spectrum plot. Energy from CSV aliases when length-matched, else frame index.

**Tech Stack:** NumPy, xarray Dataset for analysis node, FastAPI, existing ROI + nested processed children.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-11-peem-bg-preedge-design.md`
- Branch: `HTML_einstein_app` only — never merge to `main`
- Linear pre-edge only; light ensemble; no pixel-to-pixel, spatial-2D, sum-rule, I0 apply
- Whole-image apply of `bg(E)`; ROI only for spectrum estimate
- Do not overwrite flat paired `/processed` or `/raw`
- Tests: `PYTHONPATH=. TensorSpec_env/bin/pytest …`
- After ship: push + Einstein pull/restart; update roadmap/README partially

## File map

| File | Role |
|------|------|
| `tensorspec/core/peem_bg.py` | extract / fit / ensemble / apply / energy resolve / analysis Dataset builder |
| `tests/test_peem_bg.py` | Unit tests |
| `tensorspec/web/server/schemas.py` | BG request/response/meta fields |
| `tensorspec/web/server/routers/peem.py` | preview/apply/spectrum + meta/frame |
| `tests/test_peem_api.py` | API BG tests |
| `tensorspec/web/static/js/api.js` | peemBgPreview / peemBgApply |
| `tensorspec/web/static/js/peem_suite.js` | BG UI + spectrum plot |
| `tensorspec/web/templates/suites/peem_suite.html` | Enable BG controls + plot host |
| `roadmap.md` / `README.md` | Partial BG shipped note |

## Locked details

- **Energy aliases** (case-insensitive): `energy`, `E`, `hv`, `photon_energy`, `PhotonEnergy`, `eV` — first match in `beamline_table.series` with `len == n_frames`.
- **Ensemble defaults:** `ensemble_delta` default = 5% of energy span (or 1.0 if index); `ensemble_n` default 21; caps `1≤n≤101`, `delta≥0`.
- **Child name:** source `raw` or flat `processed` (3D) → `bg`; source `processed/{tag}` → `{tag}_bg`.
- **Analysis node name:** `background` via `write_analysis_data`.
- **RNG:** `numpy.random.default_rng(0)` for reproducible ensemble in tests; API may pass optional `seed`.

---

### Task 1: `peem_bg` engine + unit tests

**Files:**
- Create: `tensorspec/core/peem_bg.py`
- Create: `tests/test_peem_bg.py`

**Interfaces:**

```python
ENERGY_ALIASES = ("energy", "E", "hv", "photon_energy", "PhotonEnergy", "eV")

def resolve_energy(n_frames: int, metadata: dict) -> tuple[np.ndarray, str]:
    """Return (energy, source) where source is 'csv' or 'index'."""

def extract_spectrum(stack: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """stack (n,y,x) → spectrum (n,). mask None = all pixels."""

def fit_linear_preedge(
    energy: np.ndarray, spectrum: np.ndarray, e0: float, e1: float
) -> dict:
    """Return slope, intercept, bg (full axis). ValueError if <2 points in window."""

def ensemble_preedge(
    energy, spectrum, e0, e1, *, delta: float, n: int, seed: int = 0
) -> dict:
    """bg_mean, bg_std, subtracted_mean, subtracted_std, n_valid."""

def apply_bg_to_stack(stack: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """I'[..., i] = I[..., i] - bg[i]."""

def analysis_dataset(...) -> xr.Dataset: ...
def bg_child_name(source_node: str) -> str: ...
```

- [ ] **Step 1: Write failing tests** in `tests/test_peem_bg.py` covering: picture-wide extract, ROI mask extract, linear fit known line, invalid window, ensemble std>0, apply subtracts constant ramp, resolve_energy csv vs index, `bg_child_name`.

- [ ] **Step 2:** `PYTHONPATH=. TensorSpec_env/bin/pytest tests/test_peem_bg.py -v` → FAIL

- [ ] **Step 3: Implement `peem_bg.py`**

- [ ] **Step 4:** pytest → PASS

- [ ] **Step 5: Commit**

```bash
git add tensorspec/core/peem_bg.py tests/test_peem_bg.py
git commit -m "$(cat <<'EOF'
feat(peem): linear pre-edge BG engine

Extract/fit/ensemble/apply for mean spectra; energy CSV or index.
EOF
)"
```

---

### Task 2: Schemas + API preview/apply + meta/frame

**Files:**
- Modify: `schemas.py`, `peem.py`, `tests/test_peem_api.py`

**Interfaces:**
- `PeemBgRequest`, `PeemBgPreviewResponse`, `PeemBgApplySummary`
- Meta: `has_background`, `has_processed_bg`, `energy_source`, `processed_bg_node`
- `POST /bg/preview`, `POST /bg/apply`, `GET /bg/spectrum`
- Pull 3D stack from node (raw / processed 3D / processed/tag / paired+channel)
- Apply: `write_analysis_data(..., "background", ds)` + `write_processed_child_data(..., child, tensor)`

- [ ] **Step 1: Schemas**

```python
class PeemBgRequest(BaseModel):
    node: str = "raw"
    channel: int = Field(default=0, ge=0, le=1)
    use_roi: bool = False
    roi: PeemRoi | None = None
    e0: float
    e1: float
    ensemble_delta: float | None = None
    ensemble_n: int = Field(default=21, ge=1, le=101)
    seed: int = 0

class PeemBgPreviewResponse(BaseModel):
    energy: list[float]
    spectrum: list[float]
    bg: list[float]
    bg_std: list[float]
    subtracted: list[float]
    subtracted_std: list[float]
    slope: float
    intercept: float
    energy_source: str
    e0: float
    e1: float
    ensemble_n_valid: int

class PeemBgApplySummary(BaseModel):
    name: str
    analysis_node: str = "background"
    processed_bg_node: str
    n_frames: int
    shape: list[int]
    has_background: bool = True
    energy_source: str
```

- [ ] **Step 2: Failing API tests** — load ramp stack → preview → apply → meta flags → frame `processed/bg`; preview does not create analysis; ROI path; 422 bad window. Mirror existing `TestPeemApi` harness.

- [ ] **Step 3:** Implement router helpers + endpoints; extend `get_meta` / `get_frame` for `processed/bg` and `*_bg` children (reuse `_separated_tensor`-style or generalize processed child 3D pull).

- [ ] **Step 4:** `pytest tests/test_peem_api.py -k bg -v` and full peem suite green

- [ ] **Step 5: Commit** `feat(peem): BG preview/apply API`

---

### Task 3: Suite UI — BG panel + spectrum plot

**Files:**
- Modify: `api.js`, `peem_suite.js`, `peem_suite.html`

- [ ] **Step 1: HTML** — enable BG fieldset (keep sum-rule disabled); add e0/e1, ensemble fields, use-ROI checkbox, Preview/Apply buttons, `<canvas id="peem-bg-plot">`, toggle checkboxes for raw/bg/band/subtracted; viewer radio host for BG node.

- [ ] **Step 2: api.js** — `peemBgPreview`, `peemBgApply`, optional `peemBgSpectrum`

- [ ] **Step 3: peem_suite.js** — wire preview/apply with busy+name guards (mirror Separate); draw spectrum; drag window updates e0/e1; after apply refresh meta + add `processed/bg` radio; pass ROI when use_roi checked

- [ ] **Step 4: Commit** `feat(peem): enable linear pre-edge BG in suite UI`

---

### Task 4: Docs + push + Einstein

- [ ] Roadmap: mark linear pre-edge / apply-to-stack partial; leave multi-spectra, pixel-to-pixel, sum-rule open
- [ ] README PEEM blurb
- [ ] `pytest tests/test_peem_*.py -v`
- [ ] Commit docs, push, Einstein pull/restart, health check

---

## Spec coverage

| Spec | Task |
|------|------|
| extract/fit/ensemble/apply | 1 |
| energy csv/index | 1–2 |
| analysis + processed/bg | 2 |
| preview/apply API + meta/frame | 2 |
| plot + window drag + suite | 3 |
| docs/Einstein | 4 |

## Placeholder scan

No TBD. Ensemble caps and energy aliases locked above.
