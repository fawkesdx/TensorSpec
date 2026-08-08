# Suite Gaps Track B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Track B: QE `input_dft` XC picker; ARPES slit/defl resolution model; DFT 2D isoenergy heatmap; push + Einstein.

**Architecture:** Thin slices. B1 threads `functional` into QE generator. B2 adds pure `resolution.py` helpers + ARPES UI/schema metadata; sim still consumes `res_E` + ky bounds. B3 new `/isoenergy` endpoint on `calculate_2d_mesh` + DFT heatmap UI.

**Tech Stack:** FastAPI, Pydantic, QE input writer, numpy, unittest, DFT/ARPES suite JS.

## Global Constraints

- Branch: `HTML_einstein_app` only — never merge to `main`
- Spec: `docs/superpowers/specs/2026-08-08-suite-gaps-track-b-design.md`
- Tests: `./TensorSpec_env/bin/python -m unittest …`
- After final push: `ssh einstein` pull + restart uvicorn `:8000`
- No HSE; no pseudo XC filter; isoenergy only (no full 2D band UI)

## File map

| File | Role |
|------|------|
| `schemas.py` | `QERequest.functional`; ARPES optional metadata; `IsoenergyRequest/Result` |
| `qe_generator.py`, `qe_pipeline.py`, `dft.py` router | Emit `input_dft`; pass functional |
| `dft_suite.html/js` | `#qe-xc`; isoenergy mode + heatmap |
| `tensorspec/core/arpes/resolution.py` | NEW: ΔE + deflector Δk helpers |
| `arpes_suite.html/js`, `schemas.py`, `arpes.py` | Wire slit/defl/PE/beam/extra |
| `band_service.py` / `dft.py` | Isoenergy density from 2D mesh |
| `tests/test_qe_functional.py`, `test_arpes_resolution.py`, `test_isoenergy.py` | Unit tests |

---

### Task 1: QE XC functional → `input_dft`

**Files:**
- Modify: `tensorspec/web/server/schemas.py` (`QERequest`)
- Modify: `tensorspec/core/dft/qe_pipeline.py` (`PipelineParams`)
- Modify: `tensorspec/core/dft/qe_generator.py` (`write_scf_input`, `write_nscf_input`)
- Modify: `tensorspec/web/server/routers/dft.py` (`_params_from_request`)
- Modify: `tensorspec/web/templates/suites/dft_suite.html`, `dft_suite.js`
- Test: `tests/test_qe_functional.py`

**Interfaces:**
- Produces: `QERequest.functional: Literal["PBE","LDA","PBEsol"] = "PBE"`
- Produces: scf/nscf contain `input_dft = 'pbe'|'lda'|'pbesol'`

- [ ] **Step 1: Failing test**

```python
"""QE functional emits input_dft in scf.in."""
import tempfile
import unittest
from pathlib import Path

from pymatgen.core import Lattice, Structure

from tensorspec.core.dft.qe_generator import QEInputGenerator
from tensorspec.web.server.schemas import QERequest


class TestQEFunctional(unittest.TestCase):
    def setUp(self):
        self.structure = Structure(Lattice.cubic(4.0), ["Si"], [[0, 0, 0]])

    def test_default_pbe(self):
        self.assertEqual(QERequest().functional, "PBE")

    def test_scf_contains_input_dft_lda(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = QEInputGenerator(self.structure, pseudo_dir=None)
            # May need a fake pseudo file — if generator requires UPF, create empty Si*.upf in tmp pseudo
            # Prefer calling write_scf_input with functional= if signature added
            path = gen.write_scf_input(tmp, functional="LDA")
            text = Path(path).read_text()
            self.assertIn("input_dft = 'lda'", text)

    def test_scf_default_pbe_keyword(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = QEInputGenerator(self.structure, pseudo_dir=None)
            path = gen.write_scf_input(tmp, functional="PBE")
            self.assertIn("input_dft = 'pbe'", Path(path).read_text())


if __name__ == "__main__":
    unittest.main()
```

If `QEInputGenerator` requires real UPFs in setUp, create a minimal `Si.upf` stub in a temp `pseudo_dir` and pass it — match patterns in `tests/test_qe_slab.py` if present.

- [ ] **Step 2: Run — expect FAIL**

```bash
./TensorSpec_env/bin/python -m unittest tests.test_qe_functional -v
```

- [ ] **Step 3: Implement**

Schema:

```python
functional: Literal["PBE", "LDA", "PBEsol"] = "PBE"
```

PipelineParams + `_params_from_request` pass through.

Generator helper:

```python
_DFT_MAP = {"PBE": "pbe", "LDA": "lda", "PBEsol": "pbesol"}

def _input_dft_line(self, functional: str) -> str:
    key = _DFT_MAP.get(functional, "pbe")
    return f"\n  input_dft = '{key}'"
```

Inject into `&SYSTEM` block of scf and nscf (after `ecutrho` or with other flags). Add `functional: str = "PBE"` kwarg to `write_scf_input` / `write_nscf_input`. Thread from `qe_pipeline` generate calls.

UI: `#qe-xc` select; `readQeParameters` includes `functional: dom.qeXc?.value || "PBE"`. Hint about PBE pseudos.

- [ ] **Step 4: Tests PASS**

```bash
./TensorSpec_env/bin/python -m unittest tests.test_qe_functional -v
```

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(dft): emit QE input_dft from XC picker"
```

---

### Task 2: ARPES resolution helpers + slit/defl UI

**Files:**
- Create: `tensorspec/core/arpes/resolution.py`
- Create: `tests/test_arpes_resolution.py`
- Modify: `schemas.py` (`ArpesSimRequest` optional fields)
- Modify: `arpes_suite.html`, `arpes_suite.js`
- Modify: `arpes.py` `_experiment_kwargs` if metadata forwarded (optional)

**Interfaces:**
- Produces:
  - `analyzer_delta_e(slit_mm: float, pass_energy: float) -> float`
  - `total_delta_e(ana: float, beam: float, extra: float) -> float`
  - `deflector_dk(hv: float, work_function: float, deflector_deg: float) -> float`
- Produces: UI syncs `#ar-de`; payload `res_E` + shifted `ky`

- [ ] **Step 1: Failing tests**

```python
"""Analyzer ΔE and deflector Δk helpers."""
import math
import unittest

from tensorspec.core.arpes.resolution import (
    analyzer_delta_e,
    total_delta_e,
    deflector_dk,
)


class TestArpesResolution(unittest.TestCase):
    def test_analyzer_0p2mm_20eV(self):
        # (0.2/400)*20 = 0.01 eV
        self.assertAlmostEqual(analyzer_delta_e(0.2, 20.0), 0.01, places=6)

    def test_total_quadrature(self):
        self.assertAlmostEqual(total_delta_e(0.03, 0.04, 0.0), 0.05, places=6)

    def test_deflector_zero(self):
        self.assertAlmostEqual(deflector_dk(90.0, 4.5, 0.0), 0.0, places=9)

    def test_deflector_sign(self):
        dk = deflector_dk(90.0, 4.5, 15.0)
        self.assertGreater(dk, 0.0)
        self.assertAlmostEqual(deflector_dk(90.0, 4.5, -15.0), -dk, places=9)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run — FAIL**

```bash
./TensorSpec_env/bin/python -m unittest tests.test_arpes_resolution -v
```

- [ ] **Step 3: Implement `resolution.py`**

```python
import math

K_FACTOR = 0.5123  # Å^-1 / sqrt(eV)

def analyzer_delta_e(slit_mm: float, pass_energy: float) -> float:
    return (float(slit_mm) / 400.0) * float(pass_energy)

def total_delta_e(ana: float, beam: float = 0.0, extra: float = 0.0) -> float:
    return math.sqrt(max(ana, 0.0) ** 2 + max(beam, 0.0) ** 2 + max(extra, 0.0) ** 2)

def deflector_dk(hv: float, work_function: float, deflector_deg: float) -> float:
    ek = max(float(hv) - float(work_function), 0.0)
    return K_FACTOR * math.sqrt(ek) * math.sin(math.radians(float(deflector_deg)))
```

- [ ] **Step 4: UI + schema**

HTML (section 4):
- Enable `#ar-slitsize`, `#ar-defl`
- Add `#ar-pe` (default 20), `#ar-de-beam` (0.01), `#ar-de-extra` (0)
- Add `#ar-de-manual` checkbox; `#ar-res-status` status line
- Update hint per spec

JS:
- Keep `baseKy = {min, max}` captured once on load / when user edits ky while tracking “base” carefully: on first load set base from inputs; when user manually edits ky min/max with defl=0, update base; when defl changes, set displayed ky = base ± dk (same width).
- `syncResolution()`: compute ana/total; if !manual, set `#ar-de` and readonly; update status text.
- Listeners on slit, pe, beam, extra, manual, hv, phi, defl, ky edits.
- `simPayload`: include shifted ky; `res_E` from `#ar-de`; optional metadata fields.

Schema optional floats/bools: `deflector_angle`, `slit_size_mm`, `pass_energy`, `res_E_beam`, `res_E_extra`, `res_E_manual` — do not change Chinook path beyond existing `res_E` / ky.

- [ ] **Step 5: Tests PASS + commit**

```bash
./TensorSpec_env/bin/python -m unittest tests.test_arpes_resolution -v
git commit -m "feat(arpes): slit/PE resolution model and deflector ky shift"
```

---

### Task 3: DFT isoenergy endpoint + heatmap UI

**Files:**
- Modify: `schemas.py` — `IsoenergyRequest`, `IsoenergyResult`
- Modify: `dft.py` router — `POST /{name}/isoenergy`
- Optionally thin helper in `band_service.py`: `isoenergy_intensity(eigenvalues, energy, smear)`
- Modify: `api.js`, `dft_suite.html`, `dft_suite.js`
- Test: `tests/test_isoenergy.py`

**Interfaces:**
- Produces: `POST /api/dft/{name}/isoenergy` → `{kx, ky, intensity[ny][nx] or flat, energy, smear, …}`
- Formula: `I = Σ_n exp(−(E_n−E)²/(2σ²))` on mesh from `calculate_2d_mesh`

- [ ] **Step 1: Failing unit test for density helper**

```python
import unittest
import numpy as np
from tensorspec.core.dft.band_service import isoenergy_density  # or wherever placed

class TestIsoenergyDensity(unittest.TestCase):
    def test_peaks_at_band(self):
        # eigenvalues shape (nk, nb) with constant band at 0.5
        ev = np.full((4, 2), 0.5)
        near = isoenergy_density(ev, energy=0.5, smear=0.05, grid_shape=(2, 2))
        far = isoenergy_density(ev, energy=5.0, smear=0.05, grid_shape=(2, 2))
        self.assertGreater(near.mean(), far.mean())

    def test_resolution_schema_cap(self):
        from tensorspec.web.server.schemas import IsoenergyRequest
        with self.assertRaises(Exception):
            IsoenergyRequest(resolution=100)  # >48


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: FAIL then implement helper + schema + route**

`IsoenergyRequest`: reuse TB fields from `BandRequest` where practical (hoppings, etc.) + `energy`, `kx_min/max`, `ky_min/max`, `resolution: int = Field(24, ge=4, le=48)`, `smear: float = Field(0.05, ge=0.001, le=1)`.

Route mirrors `compute_bands` structure setup then:

```python
mesh = band_service.calculate_2d_mesh(engine, kx_min=..., ..., resolution=..., hoppings=..., ...)
evals = np.asarray(mesh["eigenvalues"]).reshape(res, res, -1)  # careful with k order
# calculate_2d_mesh uses indexing ij ravel — reshape (res,res,nb) from (res*res, nb)
intensity = isoenergy_density(...)
return IsoenergyResult(...)
```

Budget: `res*res * orbitals**3` vs `DIAGONALISATION_BUDGET`.

- [ ] **Step 3: UI**

- Enable `#tb-kgrid` options: value `path` | `isoenergy`
- Enable `#tb-isoe` when isoenergy
- `api.js`: `dftIsoenergy(name, payload)`
- `calculate()`: if isoenergy mode, call isoenergy API and render heatmap into the plot container (simple canvas: map intensity to grayscale/color; axes kx/ky). Hide or replace BandPlot for that mode.
- Default mesh bounds e.g. −2…2 if no extra inputs — add compact `#tb-kx-min` etc. OR reuse fixed defaults (−2,2) in JS payload for this pass (document in hint). Prefer small optional fields next to isoe if space; else defaults −2…2, resolution 24.

- [ ] **Step 4: Tests PASS + commit**

```bash
./TensorSpec_env/bin/python -m unittest tests.test_isoenergy -v
git commit -m "feat(dft): 2D TB isoenergy cut heatmap"
```

---

### Task 4: Verify + push + Einstein

- [ ] **Step 1: Run Track B tests**

```bash
./TensorSpec_env/bin/python -m unittest \
  tests.test_qe_functional \
  tests.test_arpes_resolution \
  tests.test_isoenergy \
  tests.test_qe_slab \
  -v
```

- [ ] **Step 2: Push**

```bash
git push -u origin HEAD
```

- [ ] **Step 3: Einstein**

```bash
ssh einstein 'cd /home/sandy/TensorSpec && git fetch && git checkout HTML_einstein_app && git pull --ff-only && (pkill -f "uvicorn tensorspec.web.server.app" || true); sleep 1; nohup TensorSpec_env/bin/uvicorn tensorspec.web.server.app:app --host 0.0.0.0 --port 8000 --reload > ~/tensorspec-uvicorn.log 2>&1 & sleep 3; curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/'
```

- [ ] **Step 4: Report smoke checklist** (manual optional)

1. QE generate → scf has `input_dft`  
2. ARPES slit/PE moves dE; manual override  
3. Deflector shifts ky  
4. DFT isoenergy heatmap  

---

## Spec coverage

| Spec | Task |
|------|------|
| B1 QE XC | 1 |
| B2 ARPES resolution/defl | 2 |
| B3 Isoenergy | 3 |
| Deploy | 4 |
