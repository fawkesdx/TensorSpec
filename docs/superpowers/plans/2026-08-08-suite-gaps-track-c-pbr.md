# Suite Gaps Track C — PBR Shiny — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable Crystal Suite `#cr-pbr` to toggle shiny vs matte PBR on atoms and bonds only.

**Architecture:** Pure `pbrMaterialParams(shiny, kind)` drives `MeshStandardMaterial` metalness/roughness in `_drawAtoms` / `_drawBonds`. `CrystalViewer.setPbrShiny(bool)` stores flag and re-renders like `setBondRadius`. Suite wires checkbox change.

**Tech Stack:** three.js, crystal_suite.js, unittest contract tests.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-08-suite-gaps-track-c-pbr-design.md`
- Branch: `HTML_einstein_app`
- Atoms+bonds only; cell/BZ/axes untouched
- Matte: atoms metalness `0.1` roughness `0.45`; bonds metalness `0.1` roughness `0.5`
- Shiny: metalness `0.85` roughness `0.2` (both)
- No server API; no Eraser/polyhedra/Matplotlib
- Tests: `./TensorSpec_env/bin/python -m unittest …`

## File map

| File | Role |
|------|------|
| `tensorspec/web/static/js/viewers/pbr_params.js` | Pure params helper (export) |
| `tensorspec/web/static/js/viewers/viewer_3d.js` | Import helper; `setPbrShiny`; use in draw |
| `tensorspec/web/static/js/crystal_suite.js` | Wire `#cr-pbr` |
| `tensorspec/web/templates/suites/crystal_suite.html` | Enable checkbox; label |
| `tests/test_pbr_params_contract.py` | Contract + parse helper numbers from JS |

---

### Task 1: `pbrMaterialParams` + viewer API

**Files:**
- Create: `tensorspec/web/static/js/viewers/pbr_params.js`
- Modify: `tensorspec/web/static/js/viewers/viewer_3d.js`
- Create: `tests/test_pbr_params_contract.py`

**Interfaces:**
- `export function pbrMaterialParams(shiny, kind)` where `kind` is `"atom"` | `"bond"` → `{ metalness, roughness }`
- `CrystalViewer.setPbrShiny(enabled: boolean)` — sets `this.pbrShiny`; if `this.geometry`, `this.render(this.geometry, { ...(this._lastOptions || {}), frame: false })`

- [ ] **Step 1: Failing contract test**

```python
"""Contract: pbr_params.js exports shiny/matte numbers from Track C PBR spec."""
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PBR_JS = REPO / "tensorspec/web/static/js/viewers/pbr_params.js"


def _read():
    return PBR_JS.read_text(encoding="utf-8")


class TestPbrParamsContract(unittest.TestCase):
    def test_file_exists(self):
        self.assertTrue(PBR_JS.is_file(), msg=str(PBR_JS))

    def test_exports_function(self):
        text = _read()
        self.assertIn("export function pbrMaterialParams", text)

    def test_shiny_numbers(self):
        text = _read()
        self.assertRegex(text, r"metalness:\s*0\.85")
        self.assertRegex(text, r"roughness:\s*0\.2\b")

    def test_matte_atom_roughness(self):
        text = _read()
        self.assertRegex(text, r"roughness:\s*0\.45")

    def test_matte_bond_roughness(self):
        text = _read()
        self.assertRegex(text, r"roughness:\s*0\.5\b")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run — expect FAIL** (file missing)

```bash
./TensorSpec_env/bin/python -m unittest tests.test_pbr_params_contract -v
```

- [ ] **Step 3: Implement `pbr_params.js`**

```javascript
/** @param {boolean} shiny
 *  @param {"atom"|"bond"} kind
 *  @returns {{ metalness: number, roughness: number }}
 */
export function pbrMaterialParams(shiny, kind) {
  if (shiny) {
    return { metalness: 0.85, roughness: 0.2 };
  }
  if (kind === "bond") {
    return { metalness: 0.1, roughness: 0.5 };
  }
  return { metalness: 0.1, roughness: 0.45 };
}
```

- [ ] **Step 4: Wire `viewer_3d.js`**

- Import: `import { pbrMaterialParams } from "/static/js/viewers/pbr_params.js";` (match existing import style in crystal_suite; if viewer uses relative imports, use `./pbr_params.js`).
- Constructor: `this.pbrShiny = false;`
- In `_drawAtoms` material: spread `pbrMaterialParams(this.pbrShiny, "atom")` instead of hard-coded roughness/metalness.
- In `_drawBonds` same with `"bond"`.
- Add:

```javascript
  setPbrShiny(on) {
    this.pbrShiny = Boolean(on);
    if (this.geometry) {
      this.render(this.geometry, { ...(this._lastOptions || {}), frame: false });
    }
  }
```

- [ ] **Step 5: Tests PASS + commit**

```bash
./TensorSpec_env/bin/python -m unittest tests.test_pbr_params_contract -v
git add tensorspec/web/static/js/viewers/pbr_params.js tensorspec/web/static/js/viewers/viewer_3d.js tests/test_pbr_params_contract.py
git commit -m "feat(crystal): PBR shiny params and viewer setPbrShiny"
```

---

### Task 2: Enable checkbox + suite wire

**Files:**
- Modify: `tensorspec/web/templates/suites/crystal_suite.html`
- Modify: `tensorspec/web/static/js/crystal_suite.js`

- [ ] **Step 1: HTML** — change checkbox to:

```html
<label class="check"><input type="checkbox" id="cr-pbr"> PBR Shiny</label>
```

Optional hint after it: `<p class="hint">Shiny metalness on atoms and bonds only.</p>`

- [ ] **Step 2: JS** — add `pbr: el("cr-pbr")` to `dom`.

In `applyViewOptions` / wherever `setShowAxes` is applied before draw (around `view.setShowAxes`):

```javascript
view.setPbrShiny(Boolean(dom.pbr?.checked));
```

Listener (mirror axes):

```javascript
dom.pbr?.addEventListener("change", () => ensureViewer().setPbrShiny(dom.pbr.checked));
```

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(crystal-ui): enable PBR Shiny checkbox"
```

---

### Task 3: Push

- [ ] **Step 1:**

```bash
./TensorSpec_env/bin/python -m unittest tests.test_pbr_params_contract -v
git push -u origin HEAD
```

- [ ] **Step 2:** Report manual smoke: load crystal → toggle PBR → atoms/bonds shine; cell unchanged. Einstein pull if Mac push path expects it (user rule: HTML_einstein_app push → Einstein pull).

---

## Spec coverage

| Spec | Task |
|------|------|
| `pbrMaterialParams` numbers | 1 |
| `setPbrShiny` + draw path | 1 |
| Enable `#cr-pbr` + wire | 2 |
| Push / smoke note | 3 |
