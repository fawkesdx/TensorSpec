# Crystal Suite Viewer Fidelity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Crystal Suite Tab 1 naming/colors follow the opened structure (CIF labels + dynamic swatches + full Jmol CPK), and wire practical three.js controls (bond thick, axes, cell, Bonds/None, camera, conventional|primitive basis); then push and document Einstein pull/restart.

**Architecture:** Hybrid — server adds `Atom.label` and optional `basis` transform; browser owns CPK colors, live overrides, camera, and axes. Colors stay off the geometry schema.

**Tech Stack:** FastAPI, pymatgen (`Structure`, `SpacegroupAnalyzer`), three.js (`CrystalViewer`), vanilla JS suite controller, unittest.

**Spec:** `docs/superpowers/specs/2026-08-07-crystal-suite-viewer-fidelity-design.md`

## Global Constraints

- Branch: `HTML_einstein_app` on local clone; push to origin; Einstein pulls after.
- Colors never added to `CrystalGeometry` JSON.
- Scope B only: no polyhedra, PBR, eraser, cut plane, Matplotlib/PyVista, 3ds/Blender, hi-res.
- Disabled stubs must show hint text `HTML viewer: not yet`.
- Follow existing unittest style in `tests/test_qe_slab.py`.
- Jmol CPK hexes from https://jmol.sourceforge.net/jscolors/ (elements 1–109). Prior custom Ta/Ir/Te hexes are replaced; users recolor via swatches.

## File map

| File | Responsibility |
|------|----------------|
| `tensorspec/web/server/schemas.py` | `Atom.label`; `GeometryRequest.basis` |
| `tensorspec/web/server/routers/crystal.py` | Site label/element helpers; basis transform; load fmt by extension |
| `tests/test_crystal_geometry.py` | Labels + basis + load-fmt unit tests |
| `tensorspec/web/static/js/viewers/viewer_3d.js` | Jmol CPK, overrides, bond radius, axes, cell/bonds options, camera, hover |
| `tensorspec/web/templates/suites/crystal_suite.html` | IDs, empty swatch container, disable+hint stubs, file accept |
| `tensorspec/web/static/js/crystal_suite.js` | Wire new DOM → API/viewer; rebuild swatches |

---

### Task 1: Schema — `Atom.label` + `GeometryRequest.basis`

**Files:**
- Modify: `tensorspec/web/server/schemas.py`
- Test: (covered in Task 2)

**Interfaces:**
- Produces: `Atom(element, label, position, radius)`; `GeometryRequest.basis: Literal["conventional","primitive"] = "conventional"`

- [ ] **Step 1: Add fields**

In `schemas.py`, ensure `Literal` is imported (already used elsewhere in file). Update:

```python
class GeometryRequest(BaseModel):
    # ... existing fields ...
    basis: Literal["conventional", "primitive"] = "conventional"
    # keep nx/ny/nz, bonds, CDW as today

class Atom(BaseModel):
    element: str
    label: str
    position: list[float]
    radius: float
```

Keep the `CrystalGeometry` docstring that colours are absent.

- [ ] **Step 2: Commit**

```bash
git add tensorspec/web/server/schemas.py
git commit -m "$(cat <<'EOF'
feat(crystal): add Atom.label and geometry basis field

EOF
)"
```

---

### Task 2: Failing tests for labels + basis

**Files:**
- Create: `tests/test_crystal_geometry.py`
- Modify: (none yet — tests call helpers that Task 3 adds)

**Interfaces:**
- Consumes (Task 3 will provide):
  - `_site_element_and_label(site) -> tuple[str, str]`
  - `_apply_basis(structure, basis: str) -> Structure`
  - `_geometry_from_structure(...)` (existing, will gain labels)
  - `_detect_structure_fmt(filename: str) -> str`

- [ ] **Step 1: Write failing tests**

```python
"""Crystal geometry labels, basis transform, and load fmt detection."""
import unittest
from io import StringIO

from pymatgen.core import Lattice, Structure

from tensorspec.web.server.routers import crystal as crystal_router


class TestSiteLabel(unittest.TestCase):
    def test_label_falls_back_to_symbol(self):
        s = Structure(Lattice.cubic(4.0), ["Si"], [[0, 0, 0]])
        element, label = crystal_router._site_element_and_label(s[0])
        self.assertEqual(element, "Si")
        self.assertEqual(label, "Si")

    def test_geometry_includes_labels(self):
        s = Structure(Lattice.cubic(4.0), ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        # pymatgen assigns Si0, Si1-style labels on construction
        geo = crystal_router._geometry_from_structure("si", s, show_bonds=False)
        self.assertEqual(len(geo.atoms), 2)
        self.assertTrue(all(a.label for a in geo.atoms))
        self.assertEqual(geo.atoms[0].element, "Si")


class TestBasis(unittest.TestCase):
    def test_primitive_differs_from_conventional_fcc(self):
        # Conventional FCC cubic cell (4 sites) → primitive has 1 site
        lattice = Lattice.cubic(3.6)
        species = ["Cu"] * 4
        frac = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
        conventional = Structure(lattice, species, frac)
        prim = crystal_router._apply_basis(conventional, "primitive")
        self.assertLess(len(prim), len(conventional))

    def test_conventional_is_identity_for_already_standard(self):
        s = Structure(Lattice.cubic(4.0), ["Si"], [[0, 0, 0]])
        out = crystal_router._apply_basis(s, "conventional")
        self.assertEqual(len(out), len(s))


class TestLoadFmt(unittest.TestCase):
    def test_detect_fmt(self):
        self.assertEqual(crystal_router._detect_structure_fmt("x.cif"), "cif")
        self.assertEqual(crystal_router._detect_structure_fmt("POSCAR"), "poscar")
        self.assertEqual(crystal_router._detect_structure_fmt("foo.vasp"), "poscar")
        self.assertEqual(crystal_router._detect_structure_fmt("bar.POSCAR"), "poscar")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests — expect FAIL (helpers missing)**

Run: `python -m unittest tests.test_crystal_geometry -v`

Expected: `AttributeError` / import failures for `_site_element_and_label`, `_apply_basis`, `_detect_structure_fmt`.

- [ ] **Step 3: Commit tests**

```bash
git add tests/test_crystal_geometry.py
git commit -m "$(cat <<'EOF'
test(crystal): add failing geometry label and basis tests

EOF
)"
```

---

### Task 3: Implement label helpers, basis, load fmt

**Files:**
- Modify: `tensorspec/web/server/routers/crystal.py`
- Test: `tests/test_crystal_geometry.py`

**Interfaces:**
- Produces: `_site_element_and_label`, `_apply_basis`, `_detect_structure_fmt`
- Updates: `_geometry_from_structure` sets `label`; `get_geometry` applies basis before supercell; `load_cif` uses detected fmt

- [ ] **Step 1: Add helpers near top of `crystal.py` (after constants)**

```python
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def _detect_structure_fmt(filename: str) -> str:
    lower = (filename or "").lower()
    if lower.endswith(".cif"):
        return "cif"
    if lower.endswith(".vasp") or lower.endswith(".poscar") or lower.endswith("poscar"):
        return "poscar"
    return "cif"


def _site_element_and_label(site) -> tuple[str, str]:
    """Return (element_symbol, display_label) for one pymatgen site."""
    try:
        element = site.specie.symbol
    except Exception:
        # Disordered: take the majority species
        if getattr(site, "species", None) is not None:
            element = sorted(site.species.items(), key=lambda kv: -kv[1])[0][0].symbol
        else:
            raise
    label = getattr(site, "label", None) or element
    return element, str(label)


def _apply_basis(structure: Structure, basis: str) -> Structure:
    if basis == "conventional":
        try:
            return SpacegroupAnalyzer(structure).get_conventional_standard_structure()
        except Exception:
            return structure.copy()
    if basis == "primitive":
        try:
            return SpacegroupAnalyzer(structure).get_primitive_standard_structure()
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Could not convert to primitive basis: {exc}",
            ) from exc
    raise HTTPException(status_code=422, detail=f"Unknown basis: {basis}")
```

- [ ] **Step 2: Update `_geometry_from_structure` atom build**

```python
    atoms = []
    for idx, site in enumerate(structure):
        try:
            element, label = _site_element_and_label(site)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Could not resolve species for site {idx}: {exc}",
            ) from exc
        radius = DEFAULT_RADIUS
        try:
            radius = float(site.specie.atomic_radius or DEFAULT_RADIUS)
        except Exception:
            radius = DEFAULT_RADIUS
        atoms.append(
            Atom(
                element=element,
                label=label,
                position=[float(v) for v in coords[idx]],
                radius=radius,
            )
        )
```

Update `elements=` to use the resolved symbols from `atoms` (or keep `site` loop with `_site_element_and_label`).

- [ ] **Step 3: Apply basis in `get_geometry` before supercell**

```python
    structure = _require_structure(session, name)
    structure = _apply_basis(structure, request.basis)
    # then projected count, supercell, CDW, _geometry_from_structure as today
```

- [ ] **Step 4: Fix `load_cif` fmt detection**

```python
    fname = file.filename or ""
    if not fname.lower().endswith((".cif", ".vasp", ".poscar")) and not fname.upper().endswith("POSCAR"):
        raise HTTPException(status_code=400, detail="Expected a .cif, .vasp, or POSCAR file.")
    ...
    fmt = _detect_structure_fmt(fname)
    try:
        structure = Structure.from_str(payload.decode("utf-8", errors="replace"), fmt=fmt)
```

- [ ] **Step 5: Run tests — expect PASS**

Run: `python -m unittest tests.test_crystal_geometry -v`

Expected: all OK.

- [ ] **Step 6: Commit**

```bash
git add tensorspec/web/server/routers/crystal.py
git commit -m "$(cat <<'EOF'
feat(crystal): emit site labels and support primitive basis

EOF
)"
```

---

### Task 4: Viewer — Jmol CPK + color overrides + bond/axes/camera/hover

**Files:**
- Modify: `tensorspec/web/static/js/viewers/viewer_3d.js`

**Interfaces:**
- Produces:
  - `elementColor(symbol)` — override → CPK → fallback
  - `CrystalViewer.setElementColor(symbol, hex)`, `setBondColor(hex)`, `setBondRadius(r)`, `setShowAxes(bool)`, `setProjection(mode)`, `lookAlong(axis)`, `setAzEl(az, el)`
  - `render(geometry, { showBonds, showCell, frame })` unchanged contract
  - Hover tooltip via raycaster

- [ ] **Step 1: Replace `CPK_COLORS` with full Jmol table**

Replace the 11-entry map with Jmol defaults (prefix `#`). Minimum required symbols (copy hex from jmol.sourceforge.net/jscolors atomic-number table):

```javascript
const CPK_COLORS = {
  H:"#FFFFFF", He:"#D9FFFF", Li:"#CC80FF", Be:"#C2FF00", B:"#FFB5B5",
  C:"#909090", N:"#3050F8", O:"#FF0D0D", F:"#90E050", Ne:"#B3E3F5",
  Na:"#AB5CF2", Mg:"#8AFF00", Al:"#BFA6A6", Si:"#F0C8A0", P:"#FF8000",
  S:"#FFFF30", Cl:"#1FF01F", Ar:"#80D1E3", K:"#8F40D4", Ca:"#3DFF00",
  Sc:"#E6E6E6", Ti:"#BFC2C7", V:"#A6A6AB", Cr:"#8A99C7", Mn:"#9C7AC7",
  Fe:"#E06633", Co:"#F090A0", Ni:"#50D050", Cu:"#C88033", Zn:"#7D80B0",
  Ga:"#C28F8F", Ge:"#668F8F", As:"#BD80E3", Se:"#FFA100", Br:"#A62929",
  Kr:"#5CB8D1", Rb:"#702EB0", Sr:"#00FF00", Y:"#94FFFF", Zr:"#94E0E0",
  Nb:"#73C2C9", Mo:"#54B5B5", Tc:"#3B9E9E", Ru:"#248F8F", Rh:"#0A7D8C",
  Pd:"#006985", Ag:"#C0C0C0", Cd:"#FFD98F", In:"#A67573", Sn:"#668080",
  Sb:"#9E63B5", Te:"#D47A00", I:"#940094", Xe:"#429EB0", Cs:"#57178F",
  Ba:"#00C900", La:"#70D4FF", Ce:"#FFFFC7", Pr:"#D9FFC7", Nd:"#C7FFC7",
  Pm:"#A3FFC7", Sm:"#8FFFC7", Eu:"#61FFC7", Gd:"#45FFC7", Tb:"#30FFC7",
  Dy:"#1FFFC7", Ho:"#00FF9C", Er:"#00E675", Tm:"#00D452", Yb:"#00BF38",
  Lu:"#00AB24", Hf:"#4DC2FF", Ta:"#4DA6FF", W:"#2194D6", Re:"#267DAB",
  Os:"#266696", Ir:"#175487", Pt:"#D0D0E0", Au:"#FFD123", Hg:"#B8B8D0",
  Tl:"#A6544D", Pb:"#575961", Bi:"#9E4FB5", Po:"#AB5C00", At:"#754F45",
  Rn:"#428296", Fr:"#420066", Ra:"#007D00", Ac:"#70ABFA", Th:"#00BAFF",
  Pa:"#00A1FF", U:"#008FFF", Np:"#0080FF", Pu:"#006BFF", Am:"#545CF2",
  Cm:"#785CE3", Bk:"#8A4FE3", Cf:"#A136D4", Es:"#B31FD4", Fm:"#B31FBA",
  Md:"#B30DA6", No:"#BD0D87", Lr:"#C70066", Rf:"#CC0059", Db:"#D1004F",
  Sg:"#D90045", Bh:"#E00038", Hs:"#E6002E", Mt:"#EB0026",
};
const FALLBACK_COLOR = "#808080";
let BOND_COLOR = "#d3d3d3";
const colorOverrides = Object.create(null);

export function elementColor(symbol) {
  return colorOverrides[symbol] || CPK_COLORS[symbol] || FALLBACK_COLOR;
}

export function setGlobalElementColor(symbol, hex) {
  colorOverrides[symbol] = hex;
}

export function setGlobalBondColor(hex) {
  BOND_COLOR = hex;
}
```

Also keep per-viewer methods that call these and re-render.

- [ ] **Step 2: Constructor — axes, raycaster, tooltip, projection state**

```javascript
    this.bondRadius = 0.1;
    this.showAxes = true;
    this._axes = new THREE.AxesHelper(5);
    this.scene.add(this._axes);

    this._raycaster = new THREE.Raycaster();
    this._pointer = new THREE.Vector2();
    this._tooltip = document.createElement("div");
    this._tooltip.className = "crystal-atom-tooltip";
    this._tooltip.style.cssText = "position:absolute;pointer-events:none;display:none;padding:4px 8px;background:#111;color:#eee;font:12px/1.3 sans-serif;border-radius:4px;z-index:5;";
    container.style.position = container.style.position || "relative";
    container.appendChild(this._tooltip);
    this.renderer.domElement.addEventListener("pointermove", (e) => this._onPointerMove(e));

    this._atomIndexByMesh = new WeakMap(); // InstancedMesh → array of atom indices
```

- [ ] **Step 3: `_drawAtoms` — store atom indices; use `elementColor`**

When building each InstancedMesh, record `entries.map(e => e.index)` on the mesh via `this._atomIndexByMesh.set(mesh, ...)`.

- [ ] **Step 4: Hover handler**

```javascript
  _onPointerMove(event) {
    if (!this.geometry) { this._tooltip.style.display = "none"; return; }
    const rect = this.renderer.domElement.getBoundingClientRect();
    this._pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this._pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this._raycaster.setFromCamera(this._pointer, this.camera);
    const meshes = [];
    this.content.traverse((o) => { if (o.isInstancedMesh && this._atomIndexByMesh.has(o)) meshes.push(o); });
    const hits = this._raycaster.intersectObjects(meshes, false);
    if (!hits.length) { this._tooltip.style.display = "none"; return; }
    const hit = hits[0];
    const indices = this._atomIndexByMesh.get(hit.object);
    const atom = this.geometry.atoms[indices[hit.instanceId]];
    this._tooltip.textContent = `${atom.label} (${atom.element})`;
    this._tooltip.style.display = "block";
    this._tooltip.style.left = `${event.clientX - rect.left + 12}px`;
    this._tooltip.style.top = `${event.clientY - rect.top + 12}px`;
  }
```

- [ ] **Step 5: Camera / axes / bond API**

```javascript
  setBondRadius(r) {
    this.bondRadius = r;
    if (this.geometry) this.render(this.geometry, { ...(this._lastOptions || {}), frame: false });
  }

  setShowAxes(on) {
    this.showAxes = on;
    this._axes.visible = on;
  }

  setProjection(mode) {
    // mode: "perspective" | "orthographic"
    const aspect = this.camera.aspect || 1;
    const dist = this.camera.position.length();
    if (mode === "orthographic") {
      const half = Math.max(dist * Math.tan((45 * Math.PI) / 360), 1);
      const cam = new THREE.OrthographicCamera(-half * aspect, half * aspect, half, -half, 0.1, 5000);
      cam.position.copy(this.camera.position);
      cam.quaternion.copy(this.camera.quaternion);
      this.camera = cam;
    } else {
      const cam = new THREE.PerspectiveCamera(45, aspect, 0.1, 5000);
      cam.position.copy(this.camera.position);
      cam.quaternion.copy(this.camera.quaternion);
      this.camera = cam;
    }
    this.controls.object = this.camera;
    this._resize();
  }

  lookAlong(which) {
    // which: "+a" | "+b" | "+c" | "111"
    if (!this.geometry) return;
    const [a, b, c] = this.geometry.cell.map((row) => new THREE.Vector3(...row));
    let dir;
    if (which === "+a") dir = a.clone();
    else if (which === "+b") dir = b.clone();
    else if (which === "+c") dir = c.clone();
    else dir = a.clone().add(b).add(c);
    dir.normalize();
    const dist = this.camera.position.length() || 30;
    this.camera.position.copy(dir.multiplyScalar(dist));
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  setAzEl(azDeg, elDeg) {
    const dist = this.camera.position.length() || 30;
    const az = (azDeg * Math.PI) / 180;
    const el = (elDeg * Math.PI) / 180;
    this.camera.position.set(
      dist * Math.cos(el) * Math.sin(az),
      dist * Math.sin(el),
      dist * Math.cos(el) * Math.cos(az),
    );
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  setElementColor(symbol, hex) {
    setGlobalElementColor(symbol, hex);
    if (this.geometry) this.render(this.geometry, { ...(this._lastOptions || {}), frame: false });
  }

  setBondColor(hex) {
    setGlobalBondColor(hex);
    if (this.geometry) this.render(this.geometry, { ...(this._lastOptions || {}), frame: false });
  }

  legend() {
    if (!this.geometry) return [];
    return this.geometry.elements.map((el) => ({ element: el, color: elementColor(el) }));
  }
```

- [ ] **Step 6: Manual smoke (browser)** — load any structure; hover shows label; change not yet wired from UI.

- [ ] **Step 7: Commit**

```bash
git add tensorspec/web/static/js/viewers/viewer_3d.js
git commit -m "$(cat <<'EOF'
feat(viewer): Jmol CPK, overrides, axes, camera, atom hover

EOF
)"
```

---

### Task 5: HTML — IDs, swatch container, stubs

**Files:**
- Modify: `tensorspec/web/templates/suites/crystal_suite.html`

- [ ] **Step 1: File accept + load label**

```html
<input type="file" id="cr-file" accept=".cif,.vasp,.poscar,.POSCAR" hidden>
<button ... id="cr-load">Load Structure File</button>
```

- [ ] **Step 2: Backend select — disable + hint**

```html
<select class="field" id="cr-backend" disabled title="HTML viewer: not yet">
  <option>three.js (GPU)</option>
</select>
<p class="hint">Graphics backends Matplotlib/PyVista: HTML viewer: not yet.</p>
```

- [ ] **Step 3: Basis radios — add ids/values**

```html
<label><input type="radio" name="cr-basis" id="cr-basis-conv" value="conventional" checked> Conventional Basis</label>
<label><input type="radio" name="cr-basis" id="cr-basis-prim" value="primitive"> Primitive Basis</label>
```

- [ ] **Step 4: Styles — ids on checkboxes / conn**

```html
<select class="field" id="cr-conn">
  <option value="bonds">Bonds (Sticks)</option>
  <option value="none">None</option>
  <option value="polyhedra" disabled>Polyhedra (Planes) — HTML viewer: not yet</option>
</select>
...
<label class="check"><input type="checkbox" id="cr-pbr" disabled> PBR Shiny (HTML viewer: not yet)</label>
<label class="check"><input type="checkbox" id="cr-axes" checked> Show Axes</label>
<label class="check"><input type="checkbox" id="cr-show-cell" checked> Show Conv. Box</label>
<label class="check"><input type="checkbox" id="cr-show-prim" disabled> Show Prim. Box (use Primitive Basis)</label>
<label class="check"><input type="checkbox" id="cr-eraser" disabled> Enable Interactive Eraser Brush (HTML viewer: not yet)</label>
```

- [ ] **Step 5: Replace static swatches**

```html
<fieldset class="fieldset">
  <legend>Dynamic Element Colors</legend>
  <p class="hint">Populated per element once a structure is loaded.</p>
  <div id="cr-swatches"></div>
</fieldset>
```

- [ ] **Step 6: Camera controls — ids + enable quick views**

```html
<select class="field" id="cr-projection" style="margin-bottom:8px">
  <option value="perspective">Perspective Projection</option>
  <option value="orthographic">Orthogonal Projection</option>
</select>
...
<button type="button" class="btn" id="cr-view-a">+a</button>
<button type="button" class="btn" id="cr-view-b">+b</button>
<button type="button" class="btn" id="cr-view-c">+c</button>
<button type="button" class="btn" id="cr-view-111">111</button>
```

- [ ] **Step 7: Crystallography tools + export stubs — keep disabled, add hint**

Add under Export / Render:

```html
<p class="hint">Cut plane, Align, 3ds/Blender, hi-res render: HTML viewer: not yet.</p>
```

Leave those buttons `disabled`.

- [ ] **Step 8: Commit**

```bash
git add tensorspec/web/templates/suites/crystal_suite.html
git commit -m "$(cat <<'EOF'
feat(crystal-ui): ids, dynamic swatch host, disable stubs

EOF
)"
```

---

### Task 6: Wire `crystal_suite.js`

**Files:**
- Modify: `tensorspec/web/static/js/crystal_suite.js`

**Interfaces:**
- Consumes: viewer APIs from Task 4; `basis` on geometry request from Task 1

- [ ] **Step 1: Extend `dom` map**

```javascript
    bondThick: el("cr-bondthick"),
    swatches: el("cr-swatches"),
    conn: el("cr-conn"),
    axes: el("cr-axes"),
    showCell: el("cr-show-cell"),
    projection: el("cr-projection"),
    viewA: el("cr-view-a"),
    viewB: el("cr-view-b"),
    viewC: el("cr-view-c"),
    view111: el("cr-view-111"),
    az: el("cr-az"),
    elv: el("cr-el"),
    basisConv: el("cr-basis-conv"),
    basisPrim: el("cr-basis-prim"),
```

- [ ] **Step 2: `geometryRequest` includes basis + show_bonds**

```javascript
        basis: dom.basisPrim?.checked ? "primitive" : "conventional",
        show_bonds: dom.conn?.value !== "none",
```

- [ ] **Step 3: `rebuildSwatches(elements)`**

```javascript
function rebuildSwatches(elements) {
    if (!dom.swatches) return;
    dom.swatches.innerHTML = "";
    elements.forEach((symbol) => {
        const row = document.createElement("div");
        row.className = "swatch-row";
        const label = document.createElement("span");
        label.textContent = `${symbol}:`;
        const input = document.createElement("input");
        input.type = "color";
        input.value = elementColor(symbol);
        input.addEventListener("input", () => {
            const view = ensureViewer();
            view.setElementColor(symbol, input.value);
            updateLegend(elements);
        });
        row.append(label, input);
        dom.swatches.appendChild(row);
    });
    const bondRow = document.createElement("div");
    bondRow.className = "swatch-row";
    const bondLabel = document.createElement("span");
    bondLabel.textContent = "Bonds:";
    const bondInput = document.createElement("input");
    bondInput.type = "color";
    bondInput.value = "#d3d3d3";
    bondInput.addEventListener("input", () => ensureViewer().setBondColor(bondInput.value));
    bondRow.append(bondLabel, bondInput);
    dom.swatches.appendChild(bondRow);
}

function updateLegend(elements) {
    dom.legend.textContent = elements.join(", ");
    if (elements[0]) dom.legend.style.color = elementColor(elements[0]);
}
```

Call `rebuildSwatches(geometry.elements)` and `updateLegend` from `refreshGeometry` (and stack/relax paths that set legend today).

- [ ] **Step 4: Apply viewer options before/after render**

```javascript
function applyViewerChrome(view) {
    view.atomScale = Number(dom.radius.value) || 0.5;
    view.setBondRadius(Number(dom.bondThick?.value) || 0.1);
    view.setShowAxes(Boolean(dom.axes?.checked));
}

// in refreshGeometry:
        applyViewerChrome(view);
        view.render(geometry, {
            frame,
            showBonds: dom.conn?.value !== "none",
            showCell: Boolean(dom.showCell?.checked),
        });
        rebuildSwatches(geometry.elements);
        updateLegend(geometry.elements);
```

- [ ] **Step 5: Event listeners (in init / bottom of file where others bind)**

```javascript
[dom.nx, dom.ny, dom.nz, dom.threshold].forEach((node) =>
    node?.addEventListener("change", () => refreshGeometry({ frame: false }))
);
dom.basisConv?.addEventListener("change", () => refreshGeometry({ frame: true }));
dom.basisPrim?.addEventListener("change", () => refreshGeometry({ frame: true }));
dom.conn?.addEventListener("change", () => refreshGeometry({ frame: false }));
dom.bondThick?.addEventListener("change", () => {
    ensureViewer().setBondRadius(Number(dom.bondThick.value) || 0.1);
});
dom.radius?.addEventListener("change", () => {
    ensureViewer().setAtomScale(Number(dom.radius.value) || 0.5);
});
dom.axes?.addEventListener("change", () => ensureViewer().setShowAxes(dom.axes.checked));
dom.showCell?.addEventListener("change", () => refreshGeometry({ frame: false }));
dom.projection?.addEventListener("change", () => ensureViewer().setProjection(dom.projection.value));
dom.viewA?.addEventListener("click", () => ensureViewer().lookAlong("+a"));
dom.viewB?.addEventListener("click", () => ensureViewer().lookAlong("+b"));
dom.viewC?.addEventListener("click", () => ensureViewer().lookAlong("+c"));
dom.view111?.addEventListener("click", () => ensureViewer().lookAlong("111"));
const syncAzEl = () => ensureViewer().setAzEl(Number(dom.az.value) || 0, Number(dom.elv.value) || 0);
dom.az?.addEventListener("change", syncAzEl);
dom.elv?.addEventListener("change", syncAzEl);
```

On primitive basis 422: `setStatus(err.message, true)` and re-check conventional radio.

- [ ] **Step 6: Manual verify checklist**

1. Load MoS₂ / graphene → swatches Mo,S or C — not Ta/Ir/Te; distinct colors.
2. Hover atom → `label (element)`.
3. Bond thick, axes, cell, +a work.
4. Toggle primitive on FCC-like cell → fewer sites / different cell.
5. Polyhedra / backend / eraser remain non-interactive.

- [ ] **Step 7: Commit**

```bash
git add tensorspec/web/static/js/crystal_suite.js
git commit -m "$(cat <<'EOF'
feat(crystal): wire swatches, basis, and three.js Tab 1 controls

EOF
)"
```

---

### Task 7: Verification + push + Einstein instructions

**Files:** none (ops)

- [ ] **Step 1: Run unit tests**

Run: `python -m unittest tests.test_crystal_geometry tests.test_qe_slab tests.test_fat_bands -v`

Expected: all PASS.

- [ ] **Step 2: Spec coverage check**

Confirm each Success criterion in the design spec maps to a Task 6 checklist item or a test.

- [ ] **Step 3: Push**

```bash
git push -u origin HEAD
```

Requires network / user approval.

- [ ] **Step 4: Give user Einstein commands**

```bash
cd ~/TensorSpec   # adjust if clone path differs
git fetch
git checkout HTML_einstein_app
git pull
# restart uvicorn (or the service you use on Einstein):
uvicorn tensorspec.web.server.app:app --reload --host 0.0.0.0 --port 8000
```

Ask user to confirm Einstein path + process manager if not bare uvicorn.

---

## Self-review (plan vs spec)

| Spec requirement | Task |
|------------------|------|
| Dynamic swatches from elements | 5, 6 |
| Full Jmol CPK + overrides | 4, 6 |
| Atom.label + hover | 1, 3, 4 |
| Bond thick / axes / cell / Bonds\|None | 4, 5, 6 |
| Camera persp/ortho, +a/+b/+c/111, az/el | 4, 5, 6 |
| Conventional\|primitive basis | 1, 2, 3, 6 |
| Load cif/vasp/poscar | 3, 5 |
| Disabled stubs + hint | 5 |
| Pytest labels + basis | 2, 3 |
| Einstein pull after push | 7 |
| Non-goals untouched | 5 (disabled) |

No TBD placeholders. Types consistent: `basis` string, `Atom.label` string, viewer method names match Task 6 consumers.
