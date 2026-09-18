# QE Relax + vdW + Heterostructure Geometry Follow-through

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ionic relaxation (default) and optional DFT-D3 to the QE pipeline, sync SCF/Wannier to the relaxed geometry, write back a CIF, then selective dynamics and clearer Push strain/moiré behavior.

**Architecture:** Extend `QEInputGenerator` to emit `relax.in` and parse QE final positions into a pymatgen `Structure`. Extend `qe_generator_panel` so Generate builds `relax → sync geometry → scf → nscf → wannier` in `run_pipeline.sh`. P2 adds `if_pos` selective dynamics; P3 tightens Crystal Tab 3 Push strain messaging and commensurate moiré use.

**Tech Stack:** Python 3, PyQt5, pymatgen, Quantum ESPRESSO `pw.x`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-17-qe-relax-vdw-heterostructure-design.md`

## Global Constraints

- Branch: `TensorSpec_GUI` only (no merge-to-main offers).
- Default geometry mode: **Relax ions (fixed cell)**; vc-relax is advanced.
- Ionic relax keeps **CELL_PARAMETERS fixed** (Push strain + vacuum preserved).
- vdW keyword when enabled: `vdw_corr = 'dft-d3'` inside `&SYSTEM`.
- Pipeline stage banners required: `=== RELAX ===`, `=== SCF ===`, etc.
- Prefer file write-back `relaxed_structure.cif` + optional workspace button; do not silently overwrite Crystal viz structure.
- Do not enable multi-twist DFT (>2 twisted layers).
- Dry-run / Einstein validation is manual; CI covers unit tests only.

## File map

| File | Responsibility |
|------|----------------|
| `tensorspec/core/dft/qe_generator.py` | `write_relax_input`, vdW/SOC flags, `if_pos` positions, apply Structure updates |
| `tensorspec/core/dft/qe_relax_io.py` (new) | Parse `relax.out` / final positions → `Structure`; write CIF |
| `tensorspec/gui/components/qe_generator_panel.py` | Geometry/vdW/selective UI; pipeline script; write-back button |
| `tensorspec/core/crystallography.py` | Isotropic strain % helper; keep moiré commensurate path |
| `tensorspec/gui/suites/crystal_suite.py` | Strain dialog copy + isotropic % |
| `tests/test_qe_relax_io.py` (new) | Parser + CIF round-trip |
| `tests/test_qe_generator_relax.py` (new) | `relax.in` contents |
| `tests/test_stack_dft_cell.py` | Strain display / commensurate assertions |

---

### Task 1: Parse relax.out final geometry → Structure

**Files:**
- Create: `tensorspec/core/dft/qe_relax_io.py`
- Create: `tests/test_qe_relax_io.py`
- Test: `tests/test_qe_relax_io.py`

**Interfaces:**
- Produces: `parse_qe_relaxed_structure(relax_out_text: str, template: Structure) -> Structure`
- Produces: `write_relaxed_cif(structure: Structure, path: str) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_qe_relax_io.py
from pymatgen.core import Lattice, Structure
from tensorspec.core.dft.qe_relax_io import parse_qe_relaxed_structure, write_relaxed_cif

FIXTURE = """
     site n.     atom                  positions (alat units)
         1           C   tau(   1 ) = (  0.3333333  0.6666667  0.4000000  )
         2           C   tau(   2 ) = (  0.6666667  0.3333333  0.4000000  )
         3           B   tau(   3 ) = (  0.3333333  0.6666667  0.6000000  )
         4           N   tau(   4 ) = (  0.6666667  0.3333333  0.6000000  )
"""

# Prefer parser that reads final ATOMIC_POSITIONS crystal block if present:
FIXTURE_CRYSTAL = """
ATOMIC_POSITIONS (crystal)
C  0.333333  0.666667  0.410000
C  0.666667  0.333333  0.410000
B  0.333333  0.666667  0.590000
N  0.666667  0.333333  0.590000
End final coordinates
"""

def test_parse_final_crystal_positions():
    lat = Lattice.hexagonal(2.5, 23.4)
    template = Structure(lat, ["C", "C", "B", "N"],
                         [[1/3, 2/3, 0.42], [2/3, 1/3, 0.42],
                          [1/3, 2/3, 0.58], [2/3, 1/3, 0.58]])
    s = parse_qe_relaxed_structure(FIXTURE_CRYSTAL, template)
    assert abs(s[0].frac_coords[2] - 0.41) < 1e-5
    assert s.lattice.a == template.lattice.a  # ionic relax: cell unchanged


def test_write_relaxed_cif(tmp_path):
    lat = Lattice.hexagonal(2.5, 23.4)
    s = Structure(lat, ["C", "C"], [[1/3, 2/3, 0.4], [2/3, 1/3, 0.4]])
    path = write_relaxed_cif(s, str(tmp_path / "relaxed_structure.cif"))
    s2 = Structure.from_file(path)
    assert len(s2) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/sandyai/Documents/GitHub/TensorSpec_GUI && PYTHONPATH=. TensorSpec_env/bin/python -m pytest tests/test_qe_relax_io.py -v`  
Expected: FAIL (module missing)

- [ ] **Step 3: Implement `qe_relax_io.py`**

```python
# tensorspec/core/dft/qe_relax_io.py
from __future__ import annotations
import re
from pymatgen.core import Structure

_FINAL_POS = re.compile(
    r"ATOMIC_POSITIONS\s*\(\s*crystal\s*\)\s*(.*?)\s*End final coordinates",
    re.IGNORECASE | re.DOTALL,
)

def parse_qe_relaxed_structure(relax_out_text: str, template: Structure) -> Structure:
    """Ionic relax: keep template lattice; replace frac coords from final crystal block."""
    m = _FINAL_POS.search(relax_out_text)
    if not m:
        raise ValueError("No 'ATOMIC_POSITIONS (crystal) ... End final coordinates' in relax output")
    species, fracs = [], []
    for line in m.group(1).strip().splitlines():
        p = line.split()
        if len(p) >= 4 and p[0][0].isalpha():
            species.append(p[0])
            fracs.append([float(p[1]), float(p[2]), float(p[3])])
    if len(species) != len(template):
        raise ValueError(f"Atom count {len(species)} != template {len(template)}")
    site_props = {k: list(v) for k, v in template.site_properties.items()} if template.site_properties else {}
    return Structure(template.lattice, species, fracs, site_properties=site_props or None)


def write_relaxed_cif(structure: Structure, path: str) -> str:
    structure.to(filename=path, fmt="cif")
    return path
```

For vc-relax later, extend parser to also read final `CELL_PARAMETERS`; P1 ionic path keeps lattice from template.

- [ ] **Step 4: Run tests — expect PASS**

Run: `PYTHONPATH=. TensorSpec_env/bin/python -m pytest tests/test_qe_relax_io.py -v`

- [ ] **Step 5: Commit**

```bash
git add tensorspec/core/dft/qe_relax_io.py tests/test_qe_relax_io.py
git commit -m "feat(dft): parse QE relax final positions into Structure"
```

---

### Task 2: `write_relax_input` + vdW flag on generator

**Files:**
- Modify: `tensorspec/core/dft/qe_generator.py`
- Create: `tests/test_qe_generator_relax.py`

**Interfaces:**
- Consumes: existing `QEInputGenerator(structure)`
- Produces: `write_relax_input(out_dir, *, ecutwfc, ecutrho, kmesh, use_soc, use_gpu, vdw_dft_d3, calculation="relax", if_pos=None) -> str`
- Produces: shared helper injecting `vdw_corr = 'dft-d3'` into SYSTEM blocks used by relax/scf/nscf when flag set

- [ ] **Step 1: Write failing tests**

```python
# tests/test_qe_generator_relax.py
import os
from pymatgen.core import Lattice, Structure
from tensorspec.core.dft.qe_generator import QEInputGenerator

def _bilayer(tmp_path):
    # Provide dummy UPFs so _generate_atomic_species works, OR monkeypatch it
    ...

def test_write_relax_ions_fixed_cell(tmp_path, monkeypatch):
    lat = Lattice.hexagonal(2.5, 23.4)
    s = Structure(lat, ["C", "C"], [[1/3, 2/3, 0.4], [2/3, 1/3, 0.4]])
    gen = QEInputGenerator(s)
    monkeypatch.setattr(gen, "_generate_atomic_species", lambda out_dir, use_soc=False: " C  12.01  C.upf")
    path = gen.write_relax_input(str(tmp_path), ecutwfc=40, kmesh=(4, 4, 1), vdw_dft_d3=True)
    text = open(path).read()
    assert "calculation = 'relax'" in text
    assert "vdw_corr = 'dft-d3'" in text
    assert "CELL_PARAMETERS" in text
    assert "ATOMIC_POSITIONS" in text


def test_write_vc_relax_flag(tmp_path, monkeypatch):
    ...
    path = gen.write_relax_input(..., calculation="vc-relax", vdw_dft_d3=False)
    assert "calculation = 'vc-relax'" in open(path).read()
```

Use the same pseudo stub pattern as any existing QE generator tests if present; otherwise monkeypatch `_generate_atomic_species`.

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `write_relax_input`**

Mirror `write_scf_input` but:
- `calculation = 'relax'` or `'vc-relax'`
- Add `&IONS` / `&CELL` (for vc-relax only, minimal defaults)
- `forc_conv_thr = 1.0d-3` in `&IONS`
- Optional `vdw_corr = 'dft-d3'`
- Optional `if_pos` triples appended per site when `if_pos` list provided (P2 can wire UI; support in API now)

Also add optional `vdw_dft_d3: bool = False` to `write_scf_input` / `write_nscf_input` so post-relax electronic steps stay consistent if user enabled D3.

- [ ] **Step 4: Tests PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(dft): write QE relax.in with optional DFT-D3"
```

---

### Task 3: Sync SCF/NSCF/Wannier inputs from relaxed Structure

**Files:**
- Modify: `tensorspec/core/dft/qe_generator.py`
- Modify: `tests/test_qe_generator_relax.py`

**Interfaces:**
- Produces: `QEInputGenerator.apply_structure(structure: Structure) -> None`  # sets `self.structure`
- Or pipeline helper: `rewrite_electronic_inputs_from_structure(gen, out_dir, ...)` that re-calls write_scf/nscf/wannier

- [ ] **Step 1: Test that rewriting updates ATOMIC_POSITIONS z**

```python
def test_rewrite_scf_uses_new_coords(tmp_path, monkeypatch):
    ...
    gen.write_scf_input(...)
    s2 = s.copy(); s2.translate_sites(list(range(len(s2))), [0, 0, 0.05], frac_coords=True)
    gen.structure = s2
    gen.write_scf_input(...)
    assert "0.450000" in open(tmp_path / "scf.in").read() or updated z appears
```

- [ ] **Step 2–4: Implement + pass + commit**

```bash
git commit -m "feat(dft): regenerate QE/Wannier inputs from relaxed Structure"
```

---

### Task 4: QE panel UI — Geometry + vdW

**Files:**
- Modify: `tensorspec/gui/components/qe_generator_panel.py` (`_setup_ui`, `generate_qe_files`)

**Interfaces:**
- UI produces: `combo_geometry` data values `"scf_only" | "relax_ions" | "vc_relax"`
- UI produces: `chk_vdw` bool

- [ ] **Step 1: Add form rows after k-mesh**

```python
self.combo_geometry = QComboBox()
self.combo_geometry.addItem("Relax ions (fixed cell)", "relax_ions")  # default index 0
self.combo_geometry.addItem("SCF only (no relax)", "scf_only")
self.combo_geometry.addItem("vc-relax (advanced)", "vc_relax")
qe_form.addRow("Geometry:", self.combo_geometry)

self.chk_vdw = QCheckBox("vdW DFT-D3 (recommended for 2D stacks)")
self.chk_vdw.setChecked(False)
qe_form.addRow("Dispersion:", self.chk_vdw)
```

- [ ] **Step 2: In `generate_qe_files`, branch on geometry**

- If `relax_ions` or `vc_relax`: call `write_relax_input(...)`; warn if not `chk_vdw` via `QMessageBox.warning` (non-blocking: Continue).
- Pass `vdw_dft_d3=self.chk_vdw.isChecked()` into relax/scf/nscf writers.
- Build bash pipeline (Mac/Linux first; PowerShell mirror):

```bash
# when relax enabled:
echo "=== RELAX ==="
${pw_mpi_cmd}${pw_exec} -in relax.in | tee relax.out
python - <<'PY'
# inline or call module: parse relax.out, rewrite scf/nscf/win using saved meta
from tensorspec.core.dft.qe_relax_io import parse_qe_relaxed_structure, write_relaxed_cif
...
PY
echo "=== SCF ==="
...
```

Prefer shipping a small script `tensorspec/core/dft/sync_after_relax.py` invoked as:

```bash
"$PYTHON" -m tensorspec.core.dft.sync_after_relax --out-dir . --template-cif structure_template.cif
```

On Generate, also write `structure_template.cif` from the current structure so the remote pipeline can reload lattice/tags without pymatgen GUI deps issues — **or** embed JSON of lattice + species. Simplest robust approach for Einstein:

1. On Generate, write `structure_template.cif` from `engine.crystal_structure`.
2. Pipeline after relax: `python -m tensorspec.core.dft.sync_after_relax` which reads `relax.out` + template CIF, writes `relaxed_structure.cif`, regenerates `scf.in`/`nscf.in`/`wannier90.win` via `QEInputGenerator` using the same CLI flags stored in `tensorspec_relax_meta.json` written at Generate time.

- [ ] **Step 3: Write `tensorspec_relax_meta.json` at Generate**

```json
{"ecutwfc": 40, "ecutrho": 160, "kmesh": [4,4,1], "nbnd": 16, "use_soc": false, "use_gpu": false, "vdw_dft_d3": true, "mlwf": false, "calculation": "relax"}
```

- [ ] **Step 4: Manual smoke** — Generate locally with bilayer CIF; confirm files exist: `relax.in`, `structure_template.cif`, `tensorspec_relax_meta.json`, pipeline contains `=== RELAX ===`.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(gui): QE geometry relax + vdW controls and pipeline hook"
```

---

### Task 5: `sync_after_relax` CLI module

**Files:**
- Create: `tensorspec/core/dft/sync_after_relax.py`
- Test: `tests/test_sync_after_relax.py`

**Interfaces:**
- CLI: `python -m tensorspec.core.dft.sync_after_relax --out-dir DIR`
- Reads: `relax.out`, `structure_template.cif`, `tensorspec_relax_meta.json`
- Writes: `relaxed_structure.cif`, regenerated `scf.in`, `nscf.in`, `wannier90.win`, `pw2wan.in`

- [ ] **Step 1: Failing test with tmp_path fixture files**

- [ ] **Step 2–4: Implement + pass**

Must monkeypatch or provide tiny fake UPF dir: set `gen.app_pseudo_dir` to a folder with stub `.upf` files named like existing patterns, or copy from `qe_dryrun_gr_hbn/pseudo` in test.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(dft): sync_after_relax rewrites electronic inputs from relax.out"
```

---

### Task 6: Write-back button in QE panel

**Files:**
- Modify: `tensorspec/gui/components/qe_generator_panel.py`
- Modify: fetch thread allowlist if needed to pull `relax.out`, `relaxed_structure.cif`

- [ ] **Step 1: Add button** `Push relaxed CIF to workspace`

```python
self.btn_push_relaxed = QPushButton("Push relaxed structure to workspace")
# enabled if os.path.isfile(os.path.join(out_dir, "relaxed_structure.cif"))
```

- [ ] **Step 2: On click** — `Structure.from_file` → `global_workspace.push_crystal_structure(name, struct)` with name like `qe_relaxed_<outdir_basename>`.

- [ ] **Step 3: Extend Fetch** to include `relax.out`, `relaxed_structure.cif`, `tensorspec_relax_meta.json` when present.

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(gui): push relaxed CIF to workspace + fetch relax artifacts"
```

---

### Task 7 (P2): Selective dynamics UI + `if_pos`

**Files:**
- Modify: `qe_generator.py` `_generate_atomic_positions(if_pos=...)`
- Modify: `qe_generator_panel.py`
- Test: `tests/test_qe_generator_relax.py`

- [ ] **Step 1: Test `if_pos` lines in relax.in**

```python
def test_relax_if_pos_fix_first_two():
    if_pos = [(0,0,0), (0,0,0), (1,1,1), (1,1,1)]
    ...
    assert "0 0 0" in text and "1 1 1" in text
```

- [ ] **Step 2: UI combo** `None | Fix reference layer | Fix bottom layer`

Resolve masks:
- bottom: sites with `z == min(z)` (tolerance)
- reference: sites whose `layer_tag` ends with `_L{ref}` — if no tags, disable and tooltip

- [ ] **Step 3: Pass mask into `write_relax_input`**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(dft): selective dynamics if_pos for QE relax"
```

---

### Task 8 (P3): Isotropic strain helper + dialog copy

**Files:**
- Modify: `tensorspec/core/crystallography.py`
- Modify: `tensorspec/gui/suites/crystal_suite.py`
- Modify: `tests/test_stack_dft_cell.py`

**Interfaces:**
- Produces: `CrystalEngine.isotropic_match_strain_percent(a_native: float, a_ref: float) -> float`  
  returns `(a_ref - a_native) / a_native * 100.0`

- [ ] **Step 1: Test Gr/hBN → ~+1.626%**

```python
def test_isotropic_gr_on_hbn():
    assert abs(CrystalEngine.isotropic_match_strain_percent(2.46, 2.5) - 1.62601626) < 1e-3
```

- [ ] **Step 2: Strain dialog text** includes:
  - “Isotropic lattice strain on \<layer\>: +1.63%”
  - “Suggestion score (includes twist): …” when twist case
  - “Run QE Relax ions + vdW after Push for better interlayer geometry.”

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(crystal): show isotropic strain % in Push dialog"
```

---

### Task 9 (P3): Commensurate moiré path enforcement

**Files:**
- Modify: `tensorspec/core/crystallography.py` `build_dft_twist_stack`
- Modify: `tests/test_stack_dft_cell.py`

- [ ] **Step 1: Assert** when `moire["status"]=="commensurate"`, returned structure atom count matches tiling expectation (existing `test_twist_commensurate_fills_moire_cell` must still pass).

- [ ] **Step 2: If any code path still force-strains despite commensurate status, remove it.** Log/info dict must keep `status: commensurate`.

- [ ] **Step 3: Incommensurate dialog** must not present Frobenius+twist ~50% as “stretch”; use Task 8 isotropic % for the stretched layer.

- [ ] **Step 4: Commit**

```bash
git commit -m "fix(crystal): keep commensurate twist on moiré cell; clarify strain copy"
```

---

### Task 10: Docs + Einstein D3 smoke note

**Files:**
- Modify: spec status already approved
- Create short note in spec Open points resolution: run on Einstein

```bash
ssh einstein 'source ... && conda activate qe && grep -i d3 $(which pw.x) || pw.x -h 2>&1 | head'
# or one-atom relax.in with vdw_corr to see accept/reject
```

- [ ] Document result in spec “Open points” section (D3 yes/no).
- [ ] Commit doc update only.

---

## Spec coverage checklist

| Spec item | Task |
|-----------|------|
| Geometry modes + default ionic relax | 2, 4 |
| DFT-D3 flag + warn | 2, 4 |
| Pipeline relax → sync → scf → nscf → wan | 4, 5 |
| Fixed cell ionic | 1, 2 |
| vc-relax advanced | 2, 4 |
| `relaxed_structure.cif` + workspace push | 1, 5, 6 |
| Selective dynamics | 7 |
| Isotropic strain UI | 8 |
| Commensurate moiré preference | 9 |
| Einstein D3 verify | 10 |

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-17-qe-relax-vdw-heterostructure-plan.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — this session with executing-plans checkpoints  

Which approach?
