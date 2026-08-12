# DFT Auto-nbnd Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore old-GUI nbnd suggestion from structure sites; ×2 when SOC is on; wire DFT structure list + suite UI.

**Architecture:** Core `suggest_nbnd_base(structure)`; expose `suggest_nbnd` on `/api/dft/structures`; UI syncs `#qe-nbnd` on structure change and SOC toggle (same pattern as `syncSlabSuggestion`).

**Tech Stack:** pymatgen Structure, FastAPI, existing `dft_suite.js`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-12-dft-auto-nbnd-design.md`
- Branch: `HTML_einstein_app` only — never merge to `main`
- Per site: TM or Z>30 → +9; else +4; SOC UI → `min(500, 2*base)`
- Cap 1…500; no popup; overwrite on structure/SOC change
- Tests: `PYTHONPATH=. TensorSpec_env/bin/pytest …`
- Push + Einstein after ship

## File map

| File | Role |
|------|------|
| `tensorspec/core/dft/nbnd_suggest.py` | `suggest_nbnd_base` |
| `tests/test_nbnd_suggest.py` | unit tests |
| `schemas.py` `StructureOption` | `suggest_nbnd: int` |
| `routers/dft.py` `list_structures` | fill field |
| `dft_suite.js` | apply suggestion + SOC |
| `tests/test_dft_api.py` (or existing) | structures include suggest_nbnd |

---

### Task 1: Core suggest_nbnd_base + tests

**Files:** `nbnd_suggest.py`, `tests/test_nbnd_suggest.py`

```python
def suggest_nbnd_base(structure) -> int:
    total = 0
    for site in structure:
        el = site.specie
        if el.is_transition_metal or el.number > 30:
            total += 9
        else:
            total += 4
    return max(1, total)
```

- [ ] Failing tests: graphene 2C → 8; single Fe → 9; empty guard if needed
- [ ] Implement + pass
- [ ] Commit `feat(dft): suggest_nbnd_base from site orbitals`

---

### Task 2: API StructureOption.suggest_nbnd

**Files:** `schemas.py`, `dft.py`, API test

- Add `suggest_nbnd: int` to `StructureOption`
- `list_structures`: `suggest_nbnd=suggest_nbnd_base(structure)`
- Test GET structures returns field for a pushed crystal

- [ ] Commit `feat(dft): expose suggest_nbnd on structures API`

---

### Task 3: Suite UI sync + SOC×2 + ship

**Files:** `dft_suite.js`

- `syncNbndSuggestion()`: read selected `suggest_nbnd`, ×2 if `#qe-soc` (or whatever id) checked, clamp 1–500, set `#qe-nbnd`, status text
- Call from structure change / refresh (with slab sync) and SOC change listener
- Brief pytest regression; push + Einstein

- [ ] Commit `feat(dft): auto-fill nbnd from suggest + SOC`
- [ ] Docs one-liner in roadmap if easy; push + Einstein

---

## Spec coverage

| Spec | Task |
|------|------|
| Core formula | 1 |
| API | 2 |
| UI + SOC×2 + ship | 3 |
