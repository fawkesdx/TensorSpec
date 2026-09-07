# Grizzly radint + Ylm CPU→GPU (B then A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop multi-GPU Full-ME jobs from rebuilding Chinook radial integrals twice (Phase B), then replace Slater/hydrogenic radint setup with a torch path in GrizzlyME (Phase A) so cold VTe₂ hydrogenic builds are order-of-magnitude faster.

**Architecture:** GrizzlyME owns `prepare_me_shell` / radint APIs and (later) `radint_torch`. TensorSpec’s `build_grizzly_me_shell` and multi-GPU runner become thin callers. Phase B uses parent pre-warm + disk cache (+ file lock) so spawn workers HIT. Phase A ports Chinook `make_radint_pointer` physics behind `GRIZZLY_RADINT=auto|torch|chinook`.

**Tech Stack:** Python ≥3.10, PyTorch ≥2.0, chinook ≥1.1.3, pytest, GrizzlyME package in `TensorSpec_GUI/GrizzlyME`, TensorSpec wire in `tensorspec/core/arpes/one_step/`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-07-grizzly-radint-ylm-gpu-design.md`
- Primary code lives in **GrizzlyME** repo/dir; TensorSpec only thin wire
- Version: **0.1.5** after Phase B; **0.2.0** after Phase A (or keep torch behind `auto` preferring chinook until A green)
- Spin / SARPES / `grid`/`fixed`/`exec` rad_type: Chinook fallback only
- Do not claim GPU when SSL holds VRAM — CPU torch / Chinook fallback OK
- Caveman chat OK; commits/docs = normal prose
- Do not commit unless user asks
- Do not merge TensorSpec to `main`

## File map

| File | Role |
|------|------|
| `GrizzlyME/grizzly/me_shell.py` | **Create:** `prepare_me_shell`, shared build+cache entry |
| `GrizzlyME/grizzly/radint_cache.py` | Extend: file lock around miss→put; optional path export |
| `GrizzlyME/grizzly/radint_torch.py` | **Create (Phase A):** Slater then hydrogenic `make_radint_pointer` |
| `GrizzlyME/grizzly/__init__.py` | Export new public symbols |
| `GrizzlyME/pyproject.toml` | Bump 0.1.5 / 0.2.0 |
| `GrizzlyME/tests/test_me_shell_prepare.py` | Phase B API + lock/HIT behavior |
| `GrizzlyME/tests/test_radint_torch_parity.py` | Phase A vs Chinook |
| `tensorspec/.../chinook_arpes_kmesh.py` | `build_grizzly_me_shell` → `prepare_me_shell` |
| `tensorspec/.../chinook_remote_runner_template.py` | Parent pre-warm before multi-GPU spawn |
| `tensorspec/tests/test_grizzly_me_shell_single_build.py` | Smoke: pre-warm + no dual build (mock/light) |
| `GrizzlyME/README.md` | Note Phase B/A + `GRIZZLY_RADINT` |

## Confirmed facts (do not re-derive)

1. Multi-GPU path: each worker calls `build_grizzly_me_shell` → both log `building radial integrals` (`chinook_remote_runner_template.py` `_multigpu_theta_worker` ~198–203).
2. Disk cache exists at `~/.cache/grizzlyme/radint` keyed by `make_radint_cache_key` (includes `rad_type`) — hydrogenic ≠ Slater.
3. Chinook `make_radint_pointer` = `define_radial_wavefunctions` → `fill_radint_dic` → `radint_dict_to_arr` (`chinook/radint_lib.py`).
4. Grizzly already has `ylm_torch` + `tests/test_ylm.py` parity vs Chinook Ylm.
5. Einstein install: `grizzlyme` 0.1.4 in TensorSpec_env (often under `/mnt/data/sandy/tensorspec_heavy/TensorSpec_env`).

---

### Task 1: Phase B — file lock on radint cache miss

**Files:**
- Modify: `GrizzlyME/grizzly/radint_cache.py`
- Test: `GrizzlyME/tests/test_radint_cache_lock.py`

**Interfaces:**
- Consumes: existing `RadintSetupCache.get` / `put`
- Produces: `RadintSetupCache.get_or_build(key, builder)` that serializes miss builders across processes via `fcntl` lock file next to disk pickle

- [ ] **Step 1: Write the failing test**

```python
# GrizzlyME/tests/test_radint_cache_lock.py
from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

from grizzly.radint_cache import RadintSetupCache


def _builder_counting(counter_path: Path):
    def _build():
        # Simulate expensive radint
        time.sleep(0.3)
        n = int(counter_path.read_text() or "0")
        counter_path.write_text(str(n + 1))
        Bfuncs = ["fake"]
        pointers = [0]
        return Bfuncs, pointers

    return _build


def _worker(disk_dir: str, key, counter: str):
    cache = RadintSetupCache(disk_dir=Path(disk_dir), use_disk=True)
    cache.get_or_build(key, _builder_counting(Path(counter)))


def test_get_or_build_only_one_builder_across_two_processes(tmp_path: Path):
    counter = tmp_path / "builds.txt"
    counter.write_text("0")
    key = ("lock-test", 1.0, "hydrogenic")
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_worker, args=(str(tmp_path), key, str(counter)))
        for _ in range(2)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0
    assert counter.read_text() == "1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd GrizzlyME && python -m pytest tests/test_radint_cache_lock.py::test_get_or_build_only_one_builder_across_two_processes -v`

Expected: FAIL (`get_or_build` missing) or FAIL assert builds == 2 if naive double get+put

- [ ] **Step 3: Implement `get_or_build`**

Add to `RadintSetupCache` in `radint_cache.py`:

```python
import fcntl
import time

def get_or_build(self, key, builder):
    """Return cached (Bfuncs, pointers); only one process runs builder for key."""
    if key is None:
        return builder()
    hit = self.get(key)
    if hit is not None:
        return hit
    self.disk_dir.mkdir(parents=True, exist_ok=True)
    lock_path = self.disk_dir / f"{_key_digest(key)}.lock"
    with lock_path.open("a+") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            hit = self.get(key)
            if hit is not None:
                return hit
            Bfuncs, pointers = builder()
            self.put(key, Bfuncs, pointers)
            return Bfuncs, pointers
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
```

Keep `get`/`put` behavior unchanged for existing tests.

- [ ] **Step 4: Run tests**

Run: `cd GrizzlyME && python -m pytest tests/test_radint_cache_lock.py tests/test_radint_cache.py tests/test_radint_disk_cache.py -v`

Expected: all PASS

- [ ] **Step 5: Commit** (only if user asked)

```bash
cd GrizzlyME
git add grizzly/radint_cache.py tests/test_radint_cache_lock.py
git commit -m "feat(radint): file-lock get_or_build for multi-process cache"
```

---

### Task 2: Phase B — `prepare_me_shell` in GrizzlyME

**Files:**
- Create: `GrizzlyME/grizzly/me_shell.py`
- Modify: `GrizzlyME/grizzly/__init__.py`
- Test: `GrizzlyME/tests/test_me_shell_prepare.py`

**Interfaces:**
- Consumes: `RadintSetupCache.get_or_build`, `make_radint_cache_key`, `dig_range_from_cube`, chinook `all_Y` / `projection_map` / `radint_lib` (Phase B)
- Produces:

```python
@dataclass
class PreparedMeShell:
    basis: Any
    prefactors: np.ndarray
    Largs: Any
    Margs: Any
    Gbasis: Any
    proj_arr: Any
    Bfuncs: Any
    radint_pointers: Any
    nstates: int
    spin: bool
    mfp: float
    radint_source: str  # "cache" | "chinook" | later "torch"

def prepare_me_shell(exp, *, radint_mode: str = "auto") -> PreparedMeShell:
    ...
```

- [ ] **Step 1: Write failing test (small stub exp or skip if no chinook orbital fixtures)**

Use the lightest path that already works in `tests/test_full_pipeline.py` / `test_second_model.py` — copy minimal TB+ARPES_dict construction from an existing test. Assert:

```python
def test_prepare_me_shell_second_call_is_cache():
    exp = _tiny_full_me_exp()  # helper from conftest or local
    s1 = prepare_me_shell(exp)
    s2 = prepare_me_shell(exp)
    assert s1.radint_source in ("chinook", "cache", "torch")
    assert s2.radint_source == "cache"
    assert len(s1.radint_pointers) == len(s1.basis)
```

- [ ] **Step 2: Run — expect FAIL import**

Run: `cd GrizzlyME && python -m pytest tests/test_me_shell_prepare.py -v`

- [ ] **Step 3: Implement `me_shell.py`**

Logic mirror of TensorSpec `build_grizzly_me_shell` radint block:

1. `exp.basis = exp.rot_basis()` if caller has not already (document: caller may pass already-rotated basis; match TensorSpec: call `rot_basis` once inside or require pre-rotated — **match TensorSpec today: rotate inside**)
2. `key = make_radint_cache_key(exp)`
3. `dig_range = dig_range_from_cube(exp.cube)`
4. `Bfuncs, pointers = DEFAULT_RADINT_CACHE.get_or_build(key, builder)` where builder calls chinook `make_radint_pointer` (ignore `radint_mode` torch until Task 5; if `radint_mode=="torch"` raise NotImplementedError until A)
5. `all_Y` + `projection_map` as today
6. Prefactors from mfp/depth

Export from `__init__.py`: `prepare_me_shell`, `PreparedMeShell`.

- [ ] **Step 4: pytest pass**

Run: `cd GrizzlyME && python -m pytest tests/test_me_shell_prepare.py -v`

- [ ] **Step 5: Commit** (if asked)

```bash
git commit -m "feat: add prepare_me_shell with locked radint build"
```

---

### Task 3: Phase B — TensorSpec wire + multi-GPU parent pre-warm

**Files:**
- Modify: `tensorspec/core/arpes/one_step/chinook_arpes_kmesh.py` (`build_grizzly_me_shell` ~733–830)
- Modify: `tensorspec/core/arpes/one_step/chinook_remote_runner_template.py` (`_run_full_cube_multigpu` ~242–297)
- Test: `tests/test_grizzly_shell_prewarm.py`

**Interfaces:**
- Consumes: `grizzly.prepare_me_shell` / `PreparedMeShell`
- Produces: `GrizzlyMeShell` unchanged dataclass filled from `PreparedMeShell`; parent warms cache before spawn

- [ ] **Step 1: Failing smoke test**

```python
# tests/test_grizzly_shell_prewarm.py
"""Ensure build_grizzly_me_shell delegates to grizzly.prepare_me_shell when available."""

def test_build_grizzly_me_shell_import_prepare():
    from tensorspec.core.arpes.one_step import chinook_arpes_kmesh as k
    assert hasattr(k, "build_grizzly_me_shell")
    # Structural: source mentions prepare_me_shell after wire
    import inspect
    src = inspect.getsource(k.build_grizzly_me_shell)
    assert "prepare_me_shell" in src
```

(After implement, this passes. Before: fail.)

- [ ] **Step 2: Refactor `build_grizzly_me_shell`**

Replace inline radint/`all_Y` with:

```python
from grizzly import prepare_me_shell
# ... build exp as today ...
prepared = prepare_me_shell(exp)
return GrizzlyMeShell(
    basis=prepared.basis,
    prefactors=prepared.prefactors,
    ...
)
```

Keep `GrizzlyMeShell` fields identical so `run_grizzly_arpes(..., me_shell=)` unchanged.

- [ ] **Step 3: Parent pre-warm in `_run_full_cube_multigpu`**

Before `procs = [...]` / `p.start()`:

```python
print("Pre-warming ME shell radint cache (once for all GPU workers)...", flush=True)
tb_model = load_tb_model_from_npz(os.path.abspath(tb_file), float(e_fermi))
from chinook_arpes_kmesh import build_grizzly_me_shell  # or kmesh module already loaded
build_grizzly_me_shell(tb_model, physics, B_matrix, e_axis)
print("ME shell radint cache warm.", flush=True)
```

Workers still call `build_shell` but should log **cache HIT** (not building). Optionally pass `payload["radint_prewarmed"]=True` for clearer logs only.

- [ ] **Step 4: Run tests**

```bash
cd /Users/sandyai/Documents/GitHub/TensorSpec_GUI
python -m pytest tests/test_grizzly_shell_prewarm.py -v
cd GrizzlyME && python -m pytest tests/test_me_shell_prepare.py tests/test_radint_cache_lock.py -v
```

- [ ] **Step 5: Bump GrizzlyME to 0.1.5 + README note**

- `pyproject.toml` version `0.1.5`
- README: multi-GPU single radint build; disk lock

- [ ] **Step 6: Commit** (if asked) in **both** repos as appropriate

---

### Task 4: Phase B — Einstein install check (manual)

**Files:** none (ops)

- [ ] **Step 1:** Build/install editable or wheel of GrizzlyME 0.1.5 into Einstein TensorSpec_env

```bash
# from Mac after release/commit
ssh einstein 'cd /path/to/GrizzlyME && /home/sandy/TensorSpec/TensorSpec_env/bin/pip install -U .'
# or pip install grizzlyme==0.1.5 from PyPI if published
```

- [ ] **Step 2:** Re-run small Full ME or watch logs: only **one** `building radial…` (or zero if HIT), then workers `radint cache HIT`, then `1/16`…

- [ ] **Step 3:** Record wall time note in `GrizzlyME/benchmarks/` or checklist — no code required

---

### Task 5: Phase A — Slater torch radint + Chinook parity

**Files:**
- Create: `GrizzlyME/grizzly/radint_torch.py`
- Modify: `GrizzlyME/grizzly/me_shell.py` (`radint_mode` / `GRIZZLY_RADINT`)
- Test: `GrizzlyME/tests/test_radint_torch_parity.py`

**Interfaces:**
- Consumes: same args as chinook `make_radint_pointer(rad_dict, basis, Eb)`
- Produces: `make_radint_pointer_torch(...)` → `(B_array, B_pointers)` Chinook-compatible (callables or interpolators matching eval at float E)

**Reference implementation path:**  
`TensorSpec_env/.../chinook/radint_lib.py` — port `define_radial_wavefunctions` (slater only), `fill_radint_dic`, `radint_dict_to_arr`. Prefer vectorized torch/numpy for energy×radius grids; may wrap scipy/numpy interpolate like Chinook for Bfuncs callables so Grizzly ME eval unchanged.

- [ ] **Step 1: Failing parity test**

```python
# GrizzlyME/tests/test_radint_torch_parity.py
import numpy as np
import chinook.radint_lib as radint_lib
from grizzly.radint_torch import make_radint_pointer_torch

def test_slater_radint_matches_chinook_tiny_basis():
    basis, rad_dict, Eb = _tiny_slater_basis_fixture()  # from chinook orbital stubs used elsewhere
    B_c, P_c = radint_lib.make_radint_pointer(rad_dict, basis, Eb)
    B_t, P_t = make_radint_pointer_torch(rad_dict, basis, Eb, device="cpu")
    assert np.array_equal(P_c, P_t)
    energies = np.linspace(Eb[0], Eb[1], 17)
    for i in range(len(B_c)):
        for j in (0, 1):
            yc = np.array([complex(B_c[i, j](float(e))) for e in energies])
            yt = np.array([complex(B_t[i, j](float(e))) for e in energies])
            assert np.allclose(yc, yt, rtol=1e-5, atol=1e-6), (i, j)
```

Build `_tiny_slater_basis_fixture` using same orbital construction as `tests/test_full_pipeline.py` (copy minimal).

- [ ] **Step 2: Run — FAIL missing module**

- [ ] **Step 3: Implement Slater path in `radint_torch.py`**

Supported: `rad_type` missing or `slater`. Else `raise NotImplementedError` → caller falls back.

Env resolution in `prepare_me_shell`:

```python
mode = radint_mode or os.environ.get("GRIZZLY_RADINT", "auto").lower()
# auto: try torch for slater/hydrogenic when implemented; else chinook
```

Default **`auto`**: use torch when function supports rad_type; on any exception / NotImplementedError → chinook + log warning.

- [ ] **Step 4: pytest parity PASS**

Run: `cd GrizzlyME && python -m pytest tests/test_radint_torch_parity.py -v`

- [ ] **Step 5: Commit** (if asked)

```bash
git commit -m "feat(radint): torch Slater make_radint_pointer with Chinook parity"
```

---

### Task 6: Phase A — Hydrogenic torch + wire default

**Files:**
- Modify: `GrizzlyME/grizzly/radint_torch.py`
- Modify: `GrizzlyME/tests/test_radint_torch_parity.py`
- Modify: `GrizzlyME/grizzly/me_shell.py`
- Modify: `GrizzlyME/README.md`, `pyproject.toml` → **0.2.0**

- [ ] **Step 1: Add hydrogenic parity test** (same energies, `rad_type="hydrogenic"`)

- [ ] **Step 2: Implement hydrogenic branch** mirroring Chinook `econ.hydrogenic_exec` path in `define_radial_wavefunctions`

- [ ] **Step 3: Ensure `auto` selects torch for hydrogenic**

- [ ] **Step 4: Full GrizzlyME pytest**

```bash
cd GrizzlyME && python -m pytest tests/ -v --ignore=benchmarks
```

- [ ] **Step 5: README + version 0.2.0**

- [ ] **Step 6: Einstein `pip install -U`**; cold hydrogenic VTe₂: expect one build ≪ prior 10+ min; log `radint_source=torch` / wall seconds

- [ ] **Step 7: Commit / tag / PyPI** (only if user asks)

---

### Task 7: Optional Ylm setup preference (still Phase B scope if time)

**Files:**
- Modify: `GrizzlyME/grizzly/me_shell.py`
- Test: extend `tests/test_ylm.py` or `test_me_shell_prepare.py`

Only if Phase B still open: where `all_Y` output can be proven equal to assembling from `ylm_torch` + Chinook Gaunt tables, prefer torch. **If parity unclear, skip — do not block A.**

- [ ] Document skip or implement with `max_err < 1e-10` gate like `test_ylm.py`

---

## Spec coverage checklist

| Spec item | Task |
|-----------|------|
| Single radint build multi-GPU | 1, 3 |
| `prepare_me_shell` API | 2 |
| Setup Ylm prefer ylm_torch | 7 (optional) |
| TensorSpec thin wire | 3 |
| Cache key / disk unchanged | 1–2 |
| Torch Slater then hydrogenic | 5–6 |
| `GRIZZLY_RADINT` flag | 5 |
| Chinook fallback unsupported types | 5–6 |
| 0.1.5 / 0.2.0 + Einstein deploy | 3–4, 6 |
| Non-goals respected | Global Constraints |

## Placeholder / consistency scan

- No TBD steps; fixture helper `_tiny_*` must be copied from existing GrizzlyME tests (not invented empty).
- `PreparedMeShell` field names match `GrizzlyMeShell` mapping in Task 3.
- `get_or_build` used by `prepare_me_shell`; parent pre-warm uses same path.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-07-grizzly-radint-ylm-gpu.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — this session with executing-plans checkpoints  

Which approach?
