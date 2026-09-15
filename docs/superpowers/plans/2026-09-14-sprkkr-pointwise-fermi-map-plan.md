# SPR-KKR point-wise FERMI MAP (2-D deflector grid) — implementation plan

> **For agentic workers:** one task each, haiku-class. Read ONLY the files + line anchors your task names. pytest green before handback. Gate = Sandy says go. Caveman talk.

Phase 1a plan: `docs/superpowers/plans/2026-09-13-sprkkr-pointwise-deflector-plan.md` (merged, `f2d7c5c` + `0490ad7`). Read it first.
Rules: `sandy_rule.md` (3 layers, no silent deletes, snippet edits only, xarray).
Planner: opus. Workers: haiku. Status: v1 2026-09-14, waiting Sandy GO.

All line numbers are against the repo as of 2026-09-14. Anchor lines are quoted. Do not rewrite files.

---

## Goal

Phase 1a gave ONE deflector cut: `Φ = -10.3` fixed, `NT` slit angles, `NT` jobs. Sandy wants the MAP. Today she gets it by a bash loop over `--deflector`: 11 cuts × 100 jobs, sequential, ~5 h each.

Phase 1b: one Run button does the whole map.

- GUI: `Φ (Deflect)` = `-15..15` NP=11, `Θ (Slit)` = `-10..10` NT=100, `E = -1.0..0.1` NE=200, point-wise ON, Run.
- 1100 single-point `kkrspec` jobs, launched in **waves of `max_concurrent`** (GUI "Cores:" spin, default 40).
- Stitch to ONE `(energy, theta=slit, phi=deflector)` cube with lab k axes.
- Display it. "Save to disk" writes a viewer `.npz` with `kx = k_slit`, `ky = k_defl`, both 1/Å.
- CLI same: `--deflector-range D1 D2 NP` (scalar `--deflector` keeps working), `--max-concurrent`.

Nothing here touches OSKI. Nothing here changes the Fermi-edge rule (GUI applies FD, CLI does not).

---

## Architecture

Layer 1 core, changed:

```
tensorspec/core/dft/sprkkr/
  pointwise.py     MOD  + AnglePoint.defl_deg, + angle_points_grid()
  outputs.py       MOD  stitch_points() -> 2-D (theta, phi) grid
  workflow.py      MOD  run_arpes(max_concurrent=), ArpesRunHandle pump, batched remote poll
  progress.py      MOD  + pointwise_eta_seconds()
  spc_results.py   MOD  NP>1 read-back (mostly free; needs a test)
  viewer_export.py MOD  arrays_to_viewer_npz grid path (kx, ky both real)
tensorspec/core/arpes/one_step/
  kkr_wrapper.py   MOD  np_>1 -> grid points, max_concurrent passthrough, k_slit/k_defl out
```

Layer 3:

```
tensorspec/gui/components/arpes_panel.py   MOD  5 anchors: drop ky_min==ky_max block, max_concurrent, job guard, ETA, k-axis save
scripts/sprkkr/sprkkr_e2e.py               MOD  --deflector-range, --max-concurrent, dry-run NT*NP
```

Data flow, one line:

```
GUI Φ range -> k_bounds["Y"] -> params.phi_e/np_ -> angle_points_grid -> NT*NP jobs (waves)
  -> stitch_points 2-D -> dataset(energy, theta, phi) + k_slit_1perA/k_defl_1perA coords
  -> wrapper dict {theta, phi, k_slit, k_defl} -> panel display (deg) + npz save (1/Å)
```

---

## Global Constraints

### Point ordering: **deflector-major**

`angle_points_grid` loops deflector OUTER, slit INNER. `index = j * nt + i` with `j` = deflector index, `i` = slit index. Both axes ascending `linspace`.

Why deflector-major: with a throttle, waves finish whole deflector CUTS first. Kill a run half way -> you still have complete cuts, same shape as today's bash loop. Slit-major would leave every cut half-done and nothing plottable. This ordering is fixed everywhere — job dirs `_pNNNN`, sidecar JSON, dry-run print, `.spc` suffix match.

### Throttle lives in the HANDLE, not in a background thread

`run_arpes` prepares + launches the first `max_concurrent` jobs, then hands the REST (un-prepared tuples) to `ArpesRunHandle` together with a launch closure. `ArpesRunHandle._pump()` launches more whenever in-flight drops below the cap. `_pump()` is called from `is_done()` and `fraction_done()`.

Why not a background launcher thread: a thread would build `.inp` files and run `sftp` uploads concurrent with `collect()`'s own ssh calls on the SAME paramiko connection — `RemoteLauncher.connect()` returns one shared client, that is a race. The pump is single-threaded, driven by whoever polls, and `collect()` already polls in a loop.

`wait=False` still works: the returned handle launches the next wave on every `is_done()` / `fraction_done()` call. Documented contract: **an async caller must poll the handle or the queue stalls.** GUI's B3 path is `wait=True` inside a QThread, so this is moot there.

`max_concurrent=None` = today's fire-all. Byte-identical path.

### Poll cost is the real risk

At 1100 jobs, one ssh round trip per job per poll = 1100 `ssh` calls every 3 s. That is worse than the calculation. Task 4 batches: ONE `ps` call for liveness, ONE `grep -c` call for row counts, plus a per-subjob `finished` latch so a done job is never re-queried. Non-negotiable — the throttle is useless without it.

### `stitch_points` back-compat rule

`stitch_points` picks its path by the points, not by a flag:

- `len({p.defl_deg for p in points}) > 1` -> 2-D grid path.
- else -> **today's 1-D path, unchanged**, `phi = [float(deflector_deg)]` from the argument.

Old sidecars written by phase 1a have no `defl_deg` key; `AnglePoint(**d)` still works because the field has default `0.0`, all equal -> 1-D path -> `deflector_deg` from `meta`. Old runs on Einstein keep loading.

### Energy reference: unchanged

One `ref_energy_eV` for the whole grid. Same `0.13 deg/eV` / `0.7 %/eV` residual as phase 1a. Do not re-derive.

### Fermi edge: unchanged

CLI does NOT apply it. GUI applies FD with `temperature_K` in `kkr_wrapper.run_simulation` (lines 252-265). Touch neither.

### Other

- No silent deletes. Keep `--raw`, keep the confirm-pot dialog, keep the legacy non-pointwise path.
- All 14 tests in `tests/sprkkr/test_pointwise.py` and 7 in `test_pointwise_stitch.py` must stay green.
- After any pytest run, overwrite `tests/sprkkr/RESULTS.md`, tiny: date / suite / pass-fail counts / failed names. No HTML.
- No Qt in tests. No binary in tests.

---

## Task 1 — `pointwise.py`: `angle_points_grid` + `defl_deg`

Depends on: nothing.

**Files**
- MODIFY `tensorspec/core/dft/sprkkr/pointwise.py`
- MODIFY `tests/sprkkr/test_pointwise.py` (append, do not edit the 14 existing tests)

**Interfaces** (exact)

```python
@dataclass(frozen=True)
class AnglePoint:
    index: int
    slit_deg: float
    theta_e_deg: float
    phi_e_deg: float
    k_par: float
    k_slit: float
    k_defl: float
    defl_deg: float = 0.0   # NEW, last, has default -> old JSON still loads

def angle_points_grid(
    theta_range_deg: Tuple[float, float],
    nt: int,
    deflector_range_deg: Tuple[float, float],
    np_: int,
    *,
    hv_eV: float,
    work_function_eV: float,
    slit_rot_deg: float = 0.0,
    manip_theta_deg: float = 0.0,
    manip_azimuth_deg: float = 0.0,
    manip_tilt_deg: float = 0.0,
    ref_energy_eV: float = 0.0,
    phi_offset_deg: float = 0.0,
) -> List[AnglePoint]
```

**Steps**
- [ ] Anchor — add `defl_deg: float = 0.0` at the END of the dataclass, after this exact line (currently 33):
  ```python
      k_defl: float  # signed lab deflector-axis k, 1/A
  ```
  Must be last: every other field is positional and has no default.
- [ ] `angle_points_grid`: deflector axis `defl = np.linspace(d0, d1, np_)`, or `[mean]` when `np_ == 1` — same `nt == 1` midpoint rule already at line 132-135. Then:
  ```python
  for j, d in enumerate(defl):        # deflector OUTER
      for i, slit in enumerate(theta_slit_deg):   # slit INNER
          index = j * nt + i
  ```
  Reuse `lab_k_vectors` and `sample_to_bulk_frame` per deflector row — call `lab_k_vectors(theta_slit_deg, d, slit_rot_deg, k)` once per `j`, `(3, nt)` matrix, exactly as today's line 138. Do NOT loop point-by-point through the rotation.
- [ ] Every point gets `defl_deg=float(d)`. `k_defl = k * sin(radians(d))` (already the formula at line 160, now per row).
- [ ] Keep the `theta_e > 90.0` `ValueError` (line 162-165), message unchanged shape but say which `(i, j)`.
- [ ] Anchor — rewrite `angle_points` (line 97) as a **thin wrapper**, body one line:
  ```python
      return angle_points_grid(
          theta_range_deg, nt, (deflector_deg, deflector_deg), 1,
          hv_eV=hv_eV, work_function_eV=work_function_eV, slit_rot_deg=slit_rot_deg,
          manip_theta_deg=manip_theta_deg, manip_azimuth_deg=manip_azimuth_deg,
          manip_tilt_deg=manip_tilt_deg, ref_energy_eV=ref_energy_eV,
          phi_offset_deg=phi_offset_deg,
      )
  ```
  Keep its signature and docstring. `np_=1` -> `index = 0*nt + i = i`, identical to today. `defl_deg` now carries `deflector_deg` instead of staying 0.0 — that is REQUIRED so `stitch_points` and `viewer_export` can read the deflector off the point.
- [ ] Module docstring: add a short "Grid ordering" paragraph — deflector-major, `index = j*nt + i`, why (partial run = whole cuts).
- [ ] Tests appended to `tests/sprkkr/test_pointwise.py`, exact names:
  - `test_grid_np1_equals_angle_points` — `angle_points_grid((-15,15), 5, (-10.3,-10.3), 1, hv_eV=84, work_function_eV=4.5)` field-for-field equal to `angle_points((-15,15), 5, hv_eV=84, work_function_eV=4.5, deflector_deg=-10.3)`
  - `test_grid_index_is_deflector_major` — `nt=4, np_=3`: `len == 12`; `p.index == j*4 + i`; `[p.defl_deg for p in pts[0:4]]` all equal, `pts[4].defl_deg != pts[0].defl_deg`
  - `test_grid_slit_repeats_per_deflector` — slit sequence of block `j` equals block `0` for every `j`
  - `test_grid_k_defl_constant_within_block`
  - `test_grid_k_slit_constant_across_blocks` — `pts[i].k_slit == pts[j*nt+i].k_slit` exactly
  - `test_grid_defl_zero_row_matches_single_cut` — the `defl=0` row of a `(-10,10,3)` grid == `angle_points(..., deflector_deg=0.0)`
  - `test_grid_json_round_trip_keeps_defl_deg` — through `write_points_json` / `read_points_json`
  - `test_old_points_json_without_defl_deg_loads` — hand-build a dict with the 7 old keys, `AnglePoint(**d).defl_deg == 0.0`
- [ ] Run `python3 -m pytest tests/sprkkr/test_pointwise.py -q`. All 14 old + 8 new green.

**Worker prompt hint:** "Read plan Task 1. Edit `tensorspec/core/dft/sprkkr/pointwise.py`: one new dataclass field at the end, one new function `angle_points_grid`, turn `angle_points` into a wrapper. Do NOT touch `k_vacuum`, `wrap_deg`, `lab_k_vectors`, `sprkkr_angles_to_lab_k`. Append 8 tests. Run `python3 -m pytest tests/sprkkr/test_pointwise.py -q` — 14 old tests must still pass."

---

## Task 2 — `outputs.stitch_points`: 2-D (slit, deflector) grid

Depends on: Task 1.

**Files**
- MODIFY `tensorspec/core/dft/sprkkr/outputs.py` (`stitch_points` only, lines 304-384)
- MODIFY `tests/sprkkr/test_pointwise_stitch.py` (append)

**Interface** — signature UNCHANGED:

```python
def stitch_points(datasets, points, deflector_deg: float = 0.0) -> xr.Dataset
```

**Steps**
- [ ] Keep the length check (line 326-329) and the `theta==1 / phi==1` asserts (line 335-338) exactly.
- [ ] After the asserts, branch:
  ```python
  defl_vals = sorted({round(float(p.defl_deg), 9) for p in points})
  grid = len(defl_vals) > 1
  ```
- [ ] **1-D path (`not grid`): leave the existing code untouched** from line 341 (`ds = ds.assign_coords(theta=[p.slit_deg], phi=[float(deflector_deg)])`) to line 382. Byte-identical.
- [ ] **2-D path:** `slit_vals = sorted({round(float(p.slit_deg), 9) for p in points})`; require `len(points) == len(slit_vals) * len(defl_vals)` else `ValueError("stitch_points: points are not a full slit x deflector grid")`.
- [ ] Align by VALUE, not position (phase-1a bug lesson). Build a dict first:
  ```python
  by_key = {}
  for ds, p in zip(datasets, points):
      ds = ds.assign_coords(theta=[float(p.slit_deg)], phi=[float(p.defl_deg)])
      ds = ds.rename({"k_par": "k_par_sprkkr"})
      by_key[(round(float(p.slit_deg), 9), round(float(p.defl_deg), 9))] = ds
  ```
  then, for each `d` in `defl_vals`, `cut = xr.concat([by_key[(s, d)] for s in slit_vals], dim="theta")`, and finally `merged = xr.concat(cuts, dim="phi")`. Missing key -> `KeyError` is fine, wrap it into a `ValueError` naming the `(slit, defl)` pair. No `sortby` needed — we built the axes sorted — but keep `.sortby("theta").sortby("phi")` as a cheap guard.
- [ ] Lab momenta, 2-D ratios:
  ```python
  ratio_slit = np.zeros((len(slit_vals), len(defl_vals)))
  ratio_defl = np.zeros_like(ratio_slit)
  ```
  filled from `p.k_slit / p.k_par` and `p.k_defl / p.k_par` (0.0 when `p.k_par == 0`), indexed by the position of `p` in `slit_vals` / `defl_vals`. Wrap as
  `xr.DataArray(ratio_slit, dims=("theta","phi"), coords={"theta": slit_vals, "phi": defl_vals})`.
  Then, same shape as today's lines 372-373:
  ```python
  merged["k_par"]  = merged["k_par_sprkkr"] * ratio_slit_da
  merged["k_defl"] = merged["k_par_sprkkr"] * ratio_defl_da
  ```
  Both stay `(energy, theta, phi)` — same as phase 1a, the GUI plot code does not change shape.
- [ ] NEW, **both paths** (additive, no existing var/coord touched):
  - 2-D data vars at ref energy: `merged["k_slit_ref"] = DataArray((theta, phi))` from `p.k_slit`, `merged["k_defl_ref"]` from `p.k_defl`.
  - 1-D coords the GUI and npz use:
    `merged = merged.assign_coords(k_slit_1perA=("theta", [p.k_slit for one deflector row]), k_defl_1perA=("phi", [p.k_defl for one slit column]))`.
    These are exact 1-D because `k_slit = k sin(slit)` never depends on the deflector and `k_defl = k sin(defl)` never depends on the slit (see `pointwise.angle_points_grid`).
- [ ] attrs (anchor lines 376-382): keep `NT = len(slit_vals)`, set `NP = len(defl_vals)`, `pointwise = True`, `phi_is_index = False`, and `deflector_deg = deflector_deg` for the 1-D path / `deflector_list = defl_vals` for the grid path. Never clobber `EF_Ry` / `NE`.
- [ ] Docstring: say the two paths, the grid ordering, and that alignment is by coordinate value.
- [ ] Tests appended to `tests/sprkkr/test_pointwise_stitch.py`, exact names (build fakes with `make_fake_spc`'s shape as the file already does; set col 7 `k_par` non-zero so ratios have teeth):
  - `test_stitch_grid_theta_axis_is_sorted_slit_angles`
  - `test_stitch_grid_phi_axis_is_sorted_deflectors`
  - `test_stitch_grid_shape_is_ne_nt_np`
  - `test_stitch_grid_shuffled_input_same_result` — shuffle `(datasets, points)` pairs with a fixed seed, assert `xr.testing.assert_allclose` against the unshuffled merge
  - `test_stitch_grid_k_slit_ref_varies_with_theta_only`
  - `test_stitch_grid_k_defl_ref_varies_with_phi_only`
  - `test_stitch_grid_k_slit_1perA_coord_matches_points`
  - `test_stitch_grid_incomplete_grid_raises`
  - `test_stitch_np1_path_unchanged` — a 3-point single-deflector stitch has `phi == [deflector_deg]`, `NP == 1`, `k_par` values identical to the phase-1a expectation
- [ ] Run `python3 -m pytest tests/sprkkr/test_pointwise_stitch.py -q`. 7 old + 9 new green.

**Worker prompt hint:** "Read plan Task 2. Edit ONLY `stitch_points` in `tensorspec/core/dft/sprkkr/outputs.py` (lines 304-384). Add a branch at the top; keep the single-deflector body exactly as it is. Do not touch `parse_spc` / `stitch_spc` / `spc_to_tensor` / `spc_to_datatree` / `write_points_json` / `read_points_json`. Append 9 tests. Run `python3 -m pytest tests/sprkkr/test_pointwise_stitch.py -q` — 7 old tests must still pass."

---

## Task 3 — `workflow.run_arpes(max_concurrent=)`: launch in waves

Depends on: nothing (does not need Task 1 or 2).

**Files**
- MODIFY `tensorspec/core/dft/sprkkr/workflow.py`
- MODIFY `tests/sprkkr/test_workflow.py` (new class `TestThrottle`)

**Interfaces** (exact — additive, every existing caller keeps working)

```python
def run_arpes(
    pot_path, params, workdir, launcher, nproc=1, mode="auto", mpi_available=True,
    wait=True, progress_cb=None, remote_workdir=None, cif_lattice=None, extra_raw=None,
    angle_points=None, deflector_deg=0.0,
    max_concurrent: Optional[int] = None,          # NEW
)

class ArpesRunHandle:
    def __init__(self, subjobs, params, launcher, start_time, progress_cb=None,
                 geometry=None, points=None, deflector_deg=0.0,
                 pending: Optional[List[tuple]] = None,          # NEW: un-launched (sub_params, subdir_name, nproc_i)
                 launch_fn: Optional[Callable[[tuple], "_SubJob"]] = None,   # NEW
                 max_concurrent: Optional[int] = None):          # NEW

    def _pump(self) -> int: ...   # NEW: launch until in-flight == max_concurrent
```

**Handle done-check API (quoted, do not invent)** — `workflow.py` lines 114-117:
```python
def _is_running(handle: Any) -> bool:
    if hasattr(handle, "is_running"):
        return bool(handle.is_running())
    return handle.poll() is None
```
`RemoteHandle.is_running()` (jobs.py 219-227) = ssh `kill -0 <pid>` (or `squeue`). `LocalHandle.poll()` (jobs.py 85-86) = `Popen.poll()`. `LocalHandle` also has `.wait(timeout)`; do NOT use it — blocking on one job stalls the pump.

**Steps**
- [ ] Anchor — pull the launch-loop body (currently lines 547-575) into a **nested closure** defined just above it, capturing `workdir`, `local_pot_path`, `launcher`, `remote`, `remote_workdir`, `extra_raw`:
  ```python
      def _prep_and_launch(entry):
          sub_params, subdir_name, nproc_i = entry
          ... exactly today's body, lines 548-565 ...
          return _SubJob(handle=handle, job=job, subdir=subdir,
                         sub_params=sub_params, expected_spc=arpes_inputs.expected_spc,
                         expected_log=arpes_inputs.expected_log)
  ```
  Move nothing else. Inputs are built and uploaded **lazily, at launch time** — that is the point: 1100 `.inp` builds and 1100 pot uploads do not all happen up front.
- [ ] Anchor — replace this exact line (currently 546) and the loop under it:
  ```python
      subjobs: List[_SubJob] = []
  ```
  with:
  ```python
      subjobs: List[_SubJob] = []
      pending = list(job_list)
      cap = int(max_concurrent) if max_concurrent else 0
      first = pending if cap <= 0 else pending[:cap]
      pending = [] if cap <= 0 else pending[cap:]
      for entry in first:
          subjobs.append(_prep_and_launch(entry))
      if pending:
          print(f"[throttle] max_concurrent={cap} launched={len(subjobs)} queued={len(pending)} "
                f"waves={-(-len(job_list) // cap)}")
  ```
  `cap <= 0` / `None` -> today's fire-all, `pending` empty, zero behaviour change.
- [ ] Anchor — the handle construction (currently 577-580): add `pending=pending, launch_fn=_prep_and_launch, max_concurrent=cap or None`.
- [ ] `ArpesRunHandle.__init__`: store `self._pending = list(pending or [])`, `self._launch_fn = launch_fn`, `self._max_concurrent = max_concurrent`.
- [ ] NEW `_pump`:
  ```python
      def _pump(self) -> int:
          if not self._pending or not self._max_concurrent or self._launch_fn is None:
              return 0
          running = sum(1 for sj in self.handles if _is_running(sj.handle))
          started = 0
          while self._pending and running + started < self._max_concurrent:
              self.handles.append(self._launch_fn(self._pending.pop(0)))
              started += 1
          return started
  ```
  Pop from the FRONT — keeps deflector-major order.
- [ ] Anchor — replace this exact line (currently 364-365):
  ```python
      def is_done(self) -> bool:
          return all(not _is_running(sj.handle) for sj in self.handles)
  ```
  with a version that pumps first and reports not-done while anything is queued:
  ```python
      def is_done(self) -> bool:
          self._pump()
          if self._pending:
              return False
          return all(not _is_running(sj.handle) for sj in self.handles)
  ```
- [ ] Anchor — progress denominator must NOT grow as waves launch. Replace this exact line (currently 347):
  ```python
          expected = sum(sj.sub_params.n_points for sj in self.handles)
  ```
  with:
  ```python
          expected = (sum(sj.sub_params.n_points for sj in self.handles)
                      + sum(e[0].n_points for e in self._pending))
  ```
  `e[0]` is the queued `sub_params`; `ArpesParams.n_points` is `params.py` line 115-116 (`ne * nt * np_`). Denominator constant, numerator only grows -> `progress_cb` stays monotonic.
- [ ] `collect()` needs no change. Its loop (lines 378-381) already drives the pump through `is_done()`:
  ```python
          poll_s = 3.0 if _is_remote(self._launcher) else 1.0
          while not self.is_done():
              self.fraction_done()
              time.sleep(poll_s)
  ```
  `time.sleep(poll_s)` means no CPU spin. Leave `poll_s` alone; Task 4 fixes the ssh cost inside it.
- [ ] `eta_s` (line 372-375) is left alone — it is already wrong for point-wise mode; the GUI uses the new formula instead (Task 7).
- [ ] `run_arpes` docstring: add a `max_concurrent` paragraph — `None` = launch everything (today), an int = keep at most that many jobs alive, prepared and launched lazily; and the **async contract**: with `wait=False` the caller MUST poll `is_done()` / `fraction_done()` or the queue never drains.
- [ ] Tests, new class `TestThrottle` appended to `tests/sprkkr/test_workflow.py` (reuse `FakeLauncher`, `make_fake_spc`, `cu_pot`, `_touch_bin`). `FakeHandle.poll()` returns `0` immediately, so add a local `SlowFakeLauncher` subclass whose handles report running for the first K polls, and count in-flight at each launch:
  - `test_throttle_never_exceeds_max_concurrent` — 10 jobs, `max_concurrent=3`; launcher records `in_flight` at every `launch()` call; `max(recorded) <= 3`
  - `test_throttle_launches_all_jobs` — all 10 land, `len(launcher.launched) == 10`
  - `test_throttle_preserves_order` — launched dataset names are `_p0000.._p0009` in order
  - `test_throttle_none_launches_all_at_once` — `max_concurrent=None`, every launch sees `in_flight == index`
  - `test_throttle_progress_is_monotonic` — record `fraction_done()` at each poll, assert non-decreasing and final `== 1.0`
  - `test_throttle_async_handle_drains_on_poll` — `wait=False`, loop `while not h.is_done(): pass` (fake handles finish instantly), then `h.collect()` returns all points
- [ ] Run `python3 -m pytest tests/sprkkr -q`.

**Worker prompt hint:** "Read plan Task 3. Edit `tensorspec/core/dft/sprkkr/workflow.py` at the anchors the plan quotes: signature (line 448), the launch loop (546-575) -> nested `_prep_and_launch` closure + wave slice, handle ctor (577-580), `ArpesRunHandle.__init__` (326-344), `is_done` (364-365), `fraction_done` line 347. Add `_pump`. Do NOT touch `collect()`, `plan_jobs`, `fanout.py`, or the pointwise branch at lines 507-544. Add `TestThrottle` to `tests/sprkkr/test_workflow.py`. Run `python3 -m pytest tests/sprkkr -q`."

---

## Task 4 — batch the remote poll (1100 jobs must not mean 1100 ssh calls)

Depends on: Task 3.

**Files**
- MODIFY `tensorspec/core/dft/sprkkr/workflow.py`
- MODIFY `tests/sprkkr/test_workflow.py` (class `TestPollBatching`)

**Why:** `collect()` polls every 3 s. Per poll it calls `_is_running(sj.handle)` per job (ssh `kill -0`, jobs.py 225) and `_remote_rows_done(sj.handle, ...)` per job (ssh `grep -c`, workflow.py 82-84). 1100 jobs = 2200 ssh round trips every 3 s. The poll costs more than the physics.

**Interfaces** (exact, new module-level helpers in `workflow.py`)

```python
def _remote_alive_pids(launcher: Any, pids: List[str]) -> set:
    """One ssh `ps -o pid= -p a,b,c` -> set of pids still alive. {} on error."""

def _remote_rows_done_many(launcher: Any, paths: List[str]) -> Dict[str, int]:
    """One ssh `grep -cE '^ *-?[0-9]' f1 f2 ...` -> {path: count}. grep prints
    `path:count` for multi-file input. {} on error (caller falls back to 0)."""
```

**Steps**
- [ ] `_SubJob` (dataclass, line 313-320): add `finished: bool = False` and `rows_cached: Optional[int] = None`. Defaults keep every existing construction valid.
- [ ] NEW `ArpesRunHandle._refresh_remote(self)`: called once per poll when `_is_remote(self._launcher)`.
  - Collect `pids = [sj.handle.pid for sj in self.handles if not sj.finished and getattr(sj.handle, "pid", None)]`; one `_remote_alive_pids` call; mark `sj.finished = True` for any whose pid is gone. A job whose handle has a `jobid` (SLURM) or no pid falls back to per-handle `_is_running` — Einstein has no SLURM, so this path is cold.
  - Collect `paths = [f"{sj.job.workdir}/{sj.expected_spc}" for sj in self.handles if sj.rows_cached is None or not sj.finished]`; ONE `_remote_rows_done_many` call; store into `sj.rows_cached`. Once `sj.finished` and rows read, never query that job again.
  - Chunk both calls at 200 items per ssh command so the command line stays sane.
- [ ] `fraction_done`: for the remote branch (lines 350-354), call `self._refresh_remote()` once, then sum `sj.rows_cached or 0` instead of the per-job `_remote_rows_done` call. Local branch (lines 355-358) unchanged — `arpes_rows_done` is a cheap local file read, but still latch `rows_cached` when `sj.finished`.
- [ ] `is_done`: for the remote case use the `sj.finished` latches set by `_refresh_remote()` instead of calling `_is_running` per job. `_pump`'s in-flight count uses the same latches. Local case keeps `_is_running`.
- [ ] Keep `_remote_rows_done` (lines 77-87) in place — it is still the single-job fallback and a public-ish helper. **No silent delete.**
- [ ] Tests, class `TestPollBatching`, with a fake remote launcher that counts `_run` calls:
  - `test_remote_poll_is_one_ssh_call_per_cycle` — 20 jobs, one `fraction_done()` -> `<= 4` `_run` calls (2 chunks max), not 40
  - `test_finished_job_not_repolled` — after a job's pid disappears, a second `fraction_done()` does not include it in the next `ps` / `grep` argument list
  - `test_rows_done_many_parses_grep_output` — feed `"/a/x.spc:12\n/a/y.spc:0\n"` -> `{"/a/x.spc": 12, "/a/y.spc": 0}`
  - `test_alive_pids_parses_ps_output`
  - `test_local_poll_unchanged` — local launcher path still counts rows off disk, same numbers as before
- [ ] Run `python3 -m pytest tests/sprkkr -q`.

**Worker prompt hint:** "Read plan Task 4. Edit `tensorspec/core/dft/sprkkr/workflow.py`: 2 new module-level helpers, 2 new `_SubJob` fields with defaults, new `ArpesRunHandle._refresh_remote`, and rewire `fraction_done` / `is_done` / `_pump` to use the latches. Do NOT delete `_remote_rows_done` or `_is_running`. Quote for the ssh commands: `RemoteHandle.is_running` is jobs.py lines 219-227, `_remote_rows_done` is workflow.py lines 77-87. Add `TestPollBatching`. Run `python3 -m pytest tests/sprkkr -q`."

---

## Task 5 — `kkr_wrapper.py`: NP>1 -> grid, `max_concurrent`, k axes out

Depends on: Tasks 1, 2, 3.

**Files**
- MODIFY `tensorspec/core/arpes/one_step/kkr_wrapper.py`
- MODIFY `tests/sprkkr/test_workflow.py` (extend `TestPointwiseKwargs`)

**Interfaces** (exact)

```python
def _map_deflector(kwargs, params) -> float                       # UNCHANGED
def _map_deflector_range(kwargs, params) -> Tuple[Tuple[float, float], int]   # NEW
def _build_angle_points(kwargs, params) -> List[AnglePoint]       # same signature, grid inside
```

**Steps**
- [ ] NEW `_map_deflector_range`:
  ```python
      defl = kwargs.get("deflector_angle")
      if defl is not None:
          d = float(defl)
          return (d, d), 1
      return (float(params.phi_e[0]), float(params.phi_e[1])), int(params.np_)
  ```
  No new GUI kwarg needed: the panel already sends `k_bounds["Y"] = [ky_min, ky_max, ky_steps]`, and `_map_k_bounds` (lines 79-89) already turns that into `params.phi_e` / `params.np_`, including the `min == max -> np_ = 1` rule at line 88-89.
- [ ] `_map_deflector` (lines 117-122) stays exactly as is — back-compat for the 1-D metadata path.
- [ ] Anchor — replace the body of `_build_angle_points` (lines 132-142) with a call to `angle_points_grid`:
  ```python
      from tensorspec.core.dft.sprkkr.pointwise import angle_points_grid
      if not kwargs.get("pointwise"):
          return []
      defl_range, np_grid = _map_deflector_range(kwargs, params)
      return angle_points_grid(
          params.theta_e, params.nt, defl_range, np_grid,
          hv_eV=params.hv_eV, work_function_eV=params.ework_eV,
          slit_rot_deg=float(kwargs.get("slit_angle", 0.0)),
          manip_theta_deg=float(kwargs.get("manip_theta", 0.0)),
          manip_azimuth_deg=float(kwargs.get("manip_azimuth", 0.0)),
          manip_tilt_deg=float(kwargs.get("manip_tilt", 0.0)),
          ref_energy_eV=float(kwargs.get("ref_energy_eV", 0.0)),
          phi_offset_deg=float(kwargs.get("phi_offset_deg", 0.0)),
      )
  ```
  Keep the `not kwargs.get("pointwise") -> []` early return first (line 129-130) so `test_pointwise_off_by_default` stays green.
- [ ] Anchor — **delete this exact guard** (currently lines 205-206), it is the whole point of 1b:
  ```python
          if points and params.np_ != 1:
              raise ValueError("pointwise mode needs nt >= 1 and np_ == 1")
  ```
  Replace with a sanity check that has teeth:
  ```python
          if points and len(points) != params.nt * max(1, params.np_) and kwargs.get("deflector_angle") is None:
              raise ValueError(f"pointwise grid mismatch: {len(points)} points for nt={params.nt} np_={params.np_}")
  ```
  This is a **behaviour change** — tell Sandy in the handback. Any test asserting that old `ValueError` must be updated, not deleted.
- [ ] Read `max_concurrent`: `max_concurrent = kwargs.get("max_concurrent")`, `int(...)` when not None.
- [ ] Anchor — the `run_arpes(...)` call (lines 226-239): add `max_concurrent=max_concurrent,`. Keep `deflector_deg=_map_deflector(kwargs, params) if points else 0.0` — it is only read by the 1-D `stitch_points` path now.
- [ ] Anchor — the returned dict (lines 267-280): add
  ```python
          "k_slit": _coord_or_none(result.dataset, "k_slit_1perA"),
          "k_defl": _coord_or_none(result.dataset, "k_defl_1perA"),
  ```
  with a tiny module helper `_coord_or_none(ds, name)` returning `ds[name].values` when present else `None`. `"theta"` and `"phi"` already come straight from the dataset coords (lines 271-272) and are now the lab slit axis and the deflector axis — no change there.
- [ ] Module docstring (lines 12-17): amend the pointwise paragraph — when `deflector_angle` is None, `k_bounds["Y"]` IS the deflector sweep in degrees, and point-wise then launches `NT*NP` jobs; add the `max_concurrent` key.
- [ ] Tests appended to `TestPointwiseKwargs` in `tests/sprkkr/test_workflow.py`:
  - `test_deflector_range_from_phi_e_when_angle_is_none` — `_map_deflector_range({}, ArpesParams(phi_e=(-15,15), np_=11)) == ((-15.0, 15.0), 11)`
  - `test_deflector_range_scalar_when_angle_given` — `-> ((-10.3,-10.3), 1)`
  - `test_build_angle_points_grid_length` — `pointwise=True`, `nt=4`, `phi_e=(-6,6)`, `np_=3` -> 12 points
  - `test_wrapper_np_gt_1_returns_grid_axes` — full `KKRWrapper().run_simulation` with `FakeLauncher`, `k_bounds={"X":[-10,10,4],"Y":[-6,6,3]}`, `pointwise=True`, no `deflector_angle`: `len(out["theta"]) == 4`, `len(out["phi"]) == 3`, `out["phi"]` close to `[-6,0,6]`, `out["intensity_broadened"].shape == (4,3,ne)`, `len(out["angle_points"]) == 12`, `out["k_slit"].shape == (4,)`, `out["k_defl"].shape == (3,)`
  - `test_wrapper_passes_max_concurrent` — monkeypatch `kkr_wrapper.run_arpes` with a recorder, assert `max_concurrent=7` arrives
  - `test_wrapper_np1_still_single_deflector` — the phase-1a case still gives `len(out["phi"]) == 1`
- [ ] Run `python3 -m pytest tests/sprkkr -q`.

**Worker prompt hint:** "Read plan Task 5. Edit `tensorspec/core/arpes/one_step/kkr_wrapper.py` only: 1 new helper `_map_deflector_range`, 1 new helper `_coord_or_none`, rewrite `_build_angle_points`'s body, delete the np_!=1 guard at lines 205-206 and put the grid-size check in its place, add `max_concurrent` to the `run_arpes` call, add 2 keys to the return dict, amend the docstring. Do NOT touch `_map_k_bounds`, `_map_energy`, `_build_arpes_params`, `_fermi_dirac_weights`, or the Fermi block at lines 252-265. Extend `TestPointwiseKwargs`. Run `python3 -m pytest tests/sprkkr -q`."

---

## Task 6 — `sprkkr_e2e.py`: `--deflector-range`, `--max-concurrent`, NT*NP dry run

Depends on: Task 5.

**Files**
- MODIFY `scripts/sprkkr/sprkkr_e2e.py`
- MODIFY `tests/sprkkr/test_e2e_pointwise_dryrun.py` (append)

**Steps**
- [ ] Anchor — this exact line (currently 153):
  ```python
      ap.add_argument("--deflector", type=float, default=0.0, help="deflector angle, deg")
  ```
  Put it and the new flag in a mutually-exclusive group (argparse errors when BOTH are supplied, defaults do not count as supplied):
  ```python
      g_defl = ap.add_mutually_exclusive_group()
      g_defl.add_argument("--deflector", type=float, default=0.0, help="deflector angle, deg")
      g_defl.add_argument("--deflector-range", type=float, nargs=3, default=None,
                          metavar=("D1", "D2", "NP"), help="deflector sweep, deg (Fermi map)")
  ```
- [ ] Next to `--nproc` (currently line 165) add:
  ```python
      ap.add_argument("--max-concurrent", type=int, default=None,
                      help="max kkrspec jobs alive at once in point-wise mode (default: --nproc)")
  ```
  Resolve after `nproc` is computed (line 219): `max_concurrent = args.max_concurrent if args.max_concurrent is not None else nproc`.
- [ ] Anchor — the point-building block (currently 315-330, `pts = angle_points(...)`). Swap to the grid call and keep the scalar as the `np_=1` case:
  ```python
      if args.deflector_range is not None:
          d1, d2, npd = float(args.deflector_range[0]), float(args.deflector_range[1]), int(args.deflector_range[2])
      else:
          d1 = d2 = float(args.deflector); npd = 1
      pts = angle_points_grid(
          arpes_params.theta_e, arpes_params.nt, (d1, d2), npd,
          hv_eV=arpes_params.hv_eV, work_function_eV=arpes_params.ework_eV,
          slit_rot_deg=args.slit, manip_theta_deg=args.manip_theta,
          manip_azimuth_deg=args.azimuth, manip_tilt_deg=args.tilt,
          ref_energy_eV=args.ref_energy, phi_offset_deg=args.phi_offset,
      )
  ```
  Per-point print line (currently 326) gains the deflector, index widened to 4 digits:
  ```python
      print(f"[pt] i={p.index:04d} slit={p.slit_deg:+.2f} defl={p.defl_deg:+.2f} "
            f"Theta={p.theta_e_deg:+.2f} Phi={p.phi_e_deg:+.2f} k_par={p.k_par:.4f} "
            f"k_slit={p.k_slit:+.4f} k_defl={p.k_defl:+.4f}")
  ```
  Summary line (currently 330) gains the wave count:
  ```python
      waves = -(-len(pts) // max(1, max_concurrent))
      print(f"[pt] N={len(pts)} NT={arpes_params.nt} NP={npd} min|k_par|={min_k_par:.4f} "
            f"max|k_par|={max_k_par:.4f} crosses_gamma={crosses_gamma} "
            f"max_concurrent={max_concurrent} waves={waves}")
  ```
- [ ] Anchor — the dry-run loop (currently 333-343) needs NO logic change; it already iterates `pts` and names subdirs `{dataset}_p{p.index:04d}`, which stays correct with `NT*NP` points. Only bump the message to say `{inp_count} .inp (NT={nt} x NP={npd})`.
- [ ] Anchor — the real call (currently 354-359): add `max_concurrent=max_concurrent,`. `deflector_deg=args.deflector` stays (only read on the `NP=1` stitch path).
- [ ] Import `angle_points_grid` in the block at lines 39-54 next to the existing `angle_points` (line 47). **Keep `angle_points` imported** — no silent delete.
- [ ] Anchor — the viewer-npz metadata dict (currently 389-398): add `"deflector_range": [d1, d2, npd]`, `"max_concurrent": max_concurrent`. Leave `"fermi_edge_applied": False` — CLI still does not apply FD (Global Constraints).
- [ ] Tests appended to `tests/sprkkr/test_e2e_pointwise_dryrun.py`, exact names (subprocess + `--dry-run`, no binary, same `pytestmark` skip shape as the file already has):
  - `test_dry_run_deflector_range_writes_nt_times_np_inp` — `--theta -10 10 4 --deflector-range -6 6 3 --pointwise --dry-run` -> rc 0, **12** `.inp` files
  - `test_dry_run_deflector_range_inp_have_distinct_theta_phi` — `read_inp_keywords` on all 12: every `SPEC_EL NT == "1"` and `NP == "1"`, and the 12 `(THETA, PHI)` pairs rounded to 2 dp are 12 distinct pairs
  - `test_dry_run_deflector_range_prints_wave_count` — stdout has `NP=3`, `waves=` and 12 `[pt] i=` lines
  - `test_dry_run_index_is_deflector_major` — `_p0000.._p0003` share one `defl=` value in the `[pt]` lines, `_p0004` has a different one
  - `test_deflector_and_range_together_is_an_error` — both flags -> rc != 0, stderr mentions `not allowed with`
  - `test_dry_run_scalar_deflector_unchanged` — `--deflector -10.3 --theta -15 15 5 --pointwise` still writes exactly 5 `.inp` (phase-1a behaviour preserved)
- [ ] Run `python3 -m pytest tests/sprkkr/test_e2e_pointwise_dryrun.py -q`.

**Worker prompt hint:** "Read plan Task 6. Edit `scripts/sprkkr/sprkkr_e2e.py` at the anchors the plan quotes: line 153 (`--deflector` -> mutually exclusive group), near line 165 (`--max-concurrent`), lines 315-330 (point block -> `angle_points_grid`), line 343 (message), lines 354-359 (`max_concurrent=`), lines 389-398 (metadata), imports at 39-54. Keep every existing flag including `--raw`. Append 6 tests. Run `python3 -m pytest tests/sprkkr/test_e2e_pointwise_dryrun.py -q`."

---

## Task 7 — read-back + export: `spc_results.py` NP>1, `viewer_export.py` real ky

Depends on: Task 2.

**Files**
- MODIFY `tensorspec/core/dft/sprkkr/viewer_export.py`
- MODIFY `tensorspec/core/dft/sprkkr/spc_results.py` (docstring + one return key; logic mostly already works)
- CREATE `tests/sprkkr/test_viewer_export_grid.py`
- MODIFY `tests/sprkkr/test_pointwise_stitch.py` (append `TestSpcResultsGrid`)

**Why `spc_results` mostly works already:** it matches `.spc` to point by the `_p{index:04d}_ARPES_data.spc` suffix (lines 41-46), and `stitch_points` now handles the grid. The GUI Fetch renames files `f"{i}_{basename}"` (`arpes_panel.py` line 2041) — the suffix still matches. Needs a test, a docstring line, and the k axes in the return dict.

**Steps**
- [ ] `spc_results.spc_paths_to_results`: anchor — the return dict (lines 80-85). Add two keys, `None` when the coords are absent:
  ```python
          'k_slit': merged["k_slit_1perA"].values if "k_slit_1perA" in merged.coords else None,
          'k_defl': merged["k_defl_1perA"].values if "k_defl_1perA" in merged.coords else None,
  ```
  Do NOT remove or rename `intensity_broadened` / `energy` / `theta` / `phi` — `arpes_panel.on_simulation_finished` (line 1596) depends on those four.
- [ ] Docstring: say the sidecar may now carry a `NT*NP` grid, matched by `_pNNNN`, and that `deflector_deg` in `meta` is only used on the single-deflector path.
- [ ] `viewer_export.arrays_to_viewer_npz` — anchor, this exact block (currently 67-74):
  ```python
      pts, side_meta = _points_from_json(points_json)
      if pts is not None and len(pts) == theta.size:
          kx = np.array([p["k_slit"] for p in pts], dtype=float)
          ky = np.array([float(pts[0]["k_defl"])], dtype=float)
  ```
  `len(pts) == theta.size` is false for a grid (`NT*NP` vs `NT`) so today a Fermi map falls into the legacy branch and loses ky. Replace the condition with a grid-aware one:
  ```python
      if pts is not None and len(pts) == theta.size * max(1, phi.size):
          defl_vals = sorted({round(float(p.get("defl_deg", 0.0)), 9) for p in pts})
          slit_vals = sorted({round(float(p["slit_deg"]), 9) for p in pts})
          kx = np.array([k_slit for slit_vals order], dtype=float)   # one point per slit, any deflector
          ky = np.array([k_defl for defl_vals order], dtype=float)   # one point per deflector, any slit
  ```
  Concretely: build `{(slit, defl): p}`, then `kx = [by[(s, defl_vals[0])]["k_slit"] for s in slit_vals]`, `ky = [by[(slit_vals[0], d)]["k_defl"] for d in defl_vals]`. `NP=1` collapses to exactly today's result (`ky` size 1, value `pts[0]["k_defl"]`) — keep that identity.
  Metadata: `meta.setdefault("deflector_list", defl_vals)` alongside the existing `deflector_deg` / `k_defl_1perA` keys (do not remove them; for `NP>1` set `deflector_deg` to `None`).
  Note `_points_from_json` sorts by `slit_deg` only (line 37) — that is now insufficient; sort by `(defl_deg, slit_deg)`, or just do not rely on the order and use the dict.
- [ ] The `if cube.shape[1] != ky.size:` fallback (lines 87-90) now only fires for a true legacy `NP>1` non-pointwise grid. Leave it.
- [ ] `stack_viewer_cubes` (lines 129-159) is untouched — it is still the way to merge cubes from Sandy's old bash-loop runs. **No silent delete.**
- [ ] Tests, `tests/sprkkr/test_viewer_export_grid.py`, exact names (write a sidecar JSON by hand with `write_points_json`, no binary):
  - `test_viewer_npz_grid_kx_ky_are_real_momenta` — `NT=4, NP=3` -> `kx.shape == (4,)`, `ky.shape == (3,)`, both match `k_vacuum * sin(angle)`
  - `test_viewer_npz_grid_intensity_shape` — `intensity.shape == (4, 3, ne)`
  - `test_viewer_npz_grid_ky_is_sorted_ascending`
  - `test_viewer_npz_np1_identical_to_phase1a` — one deflector: `ky == [k_defl]`, `kx` unchanged vs the phase-1a expectation
  - `test_viewer_npz_metadata_has_deflector_list`
  - `test_viewer_npz_legacy_no_sidecar_still_works` — `points_json=None` + `k_par_e0` -> old branch, unchanged
- [ ] Tests appended to `tests/sprkkr/test_pointwise_stitch.py`, class `TestSpcResultsGrid`:
  - `test_spc_results_grid_axes` — write 12 fake `.spc` named `<ds>_p0000_ARPES_data.spc`.. plus the sidecar; `spc_paths_to_results(paths, points_json=...)` gives `theta` size 4, `phi` size 3, `intensity_broadened.shape == (4,3,ne)`
  - `test_spc_results_grid_returns_k_axes` — `k_slit` size 4, `k_defl` size 3
  - `test_spc_results_grid_shuffled_paths` — same result with `paths` shuffled (suffix match, not sort position)
  - `test_spc_results_grid_prefixed_filenames` — prefix every file `f"{i}_"` (GUI Fetch shape, `arpes_panel.py` line 2041) and still get the same axes
- [ ] Run `python3 -m pytest tests/sprkkr -q`.

**Worker prompt hint:** "Read plan Task 7. Edit `tensorspec/core/dft/sprkkr/viewer_export.py` (only the block at lines 67-74 plus `_points_from_json`'s sort at line 37) and `tensorspec/core/dft/sprkkr/spc_results.py` (only the return dict at lines 80-85 plus docstring). Do NOT change `stack_viewer_cubes` or `run_dir_to_viewer_npz`'s legacy branch. New test file + a `TestSpcResultsGrid` class. Run `python3 -m pytest tests/sprkkr -q`."

---

## Task 8 — GUI: Φ range allowed, waves, ETA, k-axis save

Depends on: Tasks 5, 7.

**Files**
- MODIFY `tensorspec/gui/components/arpes_panel.py` (5 anchors)
- MODIFY `tensorspec/core/dft/sprkkr/progress.py` (1 new function)
- CREATE nothing. No Qt tests.

**New layer-1 helper** (exact), in `progress.py` below `EtaModel`:

```python
def pointwise_eta_seconds(ne: int, n_jobs: int, max_concurrent: Optional[int],
                          t_energy_s: float = 35.0, effective_cores: int = 40) -> float:
    """Wall seconds for point-wise mode: each job is ONE (Theta,Phi) at all NE
    energies on a single rank. Measured on Einstein 2026-09-14: ~35 s CPU per
    energy point per job, and 100 concurrent jobs took 2.5x their CPU time in
    wall -> the box delivers ~40 cores of throughput. So wall is CPU-bound:
        wall = ne * t_energy_s * n_jobs / min(effective_cores, cap)
    with cap = max_concurrent (or n_jobs when None). Setting cap above
    effective_cores does not speed things up. EtaModel.estimate_seconds does
    NOT fit here -- it models MPI ranks sharing one job (progress.py 101-111)."""
    cap = max(1, int(max_concurrent or n_jobs))
    parallel = max(1, min(cap, int(effective_cores)))
    return float(t_energy_s) * int(ne) * int(n_jobs) / parallel
```

Add it to `sprkkr/__init__.py`'s `from .progress import (...)` block (currently lines 50-56) and to `__all__`.

**Steps**
- [ ] Anchor 1 — **remove the single-deflector restriction.** This exact line (currently 1097):
  ```python
                  'deflector_angle': (ky_min if ky_min == ky_max else None),
  ```
  stays EXACTLY as written — `None` on a range is now the signal that means "sweep it", handled by `_map_deflector_range`. What changes is the comment above it (lines 1093-1096): rewrite it to say a Φ RANGE is now a Fermi map, `None` here means `kkr_wrapper` reads the sweep from `k_bounds["Y"]`, and `NT*NP` jobs get launched. **Zero code change on this line; comment only.** Do not touch the rest of the dict.
- [ ] Anchor 2 — add the throttle to the same dict, right after this exact line (currently 1113):
  ```python
                  'nproc': self.spin_kkr_nproc.value(),
  ```
  insert:
  ```python
                  'max_concurrent': self.remote_cores_spin.value(),
  ```
  `remote_cores_spin` is built at lines 237-240 (`QSpinBox`, range 1-128, value 40, prefix `"Cores: "`). Also set its tooltip there: `"Point-wise mode: max kkrspec jobs alive at once. Einstein ran 100 concurrent single-rank jobs with no slowdown (2026-09-13)."`
- [ ] Anchor 3 — the job-count guard. Replace this exact block (currently 1057-1066):
  ```python
              if self.chk_pointwise.isChecked() and kx_steps_eff > 64:
                  reply = QMessageBox.question(
                      self,
                      "Point-wise mode: many jobs",
                      f"Point-wise mode launches {kx_steps_eff} kkrspec jobs at once on {nproc} ranks. Continue?",
  ```
  with a version that counts the GRID and the waves (`kx_steps_eff` / `ky_steps_eff` / `e_steps_eff` are already computed at lines 1045-1047):
  ```python
              n_jobs = kx_steps_eff * ky_steps_eff
              max_conc = self.remote_cores_spin.value()
              waves = -(-n_jobs // max(1, max_conc))
              pw_eta = sprkkr_pointwise_eta_seconds(e_steps_eff, n_jobs, max_conc)
              if self.chk_pointwise.isChecked() and n_jobs > 64:
                  reply = QMessageBox.question(
                      self,
                      "Point-wise mode: many jobs",
                      f"Point-wise map: NT={kx_steps_eff} x NP={ky_steps_eff} = {n_jobs} kkrspec jobs, "
                      f"{waves} wave(s) of up to {max_conc}.\n"
                      f"Estimated wall time ≈ {sprkkr_format_eta(pw_eta)}.\nContinue?",
                      QMessageBox.Yes | QMessageBox.No,
                      QMessageBox.No,
                  )
                  if reply != QMessageBox.Yes:
                      return
  ```
  Leave the `if eta_seconds > 900:` guard below it (lines 1067-1078) alone, but skip it when point-wise is on (`and not self.chk_pointwise.isChecked()`) — otherwise Sandy answers two dialogs saying the same thing with two different, contradictory numbers.
  Import `pointwise_eta_seconds as sprkkr_pointwise_eta_seconds` in the import block at lines 23-33, next to the existing `sprkkr_format_eta`.
- [ ] Anchor 4 — the ETA label. `_update_kkr_eta` (lines 2473-2492) currently always uses `EtaModel.estimate_seconds`. Add a point-wise branch after `seconds = model.estimate_seconds(...)` (line 2489):
  ```python
              if self.chk_pointwise.isChecked():
                  n_jobs = kx_steps_eff * ky_steps_eff
                  max_conc = self.remote_cores_spin.value()
                  waves = -(-n_jobs // max(1, max_conc))
                  seconds = sprkkr_pointwise_eta_seconds(e_steps_eff, n_jobs, max_conc)
                  self.lbl_kkr_eta.setText(
                      f"ETA: {sprkkr_format_eta(seconds)}  ({n_jobs} jobs, {waves} wave(s) of {max_conc})"
                  )
                  return
  ```
  The local names `e_steps_eff` / `kx_steps_eff` / `ky_steps_eff` already exist at lines 2481-2483. Hook `self.remote_cores_spin.valueChanged.connect(self._update_kkr_eta)` next to the existing connects (lines 642-648).
- [ ] Anchor 5 — **k axes on save.** `on_simulation_finished` sets `self.sim_kx = results['theta']` and `self.sim_ky = results['phi']` (lines 1644-1651) — those are DEGREES, and `self.sim_axes_are_angles = True` at line 1654. **Leave the display alone**: the plot labels already say deg, the 2-D Fermi-map branch at lines 1666-1685 already handles `nx>1 and ny>1`, and switching display units mid-plan risks the axis code at lines 1723-1768. Instead, right after line 1651 add:
  ```python
              self.sim_k_slit = (np.asarray(results['k_slit'], dtype=float)
                                 if results.get('k_slit') is not None else None)
              self.sim_k_defl = (np.asarray(results['k_defl'], dtype=float)
                                 if results.get('k_defl') is not None else None)
  ```
  Then in `save_arpes_to_disk`, anchor — this exact call (currently 2107-2115):
  ```python
                  global_workspace.save_simulated_arpes(
                      name, 
                      self.sim_intensity, 
                      self.sim_kx, 
                      self.sim_ky, 
  ```
  pass the momenta when present:
  ```python
                  kx_save = self.sim_k_slit if getattr(self, "sim_k_slit", None) is not None else self.sim_kx
                  ky_save = self.sim_k_defl if getattr(self, "sim_k_defl", None) is not None else self.sim_ky
                  meta = self.get_simulation_metadata()
                  meta.update({
                      "pointwise": bool(self.chk_pointwise.isChecked()),
                      "deflector_list": (self.sim_ky.tolist() if self.sim_ky is not None else None),
                      "slit_angle": self.slit_angle_spin.value(),
                      "phi_offset_deg": self.spin_phi_offset.value(),
                      "hv_eV": float(self.photon_energy_spin.value()),
                      "fermi_edge_applied": True,
                      "axes": ("intensity(kx,ky,E); kx=k_slit, ky=k_defl [1/A]"
                               if kx_save is not self.sim_kx else
                               "intensity(theta,phi,E); angles in deg"),
                  })
                  global_workspace.save_simulated_arpes(
                      name, self.sim_intensity, kx_save, ky_save,
                      self.sim_E_axis, meta, hv=self.sim_hv,
                  )
  ```
  `get_simulation_metadata` (lines 1859-1887) already supplies `manip_theta` / `manip_azimuth` / `manip_tilt` / `slit_angle` / `work_function` — do not duplicate those, just add the new keys. `fermi_edge_applied: True` is correct for the GUI path (`kkr_wrapper` lines 252-265 apply FD with `temperature_K`).
  Do the same fallback in `push_arpes_to_workspace` (lines 2085-2094): when `sim_k_slit` is present, use it for `axes` and set `labels=["Energy", "k_slit", "k_defl"]`, `units=["eV", "1/Å", "1/Å"]`; otherwise keep today's `["Energy", "Θ (Slit)", "Φ (Deflect)"]` / `["eV","deg","deg"]` exactly.
- [ ] Fetch / local loader need no change: `load_local_spc` already hunts `pointwise_points.json` in 3 dirs plus one subdir glob (lines 1917-1928) and `fetch_sprkkr_results` already sftp-gets it (lines 2045-2052); `_spc_paths_to_results` (line 1892-1895) just forwards to the core function Task 7 extended. Leave all of it.
- [ ] **UI responsiveness note, verified:** `ARPESRunnerThread` (line 61) is a `QThread`; its `run()` (lines 81-113) calls the router synchronously, and `kkr_wrapper` uses `wait = not kwargs.get("async_", False)` -> `wait=True`, so `run_arpes` -> `collect()` blocks **inside the QThread**, not the UI thread. That is fine and is today's behaviour. The throttle adds no busy loop: `collect()` sleeps `poll_s` (workflow.py line 378) between polls, and Task 4 makes each poll 2 ssh calls instead of 2200. Known gap, NOT fixed here: `kkr_wrapper` never passes `progress_cb` to `run_arpes` (lines 226-239), so the GUI shows no live progress during the run — a 54 h run looks frozen except for the spinner. Log it for phase 2; do not fix it in this task.
- [ ] No new test file. Smoke-check only:
  `python3 -c "import ast; ast.parse(open('tensorspec/gui/components/arpes_panel.py').read())"`
  plus `python3 -m pytest tests/sprkkr -q` for the `progress.py` addition.

**Worker prompt hint:** "Read plan Task 8. First add `pointwise_eta_seconds` to `tensorspec/core/dft/sprkkr/progress.py` and export it from `__init__.py`. Then edit `tensorspec/gui/components/arpes_panel.py` at 5 anchors ONLY — jump to lines 23, 237, 642, 1057, 1093, 1113, 1651, 2085, 2107, 2473. Do NOT read the whole 2700-line file. Qt only, no physics. No deletions except the two guard lines the plan names. ast-parse to check syntax, then `python3 -m pytest tests/sprkkr -q`."

---

## Task 9 — Verify

Depends on: Tasks 1-8.

**Steps**

- [ ] Full suite on the mac, repo root:
  ```
  TensorSpec_env/bin/python -m pytest tests/sprkkr -q
  ```
  All green, including the 14 `test_pointwise.py` and 7 `test_pointwise_stitch.py` tests from phase 1a. Then overwrite `tests/sprkkr/RESULTS.md`, tiny (date / suite / pass-fail counts / failed names). No HTML.

- [ ] Dry run, Sandy's real map geometry, no binary:
  ```
  TensorSpec_env/bin/python scripts/sprkkr/sprkkr_e2e.py \
    --pot scratch/sprkkr_gui_run/scf_20260909_205106/scf.pot_new \
    --hkl 0 1 -1 --iq-surf 2 \
    --hv 84 --pol P --theta-ph 45 --ework 4.5 \
    --theta -10 10 10 --erange -1.0 0.1 8 \
    --slit 0 --deflector-range -15 15 11 --manip-theta 0 --tilt 0 --azimuth 0 \
    --max-concurrent 40 --pointwise --dry-run --out /tmp/pw_map_check
  ```
  (NT reduced to 10 / NE to 8 so the dry run is quick; the angle math does not depend on either.)
  Expected, check them:
  - 110 `[pt]` lines, 110 `.inp` files under `/tmp/pw_map_check/arpes/`
  - `[pt] N=110 NT=10 NP=11 ... max_concurrent=40 waves=3`
  - deflector-major: `i=0000..0009` all show `defl=-15.00`; `i=0010` shows `defl=-12.00`
  - `k_defl` runs `-1.1823 → 0 → +1.1823` 1/Å (`4.5679 * sin(15°) = 1.1823`); `k_slit` runs `-0.7932 → +0.7932` (`4.5679 * sin(10°)`)
  - the middle row (`defl=0`) has `min |k_par| = 0` at `slit=0` — the ONLY cut that crosses Gamma, as it should
  - every `.inp` has `SPEC_EL NT=1 NP=1`, bare scalar `THETA=` and `PHI=` (no `{a,b}`)
  - legacy checks still pass: `--deflector -10.3 --theta -15 15 40 --pointwise` -> 40 `.inp`, `min theta_e = 10.30`, `max theta_e = 18.33`, `min |k_par| = 0.8168`, `crosses_gamma=False`; drop `--pointwise` -> exactly ONE `.inp` with `THETA={-15.0,15.0}` `PHI=-10.3` `NT=40`
  - `--deflector -5 --deflector-range -15 15 11` together -> non-zero exit, argparse says `not allowed with`

- [ ] Throttle proof without a binary:
  ```
  TensorSpec_env/bin/python -m pytest tests/sprkkr/test_workflow.py -q -k "Throttle or PollBatching"
  ```
  All green. `test_throttle_never_exceeds_max_concurrent` is the load-bearing one.

- [ ] GUI smoke, local, no run: open the ARPES panel, pick B3. Set `Φ (Deflect)` to `-15..15` steps 11 and `Θ (Slit)` to `-10..10` steps 100, `E -1.0..0.1` steps 200, point-wise ON, `Cores: 40`. Before pressing Run, check the ETA label reads
  `ETA: ~2d 6h  (1100 jobs, 28 wave(s) of 40)` — the label must show waves AND the CPU-bound wall estimate from `pointwise_eta_seconds`; at Cores: 100 it must STILL read ~2d 6h (11 waves, but 40 effective cores)
  and that pressing Run pops ONE dialog saying `NT=100 x NP=11 = 1100 kkrspec jobs, 28 wave(s) of up to 40`. Answer No. Nothing launches.

- [ ] **Wall-time estimate, hand-computed, for Sandy's real run.** Measured on Einstein 2026-09-14 (100x200 cut): per job CPU = 200 x 35 s ≈ 1.95 h, but 100 concurrent jobs took **4 h 57 min wall** — i.e. Einstein delivered ~40 cores' worth of throughput to 100 processes (2.5x oversubscribed). So the budget is CPU-bound:
  - total CPU = `NT x NP x NE x 35 s` = `1100 x 7000 s` = **2140 core-hours**
  - wall ≈ `2140 / ~40 effective cores` ≈ **54 h ≈ 2 days 6 h**, whatever `max_concurrent` is, as long as it is >= 40.
  - `max_concurrent` therefore does NOT buy wall time here; it buys **politeness** (Einstein stays usable for others, no 1100-process pile-up) and **partial results** (deflector-major waves finish whole cuts). Set Cores to 40-100; both give ~54 h. Do not set 1100.
  - Same as today's bash loop (11 x ~5 h ≈ 55 h) — the win of 1b is one Run button + one stitched cube + workspace save, not speed. To be faster: fewer energies (E is the wall-time axis: NE=100 -> ~27 h), or a narrow window around E_F for a pure Fermi surface.
  - Fixed reviewer note (Fable): the planner's first draft claimed "Cores: 100 saves 33 h" from "no slowdown at 100 concurrent" — wrong; CPU per job was uniform but wall was 2.5x CPU. Corrected here.

- [ ] **Sandy runs the one real Einstein job herself.** Do not launch it from an agent.
  - pot `scratch/sprkkr_gui_run/scf_20260909_205106/scf.pot_new`, `hkl` ABAS `0 1 -1`, `IQ_AT_SURF 2`, `hv 84`, pol `P`, `theta_ph 45`, `ework 4.5`
  - slit `0`, manip theta/tilt/azimuth `0`, `phi_offset 0` (still UNCALIBRATED)
  - `Θ -10..10 NT=100`, `Φ -15..15 NP=11`, `E -1.0..0.1 NE=200`, point-wise ON, `Cores: 40` (or up to 100; wall ~54 h either way)
  - **Pass gate:** stitched dataset `theta` = 100 lab slit angles `-10..10`; `phi` = 11 deflector angles `-15..15` step 3; `intensity_broadened.shape == (100, 11, 200)`; the `phi = -15` and `phi = +15` cuts never cross `k_par = 0`; the `phi = 0` cut DOES cross it at `slit = 0`; the constant-energy slice at `E = 0` looks like a 2-D Fermi surface, not 11 copies of one line.
  - **Fail gates:**
    - all 11 cuts identical -> the deflector never reached the points; check `pointwise_points.json` has 1100 entries with 11 distinct `defl_deg`
    - `phi` axis is `[0..10]` integers -> `stitch_points` took the 1-D branch; check `defl_deg` is non-constant in the sidecar
    - more than `Cores:` jobs alive on Einstein at once (`pgrep -c kkrspec9.7`) -> throttle did not engage; check `[throttle]` printed at launch
    - run appears frozen with no progress -> expected, see the known gap in Task 8; check `pgrep -c kkrspec9.7` on Einstein instead of the GUI

- [ ] Still open after this, do not pretend otherwise: the absolute `PHI` zero axis (the Task-8 runbook from phase 1a, `docs/superpowers/runbooks/2026-09-13-sprkkr-phi-axis-calibration.md`, still NOT run); `TYP=3/4` native 2-D maps; `BETA1/BETA2/ROTAXIS`; live progress in the GUI during a point-wise run; per-energy angle re-evaluation (the `0.13 deg/eV` drift). Phase 2, not here.

---
