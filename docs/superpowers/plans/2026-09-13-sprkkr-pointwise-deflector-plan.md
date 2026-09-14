# SPR-KKR point-wise deflector cut — implementation plan

> **For agentic workers:** one task each, haiku-class. Read ONLY the files + line anchors your task names. pytest green before handback. Gate = Sandy says go. Caveman talk.

Spec: `docs/superpowers/specs/2026-09-11-sprkkr-full-geometry-design.md` §§5-7. Read §6 + §7 first.
Rules: `sandy_rule.md` (3 layers, no silent deletes, snippet edits only, xarray). `AGENT_HANDOFF.md` gates.
Planner: opus (Sandy's ask 2026-09-13). Reviewer/gate: Fable. Workers: haiku.
Status: v2 2026-09-13 — Sandy said GO. v2 change: sample azimuth IS exposed (Sandy: "I really want to be able to define the sample azimuth"). Workers running.

---

## Goal

Today B3 sends the deflector angle to `SPEC_EL PHI`. That is WRONG. Real run 2026-09-13 (`scratch/sprkkr_gui_run/arpes_20260912_222715`) proved it:

- GUI ky spinbox `-10.3..-10.3` -> `k_bounds["Y"]` -> `kkr_wrapper._map_k_bounds` -> `ArpesParams.phi_e=(-10.3,-10.3)` -> `.inp` `SPEC_EL PHI=-10.3`, `THETA={-15,15}`, `NT=40 NP=1`.
- SPR-KKR `PHI` = azimuth of the polar `THETA` sweep about the surface normal. Fixed `PHI` = a line **through Gamma**, rotated 10.3 deg in-plane.
- `.spc` col 7 confirms: `k_par = k sin(THETA)`, antisymmetric through 0. Cut passes Gamma. It must not.
- A deflector cut is a line **parallel to the slit, offset from Gamma**. In Chinook's own lab k convention (`chinook_arpes_kmesh.build_k_bulk_mesh`, lines 274-286): `K_SLIT = k sin(theta_i)`, `K_DEFL = k sin(phi_d)`, `k_lab_x = K_SLIT cos(slit) - K_DEFL sin(slit)`, `k_lab_z = K_SLIT sin(slit) + K_DEFL cos(slit)`.
- At hv 84 / WF 4.5 / E=E_F: `k = 0.512316*sqrt(79.5) = 4.5679 1/A`, offset `= k sin(10.3) = 0.8168 1/A`. Fixed-PHI and deflector agree ONLY at deflector=0 — that is exactly why the PHI=0 baseline validated and nothing else did.

**Fix (Sandy's pick, approach (a)): N single-point jobs.** Each lab slit angle `i` becomes its own SPR-KKR job with `THETA=Theta_i`, `PHI=Phi_i`, `NT=1 NP=1`. Stitch the N `.spc` back into ONE dataset whose `theta` axis is the **lab slit angle** and whose `k_par` is the **signed lab slit k**, not SPR-KKR's `k sin(Theta)`.

Nothing here touches OSKI. Phase 1a only.

---

## Architecture

Layer 1 core (zero Qt), new + changed:

```
tensorspec/core/dft/sprkkr/
  pointwise.py     NEW  lab (slit,deflector,manip) -> SPR-KKR (Theta_i, Phi_i). numpy only.
  outputs.py       MOD  + stitch_points(), write_points_json(), read_points_json()
  workflow.py      MOD  run_arpes(..., angle_points=[...]) -> N sub-jobs, new stitch branch
  __init__.py      MOD  export new names
tensorspec/core/arpes/one_step/
  kkr_wrapper.py   MOD  new experiment_kwargs -> AnglePoint list -> run_arpes
```

Layer 3 gui:

```
tensorspec/gui/components/arpes_panel.py   MOD  un-grey knobs, new kwargs, .spc loader fix
scripts/sprkkr/sprkkr_e2e.py               MOD  --pointwise + geometry flags, dry-run prints N
```

Math chain (one place, `pointwise.py`):

1. `k = 0.512316 * sqrt(hv - WF + E_ref)`  (same constant as `build_k_bulk_mesh` line 268).
2. lab vacuum k per slit point, from Chinook's own formula above; `k_lab_y = sqrt(max(0, k^2 - x^2 - z^2))`, lab `+y` = outward surface normal (`photon_momentum.py` docstring).
3. `K_sample = sample_to_bulk_frame(K_lab, manip_theta, manip_azimuth, manip_tilt)` — reuse the REAL function in `chinook_arpes_kmesh.py` line 185, `R_total = R_z(theta) R_y(azimuth) R_x(tilt) R_base`. Do not re-derive. Sample `z` = surface normal. **Worker: if `K_sample[2] < 0` for the trivial (all-zero) case, the normal sign convention is flipped — read `build_k_bulk_mesh` to see how it treats the y component, fix the convention explicitly, document it. Never silently `abs()`.**
4. `Theta_i = degrees(atan2(|k_par_i|, k_z_i))`, `Phi_i = degrees(atan2(k_y_i, k_x_i)) + phi_offset_deg`, wrapped to `(-180, 180]`.
5. Sanity, hand-checked, goes in the tests: deflector=0, slit=0, manip=0 -> `Phi_i in {0, 180}`, `Theta_i = |theta_i|`. Exactly today's working baseline. Good.

Fan-out: `run_arpes` already fans out today (`plan_jobs` -> `plan.jobs` -> `_SubJob` list -> `parse_spc(..., phi_range=...)` -> `stitch_spc`). Angle-point mode is a NEW branch in the SAME loop. Do not rewrite. Do not touch `fanout.py`.

Angle points are **orthogonal to energy chunks and not combinable**: `NT x NE` jobs is too many. Angle-point mode = one job per point, ALL energies in that job. `mode=` is ignored when `angle_points` is set.

---

## Global Constraints

- **Energy: one reference energy.** `k` depends on `E_kin = hv - WF + E` so `(Theta_i, Phi_i)` drift with E. We evaluate ONCE at `ref_energy_eV` (default `0.0` = E_F) and hold the angles fixed for the whole window. Residual error: `dTheta/dE = tan(Theta) / (2 E_kin)`. At `Theta=18 deg`, `E_kin=79.5 eV` that is `0.0023 rad/eV = 0.13 deg/eV`; over Sandy's `-1.0..0.1 eV` window `<= 0.15 deg`, and `|dk_par|/k_par = dE/(2 E_kin) = 0.7 %/eV`. Acceptable for phase 1a. `ref_energy_eV` is a parameter, so a future worker can set it to the window midpoint. **Write this number in the code docstring.** Option (2) (per-energy jobs, `NT x NE`) is rejected: too many jobs.
- **Known unknown, do NOT hide it.** SPR-KKR `PHI` is measured from an in-plane axis `x` that ase2sprkkr picks when it slices by `MILLER_HKL`. The manual does not pin it (§7: "phi_e's reference axis (x) is still not pinned down"). Empirically slit || x held for the validated `PHI=0` VTe2 (0,1,-1)/IQ 2 run. **Phase 1a ASSUMES: slit direction at azimuth 0 == SPR-KKR x.** That assumption lives in exactly ONE named parameter, `phi_offset_deg`, default `0.0`. Every docstring and the GUI label say UNCALIBRATED. Task 8 writes the calibration runbook; nobody runs it in this plan.
- **manip azimuth IS exposed (v2, Sandy's requirement).** Sample azimuth = a real rotation Sandy did on the manipulator; it goes through `sample_to_bulk_frame` like theta/tilt and rotates every `Phi_i`. `phi_offset_deg` is a DIFFERENT number: the fixed, unknown angle between the slit at azimuth 0 and SPR-KKR's x. They add in `Phi_i`, so they cannot be FITTED at the same time — calibration (Task 8) is done at azimuth 0, then azimuth is free. GUI: azimuth un-greyed with tooltip "relative to the calibrated PHI zero offset". All three manipulator knobs un-greyed.
- **Do NOT plan on `BETA1`/`BETA2`/`ROTAXIS` or `TYP=3/4`.** Unverified (§7). The `--raw` escape hatch in `sprkkr_e2e.py` + `build_arpes_inputs(extra_raw=)` already exists for probing them. Leave it alone, do not delete it.
- **Job count.** Angle-point mode launches N processes, each `nproc=1`. `NT=40` -> 40 concurrent serial `kkrspec9.7`. Point count total is unchanged vs one big job, but per-job fixed setup cost is paid N times. GUI must warn above 64 points.
- Keep the confirm-pot dialog and all `--raw` plumbing. No silent deletes.
- After any pytest run, overwrite `tests/sprkkr/RESULTS.md`, tiny: date / suite / pass-fail counts / failed names. No HTML.

---

## Task 1 — `pointwise.py`: lab angles -> SPR-KKR (Theta, Phi)

Depends on: nothing.

**Files**
- CREATE `tensorspec/core/dft/sprkkr/pointwise.py`
- CREATE `tests/sprkkr/test_pointwise.py`

**Interfaces** (exact)

```python
K_PER_SQRT_EV: float = 0.512316   # same constant as chinook_arpes_kmesh.build_k_bulk_mesh

@dataclass(frozen=True)
class AnglePoint:
    index: int
    slit_deg: float        # lab slit angle (the swept detector axis)
    theta_e_deg: float     # SPR-KKR SPEC_EL THETA, >= 0
    phi_e_deg: float       # SPR-KKR SPEC_EL PHI, wrapped to (-180, 180]
    k_par: float           # |k_par| at ref energy, 1/A, >= 0
    k_slit: float          # signed lab slit-axis k, 1/A
    k_defl: float          # signed lab deflector-axis k, 1/A

def k_vacuum(hv_eV: float, work_function_eV: float, energy_eV: float = 0.0) -> float
def wrap_deg(angle_deg: float) -> float
def lab_k_vectors(theta_slit_deg, deflector_deg: float, slit_rot_deg: float, k: float) -> np.ndarray  # (3, N)
def angle_points(
    theta_range_deg: Tuple[float, float],
    nt: int,
    *,
    hv_eV: float,
    work_function_eV: float,
    deflector_deg: float = 0.0,
    slit_rot_deg: float = 0.0,
    manip_theta_deg: float = 0.0,
    manip_azimuth_deg: float = 0.0,
    manip_tilt_deg: float = 0.0,
    ref_energy_eV: float = 0.0,
    phi_offset_deg: float = 0.0,
) -> List[AnglePoint]
def sprkkr_angles_to_lab_k(
    theta_e_deg, phi_e_deg, *, k: float, slit_rot_deg: float = 0.0,
    manip_theta_deg: float = 0.0, manip_azimuth_deg: float = 0.0,
    manip_tilt_deg: float = 0.0, phi_offset_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]   # (k_slit, k_defl), for round-trip checks
```

**Steps**
- [ ] `k_vacuum`: `E_kin = max(hv_eV - work_function_eV + energy_eV, 0.1)`; return `K_PER_SQRT_EV * sqrt(E_kin)`. Sign note in docstring: `energy_eV` is relative to E_F, negative below, same sign rule as `ArpesParams.e_min_eV`.
- [ ] `lab_k_vectors`: `theta_slit_deg` is an array. `K_SLIT = k*sin(theta)`, `K_DEFL = k*sin(deflector)`, then the two lines copied verbatim in meaning from `chinook_arpes_kmesh.py` lines 280-281, then `k_lab_y = sqrt(clip(k**2 - x**2 - z**2, 0, None))`. Return `np.vstack([x, y, z])`.
- [ ] `angle_points`: `theta = np.linspace(theta_range_deg[0], theta_range_deg[1], nt)` (if `nt == 1`, use the midpoint). `k = k_vacuum(hv_eV, work_function_eV, ref_energy_eV)`. `K_lab = lab_k_vectors(...)`. Rotate: `from tensorspec.core.arpes.one_step.chinook_arpes_kmesh import sample_to_bulk_frame` **inside the function body** (lazy import, keeps module load light and avoids a package cycle); `K_s = sample_to_bulk_frame(K_lab, manip_theta_deg, manip_azimuth_deg, manip_tilt_deg)` — it is `R_inv @ vec` so a `(3, N)` matrix works unchanged.
- [ ] Per column: `kpar = hypot(K_s[0], K_s[1])`; `theta_e = degrees(arctan2(kpar, K_s[2]))`; `phi_e = wrap_deg(degrees(arctan2(K_s[1], K_s[0])) + phi_offset_deg)`; `k_slit = k*sin(radians(theta_i))`; `k_defl = k*sin(radians(deflector_deg))`.
- [ ] Raise `ValueError` if any `theta_e_deg > 90.0` (emission into the crystal — bad geometry, do not let it reach kkrspec).
- [ ] Module docstring: state the phase-1a assumption (slit at azimuth 0 == SPR-KKR x, `phi_offset_deg` is the single calibration knob, UNCALIBRATED), and the reference-energy error `0.13 deg/eV` / `0.7 %/eV` from Global Constraints.
- [ ] Tests, exact names in `tests/sprkkr/test_pointwise.py`:
  - `test_k_vacuum_hv84` — `k_vacuum(84.0, 4.5) == approx(4.5679, abs=1e-3)`
  - `test_deflector_zero_gives_phi_0_or_180`
  - `test_deflector_zero_theta_equals_abs_slit_angle`
  - `test_slit_zero_deflector_matches_manip_tilt_axis` — deflector `d` at `slit_rot=0` and `manip_tilt=-d` at deflector 0 put `k_par` on the SAME sample axis (`k_x == 0`, `k_y != 0`), same magnitude (sign of `-d` vs `+d`: whichever matches, assert it and comment why)
  - `test_slit_ninety_deflector_matches_manip_theta_axis` — at `slit_rot=90`, the partner is `manip_theta` (`k_y == 0`, `k_x != 0`)
  - `test_offset_magnitude_vte2_case` — hv 84, WF 4.5, E 0, deflector `-10.3`, slit 0, `theta_range=(-15,15)`, `nt=40`: `min(p.k_par) == approx(0.8168, abs=5e-3)`, `min theta_e == approx(10.30, abs=0.02)`, `max theta_e == approx(18.33, abs=0.02)`
  - `test_no_point_crosses_gamma_when_deflected` — every `p.k_par > 0.8` for that same case
  - `test_round_trip_lab_to_sprkkr_to_lab` — feed each `(theta_e_deg, phi_e_deg)` back through `sprkkr_angles_to_lab_k` with the same manip/slit, get `k_slit`/`k_defl` back to `1e-9`
  - `test_phi_offset_shifts_all_phi` — `phi_offset_deg=30` shifts every `phi_e_deg` by exactly 30 (mod wrap)
  - `test_azimuth_rotates_phi_not_theta` — `manip_azimuth_deg=20`, deflector 0: every `theta_e_deg` unchanged, every `phi_e_deg` shifted by a constant `±20` (assert the sign `sample_to_bulk_frame` gives and comment it)
  - `test_theta_above_90_raises`

**Worker prompt hint:** "Read plan Task 1. New file `tensorspec/core/dft/sprkkr/pointwise.py`, pure numpy, no Qt. Copy k formula from `chinook_arpes_kmesh.py` lines 267-286 (read ONLY those lines). Reuse `sample_to_bulk_frame` (line 185), lazy import. Write tests file too. Run `python3 -m pytest tests/sprkkr/test_pointwise.py -q`. Map: `graphify-out/GRAPH_MAP.md`. Report <= 30 lines."

---

## Task 2 — `outputs.py`: stitch N single-point `.spc` into one lab-axis dataset

Depends on: Task 1.

**Files**
- MODIFY `tensorspec/core/dft/sprkkr/outputs.py` (append below `stitch_spc`, do not touch `parse_spc` / `stitch_spc` / `spc_to_tensor` / `spc_to_datatree`)
- CREATE `tests/sprkkr/test_pointwise_stitch.py`

**Interfaces** (exact)

```python
def stitch_points(
    datasets: List[xr.Dataset],
    points: List["AnglePoint"],
    deflector_deg: float = 0.0,
) -> xr.Dataset

def write_points_json(path: str, points: List["AnglePoint"], meta: Optional[Dict[str, Any]] = None) -> None
def read_points_json(path: str) -> Tuple[List["AnglePoint"], Dict[str, Any]]
```

**Steps**
- [ ] `from .pointwise import AnglePoint` at module top (one-way; `pointwise.py` must never import `outputs.py`).
- [ ] `stitch_points`: require `len(datasets) == len(points)`, else `ValueError`. Each input has `sizes["theta"] == 1` and `sizes["phi"] == 1` — assert it.
- [ ] For each `(ds, p)`: `ds = ds.assign_coords(theta=[p.slit_deg], phi=[float(deflector_deg)])`. **Lab slit angle replaces SPR-KKR Theta on the theta axis.**
- [ ] Before concat, keep SPR-KKR's own column: `ds = ds.rename({"k_par": "k_par_sprkkr"})`.
- [ ] `merged = xr.concat(list_of_ds, dim="theta").sortby("theta")`.
- [ ] Recompute the lab momenta. SPR-KKR writes `k_par_sprkkr(E, i) = k(E) sin(Theta_i)`; the angles are fixed per point, so the lab split scales the same way and the ratio is energy-independent and EXACT: build `ratio_slit[i] = p.k_slit / p.k_par` and `ratio_defl[i] = p.k_defl / p.k_par` (`0.0` when `p.k_par == 0`), broadcast along `theta`, then `merged["k_par"] = merged["k_par_sprkkr"] * ratio_slit` (signed lab slit k — this is what the GUI plots) and `merged["k_defl"] = merged["k_par_sprkkr"] * ratio_defl`.
- [ ] attrs: carry `datasets[0].attrs`, then set `NT = len(points)`, `NP = 1`, `pointwise = True`, `deflector_deg`, `phi_is_index = False`. Do not clobber `EF_Ry` / `NE`.
- [ ] `write_points_json`: `{"points": [asdict(p) for p in points], "meta": meta or {}}`, `json.dump(..., indent=2)`.
- [ ] `read_points_json`: inverse, returns `([AnglePoint(**d) for d in ...], meta)`.
- [ ] Tests, exact names:
  - `test_stitch_points_theta_axis_is_lab_slit_angle`
  - `test_stitch_points_phi_axis_is_deflector`
  - `test_stitch_points_k_par_is_signed_lab_slit_k`
  - `test_stitch_points_keeps_sprkkr_k_par`
  - `test_stitch_points_length_mismatch_raises`
  - `test_points_json_round_trip`
  - Build the N fake single-point `.spc` the same way `tests/sprkkr/test_workflow.py::make_fake_spc` does (copy that helper's shape: `NE`/`EFERMI`/`NT`/`NP` header, `#######` line, then rows of 8 floats, energy descending, theta ascending). Set col 7 (`k_par`) to a non-zero value so the ratio test has teeth.

**Worker prompt hint:** "Read plan Task 2. Add 3 functions at the END of `tensorspec/core/dft/sprkkr/outputs.py`. Do not change what is already there. New test file. Fake `.spc` builder: copy the shape of `make_fake_spc` in `tests/sprkkr/test_workflow.py` lines 70-87. Run `python3 -m pytest tests/sprkkr/test_pointwise_stitch.py -q`."

---

## Task 3 — `workflow.run_arpes(angle_points=...)`: N single-point jobs

Depends on: Tasks 1, 2.

**Files**
- MODIFY `tensorspec/core/dft/sprkkr/workflow.py`
- MODIFY `tensorspec/core/dft/sprkkr/__init__.py` (exports only)

**Interfaces** (exact — added params keep every existing caller working)

```python
def run_arpes(
    pot_path, params, workdir, launcher, nproc=1, mode="auto", mpi_available=True,
    wait=True, progress_cb=None, remote_workdir=None, cif_lattice=None, extra_raw=None,
    angle_points: Optional[List["AnglePoint"]] = None,     # NEW
    deflector_deg: float = 0.0,                            # NEW
)

class ArpesRunHandle:
    def __init__(self, subjobs, params, launcher, start_time, progress_cb=None,
                 geometry=None,
                 points: Optional[List["AnglePoint"]] = None,   # NEW
                 deflector_deg: float = 0.0):                   # NEW
```

**Steps**
- [ ] Import: extend the existing line 44 `from .outputs import ScfStatus, parse_spc, spc_to_datatree, spc_to_tensor, stitch_spc` to also import `stitch_points, write_points_json`. Add `from .pointwise import AnglePoint`.
- [ ] Anchor — replace this exact line (currently 489):
  ```python
      plan = plan_jobs(params, nproc, mode=mode, mpi_available=mpi_available)
  ```
  with a branch: when `angle_points` is truthy, build
  ```python
      job_list = [
          (
              replace(params,
                      theta_e=(p.theta_e_deg, p.theta_e_deg), nt=1,
                      phi_e=(p.phi_e_deg, p.phi_e_deg), np_=1,
                      dataset=f"{params.dataset}_p{p.index:04d}"),
              f"{params.dataset}_p{p.index:04d}",
              1,
          )
          for p in angle_points
      ]
  ```
  else `job_list = list(plan_jobs(params, nproc, mode=mode, mpi_available=mpi_available).jobs)`. `replace` is already imported (line 25). `nproc_i = 1` per point on purpose — the N jobs ARE the parallelism.
- [ ] Anchor — replace this exact line (currently 492):
  ```python
      for sub_params, subdir_name, nproc_i in plan.jobs:
  ```
  with `for sub_params, subdir_name, nproc_i in job_list:`. Everything inside the loop is unchanged.
- [ ] Right after `job_list` is built and `workdir.mkdir(parents=True, exist_ok=True)`, when `angle_points`: `write_points_json(workdir / "pointwise_points.json", angle_points, meta={...})` with `deflector_deg`, `hv_eV`, `ework_eV`, `dataset`, `nt`, plus whatever geometry kwargs the caller passed through. This sidecar is what the GUI loader and remote Fetch read back.
- [ ] `print(f"[pointwise] N={len(angle_points)} Theta=[{min:.2f},{max:.2f}] Phi=[{min:.2f},{max:.2f}] min|k_par|={...:.4f}")` — matches the existing `[geom]` print style at line 484.
- [ ] Pass `points=angle_points, deflector_deg=deflector_deg` into the `ArpesRunHandle(...)` construction (currently lines 522-524); store as `self._points`, `self._deflector_deg`.
- [ ] Anchor in `collect()` — replace this exact line (currently 396):
  ```python
          merged = stitch_spc(datasets) if len(datasets) > 1 else datasets[0]
  ```
  with: if `self._points` -> `stitch_points(datasets, self._points, self._deflector_deg)`, else the old expression unchanged. Line 388 `parse_spc(str(spc_path), phi_range=sj.sub_params.phi_e)` stays as is — each sub `phi_e` is already the single correct `Phi_i`.
- [ ] In `collect()`'s `meta` dict (line 400-409) add `"pointwise": bool(self._points)` and `"deflector_deg": self._deflector_deg`.
- [ ] Docstring of `run_arpes`: add a paragraph saying `angle_points` overrides `mode`/`plan_jobs`, is NOT combinable with energy chunking, and one job = one `(Theta_i, Phi_i)` at all energies.
- [ ] `__init__.py`: add `from .pointwise import AnglePoint, angle_points, k_vacuum, sprkkr_angles_to_lab_k` and extend the `outputs` import with `read_points_json, stitch_points, write_points_json`; add all six-plus names to `__all__` in the right comment groups. Add a `- pointwise  lab detector angles -> SPR-KKR (Theta,Phi) single-point jobs` line to the module docstring's Modules list.
- [ ] Tests, new class `TestPointwiseFanout` appended to `tests/sprkkr/test_workflow.py` (reuse `FakeLauncher`, `make_fake_spc`, `cu_pot`):
  - `test_angle_points_launch_one_job_per_point` — 5 points -> `len(launcher.launched) == 5`
  - `test_sub_params_are_single_point` — every launched `.inp` has `NT=1` and `NP=1` via `read_inp_keywords`
  - `test_sub_dataset_names_unique`
  - `test_pointwise_sidecar_written` — `workdir/pointwise_points.json` exists, `read_points_json` gives back 5 points
  - `test_pointwise_result_theta_axis_is_lab_slit_angle` — `result.dataset["theta"].values` == the 5 lab slit angles, `result.dataset["phi"].values == [deflector]`

**Worker prompt hint:** "Read plan Task 3. Edit `tensorspec/core/dft/sprkkr/workflow.py` ONLY at the 4 anchor lines the plan quotes (489, 492, 396, 522) plus signature + imports. Do not rewrite the file. Do not touch `fanout.py`. Then add exports to `__init__.py`. Add a `TestPointwiseFanout` class to `tests/sprkkr/test_workflow.py`. Run `python3 -m pytest tests/sprkkr -q`."

---

## Task 4 — `kkr_wrapper.py`: GUI kwargs -> angle points

Depends on: Task 3.

**Files**
- MODIFY `tensorspec/core/arpes/one_step/kkr_wrapper.py`

**Interfaces** (exact)

```python
_GEOMETRY_DEFAULTS = {
    "slit_angle": 0.0, "deflector_angle": None, "manip_theta": 0.0,
    "manip_azimuth": 0.0, "manip_tilt": 0.0,
    "phi_offset_deg": 0.0, "ref_energy_eV": 0.0,
}

def _map_deflector(kwargs: Dict[str, Any], params: ArpesParams) -> float
def _build_angle_points(kwargs: Dict[str, Any], params: ArpesParams) -> List[AnglePoint]
```

**Steps**
- [ ] New `experiment_kwargs` keys, all optional: `pointwise` (bool, default `False`), `slit_angle`, `deflector_angle`, `manip_theta`, `manip_azimuth`, `manip_tilt`, `phi_offset_deg`, `ref_energy_eV`. None of them are `ArpesParams` fields, so the existing `_ARPES_FIELDS` passthrough at line 117-119 cannot leak them into `ArpesParams`. Leave `_ARPES_FIELDS` / `_EXPLICIT_FIELDS` alone.
- [ ] `_map_deflector`: return `float(kwargs["deflector_angle"])` when present and not None; else fall back to `params.phi_e[0]`. Back-compat on purpose: today the ky spinbox carries the deflector, so a run with no new keys still means the same angle — just now interpreted correctly.
- [ ] `_build_angle_points`: return `[]` when `not kwargs.get("pointwise")`. Otherwise
  ```python
      return angle_points(
          params.theta_e, params.nt,
          hv_eV=params.hv_eV, work_function_eV=params.ework_eV,
          deflector_deg=_map_deflector(kwargs, params),
          slit_rot_deg=float(kwargs.get("slit_angle", 0.0)),
          manip_theta_deg=float(kwargs.get("manip_theta", 0.0)),
          manip_azimuth_deg=float(kwargs.get("manip_azimuth", 0.0)),
          manip_tilt_deg=float(kwargs.get("manip_tilt", 0.0)),
          ref_energy_eV=float(kwargs.get("ref_energy_eV", 0.0)),
          phi_offset_deg=float(kwargs.get("phi_offset_deg", 0.0)),
      )
  ```
- [ ] Bugfix, same file, line 134: `iq_at_surf=_map_iq_at_surf(kwargs.get("iq_at_surf")),` -> `iq_at_surf=_map_iq_at_surf(kwargs.get("iq_at_surf", defaults.iq_at_surf)),`. Absent key = ArpesParams default (1), not auto. Restores the 3 pre-existing red tests `TestKKRWrapper::test_returns_intensity_broadened_shape`, `test_async_returns_handle_then_collect`, `test_kkr_wrapper_fermi::test_fermi_cutoff_kills_above_ef_keeps_below` (broke 2026-09-10; GUI/CLI always pass the key so real runs were unaffected).
- [ ] In `run_simulation`, after `params.validate()` (line 167): `points = _build_angle_points(kwargs, params)`. Raise `ValueError("pointwise mode needs nt >= 1 and np_ == 1")` if `points and params.np_ != 1`.
- [ ] Anchor — the `run_arpes(...)` call at lines 187-198: add `angle_points=points or None,` and `deflector_deg=_map_deflector(kwargs, params) if points else 0.0,`. Change nothing else in that call.
- [ ] Add `"angle_points": points` to the returned dict (line 226-238). `"theta"` already comes from `result.dataset["theta"].values`, which is now the lab slit axis — no change needed there.
- [ ] Module docstring: amend the `k_bounds` unit paragraph — say that in `pointwise` mode `k_bounds["X"]` is the lab SLIT angle sweep and `k_bounds["Y"]` is ignored in favour of `deflector_angle` (falling back to `Y`), and that `SPEC_EL PHI` is then per-point, never the raw deflector.
- [ ] Tests appended to `tests/sprkkr/test_workflow.py`, class `TestPointwiseKwargs`:
  - `test_pointwise_off_by_default` — `_build_angle_points({}, ArpesParams()) == []`
  - `test_deflector_falls_back_to_phi_e`
  - `test_deflector_angle_key_wins`
  - `test_wrapper_pointwise_returns_lab_theta_axis` — full `KKRWrapper().run_simulation` with `FakeLauncher`, `pointwise=True`, `nt=5`: `out["theta"]` == the 5 lab slit angles, `out["phi"] == [deflector]`, `len(out["angle_points"]) == 5`

**Worker prompt hint:** "Read plan Task 4. Edit `tensorspec/core/arpes/one_step/kkr_wrapper.py` only: 2 new module-level helpers, 3 new lines inside `run_simulation`, 1 new return key, docstring. Do not touch `_map_k_bounds` / `_map_energy` / `_build_arpes_params`. Add `TestPointwiseKwargs` to `tests/sprkkr/test_workflow.py`. Run `python3 -m pytest tests/sprkkr -q`."

---

## Task 5 — `sprkkr_e2e.py`: CLI flags + N-file dry run

Depends on: Task 4.

**Files**
- MODIFY `scripts/sprkkr/sprkkr_e2e.py`
- CREATE `tests/sprkkr/test_e2e_pointwise_dryrun.py`

**Steps**
- [ ] New args, next to the existing `--phi` line (currently 149):
  ```
  --slit  (float, default 0.0)        analyzer slit orientation in lab, deg
  --deflector (float, default 0.0)    deflector angle, deg
  --manip-theta (float, default 0.0)
  --tilt (float, default 0.0)
  --azimuth (float, default 0.0)
  --phi-offset (float, default 0.0)   UNCALIBRATED SPR-KKR PHI zero offset
  --ref-energy (float, default 0.0)   eV rel. E_F where (Theta,Phi) are evaluated
  --pointwise (store_true)            one kkrspec job per slit angle
  ```
- [ ] After `arpes_params` is built (currently line 293-303), when `args.pointwise`: build `pts = angle_points(arpes_params.theta_e, arpes_params.nt, hv_eV=..., work_function_eV=..., deflector_deg=args.deflector, slit_rot_deg=args.slit, manip_theta_deg=args.manip_theta, manip_azimuth_deg=args.azimuth, manip_tilt_deg=args.tilt, ref_energy_eV=args.ref_energy, phi_offset_deg=args.phi_offset)` and print one line per point: `[pt] i=NN slit=+xx.xx Theta=+xx.xx Phi=+xxx.xx k_par=x.xxxx k_slit=+x.xxxx`. Then a summary line `[pt] N=.. min|k_par|=.. max|k_par|=.. crosses_gamma=<bool>`.
- [ ] Anchor — the dry-run block (currently lines 305-312, `if args.dry_run:` ... `build_arpes_inputs(...)`): when `args.pointwise`, loop the points and call `build_arpes_inputs(pot_path, replace(arpes_params, theta_e=(p.theta_e_deg,)*2, nt=1, phi_e=(p.phi_e_deg,)*2, np_=1, dataset=f"{arpes_params.dataset}_p{p.index:04d}"), arpes_workdir / f"{arpes_params.dataset}_p{p.index:04d}", extra_raw=extra_raw)` — same subdir naming as `workflow.run_arpes` so dry-run output matches a real run byte for byte. Print `[arpes] dry-run: wrote N .inp under <dir>`. Non-pointwise dry run keeps today's single-file path untouched.
- [ ] Anchor — the real `run_arpes(...)` call (currently lines 315-319): add `angle_points=pts if args.pointwise else None, deflector_deg=args.deflector`.
- [ ] Import `angle_points` from `tensorspec.core.dft.sprkkr` (add to the existing block at lines 39-54) and `replace` from `dataclasses`.
- [ ] Test `tests/sprkkr/test_e2e_pointwise_dryrun.py`, exact names:
  - `test_dry_run_pointwise_writes_n_inp` — `subprocess.run([sys.executable, "scripts/sprkkr/sprkkr_e2e.py", "--pot", str(VTE2_POT), "--hkl", "0", "1", "-1", "--iq-surf", "2", "--hv", "84", "--theta", "-15", "15", "5", "--deflector", "-10.3", "--slit", "0", "--pointwise", "--dry-run", "--out", str(tmp_path)])` -> rc 0, 5 `.inp` files exist
  - `test_dry_run_pointwise_inp_theta_phi_lines` — `read_inp_keywords` on each: `SPEC_EL NT == "1"`, `NP == "1"`, `THETA` is a bare scalar (no `{`), matching `p.theta_e_deg` to 2 dp
  - `test_dry_run_pointwise_prints_every_point` — stdout has 5 `[pt] i=` lines and one `crosses_gamma=False`
  - `test_dry_run_non_pointwise_unchanged` — without `--pointwise`, exactly ONE `.inp`, `NT=5`
  - Skip the whole module when `ase2sprkkr` is missing, same `pytestmark` shape as `tests/sprkkr/test_workflow.py` lines 16-21.

**Worker prompt hint:** "Read plan Task 5. Edit `scripts/sprkkr/sprkkr_e2e.py`: add 8 argparse args, one point-printing block, one dry-run loop, 2 kwargs on the `run_arpes` call. Keep `--raw` and every existing flag. New test file uses subprocess + `--dry-run`, no binary. Run `python3 -m pytest tests/sprkkr/test_e2e_pointwise_dryrun.py -q`."

---

## Task 6 — GUI: un-grey the real knobs, send the new kwargs

Depends on: Task 4. Same file as Task 7 — do Task 6 FIRST, hand back, then Task 7.

**Files**
- MODIFY `tensorspec/gui/components/arpes_panel.py` (4 anchors only)

**Steps**
- [ ] Anchor 1 — `_sync_sprkkr_irrelevant_knobs`, the `chinook_only` list (currently lines 2145-2158). **Delete only these three entries:**
  ```python
              self.manip_theta_spin,
              self.manip_azi_spin,
              self.manip_tilt_spin,
  ```
  Update the docstring: manipulator theta/azimuth/tilt now reach SPR-KKR via `pointwise` angle points (`sample_to_bulk_frame`); azimuth is the physical sample rotation, `phi_offset_deg` is the separate uncalibrated SPR-KKR x-axis constant (design doc §7). Set `self.manip_azi_spin.setToolTip("Sample azimuth (physical manipulator rotation about the surface normal). For SPR-KKR it adds to the PHI zero offset; calibrate that offset at azimuth 0.")`. Slit Angle and Deflector Angle are already visible for B3 today (they were never in this list) — leave them.
- [ ] Anchor 2 — the SPR-KKR group form, right after this block (currently lines 283-285):
  ```python
          self.combo_kkr_mode = QComboBox()
          self.combo_kkr_mode.addItems(["auto", "mpi", "chunks"])
          sprkkr_form.addRow("Fan-out mode:", self.combo_kkr_mode)
  ```
  add three widgets:
  ```python
          self.chk_pointwise = QCheckBox("Point-wise deflector cut (1 job per slit angle)")
          self.chk_pointwise.setChecked(True)
          self.chk_pointwise.setToolTip(
              "Off = legacy: deflector goes straight to SPEC_EL PHI -> a cut THROUGH Gamma "
              "(only correct at deflector 0). On = one kkrspec job per slit angle, each with "
              "its own (THETA, PHI) -> a cut PARALLEL to the slit, offset from Gamma. "
              "Costs NT jobs."
          )
          sprkkr_form.addRow(self.chk_pointwise)

          self.spin_phi_offset = QDoubleSpinBox(); self.spin_phi_offset.setRange(-180.0, 180.0)
          self.spin_phi_offset.setValue(0.0); self.spin_phi_offset.setSuffix(" °")
          self.spin_phi_offset.setToolTip(
              "UNCALIBRATED. Angle between the analyzer slit (at azimuth 0) and SPR-KKR's own "
              "in-plane x axis, which ase2sprkkr picks when it slices by MILLER_HKL. The manual "
              "does not pin it. 0 assumes slit || x (held empirically for the VTe2 PHI=0 run)."
          )
          sprkkr_form.addRow("PHI zero offset (UNCALIBRATED):", self.spin_phi_offset)

          self.spin_ref_energy = QDoubleSpinBox(); self.spin_ref_energy.setRange(-20.0, 5.0)
          self.spin_ref_energy.setValue(0.0); self.spin_ref_energy.setSuffix(" eV")
          self.spin_ref_energy.setToolTip(
              "Energy (rel. E_F) at which (THETA_i, PHI_i) are evaluated; held fixed over the "
              "whole window. Residual drift ~0.13 deg/eV, ~0.7 %/eV in k_par."
          )
          sprkkr_form.addRow("Angle reference energy:", self.spin_ref_energy)
  ```
  `QCheckBox` is already imported (used at line 560).
- [ ] Anchor 3 — the B3 `experiment_kwargs` dict (currently lines 1039-1064). Add, right after the `'theta_ph': self.incidence_angle_spin.value(),` line:
  ```python
                  'pointwise': self.chk_pointwise.isChecked(),
                  'slit_angle': self.slit_angle_spin.value(),
                  'deflector_angle': (ky_min if ky_min == ky_max else None),  # v2: the "Φ (Deflect)" range row IS the deflector knob Sandy uses; analyzer "Deflector Angle" spin is schematic-only
                  'manip_theta': self.manip_theta_spin.value(),
                  'manip_azimuth': self.manip_azi_spin.value(),
                  'manip_tilt': self.manip_tilt_spin.value(),
                  'phi_offset_deg': self.spin_phi_offset.value(),
                  'ref_energy_eV': self.spin_ref_energy.value(),
  ```
  Change nothing else in that dict. `k_bounds` keeps its current `X`/`Y` shape (X = slit sweep, Y = the legacy deflector fallback).
- [ ] Anchor 4 — the long-run guard (currently lines 1026-1037, `if eta_seconds > 900:`). Add above it: when `self.chk_pointwise.isChecked() and kx_steps_eff > 64`, a `QMessageBox.question` "Point-wise mode launches {kx_steps_eff} kkrspec jobs at once on {nproc} ranks. Continue?" defaulting to No, `return` on No.
- [ ] Also hook `self.chk_pointwise.toggled` and `self.deflector_angle_spin.valueChanged` to `self._update_kkr_eta` next to the existing connects at lines 613-616. `_update_kkr_eta` itself is unchanged — point count is the same either way; only the job count changes.
- [ ] No new test file. Smoke-check only: `python3 -c "import ast,sys; ast.parse(open('tensorspec/gui/components/arpes_panel.py').read())"`.

**Worker prompt hint:** "Read plan Task 6. Edit `tensorspec/gui/components/arpes_panel.py` at 4 anchors ONLY (the plan quotes each). Do NOT read the whole 2615-line file — jump to lines 283, 613, 1026, 1039, 2145. No deletions except the two named list entries. Qt only, no physics. ast-parse to check syntax."

---

## Task 7 — GUI: `.spc` loader + remote Fetch carry the lab axes

Depends on: Tasks 2, 6. Same file as Task 6 — start only after Task 6 is handed back.

**Files**
- MODIFY `tensorspec/gui/components/arpes_panel.py` (3 methods)

**Why:** `_spc_paths_to_results` calls `sprkkr_parse_spc(lp)` with **no** `phi_range`, so `phi` comes back as `0.0` (`outputs.parse_spc` line 122). A fetched or locally loaded point-wise run would land on the wrong axes entirely.

**Interfaces** (exact)

```python
    @staticmethod
    def _spc_paths_to_results(local_paths, points_json: Optional[str] = None) -> dict
```

**Steps**
- [ ] Extend the import block at lines 23-33 with `read_points_json as sprkkr_read_points_json`, `stitch_points as sprkkr_stitch_points`, `read_inp_keywords as sprkkr_read_inp_keywords`.
- [ ] Anchor — `_spc_paths_to_results` (currently lines 1838-1851). Replace its body:
  - If `points_json` is given and the file exists: `points, meta = sprkkr_read_points_json(points_json)`; parse each `.spc` with `phi_range=(p.phi_e_deg, p.phi_e_deg)` in point order (match a `.spc` to its point by the `_pNNNN` suffix in its filename, not by sort position); `merged = sprkkr_stitch_points(datasets, points, meta.get("deflector_deg", 0.0))`.
  - Else, per file: look for the sibling `<dataset>.inp` (same dir, stem = filename minus `_ARPES_data.spc`); if found, `sprkkr_read_inp_keywords(...)["SPEC_EL"]["PHI"]` -> float -> pass as `phi_range=(v, v)`. If not found, today's `sprkkr_parse_spc(lp)` call, unchanged. Then `sprkkr_stitch_spc` as today.
  - Keep the return dict shape exactly as it is (`intensity_broadened`, `energy`, `theta`, `phi`) — `on_simulation_finished` depends on it.
- [ ] Anchor — `load_local_spc` (currently line 1853, the `results = self._spc_paths_to_results(sorted(paths))` line at 1873): before it, look for `pointwise_points.json` in the chosen files' directory and one level up; pass it as `points_json=`.
- [ ] Anchor — `fetch_sprkkr_results`, right after the `spc_remote_paths = sorted(_find_spc(search_dir))` block (currently line 1975) and before `sftp.close()` (line 1986): try `sftp.get(f"{search_dir}/pointwise_points.json", os.path.join(local_dir, "pointwise_points.json"))` inside a `try/except IOError: points_local = None`. Pass it into `_spc_paths_to_results` at line 1989.
- [ ] `_find_spc` already recurses into subdirs, so it picks up the N `<dataset>_pNNNN/` dirs with no change. Leave it.
- [ ] No new test file. ast-parse smoke check.

**Worker prompt hint:** "Read plan Task 7. Edit `tensorspec/gui/components/arpes_panel.py`, methods `_spc_paths_to_results` (line 1838), `load_local_spc` (1853), `fetch_sprkkr_results` (1975-1989), plus the import block at line 23. Do NOT read the whole file. Keep the returned dict keys identical. ast-parse to check."

---

## Task 8 — Calibration runbook (WRITE the doc, do NOT run it)

Depends on: Task 5.

**Files**
- CREATE `docs/superpowers/runbooks/2026-09-13-sprkkr-phi-axis-calibration.md`

**Steps**
- [ ] State the open question in one paragraph, citing design doc §7: SPR-KKR `PHI` is measured from an in-plane `x` that ase2sprkkr picks when slicing by `MILLER_HKL`; the Oct-2023 manual does not pin it; phase 1a ASSUMES slit-at-azimuth-0 == `x`, encoded as `phi_offset_deg = 0`.
- [ ] Procedure, two cheap runs on the existing VTe2 pot, deflector 0, `NT=21`, `NE=8`, coarse: run A `PHI=0`, run B `PHI=90`, `THETA={-15,15}`. Give the exact `sprkkr_e2e.py` command lines for both (non-pointwise — this calibrates the raw `PHI` axis, so point-wise must stay OFF).
- [ ] Comparison: overlay each run's `I_tot(E, k_par)` against (a) the Chinook B1 cut along the same crystal direction, and (b) the known `PHI=0 -> Gamma-K` / `PHI=90 -> Gamma-M` fact recorded in `SPRKKR_GUI_THREAD.md`. Whichever `PHI` reproduces Gamma-K fixes the axis.
- [ ] Result -> set `phi_offset_deg` in the GUI ("PHI zero offset") and as the `--phi-offset` default; record the value, the material, and the `hkl_abas`/`IQ_AT_SURF` it was measured at, because the axis is per-surface, not global.
- [ ] Explicit "NOT DONE" banner at the top. This task writes the doc only. Sandy decides when to run it.
- [ ] Add a one-line pointer to this runbook at the end of design doc §7 (a single appended line — no rewrite of §7).

**Worker prompt hint:** "Read plan Task 8. Write ONE new markdown file under `docs/superpowers/runbooks/`. No code, no runs. Quote design doc §7's 'phi_e's reference axis (x) is still not pinned down' paragraph. Append exactly one pointer line to §7."

---

## Task 9 — Verify

Depends on: Tasks 1-8.

**Steps**
- [ ] Full suite on the mac, repo root:
  ```
  TensorSpec_env/bin/python -m pytest tests/sprkkr -q
  ```
  All green. Then overwrite `tests/sprkkr/RESULTS.md`, tiny (date / suite / pass-fail counts / failed names). No HTML.
- [ ] Dry run, the exact real geometry, no binary:
  ```
  TensorSpec_env/bin/python scripts/sprkkr/sprkkr_e2e.py \
    --pot scratch/sprkkr_gui_run/scf_20260909_205106/scf.pot_new \
    --hkl 0 1 -1 --iq-surf 2 \
    --hv 84 --pol P --theta-ph 45 --ework 4.5 \
    --theta -15 15 40 --erange -1.0 0.1 40 \
    --slit 0 --deflector -10.3 --manip-theta 0 --tilt 0 --azimuth 0 \
    --pointwise --dry-run --out /tmp/pw_check
  ```
  Expected, hand-computed, check them:
  - 40 `[pt]` lines, 40 `.inp` files under `/tmp/pw_check/arpes/`
  - `min theta_e = 10.30 deg` (at slit 0), `max theta_e = 18.33 deg` (at slit ±15)
  - `phi_e` runs `34.65 -> 90.00 -> 145.35 deg` (sign may come out mirrored depending on the normal convention — either way it is NOT one fixed value; that spread IS the bug this plan fixes)
  - `min |k_par| = 0.8168 1/A`, `max |k_par| = 1.4369 1/A`, `crosses_gamma=False`, no point with `k_par < 0.8`
  - every `.inp` has `SPEC_EL` `NT=1`, `NP=1`, bare-scalar `THETA=` and `PHI=` (no `{a,b}`)
  - legacy check: drop `--pointwise`, get back exactly ONE `.inp` with `THETA={-15.0,15.0}` `PHI=-10.3` `NT=40` — the old, wrong-but-unchanged behaviour still reachable
- [ ] GUI smoke, local, no run: open ARPES panel, pick B3. Manipulator Theta, Azimuth and Tilt are now VISIBLE; "PHI zero offset (UNCALIBRATED)", "Angle reference energy" and the "Point-wise deflector cut" checkbox are present in the SPR-KKR group.
- [ ] **Sandy runs the one real Einstein job herself.** Do not launch it from an agent.
  - pot `scratch/sprkkr_gui_run/scf_20260909_205106/scf.pot_new`
  - `hkl` ABAS `0 1 -1`, `IQ_AT_SURF 2`, `hv 84`, pol `P`, `theta_ph 45`, `ework 4.5`
  - slit `0`, deflector `-10.3`, manip theta/tilt/azimuth `0`
  - `theta -15..15 NT=40`, `E -1.0..0.1 NE=40`, point-wise ON -> 40 jobs
  - Pass gate: stitched dataset `theta` axis = the 40 LAB slit angles `-15..15`; `phi` axis = `[-10.3]`; `k_par` monotone-ish and **never crosses zero**, `min |k_par| ~ 0.82 1/A` at slit 0; the band pattern is NOT symmetric about slit 0 in the way the old fixed-PHI run was.
  - Fail gate: if `k_par` still passes through 0, the point-wise branch did not engage — check `pointwise_points.json` exists in the run workdir and that the per-point `.inp` files say `NT=1`.
- [ ] Still open after this, do not pretend otherwise: the absolute `PHI` zero axis (Task 8's runbook), `TYP=3/4` native 2-D maps, and `BETA1/BETA2/ROTAXIS`. Phase 2, not here.
