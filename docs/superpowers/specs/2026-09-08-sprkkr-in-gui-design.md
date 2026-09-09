# SPR-KKR in TensorSpec_GUI — design (roadmap ARPES B3 + DFT Suite SPR-KKR)

Date 2026-09-08. Owner Sandy. Planner Fable. Workers cheap model.
Scope = run the REAL Munich SPR-KKR 9.7 (`kkrscf`, `kkrspec`) from the GUI, end to end:
structure → SCF → converged potential → ARPES (one-step, TR-LEED slab) → `.spc` → DataTree `/simulated` → viewer.
NOT this doc: GPU KKR clone (OSKI, other chat).

## 0. Facts (verified in sandbox build, gfortran + openmpi, 2026-09-08)

- Package builds clean: `make scf gen spec scfmpi specmpi`. Only make.inc edits (BIN, LIB, `$(INCLUDE)` on FC line for `mpif.h`).
- Binaries take input file as **argument** (`kkrscf9.7 scf.inp > scf.out`), not stdin.
- Cu fcc SCF (NL=3, NE=30, NKTAB=250): 12 iter, **22 s**, EF=0.65979 Ry. Output pot = `<DATASET>.pot_new`.
- Cu(001) ARPES hv=21.2, NE=71 × NT=21 = 1491 pts: **5859 s serial** on 2 slow cores ≈ **4 s per (E,θ) point per core**. Default `N_LAYER=50`, `NLAT_G_VEC=57`.
- `kkrspec9.7MPI` -np 4: ~linear speedup, **bit-identical** `.spc`. MPI is free win.
- Output data file = `<DATASET>_ARPES_data.spc`. 23 header lines then rows: `theta(deg) E(eV) I_tot I_up I_dn pol k_par(1/Å) det`. Shape (NE, NT[, NP]). Log = `<DATASET>_ARPES_SPEC.out`.
- `ase2sprkkr` 3.4.2: correct `.pot` skeleton (Bravais, ALAT, mesh) from ASE Atoms; ARPES task def knows real sections (`SPEC_PH/SPEC_EL/SPEC_STR/ENERGY/TASK`); `ARPESOutputFile` parses `.spc` → arrays; `ScfOutputReader` parses EF/converged. **Its run wrapper is flaky** (`NameError stread_reader` on MPI/async). Use it for I/O only, we launch binaries ourselves.
- Real ARPES keywords (from `spec_inputs.f`): `ENERGY{EMINEV,EMAXEV,NE,EWORK_EV,IMV_INI_EV,IMV_FIN_EV}`, `TASK{ARPES,IQ_AT_SURF,MILLER_HKL,STRVER,INPVER}`, `SPEC_PH{THETA,PHI,EPHOT,POL_P=P|S|C+|C-}`, `SPEC_EL{THETA={a,b},PHI={a,b},NT,NP,POL_E,SPOL,PSPIN}`, `SPEC_STR{N_LAYDBL,NLAT_G_VEC,N_LAYER,SURF_BAR}`.
- `MILLER_HKL` default frame = **conventional** cell axes (Fortran `CREATE_3D_SURFACE` rebuilds cubic axes for I/F/C Bravais). Keyword `CRYS_VECS` flips to raw primitive ABAS. ase2sprkkr writes `CRYS_VEC` (no S) → keyword matcher needs exact ` KW ` → ignored → harmless. Our `inputs.py` must NOT emit `CRYS_VECS` unless hkl given in primitive basis. Cu bench was true Cu(001).
- Current repo code writes WRONG keys (`IQ_SURF, MILLER_HKL={h k l}, POLAR, WORKF, TEMP`, `EMIN/EMAX` in Ry) and reads wrong file (`sys_ARPES_SPEC.out`). Spawns one `kkrspec` per angle = throws away native NT×NP loop.

## 1. Goals / non-goals

Goals
1. Correct inputs (via ase2sprkkr) for SCF + ARPES from Crystal Suite structure.
2. Run local (Mac, if binaries) or remote (Einstein) with same job object. Live progress.
3. Parse `.spc` → `xarray.DataTree` `/simulated/arpes` (coords energy_eV, theta_deg, phi_deg; data I_tot, I_up, I_dn, pol, k_par; attrs hv, pol_p, EF_Ry, ework, N_LAYER…). Show in existing N-D viewer.
4. Speed: native NT×NP grid, MPI, energy-chunk + hv-list fan-out, potential vault reuse, ETA estimate before submit.
5. Tests without binaries (fixtures). Smoke test if binary present.

Non-goals: editing Fortran; copying Munich source to public repo; GPU KKR; BSF/DOS GUI (later, same plumbing works).

## 2. Architecture (3 layers, per sandy_rule.md)

```
tensorspec/core/dft/sprkkr/            NEW subpackage. Zero Qt.
  __init__.py
  params.py     dataclasses ScfParams, ArpesParams, StructureSpec (pure data, defaults = ase2sprkkr defaults)
  inputs.py     build_scf_inputs(structure, ScfParams, dir) -> paths ; build_arpes_inputs(pot_path, ArpesParams, dir) -> path
                thin wrapper over ase2sprkkr save_input. Fallback flag use_legacy -> old sprkkr_generator (kept, untouched).
  outputs.py    parse_spc(path) -> xarray.Dataset ; spc_to_datatree(ds, meta) -> DataTree /simulated/arpes
                parse_scf_log(path) -> ScfStatus(iter, err, EF, converged, etot)
  jobs.py       JobSpec(kind scf|arpes, workdir, inp, binary, nproc, mpi) ; LocalLauncher (subprocess, mpirun) ;
                RemoteLauncher (paramiko via core/compute/cluster_paths; nohup or sbatch). Same JobSpec both.
  fanout.py     split ArpesParams by energy chunks and/or hv list -> N JobSpecs ; stitch_spc(list) -> one Dataset
  progress.py   tail SPEC.out / count .spc rows -> fraction done ; ETA model t_point * NE*NT*NP / ncores
  vault.py      converged pot registry: hash(structure, ScfParams) -> local path or remote path (uses workspace.push_remote_run)
tensorspec/core/arpes/one_step/kkr_wrapper.py    FILL stub: exp kwargs -> ArpesParams -> jobs -> DataTree. Router B3 done.
tensorspec/gui/components/sprkkr_panels.py       SCF panel + ARPES panel rewired to core/dft/sprkkr. Local/remote combo.
tensorspec/gui/components/arpes_panel.py         B3 branch: delete per-angle runner call, call KKRWrapper. Progress bar.
tests/sprkkr/                                    fixtures: REF_scf.inp, REF_arpes.inp, small .spc (95 lines), scf log tail.
```

Old files kept: `sprkkr_generator.py` (legacy path), `arpes_map_runner_template.py` (unused after P5; mark deprecated in docstring, no delete).

## 3. Data flow

1. GUI grabs `pymatgen.Structure` from `global_workspace` (Crystal Suite).
2. `inputs.build_scf_inputs` → `scf.inp`, `<sys>.pot`. `jobs` launch `kkrscf9.7[MPI]`. `progress` tails log. On converged → `vault.register(pot_new)`.
3. ARPES: user picks vault + hv, pol, θ/φ range, E range, N_LAYER… → `ArpesParams`. `fanout` makes 1..N JobSpecs. Launch. Tail.
4. Fetch `.spc` (local read / sftp) → `outputs.parse_spc` → `stitch` → DataTree `/simulated/arpes` + `/history` entry (inputs text, binary, host, wall time). Push to workspace as TensorData. Viewer opens.

## 4. Speed levers (attach-to-SPR-KKR, no Fortran edits)

| lever | where | gain |
|---|---|---|
| native NT×NP grid in one run (kill per-angle spawn) | inputs | kills ~NT×NP process startups + slab setup repeats |
| `kkrspec9.7MPI` | jobs | ~linear in cores (verified 1.96× on 2 cores) |
| energy-chunk fan-out: split [EMIN,EMAX] into K chunks → K independent runs → stitch | fanout | K× when MPI absent or across nodes (SLURM array) |
| hv-list fan-out (kz scans) | fanout | one job per hv, parallel |
| vault reuse of converged pot | vault | skip SCF (22 s Cu, minutes-hours real systems) |
| ETA before submit: `t_point × NE·NT·NP / ncores`, t_point calibrated per host from last run | progress | user sizes grid sanely |
| knob exposure: `N_LAYER`, `NLAT_G_VEC`, `IMV_FIN_EV`, `NL` with tooltips "cost ∝" | GUI | user trades accuracy/time |
| optional convergence probe: 1 energy × 3 angles at N_LAYER {20,35,50} → pick smallest within tol | fanout | 2–3× on thick-slab defaults |

## 5. Interfaces (sketch)

```python
@dataclass
class ArpesParams:
    hv_eV: float = 21.2; pol_p: str = "P"           # P S C+ C-
    theta_ph: float = 45.0; phi_ph: float = 0.0      # light incidence
    theta_e: tuple = (-20.0, 20.0); nt: int = 21     # emission grid
    phi_e: tuple = (0.0, 0.0); np_: int = 1
    e_min_eV: float = -6.0; e_max_eV: float = 1.0; ne: int = 71
    ework_eV: float = 4.5; imv_ini_eV: float = 0.05; imv_fin_eV: float = 2.0
    hkl: tuple = (0, 0, 1); iq_at_surf: int = 1
    n_layer: int = 50; nlat_g_vec: int = 57; n_laydbl: tuple = (10, 10); surf_bar: tuple = (0.25, 0.25)
    nl: int = 3; nktab: int = 250; spol: int | None = None; pol_e: str = "PZ"
    dataset: str = "arpes"

def parse_spc(path) -> xr.Dataset   # dims (energy, theta, phi); vars I_tot I_up I_dn pol k_par det; attrs EF_Ry, NE, NT, NP
```

## 6. Testing

- Unit (no binary): inputs text == REF fixtures (key/value compare, not byte); parse_spc on 95-line fixture → shapes (8,9,1), EF; stitch of 2 chunks == full; fanout splits cover range w/o overlap; ETA math.
- Smoke (skip if no `kkrspec9.7` on PATH/config): Cu SCF 22 s + ARPES NE=3 NT=3.
- After pytest: overwrite `GrizzlyME/tests/RESULTS.md`? No — SPRKKR tests live in `tests/sprkkr/`; write `tests/sprkkr/RESULTS.md` tiny MD (same rule).

## 7. Deploy targets

- Einstein: rebuild 9.7 with MPI (`make scfmpi specmpi`), bin dir in `~/.tensorspec_clusters.json` `paths.sprkkr_bin`. Use ssh KEY. Remove plaintext-password scratch scripts from repo root.
- Mac Studio (optional): `brew install gcc open-mpi openblas`; same make.inc recipe. Enables local runs + fast dev loop.
- Sandbox: my test bench only, ephemeral.

## 8. Phases (each = gate, Sandy approves before next)

P1 params+inputs (+tests) → P2 outputs parse → DataTree (+tests) → P3 jobs+progress+fanout (local first) → P4 kkr_wrapper + router B3 → P5 GUI panels rewire + roadmap.md edits → P6 Einstein deploy + Cu(001) real run → P7 VTe2.

## 9. roadmap.md edits (tell Sandy, do not rewrite roadmap)

- ARPES Suite › Option B › B3 line: change text to "SPR-KKR execution wrapper (kkrspec, TR-LEED slab). Local/remote. Native θ×φ grid, MPI, energy/hv fan-out." (drop "oscarpes API").
- DFT Suite › Full DFT Capability: add sub-items `[ ] SPR-KKR SCF runner (local/remote) + converged-potential vault`, `[ ] SPR-KKR live convergence monitor (EF, RMS per iter)`.
- General Rule: SPR-KKR = CPU only (Fortran). GPU/CPU toggle N/A for B3; note it.

## 10. Risks

- ase2sprkkr version drift vs 9.7 keywords → pin version; keyword table in `params.py` is ours; fixtures catch drift.
- Real systems: SCF convergence, `dlm1 not converged in gikamm` crash seen before (bad inputs). Expose `IMV_*`, `N_LAYER` knobs; surface Fortran error lines in GUI.
- Cost: 4 s/pt/core. 200×200 Fermi map at 1 E = 40k pts ≈ 44 core-h. Fan-out + Einstein needed; ETA panel must warn.
