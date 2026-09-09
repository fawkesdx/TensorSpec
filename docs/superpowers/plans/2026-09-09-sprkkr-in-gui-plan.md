# SPR-KKR in GUI — implementation plan

Spec: `docs/superpowers/specs/2026-09-08-sprkkr-in-gui-design.md`. Read it first.
Rules: sandy_rule.md (3 layers, no silent deletes, snippet edits, xarray DataTree). Caveman prompts to workers.
Gate = Sandy approves before next gate. Planner = Fable. Workers = sonnet, one chunk each, pytest green before handback.

Dev loop: workers code + pytest in sandbox copy `/home/claude/work/TensorSpec_GUI` (python3 + ase2sprkkr 3.4.2 + pymatgen + xarray, SPR-KKR bins at `/home/claude/sprkkr_bench/bin`). Planner commits files to mac repo via device_commit_files. Sandy runs `TensorSpec_env/bin/pip install ase2sprkkr==3.4.2` on mac once (also add to requirements.txt).

## Gate A = P1 + P2 (core I/O, pure python, no Qt, no binary)

### P1 params + inputs
Files NEW:
- `tensorspec/core/dft/sprkkr/__init__.py` — exports ScfParams, ArpesParams, build_scf_inputs, build_arpes_inputs, parse_spc, parse_scf_log.
- `tensorspec/core/dft/sprkkr/params.py` — `@dataclass ScfParams` (nl=3, ne=30, nktab=250, niter=200, mix=0.2, tol=1e-5, vxc="VWN", alg="BROYDEN2", nonmag=False, dataset="scf"); `@dataclass ArpesParams` (spec §5, exact field list). `to_ase2sprkkr_dict()` on each → `{section: {KEY: value}}` using REAL keywords. `ArpesParams.n_points = ne*nt*np_`. `validate()` (nt≥1, theta range, pol_p in P/S/C+/C-, hkl ints).
- `tensorspec/core/dft/sprkkr/inputs.py` —
  `build_scf_inputs(structure: pymatgen.Structure, params: ScfParams, out_dir) -> ScfInputs(inp_path, pot_path, dataset)`. pymatgen → ASE Atoms (`AseAtomsAdaptor`) → `ase2sprkkr.SPRKKR(atoms=...)` → `InputParameters.create_input_parameters('SCF')`, set values, `calc.save_input(directory, input_file, potential_file, input_parameters, atoms)`. No run.
  `build_arpes_inputs(pot_path, params: ArpesParams, out_dir) -> ArpesInputs(inp_path, dataset, expected_spc, expected_log)`. `SPRKKR(potential=pot_path)` (NOT atoms=). `create_input_parameters('ARPES')`. Set every field. `POTFIL` relative path ok. Do NOT emit `CRYS_VECS`.
  `read_inp_keywords(path) -> dict[section][key]=str` tiny parser (tabs/`=`, `{a,b}` arrays) for tests + legacy fallback compare.
  Keep old `sprkkr_generator.py` untouched. `core/dft/__init__.py`: ADD export of new package, keep old lines.
Tests NEW `tests/sprkkr/test_inputs.py` + fixtures `tests/sprkkr/fixtures/{REF_scf.inp, REF_arpes.inp, Cu.pot.head}`:
- Cu fcc (a=3.615) → scf.inp keys == REF keys (section/key/value compare via read_inp_keywords; ignore DATASET/POTFIL names). pot file: first block has `BRAVAIS` 13, `ALAT` ≈ 6.8314, NQ=1, NT=1.
- ArpesParams defaults → arpes.inp keys == REF_arpes.inp keys. Theta range renders `{-20.0,20.0}`, NT=21, EPHOT=21.2, POL_P=P, EWORK_EV=4.5, N_LAYER=50. No `CRYS_VECS` string.
- validate() raises on bad pol, nt=0.
- Mark `@pytest.mark.skipif(no ase2sprkkr)`.

### P2 outputs
Files NEW:
- `tensorspec/core/dft/sprkkr/outputs.py` —
  `parse_spc(path) -> xr.Dataset` own parser (no ase2sprkkr dependency here — 23-line header: KEYWORD, NE, EFERMI, NT, NP, then `#` comment lines, then rows). dims `("energy","theta","phi")` coords energy eV (from col2 unique), theta deg (col1 unique), phi deg (NP=1 → [phi_e0] attr or 0). Data vars `I_tot,I_up,I_dn,pol,k_par,det`. attrs `EF_Ry, NE, NT, NP, source_file`. Robust to NP>1 (row order: check fixture; if NP>1 rows likely theta-fastest per energy then phi blocks — assert count == NE*NT*NP, reshape, document assumption).
  `spc_to_tensor(ds, meta: dict) -> TensorData` (value (E,θ[,φ]) squeeze φ if NP=1; labels ["Energy","Theta"(,"Phi")], units ["eV","deg"], data_type "Simulated ARPES (SPR-KKR)", metadata = meta + attrs). Reuse `DataTreeBuilder.build_from_tensor` for tree — do not re-implement.
  `spc_to_datatree(ds, meta) -> DataTree` = builder + put `I_up,I_dn,pol,k_par` into `/processed` as Dataset (`DataTreeBuilder.dataset_from_tensor` or direct xr) + `/history` log line "SPR-KKR kkrspec ARPES parsed from <file>".
  `parse_scf_log(path) -> ScfStatus(iterations, converged: bool, ef_ry, etot_ry, last_err, history: list[(it, err, ef)])` regex on lines like `  12 ERR 0.948E-05 0.181E-13 EF  0.65979 ...` + `SCF - cycle converged`.
  `stitch_spc(datasets: list[xr.Dataset]) -> xr.Dataset` concat along energy (sorted, dedupe boundary point) — used by fanout later; test now with fixture split in 2.
Tests NEW `tests/sprkkr/test_outputs.py` + fixtures `_Cu_ARPES_ARPES_data.spc` (95 lines: NE=8,NT=9,NP=1), `Cu_SCF.out.tail`:
- parse_spc shapes (8,9,1), EF_Ry==0.65979, energy ascending, theta [-20..20], I_tot ≥ 0, k_par sign follows theta.
- spc_to_tensor → TensorData ndim 2, labels, units; build_from_tensor works; /processed has I_up.
- parse_scf_log → converged True, iterations 12, ef ≈ 0.65979, etot ≈ -3304.998.
- stitch(split fixture energies [0:4],[4:8]) equals full.
After tests: write `tests/sprkkr/RESULTS.md` (tiny: date, suite, pass/fail counts, failed names).

## Gate B = P3 jobs + progress + fanout (core, no Qt)

- `jobs.py`: `@dataclass JobSpec(kind, workdir, inp_name, binary, nproc=1, mpi=False, env: dict, log_name)`. `resolve_binary(kind, nproc, bin_dir, suffix="9.7") -> path` (adds `MPI` when nproc>1 and file exists, else serial). `LocalLauncher.launch(job) -> LocalHandle(pid, popen, log_path)`; `poll()`, `wait()`, `kill()`. Uses `subprocess.Popen([binary, inp], stdout=log, stderr=STDOUT, cwd=workdir)`; mpirun prefix via `cluster_paths.mpi_launch_prefix` style (`mpirun -np N`). `RemoteLauncher(cluster)` — paramiko via `cluster_paths.ssh_connect`; `upload(files)`, `launch(job) -> RemoteHandle(pid or slurm jobid)` nohup or `sbatch` (use `slurm_sbatch_header`, `shell_export_tmp`, `shell_thread_limits`, `qe_env_exports`); `fetch(remote_paths, local_dir)`; `tail(log, n)`.
- `progress.py`: `arpes_fraction_done(spc_path, expected_rows) -> float` (count data rows); `scf_progress(log_path) -> ScfStatus` (reuse parse_scf_log); `EtaModel(t_point_s=4.0)`: `estimate(params, nproc) -> seconds`; `calibrate(params, nproc, wall_s)` update t_point; persist per host in `~/.tensorspec_sprkkr_eta.json`.
- `fanout.py`: `split_energy(params, k) -> list[ArpesParams]` (contiguous energy chunks, NE split ≈ even, 1-pt overlap none, each ≥2 pts); `split_hv(params, hv_list)`; `plan_jobs(params, pot_path, workdir, nproc_total, mode: "mpi"|"chunks"|"auto") -> list[(ArpesParams, subdir)]`.
- `vault.py`: `pot_key(structure, ScfParams) -> sha1`; `Vault(root)`: `register(key, pot_path, meta)`, `lookup(key)`, `list()`; JSON index; remote entries = `{"cluster": name, "remote_path": ...}` (also mirror via `global_workspace.push_remote_run` for GUI list).
Tests `tests/sprkkr/test_jobs.py`: resolve_binary picks MPI when nproc>1; LocalLauncher with fake binary (`bash` script that writes log) works; fanout splits cover energies; ETA math; vault roundtrip in tmp. Smoke `@skipif(no kkrspec9.7)`: Cu SCF (22 s) then ARPES NE=2 NT=3 → parse_spc shape (2,3,1).

## Gate C = P4 wrapper/router + P5 GUI + roadmap

- `kkr_wrapper.py`: fill `run_simulation(crystal_data, experiment_kwargs)`: map kwargs (`photon_energy`, `polarization` text → pol_p, `k_bounds` X/Y → theta/phi range + NT/NP, `e_range` → EMINEV/EMAXEV/NE, `work_function`, `hkl`, `n_layer`, `pot_path` or vault key, `target` local/remote, `nproc`) → ArpesParams → plan_jobs → launch → wait (or return handle when `async_=True`) → parse → stitch → `TensorData` (`{'intensity_broadened': value, 'tensor': TensorData, 'datatree': tree}` to match router consumers). Router unchanged except nothing to do.
- `sprkkr_panels.py`: SCF panel → `build_scf_inputs` + `LocalLauncher/RemoteLauncher`; progress via `scf_progress` in QTimer; on converged → vault register + "Save vault" (keep existing button). Show `QTextEdit` tail. ARPES panel: fields for every ArpesParams knob w/ tooltips ("cost ∝ NE·NT·NP", "N_LAYER thicker = slower"), ETA label updates on any change, Run → KKRWrapper async, progress bar from `arpes_fraction_done`, Load → `global_workspace.push_spectroscopy_data` → viewer. `arpes_panel.py` B3 branch: replace runner-upload block with call into KKRWrapper (snippet edit; leave runner file on disk, add DEPRECATED docstring).
- `roadmap.md`: edits per spec §9 (planner tells Sandy exact lines; Sandy or worker pastes).
- Tests: `tests/sprkkr/test_kkr_wrapper.py` with monkeypatched launcher (fake .spc copy) → returns TensorData; panel smoke with `qapp` fixture (construct only).

## Gate D = P6 Einstein deploy + Cu(001) real
- Sandy: `ssh-copy-id` to Einstein; `~/.tensorspec_clusters.json` → `paths.ssh_key`, drop `password`; `paths.sprkkr_bin`.
- Rebuild on Einstein: upload tgz (sftp via GUI compute panel or scp), make.inc recipe from sandbox (`/mnt/user-data/outputs/sprkkr_fixtures/make.inc`, adjust INCLUDE + BIN), `make scf gen spec scfmpi specmpi`.
- Run Cu SCF + ARPES 71×21 with `-np 32` from GUI. Calibrate ETA. Compare .spc with sandbox (bit-identical expected up to compiler).

## Gate E = P7 VTe2 (Sandy's system)
- Structure from Crystal Suite → SCF (magnetic? NONMAG flag) → ARPES at Sandy's hv/pol → compare with data. Tune N_LAYER/IMV. Out of scope here: physics conclusions.

## Worker prompt template (caveman)
"Read spec §X + plan Gate Y item Z. Work in /home/claude/work/TensorSpec_GUI. Create only listed files. No Qt in core. Run `python3 -m pytest tests/sprkkr -q`. Green or report exact failure. Report ≤30 lines: files, test counts, open questions."
