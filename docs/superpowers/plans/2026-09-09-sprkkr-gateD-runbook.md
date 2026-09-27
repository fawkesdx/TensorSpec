# Gate D runbook — SPR-KKR on the remote host + Cu(001) real run (Sandy runs, mac terminal)

All commands from repo root on the mac. Paste outputs back to planner.

## 0. ssh key (once)
```
ssh-copy-id <user>@<host>          # type pw one last time
ssh -o BatchMode=yes <user>@<host> hostname   # must print host, no pw prompt
```
Then edit `~/.tensorspec_clusters.json`: delete the `"password"` line of the remote-host entry. Add/ensure
```
"paths": {"sprkkr_bin": "<heavy_root>/SPRKKR/bin",
          "qe_env": "<remote_conda>/envs/qe"}
```
(`ssh_key` optional — default `~/.ssh/id_*` picked up automatically.) Rotate the old password.

## 1. check remote
```
bash scripts/sprkkr/remote_check.sh <user>@<host>
```
Shows cores, disk, existing bins, mpif90/mpirun, ldd. If `kkrspec9.7MPI` missing → step 2.

## 2. build (serial + MPI), ~5–10 min
```
bash scripts/sprkkr/remote_build.sh <user>@<host>
```
Uploads tgz only if remote lacks `makefile`. Writes `make.inc` (BIN, LIB, INCLUDE for mpif.h). `make scf gen spec scfmpi specmpi`. Ends with `ls bin/` + ldd check. If it fails, paste the last 30 lines.

## 3. mac python deps (once)
```
TensorSpec_env/bin/pip install ase2sprkkr==3.4.2
TensorSpec_env/bin/python -m pytest tests/sprkkr -q      # expect 99 pass, 2 skip
```

## 4. Cu(001) end to end (CLI, no GUI) — small first
```
TensorSpec_env/bin/python scripts/sprkkr/cu001_e2e.py --target <cluster name in json> --nproc 8 --small
```
Expect: SCF 12 iter, EF≈0.6598 Ry, ARPES dims energy 2 × theta 3, files in `scratch/sprkkr_e2e/<ts>/`. ~1–2 min.
Then full (71 E × 21 θ, ≈ 1491 pts × 4 s / nproc → 32 ranks ≈ 3–4 min):
```
TensorSpec_env/bin/python scripts/sprkkr/cu001_e2e.py --target <name> --nproc 32 --mode mpi --full --pot scratch/sprkkr_e2e/<ts>/scf/scf.pot_new
```
Prints calibrated `t_point_s` for host (feeds GUI ETA label).

## 5. GUI check
DFT Suite → engine "SPRKKR" → target remote host → Run SCF → watch iter/EF → converged → vault entry.
ARPES Suite → engine B3 → target remote host → vault → set θ range/NE → ETA label sane → Run → result in viewer. Fetch button re-pulls `.spc` if GUI restarted.

## Done when
Cu(001) hv=21.2 eV θ-scan shows sp-band dispersion + d-band block (−2…−5 eV). Paste `cu001_arpes.npz` path; planner plots + compares with sandbox result (bit-identical expected up to compiler).
