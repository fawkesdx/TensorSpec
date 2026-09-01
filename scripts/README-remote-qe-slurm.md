# Remote QE on Slurm clusters

Run a prepared Quantum ESPRESSO (optional Wannier90) job on a remote Slurm HPC system from a Mac run directory via rsync + SSH + `sbatch`. Same pull/scratch policy as `remote_qe.sh`, but compute goes through the site batch scheduler instead of immediate `mpirun`.

## Prerequisites

- SSH to the cluster login node (host alias in `~/.ssh/config`).
- Slurm account with CPU (or GPU) allocation.
- Quantum ESPRESSO on the cluster: `module load …` and/or `~/tensorspec-solvers.env`.
- Local run directory from TensorSpec DFT Suite with at least `scf.in`.

## One-time local setup (not committed)

1. Copy the example env file:

   ```bash
   cp scripts/tensorspec-remote.env.example scripts/tensorspec-remote.env
   ```

2. Edit `scripts/tensorspec-remote.env` with your site values (host alias, Slurm account, module name, etc.). This file is gitignored.

3. Ensure SSH works:

   ```bash
   ssh <your-host-alias> true
   ```

Sites that require MFA (e.g. NERSC) need a short-lived SSH key from their proxy tool before batch runs. That stays on your machine; nothing auth-related goes in the repo.

## Usage

```bash
chmod +x scripts/remote_qe_slurm.sh

./scripts/remote_qe_slurm.sh /path/to/run_dir
./scripts/remote_qe_slurm.sh /path/to/run_dir --host mycluster --account myproj --np 128
./scripts/remote_qe_slurm.sh /path/to/run_dir --qos debug --time 00:30:00 --dry-run
./scripts/remote_qe_slurm.sh /path/to/run_dir --module espresso/7.5-libxc-7.0.0-cpu
```

| Flag | Env var | Meaning |
|------|---------|---------|
| `--host` | `TENSORSPEC_SLURM_HOST` | SSH host alias (required live) |
| `--account` | `TENSORSPEC_SLURM_ACCOUNT` | Slurm `-A` (required live) |
| `--qos` | `TENSORSPEC_SLURM_QOS` | Slurm `-q` (default `regular`) |
| `--constraint` | `TENSORSPEC_SLURM_CONSTRAINT` | Slurm `-C` (default `cpu`) |
| `--nodes` | `TENSORSPEC_SLURM_NODES` | Slurm `-N` (default `1`) |
| `--np` | `TENSORSPEC_SLURM_NTASKS` | MPI tasks / `--ntasks` (default `128`) |
| `--time` | `TENSORSPEC_SLURM_TIME` | Walltime `-t` (default `02:00:00`) |
| `--module` | `TENSORSPEC_SLURM_MODULE` | Optional `module load` argument |
| `--scratch` | `TENSORSPEC_SLURM_SCRATCH` | Remote scratch root override |
| `--keep-scratch` | — | Do not delete remote job dir after success |
| `--dry-run` | — | Print plan; no network |

Priority: CLI flag > environment variable > `scripts/tensorspec-remote.env`.

## Scratch policy

| | |
|--|--|
| Prefer | `$SCRATCH/qe_runs` when remote `$SCRATCH` is set |
| Override | `--scratch` / `TENSORSPEC_SLURM_SCRATCH` |
| Fallback | `$HOME/qe_scratch` |
| Job path | `<scratch_root>/<YYYYMMDD-HHMMSS>-<basename>/` |
| Sidecars | `.tensorspec_remote_scratch` (host + path); `.tensorspec_slurm_job` (host + path + Slurm job id) |
| Success | Wipe remote dir unless `--keep-scratch` |
| Failure | Keep remote dir; still pull allowlist logs/outs |

## Pull allowlist

- `scf.out`, `nscf.out`, `pw2wan.out`, `wannier90.wout`, `wannier90_hr.dat`
- `remote_qe.log`, `slurm-<jobid>.out`

Never pulled by default: `outdir/`, `*.wfc`, `UNK*`, `*.chk`.

## Exit codes

| Code | Case |
|------|------|
| 0 | Success |
| 1 | SSH / rsync pull failed |
| 2 | Local validation or missing host/account |
| 3 | Remote `pw.x` missing |
| 4 | Slurm job or QE chain failed |
| 5 | Scratch resolve / disk preflight failed |
| 6 | `sbatch` submit failed |

## Adding another cluster

No code change needed: new `tensorspec-remote.env` values (or CLI flags) for host, account, module, constraint. Only repeat site-specific SSH/MFA setup on your Mac.

## Related

- Interactive SSH runner (no Slurm): `scripts/remote_qe.sh`, `scripts/README-remote-qe.md`
- Web Queue backend (Phase 2): `nersc_slurm` / generic Slurm backend in DFT Suite
