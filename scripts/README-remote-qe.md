# Remote QE on Einstein

Run a prepared Quantum ESPRESSO (optional Wannier90) job on Einstein from a Mac run directory via rsync + SSH. Heavy intermediates stay on Einstein scratch during the run; only a small allowlist is pulled back.

## Prerequisites

- SSH host `einstein` works from this machine (`ssh einstein` / `Host einstein` in `~/.ssh/config`).
- Einstein QE environment available:
  - Prefer `~/tensorspec-solvers.env` (sourced if present), else
  - `$HOME/miniconda3/envs/qe/bin` on `PATH` (`pw.x`, `mpirun`, optional `pw2wannier90.x`, `wannier90.x`).
- Local run directory produced by TensorSpec DFT Suite (Generate / Bundle), containing at least `scf.in`.

## Prepare run dir

1. In TensorSpec DFT Suite, generate inputs for your structure (SCF, optional NSCF / Wannier).
2. Bundle or export so the directory has:
   - `scf.in` (required)
   - `pseudo/` or UPF paths valid relative to the run dir
   - Optional: `nscf.in`, `pw2wan.in`, `wannier90.win`
3. Point `remote_qe.sh` at that directory (not the git repo root).

Local Generate/Bundle builds inputs; this script only syncs and executes remotely.

## Usage

```bash
./scripts/remote_qe.sh /path/to/run_dir
./scripts/remote_qe.sh /path/to/run_dir --np 8
./scripts/remote_qe.sh /path/to/run_dir --host einstein --keep-scratch
./scripts/remote_qe.sh /path/to/run_dir --dry-run
```

| Flag | Meaning |
|------|---------|
| `--np N` | MPI ranks (default 4, max 32) |
| `--host` | SSH host (default `einstein`) |
| `--keep-scratch` | Do not delete remote scratch after success |
| `--dry-run` | Print planned steps; **no network** (no SSH) |

## Scratch policy

| | |
|--|--|
| Prefer | `/data/sandy/qe_scratch` if `/data/sandy` exists and is writable |
| Fallback | `$HOME/qe_scratch` |
| Job path | `$SCRATCH_ROOT/<YYYYMMDD-HHMMSS>-<basename>/` |
| Success | Wipe remote scratch unless `--keep-scratch` |
| Failure | **Keep** remote scratch; still pull allowlist logs/outs |

`--dry-run` assumes `$HOME/qe_scratch` in the printed plan (no live resolve).

## Pull allowlist

Pulled if present on remote:

- `scf.out`, `nscf.out`
- `pw2wan.out`
- `wannier90.wout`
- `wannier90_hr.dat`
- `remote_qe.log`

**Never** pulled by default: `outdir/`, `*.wfc`, `UNK*`, `*.chk`.

## Exit codes

| Code | Case |
|------|------|
| 0 | Success |
| 1 | SSH / connectivity / allowlist rsync pull failed (scratch kept, not wiped) |
| 2 | Local validation (`scf.in` missing, bad path, bad `--np`) |
| 3 | Remote `pw.x` missing |
| 4 | SCF (or chain) failed — scratch kept; outs pulled if present |
| 5 | Remote free disk &lt; ~1 GB before start, **or** scratch-root create/resolve failed |

Any nonzero QE step fails the chain hard (exit 4). Scratch is wiped only after a successful allowlist pull (unless `--keep-scratch`).

## Limits

- Einstein root disk is tight; prefer `/data/sandy` when available. Preflight requires ≥ 1 GB free on the scratch filesystem.
- No GPU path in this runner (CPU `mpirun` only).
- No web “Queue on Einstein” UI yet — CLI only.
- Do not use this to sync the whole TensorSpec repo; `scf.in` is required as a guard.
