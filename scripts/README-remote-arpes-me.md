# Remote ARPES ME on Einstein

Run a prepared ARPES Option A (Simple Scalar matrix-element) or Option B1 (Chinook TB) job on Einstein from a Mac job directory via rsync + SSH. Heavy mesh work stays on Einstein scratch; only `intensity.npz`, `meta.json`, and `remote_arpes_me.log` are pulled back.

## Prerequisites

- SSH host `einstein` works from this machine (`ssh einstein` / `Host einstein` in `~/.ssh/config`).
- Einstein TensorSpec checkout at `~/TensorSpec` (override with `TENSORSPEC_ROOT` on the remote).
- Remote Python env: `~/TensorSpec/TensorSpec_env/bin/python` with repo dependencies installed.
- **Chinook is not required** for `tb_mode: "Simple Scalar (Isotropic)"` (Option A / default in prepared jobs).
- **Chinook is required** for Option B1 and Slater-Koster matrix-element paths.

### Chinook on Einstein

Install into the remote env (once per Einstein checkout):

```bash
ssh einstein 'cd ~/TensorSpec && TensorSpec_env/bin/pip install chinook'
```

Option A with Simple Scalar runs without Chinook. Option B1 and non-scalar ME modes fail at remote validation if Chinook is missing.

## Job dir layout

Required:

- `request.json` — Option A or B1 parameters (`model`: `"A"` or `"B1"`, k/energy grids, mesh, hoppings, etc.)
- Exactly one of:
  - `structure.cif`, or
  - `structure.json`

Prepared by `prepare_arpes_me_job.py` or equivalent export.

## Prepare job dir

```bash
./TensorSpec_env/bin/python scripts/prepare_arpes_me_job.py /path/to/crystal.cif /path/to/job_dir
# optional overrides:
./TensorSpec_env/bin/python scripts/prepare_arpes_me_job.py crystal.cif job_dir --request custom.json
```

This writes `request.json`, copies `structure.cif`, and sets conservative default grids.

## Usage

```bash
./scripts/remote_arpes_me.sh /path/to/job_dir
./scripts/remote_arpes_me.sh /path/to/job_dir --host einstein --keep-scratch
./scripts/remote_arpes_me.sh /path/to/job_dir --dry-run
```

| Flag | Meaning |
|------|---------|
| `--host` | SSH host (default `einstein`) |
| `--keep-scratch` | Do not delete remote scratch after success |
| `--dry-run` | Print planned steps; **no network** (no SSH/rsync) |

## Scratch policy

| | |
|--|--|
| Prefer | `/data/sandy/arpes_me_scratch` if `/data/sandy` exists and is writable |
| Fallback | `$HOME/arpes_me_scratch` |
| Job path | `$SCRATCH_ROOT/<YYYYMMDD-HHMMSS>-<basename>/` |
| Sidecar | Live runs write `$JOB_DIR/.tensorspec_remote_scratch` (`host<TAB>scratch_path`) for cancel/wipe |
| Success | Wipe remote scratch unless `--keep-scratch` |
| Failure | **Keep** remote scratch; still pull allowlist outputs/logs |

`--dry-run` assumes `$HOME/arpes_me_scratch` in the printed plan (no live resolve).

## Pull allowlist

Pulled if present on remote:

- `intensity.npz` — `(E, kx, ky)` cube + axes
- `meta.json` — shape, formula, timestamp, tb_mode
- `remote_arpes_me.log` — remote transcript

## Exit codes

| Code | Case |
|------|------|
| 0 | Success |
| 1 | SSH / connectivity / allowlist rsync pull failed (scratch kept, not wiped) |
| 2 | Local validation (`request.json` or structure missing, bad path) |
| 4 | Remote simulation failed — scratch kept; allowlist pulled if present |
| 5 | Scratch-root create/resolve failed |
| 6 | Remote missing dependency (from `run_arpes_me_a.py`) |

Remote validation/simulation exit codes 2, 4, and 6 are preserved. Scratch is wiped only after a successful allowlist pull (unless `--keep-scratch`).

## Scope limits

- **Option B1** (Chinook TB / Slater-Koster) is supported when Chinook is installed in Einstein `TensorSpec_env`.
- **Web Queue** can submit Option A or B1 to Einstein (SSH) when `remote_arpes_me.sh` is present.
- Do not point this script at the git repo root; `request.json` + structure guard against accidental sync.
