# Remote QE Einstein Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `scripts/remote_qe.sh` + README so Mac run dirs execute QE on Einstein via rsync/SSH with minimal pullback and scratch cleanup.

**Architecture:** Bash CLI on Mac; Einstein conda QE; scratch under `/data/sandy/qe_scratch` or `~/qe_scratch`; allowlist rsync back; wipe scratch on success.

**Tech Stack:** bash, ssh, rsync, OpenMPI/`pw.x` on Einstein.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-08-remote-qe-einstein-design.md`
- Branch: `HTML_einstein_app`
- No web UI; no ARPES remote; no default pull of wfc/UNK/`.chk`
- Default host `einstein`; QE from `~/tensorspec-solvers.env` or `~/miniconda3/envs/qe/bin`
- Do not require sudo

## File map

| File | Role |
|------|------|
| `scripts/remote_qe.sh` | Main CLI |
| `scripts/README-remote-qe.md` | Usage + policies |

---

### Task 1: `remote_qe.sh` + README

**Files:**
- Create: `scripts/remote_qe.sh` (executable)
- Create: `scripts/README-remote-qe.md`

**Interfaces:**
- CLI: `./scripts/remote_qe.sh <local_run_dir> [--np 4] [--host einstein] [--keep-scratch] [--dry-run]`
- Exit codes 1–5 per spec

- [ ] **Step 1: Create `scripts/remote_qe.sh`**

Implement with `set -euo pipefail` where safe; use explicit checks for exit codes.

Core structure:

```bash
#!/usr/bin/env bash
set -euo pipefail

usage() { ... }

NP=4
HOST=einstein
KEEP=0
DRY=0
RUN_DIR=""

# parse args...
# validate RUN_DIR/scf.in
# JOB_ID=$(date +%Y%m%d-%H%M%S)-$(basename "$RUN_DIR")

resolve_scratch_root() {
  ssh "$HOST" 'bash -s' <<'EOS'
if [ -d /data/sandy ] && [ -w /data/sandy ]; then
  mkdir -p /data/sandy/qe_scratch && echo /data/sandy/qe_scratch
elif mkdir -p "$HOME/qe_scratch" 2>/dev/null; then
  echo "$HOME/qe_scratch"
else
  exit 5
fi
EOS
}

# preflight: ssh true; remote pw.x; df -BG free on scratch fs >= 1
# if DRY: print plan and exit 0

# rsync -az --delete "$RUN_DIR/" "$HOST:$SCRATCH/"
# remote run script via ssh bash -s with env:
#   source ~/tensorspec-solvers.env 2>/dev/null || export PATH=$HOME/miniconda3/envs/qe/bin:$PATH
#   mpirun -np $NP pw.x -in scf.in | tee scf.out
#   optional nscf, pw2wannier90, wannier90
# capture remote status

# rsync allowlist files that exist back to RUN_DIR
# on success and !KEEP: ssh rm -rf SCRATCH
# on failure: keep scratch, exit 4
```

Wannier step: if both `pw2wan.in` and `wannier90.win` exist:

```bash
pw2wannier90.x < pw2wan.in > pw2wan.out 2>&1 || true  # prefer fail the chain if nonzero — match QE pipeline norms: fail hard
wannier90.x wannier90 > wannier90.wout 2>&1
```

Prefer **fail hard** on any step nonzero (exit 4), still pull logs.

Allowlist rsync from remote using a file list built on remote:

```bash
ssh "$HOST" "cd '$SCRATCH' && ls scf.out nscf.out pw2wan.out wannier90.wout wannier90_hr.dat remote_qe.log 2>/dev/null"
# rsync each existing file
```

Also append a local `remote_qe.log` copy of the transcript.

- [ ] **Step 2: Create `scripts/README-remote-qe.md`**

Sections: Prerequisites (SSH `einstein`, Einstein QE env), Prepare run dir (DFT Generate/Bundle), Usage examples, Scratch policy, Pull allowlist, Exit codes, Limits (disk, no GPU).

- [ ] **Step 3: `chmod +x scripts/remote_qe.sh`**

- [ ] **Step 4: Dry-run smoke**

```bash
mkdir -p /tmp/fake_qe_run && echo "&CONTROL /" > /tmp/fake_qe_run/scf.in && mkdir -p /tmp/fake_qe_run/pseudo
./scripts/remote_qe.sh /tmp/fake_qe_run --dry-run
```

Expected: prints scratch resolution plan / rsync / run steps; exit 0; no SSH execute of pw (dry-run may still ssh for scratch root resolve — acceptable, or mock scratch in dry-run as `$HOME/qe_scratch` without ssh). Prefer dry-run **skip live ssh** and print assumed scratch `$HOME/qe_scratch` unless `--probe` — simpler: dry-run prints commands only, zero network.

Spec says dry-run prints steps without SSH execute — **no network in --dry-run**.

- [ ] **Step 5: Commit**

```bash
git add scripts/remote_qe.sh scripts/README-remote-qe.md
git commit -m "feat(scripts): remote QE runner via rsync/SSH to Einstein"
```

---

### Task 2: Push (optional note)

- [ ] **Step 1:** `git push -u origin HEAD` so script is on the branch (Einstein pull not required for Mac→SSH use).

---

## Spec coverage

| Spec item | Task |
|-----------|------|
| remote_qe.sh flow | 1 |
| scratch fallback | 1 |
| minimal pull + wipe | 1 |
| exit codes / guards | 1 |
| README | 1 |
| dry-run | 1 |
