#!/usr/bin/env bash
# remote_qe.sh — sync a local QE run dir to Einstein, run pw.x (+ optional chain), pull allowlist.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/remote_qe.sh <local_run_dir> [--np 4] [--host einstein] [--keep-scratch] [--dry-run]

Run Quantum ESPRESSO (and optional Wannier90) on Einstein via rsync + SSH.
Pulls a minimal allowlist of outputs back to local_run_dir.

Options:
  --np N           MPI ranks (default 4, max 32)
  --host HOST      SSH host (default einstein)
  --keep-scratch   Keep remote scratch after success
  --dry-run        Print plan only; no network, no remote run

Exit codes:
  1  SSH / connectivity / rsync pull failed (scratch kept)
  2  Local validation (scf.in missing, bad path)
  3  Remote pw.x missing
  4  SCF (or chain) failed — scratch kept; outs pulled if present
  5  Remote free disk < ~1 GB, or scratch-root create/resolve failed
EOF
}

NP=4
HOST=einstein
KEEP=0
DRY=0
RUN_DIR=""

die() {
  local code="$1"
  shift
  echo "error: $*" >&2
  exit "$code"
}

log() {
  echo "$@"
  if [[ -n "${LOCAL_LOG:-}" ]]; then
    echo "$@" >>"$LOCAL_LOG"
  fi
}

# --- arg parse ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --np)
      [[ $# -ge 2 ]] || die 2 "--np requires a value"
      NP="$2"
      shift 2
      ;;
    --host)
      [[ $# -ge 2 ]] || die 2 "--host requires a value"
      HOST="$2"
      shift 2
      ;;
    --keep-scratch)
      KEEP=1
      shift
      ;;
    --dry-run)
      DRY=1
      shift
      ;;
    -*)
      die 2 "unknown option: $1"
      ;;
    *)
      if [[ -n "$RUN_DIR" ]]; then
        die 2 "unexpected argument: $1"
      fi
      RUN_DIR="$1"
      shift
      ;;
  esac
done

[[ -n "$RUN_DIR" ]] || { usage >&2; die 2 "local_run_dir required"; }

# Validate NP
if ! [[ "$NP" =~ ^[0-9]+$ ]] || [[ "$NP" -lt 1 ]]; then
  die 2 "--np must be a positive integer (got: $NP)"
fi
if [[ "$NP" -gt 32 ]]; then
  die 2 "--np capped at 32 (got: $NP)"
fi

# Resolve absolute run dir
if [[ ! -d "$RUN_DIR" ]]; then
  die 2 "not a directory: $RUN_DIR"
fi
RUN_DIR="$(cd "$RUN_DIR" && pwd)"

if [[ ! -f "$RUN_DIR/scf.in" ]]; then
  die 2 "scf.in missing in $RUN_DIR (refuse syncing non-run dirs)"
fi

JOB_ID="$(date +%Y%m%d-%H%M%S)-$(basename "$RUN_DIR")"
LOCAL_LOG="$RUN_DIR/remote_qe.log"
: >"$LOCAL_LOG"

ALLOWLIST=(scf.out nscf.out pw2wan.out wannier90.wout wannier90_hr.dat remote_qe.log)

resolve_scratch_root() {
  # shellcheck disable=SC2029
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

pull_allowlist() {
  local remote_list
  remote_list="$(ssh "$HOST" "cd '$SCRATCH' && ls scf.out nscf.out pw2wan.out wannier90.wout wannier90_hr.dat remote_qe.log 2>/dev/null" || true)"
  if [[ -z "${remote_list// }" ]]; then
    log "pull: no allowlist files present on remote"
    return 0
  fi
  local listfile
  listfile="$(mktemp)"
  # shellcheck disable=SC2001
  echo "$remote_list" | tr ' ' '\n' | sed '/^$/d' >"$listfile"
  log "pull: $(tr '\n' ' ' <"$listfile")"
  if ! rsync -az --files-from="$listfile" "$HOST:$SCRATCH/" "$RUN_DIR/"; then
    rm -f "$listfile"
    log "error: rsync allowlist pull failed from $HOST:$SCRATCH"
    return 1
  fi
  rm -f "$listfile"
  return 0
}

# --- dry-run: zero network ---
if [[ "$DRY" -eq 1 ]]; then
  ASSUMED_ROOT="\$HOME/qe_scratch"
  ASSUMED_SCRATCH="${ASSUMED_ROOT}/${JOB_ID}"
  log "=== remote_qe dry-run (no network) ==="
  log "local_run_dir: $RUN_DIR"
  log "host:          $HOST"
  log "np:            $NP"
  log "keep_scratch:  $KEEP"
  log "job_id:        $JOB_ID"
  log "scratch_root:  $ASSUMED_ROOT  (assumed; live resolve prefers /data/sandy/qe_scratch if writable)"
  log "scratch:       $ASSUMED_SCRATCH"
  log ""
  log "plan:"
  log "  1. ssh $HOST 'true'                          # preflight connectivity"
  log "  2. resolve scratch root via ssh"
  log "  3. remote: command -v pw.x (after tensorspec-solvers.env / conda qe PATH)"
  log "  4. remote: df -BG free on scratch fs >= 1"
  log "  5. rsync -az --delete $RUN_DIR/ $HOST:$ASSUMED_SCRATCH/"
  log "  6. remote pipeline:"
  log "       mpirun -np $NP pw.x -in scf.in | tee scf.out"
  if [[ -f "$RUN_DIR/nscf.in" ]]; then
    log "       mpirun -np $NP pw.x -in nscf.in | tee nscf.out"
  else
    log "       (skip nscf — no nscf.in)"
  fi
  if [[ -f "$RUN_DIR/pw2wan.in" && -f "$RUN_DIR/wannier90.win" ]]; then
    log "       pw2wannier90.x < pw2wan.in > pw2wan.out 2>&1"
    log "       wannier90.x wannier90 > wannier90.wout 2>&1"
  else
    log "       (skip wannier — need both pw2wan.in and wannier90.win)"
  fi
  log "  7. rsync allowlist -> $RUN_DIR  (${ALLOWLIST[*]})"
  if [[ "$KEEP" -eq 1 ]]; then
    log "  8. keep remote scratch (--keep-scratch)"
  else
    log "  8. on success + pull OK: ssh rm -rf $ASSUMED_SCRATCH"
  fi
  log "  on QE failure: keep scratch; still pull allowlist; exit 4"
  log "  on pull failure: keep scratch; exit 1 (never wipe without successful pull)"
  log "=== end dry-run ==="
  exit 0
fi

# --- live path ---
log "=== remote_qe start job_id=$JOB_ID host=$HOST np=$NP ==="
log "local_run_dir=$RUN_DIR"

# Preflight SSH
if ! ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" true; then
  die 1 "SSH connectivity failed for host=$HOST"
fi

SCRATCH_ROOT=""
RESOLVE_RC=0
SCRATCH_ROOT="$(resolve_scratch_root)" || RESOLVE_RC=$?
if [[ "$RESOLVE_RC" -eq 5 ]]; then
  die 5 "could not create/resolve scratch root on $HOST"
elif [[ "$RESOLVE_RC" -ne 0 ]]; then
  die 1 "SSH failed while resolving scratch root on $HOST (rc=$RESOLVE_RC)"
fi
SCRATCH_ROOT="$(echo "$SCRATCH_ROOT" | tr -d '\r' | tail -n1)"
[[ -n "$SCRATCH_ROOT" ]] || die 5 "empty scratch root from $HOST"
SCRATCH="${SCRATCH_ROOT}/${JOB_ID}"
log "scratch=$SCRATCH"

# Remote env + pw.x preflight + disk check
PREFLIGHT_RC=0
# shellcheck disable=SC2029
ssh "$HOST" "bash -s" <<EOF || PREFLIGHT_RC=$?
set -euo pipefail
source "\$HOME/tensorspec-solvers.env" 2>/dev/null || export PATH="\$HOME/miniconda3/envs/qe/bin:\$PATH"
if ! command -v pw.x >/dev/null 2>&1; then
  echo "pw.x not found in PATH" >&2
  exit 3
fi
command -v pw.x
command -v mpirun || true
mkdir -p '$SCRATCH_ROOT'
# Prefer tr over awk sub(): nested ssh/heredoc quoting breaks \"\" in awk.
FREE_G=\$(df -BG '$SCRATCH_ROOT' | awk 'NR==2 {print \$4}' | tr -d 'G')
echo "free_gb=\$FREE_G"
if [ -z "\$FREE_G" ] || [ "\$FREE_G" -lt 1 ]; then
  echo "insufficient free disk on scratch fs (need >= 1G)" >&2
  exit 5
fi
EOF

case "$PREFLIGHT_RC" in
  0) ;;
  3) die 3 "remote pw.x missing on $HOST" ;;
  5) die 5 "remote free disk < ~1 GB on scratch filesystem" ;;
  *) die 1 "remote preflight failed (rc=$PREFLIGHT_RC)" ;;
esac

# Sync run dir to remote scratch
log "rsync -> $HOST:$SCRATCH/"
ssh "$HOST" "mkdir -p '$SCRATCH'"
rsync -az --delete "$RUN_DIR/" "$HOST:$SCRATCH/" || die 1 "rsync to remote failed"

# Build remote run script
HAS_NSCF=0
HAS_WAN=0
[[ -f "$RUN_DIR/nscf.in" ]] && HAS_NSCF=1
[[ -f "$RUN_DIR/pw2wan.in" && -f "$RUN_DIR/wannier90.win" ]] && HAS_WAN=1

REMOTE_STATUS=0
# shellcheck disable=SC2029
ssh "$HOST" "bash -s" <<EOF || REMOTE_STATUS=$?
set -euo pipefail
cd '$SCRATCH'
exec > >(tee -a remote_qe.log) 2>&1
source "\$HOME/tensorspec-solvers.env" 2>/dev/null || export PATH="\$HOME/miniconda3/envs/qe/bin:\$PATH"

echo "=== remote pipeline start \$(date -Is) np=$NP ==="
echo "PATH=\$PATH"
command -v pw.x
command -v mpirun

echo "--- SCF ---"
mpirun -np $NP pw.x -in scf.in | tee scf.out
if ! grep -q "JOB DONE" scf.out; then
  echo "SCF missing JOB DONE" >&2
  exit 4
fi

if [ "$HAS_NSCF" -eq 1 ]; then
  echo "--- NSCF ---"
  mpirun -np $NP pw.x -in nscf.in | tee nscf.out
  if ! grep -q "JOB DONE" nscf.out; then
    echo "NSCF missing JOB DONE" >&2
    exit 4
  fi
fi

if [ "$HAS_WAN" -eq 1 ]; then
  echo "--- pw2wannier90 ---"
  pw2wannier90.x < pw2wan.in > pw2wan.out 2>&1
  echo "--- wannier90 ---"
  wannier90.x wannier90 > wannier90.wout 2>&1
fi

echo "=== remote pipeline OK \$(date -Is) ==="
EOF

# Always attempt pull (success or failure). Never wipe if pull fails.
log "pulling allowlist (remote status=$REMOTE_STATUS)"
PULL_RC=0
pull_allowlist || PULL_RC=$?

# Merge: ensure local transcript notes remote status
{
  echo ""
  echo "=== local wrapper remote_status=$REMOTE_STATUS pull_rc=$PULL_RC $(date -Is) ==="
} >>"$LOCAL_LOG"

if [[ "$REMOTE_STATUS" -ne 0 ]]; then
  log "QE chain failed (rc=$REMOTE_STATUS); keeping scratch=$SCRATCH"
  if [[ "$PULL_RC" -ne 0 ]]; then
    log "also: allowlist pull failed (rc=$PULL_RC); scratch kept"
  fi
  exit 4
fi

if [[ "$PULL_RC" -ne 0 ]]; then
  die 1 "allowlist pull failed after successful QE (rc=$PULL_RC); keeping scratch=$SCRATCH (not wiped)"
fi

if [[ "$KEEP" -eq 1 ]]; then
  log "success; keeping scratch (--keep-scratch): $SCRATCH"
else
  log "success; wiping scratch: $SCRATCH"
  ssh "$HOST" "rm -rf '$SCRATCH'" || log "warning: failed to wipe scratch $SCRATCH"
fi

log "=== remote_qe done ==="
exit 0
