#!/usr/bin/env bash
# remote_arpes_me.sh — sync a local ARPES ME job dir to Einstein, run Option A, pull allowlist.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/remote_arpes_me.sh <local_job_dir> [--host einstein] [--keep-scratch] [--dry-run]

Run ARPES Option A (Simple Scalar ME) on Einstein via rsync + SSH.
Pulls intensity.npz, meta.json, and remote_arpes_me.log back to local_job_dir.

Options:
  --host HOST      SSH host (default einstein)
  --keep-scratch   Keep remote scratch after success
  --dry-run        Print plan only; no network, no remote run

Exit codes:
  1  SSH / connectivity / rsync pull failed (scratch kept)
  2  Local validation (request.json or structure missing, bad path)
  4  Remote simulation failed — scratch kept; allowlist pulled if present
  5  Remote scratch-root create/resolve failed
  6  Remote missing dependency (from run_arpes_me_a.py)
EOF
}

HOST=einstein
KEEP=0
DRY=0
JOB_DIR=""

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
      if [[ -n "$JOB_DIR" ]]; then
        die 2 "unexpected argument: $1"
      fi
      JOB_DIR="$1"
      shift
      ;;
  esac
done

[[ -n "$JOB_DIR" ]] || { usage >&2; die 2 "local_job_dir required"; }

# Resolve absolute job dir
if [[ ! -d "$JOB_DIR" ]]; then
  die 2 "not a directory: $JOB_DIR"
fi
JOB_DIR="$(cd "$JOB_DIR" && pwd)"

if [[ ! -f "$JOB_DIR/request.json" ]]; then
  die 2 "request.json missing in $JOB_DIR (refuse syncing non-job dirs)"
fi

HAS_CIF=0
HAS_JSON=0
[[ -f "$JOB_DIR/structure.cif" ]] && HAS_CIF=1
[[ -f "$JOB_DIR/structure.json" ]] && HAS_JSON=1
if [[ "$HAS_CIF" -eq 0 && "$HAS_JSON" -eq 0 ]]; then
  die 2 "structure.cif or structure.json required in $JOB_DIR"
fi
if [[ "$HAS_CIF" -eq 1 && "$HAS_JSON" -eq 1 ]]; then
  die 2 "provide exactly one of structure.cif or structure.json in $JOB_DIR"
fi

JOB_ID="$(date +%Y%m%d-%H%M%S)-$(basename "$JOB_DIR")"
# Sanitize for safe embedding in ssh single-quoted remote paths (reject shell metacharacters).
JOB_ID="$(printf '%s' "$JOB_ID" | sed 's/[^A-Za-z0-9._-]/_/g')"
LOCAL_LOG="$JOB_DIR/remote_arpes_me.log"
: >"$LOCAL_LOG"

ALLOWLIST=(intensity.npz meta.json remote_arpes_me.log)

resolve_scratch_root() {
  # shellcheck disable=SC2029
  ssh "$HOST" 'bash -s' <<'EOS'
if [ -d /data/sandy ] && [ -w /data/sandy ]; then
  mkdir -p /data/sandy/arpes_me_scratch && echo /data/sandy/arpes_me_scratch
elif mkdir -p "$HOME/arpes_me_scratch" 2>/dev/null; then
  echo "$HOME/arpes_me_scratch"
else
  exit 5
fi
EOS
}

pull_allowlist() {
  local remote_list
  remote_list="$(ssh "$HOST" "cd '$SCRATCH' && ls intensity.npz meta.json remote_arpes_me.log 2>/dev/null" || true)"
  if [[ -z "${remote_list// }" ]]; then
    log "pull: no allowlist files present on remote"
    return 0
  fi
  local listfile
  listfile="$(mktemp)"
  # shellcheck disable=SC2001
  echo "$remote_list" | tr ' ' '\n' | sed '/^$/d' >"$listfile"
  log "pull: $(tr '\n' ' ' <"$listfile")"
  if ! rsync -az --files-from="$listfile" "$HOST:$SCRATCH/" "$JOB_DIR/"; then
    rm -f "$listfile"
    log "error: rsync allowlist pull failed from $HOST:$SCRATCH"
    return 1
  fi
  rm -f "$listfile"
  return 0
}

# --- dry-run: zero network ---
if [[ "$DRY" -eq 1 ]]; then
  ASSUMED_ROOT="\$HOME/arpes_me_scratch"
  ASSUMED_SCRATCH="${ASSUMED_ROOT}/${JOB_ID}"
  log "=== remote_arpes_me dry-run (no network) ==="
  log "local_job_dir: $JOB_DIR"
  log "host:          $HOST"
  log "keep_scratch:  $KEEP"
  log "job_id:        $JOB_ID"
  log "scratch_root:  $ASSUMED_ROOT  (assumed; live resolve prefers /data/sandy/arpes_me_scratch if writable)"
  log "scratch:       $ASSUMED_SCRATCH"
  log ""
  log "plan:"
  log "  1. ssh $HOST 'true'                          # preflight connectivity"
  log "  2. resolve scratch root via ssh"
  log "  3. rsync -az --delete $JOB_DIR/ $HOST:$ASSUMED_SCRATCH/"
  log "  4. remote:"
  log "       cd \$SCRATCH"
  log "       export PYTHONPATH=\${TENSORSPEC_ROOT:-\$HOME/TensorSpec}"
  log "       PY=\${TENSORSPEC_ROOT:-\$HOME/TensorSpec}/TensorSpec_env/bin/python"
  log "       \$PY \$PYTHONPATH/scripts/run_arpes_me_a.py ."
  log "  5. rsync allowlist -> $JOB_DIR  (${ALLOWLIST[*]})"
  if [[ "$KEEP" -eq 1 ]]; then
    log "  6. keep remote scratch (--keep-scratch)"
  else
    log "  6. on success + pull OK: ssh rm -rf $ASSUMED_SCRATCH"
  fi
  log "  on simulation failure: keep scratch; still pull allowlist; exit remote rc (2/4/6)"
  log "  on pull failure: keep scratch; exit 1 (never wipe without successful pull)"
  log "=== end dry-run ==="
  exit 0
fi

# --- live path ---
log "=== remote_arpes_me start job_id=$JOB_ID host=$HOST ==="
log "local_job_dir=$JOB_DIR"

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
printf '%s\t%s\n' "$HOST" "$SCRATCH" >"$JOB_DIR/.tensorspec_remote_scratch"
log "sidecar: $JOB_DIR/.tensorspec_remote_scratch -> $HOST $SCRATCH"

# Remote mkdir on scratch root
PREFLIGHT_RC=0
# shellcheck disable=SC2029
ssh "$HOST" "bash -s" <<EOF || PREFLIGHT_RC=$?
set -euo pipefail
mkdir -p '$SCRATCH_ROOT'
EOF

if [[ "$PREFLIGHT_RC" -ne 0 ]]; then
  die 1 "remote preflight failed (rc=$PREFLIGHT_RC)"
fi

# Sync job dir to remote scratch
log "rsync -> $HOST:$SCRATCH/"
ssh "$HOST" "mkdir -p '$SCRATCH'"
rsync -az --delete "$JOB_DIR/" "$HOST:$SCRATCH/" || die 1 "rsync to remote failed"

REMOTE_STATUS=0
# shellcheck disable=SC2029
ssh "$HOST" "bash -s" <<EOF || REMOTE_STATUS=$?
set -euo pipefail
cd '$SCRATCH'
export PYTHONPATH="\${TENSORSPEC_ROOT:-\$HOME/TensorSpec}"
PY="\${TENSORSPEC_ROOT:-\$HOME/TensorSpec}/TensorSpec_env/bin/python"
"\$PY" "\$PYTHONPATH/scripts/run_arpes_me_a.py" .
EOF

# Always attempt pull (success or failure). Never wipe if pull fails.
log "pulling allowlist (remote status=$REMOTE_STATUS)"
PULL_RC=0
pull_allowlist || PULL_RC=$?

# Merge: ensure local transcript notes remote status
{
  echo ""
  echo "=== local wrapper remote_status=$REMOTE_STATUS pull_rc=$PULL_RC $(date '+%Y-%m-%dT%H:%M:%S%z') ==="
} >>"$LOCAL_LOG"

if [[ "$REMOTE_STATUS" -ne 0 ]]; then
  log "ARPES ME failed (rc=$REMOTE_STATUS); keeping scratch=$SCRATCH"
  if [[ "$PULL_RC" -ne 0 ]]; then
    log "also: allowlist pull failed (rc=$PULL_RC); scratch kept"
  fi
  case "$REMOTE_STATUS" in
    2|4|6) exit "$REMOTE_STATUS" ;;
    *) exit 4 ;;
  esac
fi

if [[ "$PULL_RC" -ne 0 ]]; then
  die 1 "allowlist pull failed after successful ARPES ME (rc=$PULL_RC); keeping scratch=$SCRATCH (not wiped)"
fi

if [[ "$KEEP" -eq 1 ]]; then
  log "success; keeping scratch (--keep-scratch): $SCRATCH"
else
  log "success; wiping scratch: $SCRATCH"
  ssh "$HOST" "rm -rf '$SCRATCH'" || log "warning: failed to wipe scratch $SCRATCH"
fi

log "=== remote_arpes_me done ==="
exit 0
