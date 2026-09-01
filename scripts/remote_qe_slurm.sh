#!/usr/bin/env bash
# remote_qe_slurm.sh — sync a local QE run dir to a Slurm cluster, sbatch pw.x (+ chain), pull allowlist.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${TENSORSPEC_REMOTE_ENV:-$SCRIPT_DIR/tensorspec-remote.env}"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

usage() {
  cat <<'EOF'
Usage: ./scripts/remote_qe_slurm.sh <local_run_dir> [options]

Run Quantum ESPRESSO (and optional Wannier90) on a remote Slurm cluster via rsync + SSH + sbatch.
Pulls a minimal allowlist of outputs back to local_run_dir.

Configuration (first match wins): CLI flag > environment variable > tensorspec-remote.env

Options:
  --host HOST         SSH host alias (TENSORSPEC_SLURM_HOST)
  --account ACCT      Slurm account -A (TENSORSPEC_SLURM_ACCOUNT)
  --qos QOS           Slurm QoS -q (default: regular)
  --constraint C      Slurm constraint -C (default: cpu)
  --nodes N           Slurm nodes -N (default: 1)
  --np N              MPI tasks / --ntasks (default: 128)
  --time HH:MM:SS     Walltime -t (default: 02:00:00)
  --module MOD        module load argument (default: none; use remote PATH)
  --scratch PATH      Remote scratch root override (default: remote $SCRATCH/qe_runs)
  --job-name NAME     Slurm job name (default: ts_qe)
  --poll-interval S   Seconds between sacct polls (default: 30)
  --keep-scratch      Keep remote scratch after success
  --dry-run           Print plan only; no network

Exit codes:
  1  SSH / connectivity / rsync pull failed (scratch kept)
  2  Local validation (scf.in missing, bad path, missing host/account)
  3  Remote pw.x missing (preflight)
  4  QE chain or Slurm job failed — scratch kept; outs pulled if present
  5  Remote free disk < ~1 GB, or scratch-root create/resolve failed
  6  sbatch submit failed
EOF
}

HOST="${TENSORSPEC_SLURM_HOST:-}"
ACCOUNT="${TENSORSPEC_SLURM_ACCOUNT:-}"
QOS="${TENSORSPEC_SLURM_QOS:-regular}"
CONSTRAINT="${TENSORSPEC_SLURM_CONSTRAINT:-cpu}"
NODES="${TENSORSPEC_SLURM_NODES:-1}"
NTASKS="${TENSORSPEC_SLURM_NTASKS:-128}"
TIME="${TENSORSPEC_SLURM_TIME:-02:00:00}"
MODULE="${TENSORSPEC_SLURM_MODULE:-}"
SCRATCH_OVERRIDE="${TENSORSPEC_SLURM_SCRATCH:-}"
MAX_NTASKS="${TENSORSPEC_SLURM_MAX_NTASKS:-512}"
POLL_INTERVAL="${TENSORSPEC_SLURM_POLL_INTERVAL:-30}"
SLURM_JOB_NAME="${TENSORSPEC_SLURM_JOB_NAME:-ts_qe}"
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
    --account)
      [[ $# -ge 2 ]] || die 2 "--account requires a value"
      ACCOUNT="$2"
      shift 2
      ;;
    --qos)
      [[ $# -ge 2 ]] || die 2 "--qos requires a value"
      QOS="$2"
      shift 2
      ;;
    --constraint)
      [[ $# -ge 2 ]] || die 2 "--constraint requires a value"
      CONSTRAINT="$2"
      shift 2
      ;;
    --nodes)
      [[ $# -ge 2 ]] || die 2 "--nodes requires a value"
      NODES="$2"
      shift 2
      ;;
    --np|--ntasks)
      [[ $# -ge 2 ]] || die 2 "$1 requires a value"
      NTASKS="$2"
      shift 2
      ;;
    --time)
      [[ $# -ge 2 ]] || die 2 "--time requires a value"
      TIME="$2"
      shift 2
      ;;
    --module)
      [[ $# -ge 2 ]] || die 2 "--module requires a value"
      MODULE="$2"
      shift 2
      ;;
    --scratch)
      [[ $# -ge 2 ]] || die 2 "--scratch requires a value"
      SCRATCH_OVERRIDE="$2"
      shift 2
      ;;
    --job-name)
      [[ $# -ge 2 ]] || die 2 "--job-name requires a value"
      SLURM_JOB_NAME="$2"
      shift 2
      ;;
    --poll-interval)
      [[ $# -ge 2 ]] || die 2 "--poll-interval requires a value"
      POLL_INTERVAL="$2"
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

if ! [[ "$NTASKS" =~ ^[0-9]+$ ]] || [[ "$NTASKS" -lt 1 ]]; then
  die 2 "--np must be a positive integer (got: $NTASKS)"
fi
if [[ "$NTASKS" -gt "$MAX_NTASKS" ]]; then
  die 2 "--np capped at $MAX_NTASKS (got: $NTASKS)"
fi
if ! [[ "$NODES" =~ ^[0-9]+$ ]] || [[ "$NODES" -lt 1 ]]; then
  die 2 "--nodes must be a positive integer (got: $NODES)"
fi
if ! [[ "$POLL_INTERVAL" =~ ^[0-9]+$ ]] || [[ "$POLL_INTERVAL" -lt 5 ]]; then
  die 2 "--poll-interval must be an integer >= 5 (got: $POLL_INTERVAL)"
fi

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

HAS_NSCF=0
HAS_WAN=0
[[ -f "$RUN_DIR/nscf.in" ]] && HAS_NSCF=1
[[ -f "$RUN_DIR/pw2wan.in" && -f "$RUN_DIR/wannier90.win" ]] && HAS_WAN=1

resolve_scratch_root() {
  local override="$1"
  # shellcheck disable=SC2029
  ssh "$HOST" "bash -s" <<EOS
set -euo pipefail
override='${override}'
if [[ -n "\$override" ]]; then
  mkdir -p "\$override"
  echo "\$override"
elif [[ -n "\${SCRATCH:-}" ]]; then
  mkdir -p "\${SCRATCH}/qe_runs"
  echo "\${SCRATCH}/qe_runs"
elif mkdir -p "\$HOME/qe_scratch" 2>/dev/null; then
  echo "\$HOME/qe_scratch"
else
  exit 5
fi
EOS
}

pull_allowlist() {
  local slurm_out="${1:-}"
  local remote_list extra
  remote_list="$(ssh "$HOST" "cd '$SCRATCH' && ls scf.out nscf.out pw2wan.out wannier90.wout wannier90_hr.dat remote_qe.log 2>/dev/null" || true)"
  extra=""
  if [[ -n "$slurm_out" ]]; then
    extra="$(ssh "$HOST" "test -f '$SCRATCH/$slurm_out' && echo '$slurm_out'" || true)"
  fi
  if [[ -z "${remote_list// }" && -z "${extra// }" ]]; then
    log "pull: no allowlist files present on remote"
    return 0
  fi
  local listfile
  listfile="$(mktemp)"
  {
    echo "$remote_list" | tr ' ' '\n'
    [[ -n "${extra// }" ]] && echo "$extra"
  } | sed '/^$/d' | sort -u >"$listfile"
  log "pull: $(tr '\n' ' ' <"$listfile")"
  if ! rsync -az --files-from="$listfile" "$HOST:$SCRATCH/" "$RUN_DIR/"; then
    rm -f "$listfile"
    log "error: rsync allowlist pull failed from $HOST:$SCRATCH"
    return 1
  fi
  rm -f "$listfile"
  return 0
}

write_batch_script() {
  local dest="$1"
  local module_block=""
  if [[ -n "$MODULE" ]]; then
    module_block="module load ${MODULE}"
  fi
  cat >"$dest" <<SLURM
#!/bin/bash
#SBATCH -A ${ACCOUNT}
#SBATCH -C ${CONSTRAINT}
#SBATCH -q ${QOS}
#SBATCH -N ${NODES}
#SBATCH -t ${TIME}
#SBATCH -J ${SLURM_JOB_NAME}
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --ntasks=${NTASKS}
#SBATCH --cpus-per-task=1

set -euo pipefail
cd "\${SLURM_SUBMIT_DIR:-\$PWD}"
exec >> remote_qe.log 2>&1

echo "=== remote_qe_slurm batch start \$(date '+%Y-%m-%dT%H:%M:%S%z') ==="
echo "SLURM_JOB_ID=\${SLURM_JOB_ID:-unknown}"
echo "ntasks=${NTASKS} nodes=${NODES} constraint=${CONSTRAINT} qos=${QOS}"

${module_block}
source "\$HOME/tensorspec-solvers.env" 2>/dev/null || true

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
echo "PATH=\$PATH"
command -v pw.x

echo "--- SCF ---"
srun -n ${NTASKS} --cpu-bind=cores pw.x -in scf.in | tee scf.out
if ! grep -q "JOB DONE" scf.out; then
  echo "SCF missing JOB DONE" >&2
  exit 4
fi
SLURM

  if [[ "$HAS_NSCF" -eq 1 ]]; then
    cat >>"$dest" <<SLURM
echo "--- NSCF ---"
srun -n ${NTASKS} --cpu-bind=cores pw.x -in nscf.in | tee nscf.out
if ! grep -q "JOB DONE" nscf.out; then
  echo "NSCF missing JOB DONE" >&2
  exit 4
fi
SLURM
  fi

  if [[ "$HAS_WAN" -eq 1 ]]; then
    cat >>"$dest" <<'SLURM'
echo "--- pw2wannier90 ---"
srun -n 1 pw2wannier90.x < pw2wan.in > pw2wan.out 2>&1
echo "--- wannier90 ---"
srun -n 1 wannier90.x wannier90 > wannier90.wout 2>&1
SLURM
  fi

  cat >>"$dest" <<SLURM
echo "=== remote_qe_slurm batch OK \$(date '+%Y-%m-%dT%H:%M:%S%z') ==="
SLURM
}

poll_slurm_job() {
  local job_id="$1"
  local state=""
  while true; do
    state="$(ssh "$HOST" "sacct -j ${job_id} -n -X --format=State -p 2>/dev/null | head -1 | tr -d ' |'" || true)"
    state="${state%%+*}"
    case "$state" in
      COMPLETED)
        log "slurm: job ${job_id} COMPLETED"
        return 0
        ;;
      FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|BOOT_FAIL|DEADLINE)
        log "slurm: job ${job_id} ended state=${state:-UNKNOWN}"
        return 4
        ;;
      "")
        log "slurm: sacct returned empty state for ${job_id}; retry in ${POLL_INTERVAL}s"
        ;;
      *)
        log "slurm: job ${job_id} state=${state}; poll again in ${POLL_INTERVAL}s"
        ;;
    esac
    sleep "$POLL_INTERVAL"
  done
}

if [[ "$DRY" -eq 1 ]]; then
  ASSUMED_ROOT='${SCRATCH}/qe_runs or $HOME/qe_scratch'
  ASSUMED_SCRATCH="${ASSUMED_ROOT}/${JOB_ID}"
  log "=== remote_qe_slurm dry-run (no network) ==="
  log "local_run_dir: $RUN_DIR"
  log "host:          ${HOST:-<unset>}"
  log "account:       ${ACCOUNT:-<unset>}"
  log "qos:           $QOS"
  log "constraint:    $CONSTRAINT"
  log "nodes:         $NODES"
  log "ntasks:        $NTASKS"
  log "time:          $TIME"
  log "module:        ${MODULE:-<none>}"
  log "keep_scratch:  $KEEP"
  log "job_id:        $JOB_ID"
  log "scratch:       $ASSUMED_SCRATCH"
  log ""
  log "plan:"
  log "  1. ssh \${HOST} preflight (pw.x, disk)"
  log "  2. rsync run dir -> remote scratch"
  log "  3. sbatch run_qe.slurm on remote"
  log "  4. poll sacct until COMPLETED or failure"
  log "  5. rsync allowlist -> local (scf.out, wannier90_hr.dat, remote_qe.log, slurm-*.out)"
  log "  6. wipe remote scratch on success unless --keep-scratch"
  log "=== end dry-run ==="
  exit 0
fi

[[ -n "$HOST" ]] || die 2 "SSH host required: set TENSORSPEC_SLURM_HOST, tensorspec-remote.env, or --host"
[[ -n "$ACCOUNT" ]] || die 2 "Slurm account required: set TENSORSPEC_SLURM_ACCOUNT, tensorspec-remote.env, or --account"

log "=== remote_qe_slurm start job_id=$JOB_ID host=$HOST account=$ACCOUNT ntasks=$NTASKS ==="
log "local_run_dir=$RUN_DIR"

if ! ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" true; then
  die 1 "SSH connectivity failed for host=$HOST"
fi

SCRATCH_ROOT=""
RESOLVE_RC=0
SCRATCH_ROOT="$(resolve_scratch_root "$SCRATCH_OVERRIDE")" || RESOLVE_RC=$?
if [[ "$RESOLVE_RC" -eq 5 ]]; then
  die 5 "could not create/resolve scratch root on $HOST"
elif [[ "$RESOLVE_RC" -ne 0 ]]; then
  die 1 "SSH failed while resolving scratch root on $HOST (rc=$RESOLVE_RC)"
fi
SCRATCH_ROOT="$(echo "$SCRATCH_ROOT" | tr -d '\r' | tail -n1)"
[[ -n "$SCRATCH_ROOT" ]] || die 5 "empty scratch root from $HOST"
SCRATCH="${SCRATCH_ROOT}/${JOB_ID}"
log "scratch=$SCRATCH"
printf '%s\t%s\n' "$HOST" "$SCRATCH" >"$RUN_DIR/.tensorspec_remote_scratch"
log "sidecar: $RUN_DIR/.tensorspec_remote_scratch -> $HOST $SCRATCH"

PREFLIGHT_RC=0
# shellcheck disable=SC2029
ssh "$HOST" "bash -s" <<EOF || PREFLIGHT_RC=$?
set -euo pipefail
if [[ -n "${MODULE}" ]]; then
  module load ${MODULE}
fi
source "\$HOME/tensorspec-solvers.env" 2>/dev/null || true
if ! command -v pw.x >/dev/null 2>&1; then
  echo "pw.x not found (module=${MODULE:-none})" >&2
  exit 3
fi
command -v pw.x
command -v sbatch
mkdir -p '${SCRATCH_ROOT}'
FREE_G=\$(df -BG '${SCRATCH_ROOT}' | awk 'NR==2 {print \$4}' | tr -d 'G')
echo "free_gb=\$FREE_G"
if [[ -z "\$FREE_G" || "\$FREE_G" -lt 1 ]]; then
  echo "insufficient free disk on scratch fs (need >= 1G)" >&2
  exit 5
fi
EOF

case "$PREFLIGHT_RC" in
  0) ;;
  3) die 3 "remote pw.x missing on $HOST (check --module or remote PATH)" ;;
  5) die 5 "remote free disk < ~1 GB on scratch filesystem" ;;
  *) die 1 "remote preflight failed (rc=$PREFLIGHT_RC)" ;;
esac

BATCH_LOCAL="$RUN_DIR/.tensorspec_run_qe.slurm"
write_batch_script "$BATCH_LOCAL"

log "rsync -> $HOST:$SCRATCH/"
ssh "$HOST" "mkdir -p '$SCRATCH'"
rsync -az --delete \
  --exclude '.tensorspec_run_qe.slurm' \
  "$RUN_DIR/" "$HOST:$SCRATCH/" || die 1 "rsync to remote failed"
rsync -az "$BATCH_LOCAL" "$HOST:$SCRATCH/run_qe.slurm" || die 1 "rsync batch script failed"

SUBMIT_OUT=""
SUBMIT_RC=0
SUBMIT_OUT="$(ssh "$HOST" "cd '$SCRATCH' && sbatch run_qe.slurm" 2>&1)" || SUBMIT_RC=$?
log "$SUBMIT_OUT"
if [[ "$SUBMIT_RC" -ne 0 ]]; then
  die 6 "sbatch failed (rc=$SUBMIT_RC)"
fi

SLURM_ID=""
SLURM_ID="$(echo "$SUBMIT_OUT" | awk '/Submitted batch job/ {print $NF}' | tail -1)"
[[ -n "$SLURM_ID" ]] || die 6 "could not parse Slurm job id from sbatch output"
printf '%s\t%s\t%s\n' "$HOST" "$SCRATCH" "$SLURM_ID" >"$RUN_DIR/.tensorspec_slurm_job"
log "slurm_job_id=$SLURM_ID (sidecar: .tensorspec_slurm_job)"

REMOTE_STATUS=0
poll_slurm_job "$SLURM_ID" || REMOTE_STATUS=$?

SLURM_OUT="slurm-${SLURM_ID}.out"
log "pulling allowlist (remote status=$REMOTE_STATUS)"
PULL_RC=0
pull_allowlist "$SLURM_OUT" || PULL_RC=$?

{
  echo ""
  echo "=== local wrapper slurm_job=$SLURM_ID remote_status=$REMOTE_STATUS pull_rc=$PULL_RC $(date '+%Y-%m-%dT%H:%M:%S%z') ==="
} >>"$LOCAL_LOG"

if [[ "$REMOTE_STATUS" -ne 0 ]]; then
  log "Slurm/QE failed (rc=$REMOTE_STATUS); keeping scratch=$SCRATCH"
  [[ "$PULL_RC" -ne 0 ]] && log "also: allowlist pull failed (rc=$PULL_RC)"
  exit 4
fi

if [[ "$PULL_RC" -ne 0 ]]; then
  die 1 "allowlist pull failed after successful job (rc=$PULL_RC); keeping scratch=$SCRATCH"
fi

if [[ "$KEEP" -eq 1 ]]; then
  log "success; keeping scratch (--keep-scratch): $SCRATCH"
else
  log "success; wiping scratch: $SCRATCH"
  ssh "$HOST" "rm -rf '$SCRATCH'" || log "warning: failed to wipe scratch $SCRATCH"
fi

log "=== remote_qe_slurm done ==="
exit 0
