#!/usr/bin/env bash
# Wait for active chinook job (parent PID) to finish, then run Grizzly A/B plan.
# Idempotent via DONE / STARTED markers.
set -euo pipefail

RUN_DIR=/scratch/YOUR_USER/tensorspec_heavy/chinook_gui_run
PY=/home/YOUR_USER/TensorSpec/TensorSpec_env/bin/python
PARENT_PID=${1:-4077260}
MARKER_DIR="$RUN_DIR/post_chinook_auto"
LOG="$MARKER_DIR/watcher.log"

mkdir -p "$MARKER_DIR"
exec >>"$LOG" 2>&1

echo "==== watcher start $(date -Is) parent=$PARENT_PID ===="

if [[ -f "$MARKER_DIR/DONE" ]]; then
  echo "Already DONE — exiting."
  exit 0
fi
if [[ -f "$MARKER_DIR/STARTED_GRIZZLY" ]]; then
  echo "Grizzly already started earlier — not double-launching. Check sys.out.grizzly."
  exit 0
fi

echo "Waiting for chinook parent PID $PARENT_PID to exit..."
while kill -0 "$PARENT_PID" 2>/dev/null; do
  sleep 60
done
echo "Parent $PARENT_PID gone at $(date -Is)"

# Wait briefly for workers / final save
for i in 1 2 3 4 5 6 7 8 9 10; do
  if grep -a -q "completed successfully" "$RUN_DIR/sys.out.full" 2>/dev/null; then
    echo "Found 'completed successfully' in sys.out.full"
    break
  fi
  # any remaining active workers of the finished tree?
  if ! pgrep -f "chinook_remote_runner.py --tb_file" >/dev/null 2>&1; then
    echo "No chinook_remote_runner processes left"
    break
  fi
  # ignore stale 0%CPU zombie-ish parents older than 1 day if only those remain
  sleep 30
done

cd "$RUN_DIR"

# Rename fresh cube if present and not already archived
if [[ -f chinook_arpes_cube.npz ]]; then
  if [[ ! -f chinook_arpes_cube_chinook.npz ]]; then
    # Prefer rename when cube mtime is newer than Aug 27 (old cube was Aug 26)
    cube_mtime=$(stat -c %Y chinook_arpes_cube.npz)
    if [[ "$cube_mtime" -gt 1756500000 ]]; then
      cp -a chinook_arpes_cube.npz chinook_arpes_cube_chinook.npz
      echo "Archived cube -> chinook_arpes_cube_chinook.npz (mtime=$cube_mtime)"
    else
      echo "WARNING: cube mtime still old ($cube_mtime) — copying anyway as chinook ref"
      cp -a chinook_arpes_cube.npz chinook_arpes_cube_chinook.npz
    fi
  else
    echo "chinook_arpes_cube_chinook.npz already exists — keeping it"
  fi
else
  echo "WARNING: no chinook_arpes_cube.npz after job exit"
fi

echo "==== Phase 0: one-slice timing $(date -Is) ===="
$PY -u time_grizzly_vs_chinook_slice.py \
  --tb_file tb_data.npz \
  --theta 0.0 \
  --phi_min -15 --phi_max 15 --nphi 40 \
  --e_min -2 --e_max 0.5 --ne 40 \
  --hv 84 --workf 4.5 --temp 10 --polar P \
  --device cuda \
  | tee "$MARKER_DIR/phase0_timing.txt" || echo "Phase 0 timing failed (exit $?)"

echo "==== Phase 1: full-cube Grizzly CUDA $(date -Is) ===="
touch "$MARKER_DIR/STARTED_GRIZZLY"
# Move aside default out path so we don't overwrite chinook archive mid-write
nohup $PY -u chinook_remote_runner.py \
  --tb_file tb_data.npz \
  --theta_min -15.0 --theta_max 15.0 --ntheta 200 \
  --phi_min -15.0 --phi_max 15.0 --nphi 200 \
  --e_min -2.0 --e_max 0.5 --ne 200 \
  --hv 84.0 --workf 4.5 --v0 12.0 --temp 10.0 --polar P \
  --cores 2 \
  --engine grizzly --device cuda --layout full \
  --out_file chinook_arpes_cube_grizzly.npz \
  > "$RUN_DIR/sys.out.grizzly" 2>&1 &
GRIZZLY_PID=$!
echo "$GRIZZLY_PID" > "$MARKER_DIR/grizzly.pid"
echo "Launched Grizzly full layout PID=$GRIZZLY_PID"

echo "Waiting for Grizzly PID $GRIZZLY_PID..."
wait "$GRIZZLY_PID" || echo "Grizzly exited non-zero: $?"

echo "==== Compare $(date -Is) ===="
if [[ -f chinook_arpes_cube_chinook.npz && -f chinook_arpes_cube_grizzly.npz ]]; then
  $PY -u compare_arpes_cubes.py \
    chinook_arpes_cube_chinook.npz chinook_arpes_cube_grizzly.npz \
    | tee "$MARKER_DIR/compare.txt" || true
else
  echo "Missing cubes for compare:" 
  ls -lah chinook_arpes_cube_chinook.npz chinook_arpes_cube_grizzly.npz 2>&1 || true
  echo "--- sys.out.grizzly tail ---"
  tail -n 80 "$RUN_DIR/sys.out.grizzly" || true
fi

date -Is > "$MARKER_DIR/DONE"
echo "==== ALL DONE $(date -Is) ===="
