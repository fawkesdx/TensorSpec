#!/bin/bash
# Unattended post-ladder: wait, summarize, retry OOM with smaller theta_chunk.
set -u
cd /scratch/YOUR_USER/tensorspec_heavy/chinook_gui_run
OUT=scale_bench
LOG=$OUT/autonomous_watch.log
STATUS=$OUT/AUTONOMOUS_STATUS.md
PY=/home/YOUR_USER/TensorSpec/TensorSpec_env/bin/python
exec >>"$LOG" 2>&1
echo "==== watch start $(date -Is) ===="

# Wait for ladder process to exit (up to ~24h)
for i in $(seq 1 1440); do
  if ! pgrep -f "bench_grizzly_ladder.py" >/dev/null; then
    echo "ladder gone at iter $i $(date -Is)"
    break
  fi
  sleep 60
done

sleep 5

echo "==== writing status $(date -Is) ===="
{
  echo "# Autonomous ladder status"
  echo
  echo "Updated: $(date -Is)"
  echo
  echo "## LADDER.jsonl"
  echo '```'
  cat "$OUT/LADDER.jsonl" 2>/dev/null || echo "(empty)"
  echo '```'
  echo
  echo "## ladder.log tail"
  echo '```'
  tail -60 "$OUT/ladder.log" 2>/dev/null
  echo '```'
} > "$STATUS"

if grep -q "HARD WALL" "$OUT/ladder.log" 2>/dev/null; then
  FAIL=$(grep "HARD WALL" "$OUT/ladder.log" | tail -1)
  echo "Detected: $FAIL"
  TAG=$(echo "$FAIL" | grep -oE '[0-9]+x[0-9]+x[0-9]+' | head -1 || true)
  if [ -n "${TAG:-}" ]; then
    NT=${TAG%%x*}
    REST=${TAG#*x}
    NP=${REST%%x*}
    NE=${REST##*x}
    CHUNK=$($PY -c "
import json
chunk = 20
for line in open('$OUT/LADDER.jsonl'):
    d = json.loads(line)
    if d.get('tag') == '$TAG':
        chunk = d.get('theta_chunk_requested') or 20
print(chunk)
" 2>/dev/null | tail -1)
    CHUNK=${CHUNK:-20}
    if [ "$CHUNK" -le 0 ]; then CHUNK=20; fi
    NEW=$((CHUNK / 2))
    if [ "$NEW" -lt 5 ]; then NEW=5; fi
    if [ "$NEW" -lt "$CHUNK" ]; then
      echo "AUTO-RETRY $TAG with theta_chunk=$NEW (was $CHUNK)"
      rm -f "$OUT/result_${TAG}_grizzly_cuda_full.json"
      CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        $PY -u bench_grizzly_scale.py \
        --ntheta "$NT" --nphi "$NP" --ne "$NE" \
        --phi_min -15 --phi_max 15 --e_min -1 --e_max 0.1 \
        --device cuda --layout full --theta_chunk "$NEW" \
        --cuda_visible 0 --out_dir "$OUT" || true
      echo "retry exit=$?"
      {
        echo
        echo "## Auto-retry"
        echo "- tag: $TAG chunk $CHUNK -> $NEW"
        echo "- finished: $(date -Is)"
        echo
        echo '```'
        tail -1 "$OUT/LADDER.jsonl" 2>/dev/null
        tail -40 "$OUT/log_${TAG}_grizzly_cuda_full.txt" 2>/dev/null
        echo '```'
      } >> "$STATUS"
    fi
  fi
fi

if grep -q "LADDER complete" "$OUT/ladder.log" 2>/dev/null; then
  echo "Ladder completed cleanly."
fi

echo "==== watch done $(date -Is) ===="
{
  echo
  echo "## Watch finished"
  echo "$(date -Is)"
} >> "$STATUS"
