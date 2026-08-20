#!/usr/bin/env bash
# Durable wrapper for the long totals evaluation run.
#
# The evaluation only writes its database rows at the very end, so a mid-run death would lose
# everything. The per-game stdout lines carry the full result for each finished game
# (game_pk, point, sim_over, mkt_over, actual), so tee-ing them to a file makes partial progress
# recoverable and independent of the process supervisor.
#
# Writes a STATUS line on completion so progress can be checked with a single read.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.

# Pin thread pools. CatBoost, polars and BLAS each spawn workers per call, and these batches are
# a few dozen rows, far too small to benefit. Unpinned, a 4-game 60-sim run spent 20 minutes of
# system time against 3 minutes of wall clock on thread synchronisation; pinning cut wall clock
# from 181s to 42s, a 4.3x speedup.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export POLARS_MAX_THREADS=1

OUT_DIR=data/analysis
mkdir -p "$OUT_DIR"

GAMES=${GAMES:-600}
SIMS=${SIMS:-200}
SEED=${SEED:-2026}
SEASON=${SEASON:-2025}
RUN_ID=${RUN_ID:-totals_eval_600g_200s}
LOG="$OUT_DIR/${RUN_ID}.log"
STATUS="$OUT_DIR/${RUN_ID}_STATUS.txt"

{
  echo "STATUS=RUNNING"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "season=$SEASON games=$GAMES sims=$SIMS seed=$SEED run_id=$RUN_ID"
  echo "expect roughly 46 seconds per game, so about $((GAMES * 46 / 60)) minutes"
  echo "per-game lines accumulate in $LOG"
} > "$STATUS"

uv run python scripts/sim_totals_eval.py \
  --season "$SEASON" --games "$GAMES" --sims "$SIMS" --seed "$SEED" \
  --edge-buckets 0.02,0.03,0.05,0.07 \
  --run-id "$RUN_ID" \
  --out-csv "$OUT_DIR/${RUN_ID}.csv" \
  --out-json "$OUT_DIR/${RUN_ID}.json" \
  > "$LOG" 2>&1
RC=$?

DONE_GAMES=$(grep -c ' pt=' "$LOG" 2>/dev/null || echo 0)
{
  if [ "$RC" -eq 0 ]; then echo "STATUS=COMPLETE"; else echo "STATUS=FAILED rc=$RC"; fi
  echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "games_simulated=$DONE_GAMES"
  echo
  echo "--- summary ---"
  grep -E 'Brier|log loss|flat-bet|diagnostics|Season |inserted' "$LOG" 2>/dev/null | tail -20
} >> "$STATUS"
exit "$RC"
