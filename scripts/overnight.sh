#!/bin/zsh
# Unattended experiment queue. Every step is resumable: each sweep keeps its own
# CSV and skips cells already in it, so interrupting and re-running is safe.
#
#   scripts/overnight.sh            run the queue once, in order
#   tail -f logs/overnight.log      follow progress
#
# Steps are ordered by value: the temporal optimum first (it is the open
# methodological gap), then breadth on the spatial side, then figures.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=venv/bin/python
mkdir -p logs
LOG=logs/overnight.log

step () {                       # step <name> <command...>
  local name=$1; shift
  echo "\n=== [$(date '+%F %T')] $name" >> $LOG
  "$@" >> $LOG 2>&1
  echo "=== [$(date '+%F %T')] $name finished (exit $?)" >> $LOG
}

echo "\n########## queue started $(date '+%F %T')" >> $LOG

# 1. temporal optimum: the headline gap. Widen beyond the first sweep.
step "time optimum: base grid" \
  $PY -u milp/time_sweep.py --pvs 20 30 50 --ratios 0.1 0.2 0.4 0.6 \
      --seeds 42 43 44 45 46 --time-limit 300
step "time optimum: more seeds" \
  $PY -u milp/time_sweep.py --pvs 20 30 --ratios 0.1 0.2 0.4 0.6 \
      --seeds 47 48 49 50 51 --time-limit 300
step "time optimum: low ratios" \
  $PY -u milp/time_sweep.py --pvs 30 50 --ratios 0.02 0.05 \
      --seeds 42 43 44 45 46 --time-limit 300

# 2. spatial side: fill the thin high-ratio cells and add seeds
step "spatial: high ratios, more seeds" \
  $PY -u milp/lowratio_sweep.py --pv 50 --seeds 10 --ratios 0.4 0.6 1.0 --time-limit 900
step "spatial: PV=100 extra seeds" \
  $PY -u milp/lowratio_sweep.py --pv 100 --seeds 15 --ratios 0.01 0.02 0.05 0.10 --time-limit 900
step "spatial: PV=200" \
  $PY -u milp/lowratio_sweep.py --pv 200 --seeds 5 --ratios 0.01 0.02 0.05 --time-limit 900

# 3. score the refinement on everything new
step "refinement eval: synthetic" $PY -u milp/refine_eval.py --source synthetic --time-budget 15
step "refinement eval: M1"        $PY -u milp/refine_eval.py --source m1 --time-budget 15

# 4. refresh figures from whatever is now in the CSVs
step "figures: ILA vs optimum"  $PY analysis/plot_lowratio.py
step "figures: refinement"      $PY analysis/plot_refine.py
step "figures: capacity fill"   $PY analysis/plot_saturation.py

echo "########## queue finished $(date '+%F %T')" >> $LOG
