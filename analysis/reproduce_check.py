"""Step 0 reproduction check.

Re-runs a small subset of the stored pv_av_sweep configurations with the
CURRENT environment and compares against data/results/*/pv_av_sweep.csv,
which was produced with NumPy 1.26 / SciPy 1.11 (per the paper).

Passes if saving_percent and baseline_total_distance match exactly.
Runtime is expected to differ and is not compared.
"""
import csv, sys
from experiments.common import ScenarioParams, run_one_scenario
from core.greedy_multi import greedy_multi_av_matching
from core.hungarian_multi import hungarian_multi_av_matching

MATCHERS = {"greedy": greedy_multi_av_matching, "hungarian": hungarian_multi_av_matching}

# small, fast cells only (both scenario types, low + mid ratio)
CELLS = [("av_sweep", 200, 10, 200), ("av_sweep", 200, 20, 200),
         ("av_sweep", 400, 40, 400), ("pv_sweep", 20, 20, 100),
         ("pv_sweep", 80, 80, 100), ("pv_sweep", 160, 160, 200)]
SEEDS = [42, 43, 44, 45]

def stored(alg):
    rows = list(csv.DictReader(open(f"data/results/{alg}/pv_av_sweep.csv")))
    return {(r["scenario_type"], int(r["fixed_value"]), int(r["num_av"]),
             int(r["num_pv"]), int(r["seed"])): r for r in rows}

fails, checked = [], 0
for alg, matcher in MATCHERS.items():
    S = stored(alg)
    for scen, fixed, nav, npv in CELLS:
        for seed in SEEDS:
            k = (scen, fixed, nav, npv, seed)
            if k not in S:
                print(f"  (skip, not in CSV: {k})"); continue
            old = S[k]
            row = run_one_scenario(
                params=ScenarioParams(
                    num_av=nav, num_pv=npv, highway_length=100,
                    av_capacity_range=(1, 3), min_trip_length=10, seed=seed,
                    enable_time_constraints=True, time_tolerance=5.0,
                    time_window=100.0, av_speed_range=(0.8, 1.2),
                    pv_speed_range=(0.8, 1.2)),
                matcher=matcher, run_task2_checks=False)
            checked += 1
            for col in ("baseline_total_distance", "total_saving", "saving_percent",
                        "matched_pv"):
                a, b = float(old[col]), float(row[col])
                if abs(a - b) > 1e-9:
                    fails.append((alg, k, col, a, b))

print(f"\n검사한 (알고리즘 x 조건 x seed) 조합: {checked}")
if fails:
    print(f"불일치 {len(fails)}건:")
    for f in fails[:20]:
        print(f"  {f[0]:9} {f[1]} {f[2]}: 저장={f[3]} 재실행={f[4]}")
    sys.exit(1)
print("전부 일치 — 현재 환경에서 논문 결과 재현 가능")
