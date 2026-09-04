"""Collect the exact-solver results into CSVs for plotting.

Writes two files under data/results/milp/:

  exact_comparison.csv  one row per small instance: the optimum, what greedy and
                        ILA achieve on it, and how many segments of the optimal
                        solution are partial (i.e. strictly inside the pair's
                        maximal overlap, and so unreachable by either algorithm).

  time_effect.csv       greedy and ILA at paper-scale with the temporal
                        constraint switched off and on, to see where ILA's
                        advantage actually comes from.
"""
from __future__ import annotations

import csv
import os
from typing import List, Tuple

from core.data import generate_mock_data
from core.greedy_multi import greedy_multi_av_matching
from core.hungarian_multi import hungarian_multi_av_matching
from milp.exact import solve, extract_segments

OUT_DIR = "data/results/milp"
SIZES: List[Tuple[int, int, int]] = [
    (4, 8, 50), (5, 10, 60), (6, 12, 80), (8, 15, 100),
    (10, 20, 100), (12, 25, 100), (15, 30, 100),
]
SEEDS = [42, 43, 44]
TIME_LIMIT = 120.0

PAPER_CONFIGS = [
    ("capacity sweep", 50, 200),   # paper default, ratio 0.25
    ("length sweep", 80, 400),     # paper default, ratio 0.20
    ("ratio 0.8", 160, 200),
    ("ratio 0.8 (large)", 320, 400),
]


def towed(assignments) -> float:
    return float(sum(a.dp - a.cp for a in assignments))


def exact_comparison() -> None:
    path = os.path.join(OUT_DIR, "exact_comparison.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num_av", "num_pv", "highway_length", "seed", "reference",
                    "proven_optimal", "greedy", "ila", "greedy_pct", "ila_pct",
                    "n_segments", "n_partial", "partial_pct", "n_vars",
                    "n_constraints", "seconds"])
        for na, npv, L in SIZES:
            for seed in SEEDS:
                avs, pvs, lmin = generate_mock_data(
                    num_av=na, num_pv=npv, highway_length=L,
                    av_capacity_range=(1, 3), min_trip_length=10, seed=seed,
                    enable_time_constraints=False)
                g = towed(greedy_multi_av_matching(
                    avs, pvs, lmin, enable_time_constraints=False)[0])
                h = towed(hungarian_multi_av_matching(
                    avs, pvs, lmin, enable_time_constraints=False)[0])
                r = solve(avs, pvs, lmin, time_limit=TIME_LIMIT)
                ref = r.reference
                nseg = npart = 0
                if r.proven_optimal:
                    segs, npart = extract_segments(avs, pvs, lmin, time_limit=TIME_LIMIT)
                    nseg = len(segs)
                w.writerow([na, npv, L, seed, f"{ref:.0f}", r.proven_optimal,
                            f"{g:.0f}", f"{h:.0f}",
                            f"{100 * g / ref:.4f}", f"{100 * h / ref:.4f}",
                            nseg, npart,
                            f"{100 * npart / nseg:.2f}" if nseg else "",
                            r.n_binary, r.n_constraints, f"{r.seconds:.2f}"])
                print(f"  {na:>3}/{npv:<4} seed {seed}  "
                      f"greedy {100 * g / ref:6.2f}%  ILA {100 * h / ref:6.2f}%  "
                      f"{'optimal' if r.proven_optimal else 'LP bound'}", flush=True)
    print(f"wrote {path}")


def time_effect() -> None:
    path = os.path.join(OUT_DIR, "time_effect.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "num_av", "num_pv", "ratio",
                    "time_constraints", "seed", "greedy", "ila"])
        for label, na, npv in PAPER_CONFIGS:
            for tc in (False, True):
                for seed in (42, 43, 44, 45):
                    kw = dict(num_av=na, num_pv=npv, highway_length=100,
                              av_capacity_range=(1, 3), min_trip_length=10,
                              seed=seed, enable_time_constraints=tc)
                    if tc:
                        kw.update(av_speed_range=(0.8, 1.2),
                                  pv_speed_range=(0.8, 1.2), time_window=100.0)
                    avs, pvs, lmin = generate_mock_data(**kw)
                    g = towed(greedy_multi_av_matching(
                        avs, pvs, lmin, enable_time_constraints=tc,
                        time_tolerance=5.0)[0])
                    h = towed(hungarian_multi_av_matching(
                        avs, pvs, lmin, enable_time_constraints=tc,
                        time_tolerance=5.0)[0])
                    w.writerow([label, na, npv, f"{na / npv:.2f}", tc, seed,
                                f"{g:.0f}", f"{h:.0f}"])
            print(f"  {label} done", flush=True)
    print(f"wrote {path}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("exact comparison:")
    exact_comparison()
    print("time effect:")
    time_effect()
