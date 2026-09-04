"""Compare greedy and ILA against the exact optimum on small instances.

Reviewer item #4. Answers the question the paper cannot currently answer:
is the small greedy-ILA gap because ILA is weak, or because greedy is already
close to the best any algorithm could do?

Runs with time constraints disabled, matching the scope of milp.exact.
"""

from __future__ import annotations

import argparse
from typing import List, Sequence, Tuple

from core.data import generate_mock_data
from core.greedy_multi import greedy_multi_av_matching
from core.hungarian_multi import hungarian_multi_av_matching
from milp.exact import solve

SIZES: List[Tuple[int, int, int]] = [   # (num_av, num_pv, highway_length)
    (4, 8, 50),
    (5, 10, 60),
    (6, 12, 80),
    (8, 15, 100),
    (10, 20, 100),
]
SEEDS = [42, 43, 44]


def towed(assignments) -> float:
    return float(sum(a.dp - a.cp for a in assignments))


def run(sizes: Sequence[Tuple[int, int, int]], seeds: Sequence[int],
        time_limit: float) -> None:
    header = (f"{'AV':>3}{'PV':>4}{'L':>5} {'seed':>5} "
              f"{'optimum':>9}{'greedy':>8}{'ILA':>7} "
              f"{'greedy/opt':>11}{'ILA/opt':>9} {'vars':>7}{'rows':>7}{'sec':>7}  status")
    print(header)
    print("-" * len(header))

    for num_av, num_pv, length in sizes:
        for seed in seeds:
            avs, pvs, l_min = generate_mock_data(
                num_av=num_av, num_pv=num_pv, highway_length=length,
                av_capacity_range=(1, 3), min_trip_length=10, seed=seed,
                enable_time_constraints=False)

            g_assign, _, _ = greedy_multi_av_matching(
                avs, pvs, l_min, enable_time_constraints=False)
            h_assign, _, _ = hungarian_multi_av_matching(
                avs, pvs, l_min, enable_time_constraints=False)
            g, h = towed(g_assign), towed(h_assign)

            r = solve(avs, pvs, l_min, time_limit=time_limit)
            ref = r.reference
            label = "optimal" if r.proven_optimal else "LP bound"
            gp = f"{100 * g / ref:9.2f}%" if ref else "        -"
            hp = f"{100 * h / ref:7.2f}%" if ref else "      -"

            print(f"{num_av:>3}{num_pv:>4}{length:>5} {seed:>5} "
                  f"{ref:>9.0f}{g:>8.0f}{h:>7.0f} {gp:>11}{hp:>9} "
                  f"{r.n_binary:>7}{r.n_constraints:>7}{r.seconds:>7.1f}  {label}")

            if r.proven_optimal and (g > ref + 1e-6 or h > ref + 1e-6):
                print("   !! heuristic beat the proven optimum - formulation bug")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--time-limit", type=float, default=600.0)
    p.add_argument("--quick", action="store_true",
                   help="smallest size and one seed only")
    a = p.parse_args()
    sizes, seeds = (SIZES[:1], SEEDS[:1]) if a.quick else (SIZES, SEEDS)
    run(sizes, seeds, a.time_limit)


if __name__ == "__main__":
    main()
