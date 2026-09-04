"""Where do greedy and ILA lose ground against the optimum?

Both heuristics reach ~93% of the exact optimum and neither beats the other.
This compares the solutions themselves rather than only their values, to find
which decisions the optimum makes differently.

Three exact solves per instance isolate how much of the shortfall each degree of
freedom accounts for:

  full-only    every segment covers a pair's whole overlap - the shape both
               algorithms commit when nothing has been towed yet
  suffix-only  a segment may start late but must run to the end of the overlap -
               what they can produce once earlier tows have advanced l_j
  unrestricted any contiguous sub-interval

Alongside the values it reports structural statistics: how many segments each
solution uses, how long they are, how many PVs are served, and how much AV
capacity is actually consumed.
"""
from __future__ import annotations

import statistics as st
from collections import defaultdict

from core.data import generate_mock_data
from core.greedy_multi import greedy_multi_av_matching
from core.hungarian_multi import hungarian_multi_av_matching
from milp.exact import solve, extract_segments, _overlap

SIZES = [(5, 10, 60), (6, 12, 80), (8, 15, 100), (10, 20, 100), (12, 25, 100)]
SEEDS = [42, 43, 44]
TL = 180.0


def stats(segs, pvs, avs):
    """(count, total distance, mean length, PVs served, AV capacity used)"""
    if not segs:
        return 0, 0.0, 0.0, 0, 0.0
    lens = [e - s for (_, _, s, e) in segs]
    served = len({j for (_, j, _, _) in segs})
    cap_used = sum(lens)
    cap_avail = sum(av.capacity * (av.exit_point - av.entry_point) for av in avs)
    return len(segs), float(sum(lens)), st.mean(lens), served, 100 * cap_used / cap_avail


def as_tuples(assignments, avs, pvs):
    ai = {a.id: i for i, a in enumerate(avs)}
    pi = {p.id: j for j, p in enumerate(pvs)}
    return [(ai[a.av.id], pi[a.pv.id], a.cp, a.dp) for a in assignments]


print(f"{'AV/PV':>8}{'seed':>5} | {'OPT':>6}{'suffix':>8}{'full':>7} | "
      f"{'greedy':>7}{'ILA':>6} | {'g/full':>8}{'ILA/full':>9}", flush=True)
print("-" * 76)

acc = defaultdict(list)
struct = defaultdict(lambda: defaultdict(list))

for na, npv, L in SIZES:
    for seed in SEEDS:
        avs, pvs, lmin = generate_mock_data(
            num_av=na, num_pv=npv, highway_length=L, av_capacity_range=(1, 3),
            min_trip_length=10, seed=seed, enable_time_constraints=False)

        ga, _, _ = greedy_multi_av_matching(avs, pvs, lmin, enable_time_constraints=False)
        ha, _, _ = hungarian_multi_av_matching(avs, pvs, lmin, enable_time_constraints=False)

        r_any = solve(avs, pvs, lmin, time_limit=TL)
        r_sfx = solve(avs, pvs, lmin, time_limit=TL, suffix_only=True)
        r_full = solve(avs, pvs, lmin, time_limit=TL, full_only=True)
        if not all(r.proven_optimal for r in (r_any, r_sfx, r_full)):
            print(f"{na}/{npv:<5}{seed:>5} | not all proven, skipped", flush=True)
            continue

        opt_segs, _ = extract_segments(avs, pvs, lmin, time_limit=TL)
        g = as_tuples(ga, avs, pvs)
        h = as_tuples(ha, avs, pvs)
        o = [(i, j, s, e) for (i, j, s, e, _) in opt_segs]

        gd = sum(e - s for _, _, s, e in g)
        hd = sum(e - s for _, _, s, e in h)
        acc["opt"].append(r_any.optimal)
        acc["sfx"].append(100 * r_sfx.optimal / r_any.optimal)
        acc["full"].append(100 * r_full.optimal / r_any.optimal)
        acc["g_any"].append(100 * gd / r_any.optimal)
        acc["h_any"].append(100 * hd / r_any.optimal)
        acc["g_full"].append(100 * gd / r_full.optimal)
        acc["h_full"].append(100 * hd / r_full.optimal)

        for name, segs in (("optimum", o), ("greedy", g), ("ILA", h)):
            n, d, ml, sv, cap = stats(segs, pvs, avs)
            struct[name]["n"].append(n)
            struct[name]["len"].append(ml)
            struct[name]["served"].append(100 * sv / len(pvs))
            struct[name]["cap"].append(cap)

        print(f"{na}/{npv:<5}{seed:>5} | {r_any.optimal:>6.0f}"
              f"{100 * r_sfx.optimal / r_any.optimal:>7.1f}%"
              f"{100 * r_full.optimal / r_any.optimal:>6.1f}% | "
              f"{100 * gd / r_any.optimal:>6.1f}%{100 * hd / r_any.optimal:>5.1f}% | "
              f"{100 * gd / r_full.optimal:>7.1f}%{100 * hd / r_full.optimal:>8.1f}%",
              flush=True)

n = len(acc["opt"])
print(f"\n=== {n} instances, means ===")
print(f"  suffix-only optimum   : {st.mean(acc['sfx']):6.2f}% of unrestricted")
print(f"  full-overlap optimum  : {st.mean(acc['full']):6.2f}% of unrestricted")
print(f"  greedy                : {st.mean(acc['g_any']):6.2f}% of unrestricted, "
      f"{st.mean(acc['g_full']):6.2f}% of full-overlap optimum")
print(f"  ILA                   : {st.mean(acc['h_any']):6.2f}% of unrestricted, "
      f"{st.mean(acc['h_full']):6.2f}% of full-overlap optimum")

print(f"\n=== solution structure (means) ===")
print(f"{'':>9}{'segments':>10}{'mean len':>10}{'PVs served':>12}{'AV cap used':>13}")
for name in ("optimum", "greedy", "ILA"):
    s = struct[name]
    print(f"{name:>9}{st.mean(s['n']):>10.1f}{st.mean(s['len']):>10.1f}"
          f"{st.mean(s['served']):>11.1f}%{st.mean(s['cap']):>12.1f}%")
