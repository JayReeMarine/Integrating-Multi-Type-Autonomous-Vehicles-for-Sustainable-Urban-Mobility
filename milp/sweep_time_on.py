"""Do the congestion variants change the picture once time is switched on?

Everything measured against the exact optimum is necessarily time-free, and that
is the setting where ILA is weakest: its advantage over greedy at paper scale
shows up mainly with the temporal constraint active. This compares the variants
head to head with time on, where no exact reference exists but the relative
ordering is still meaningful.
"""
from __future__ import annotations

import statistics as st

from core.data import generate_mock_data
from core.greedy_multi import greedy_multi_av_matching
from core.hungarian_multi import hungarian_multi_av_matching
from core.congestion import (greedy_congestion_matching, ila_congestion_matching,
                             ila_filter_matching)

CONFIGS = [("capacity sweep", 50, 200), ("length sweep", 80, 400),
           ("ratio 0.8", 160, 200), ("ratio 0.8 large", 320, 400)]
SEEDS = [42, 43, 44, 45]
KW = dict(enable_time_constraints=True, time_tolerance=5.0)


def towed(a):
    return float(sum(x.dp - x.cp for x in a))


def instance(na, npv, seed):
    return generate_mock_data(
        num_av=na, num_pv=npv, highway_length=100, av_capacity_range=(1, 3),
        min_trip_length=10, seed=seed, enable_time_constraints=True,
        av_speed_range=(0.8, 1.2), pv_speed_range=(0.8, 1.2), time_window=100.0)


VARIANTS = [
    ("greedy", lambda a, p, l: greedy_multi_av_matching(a, p, l, **KW)),
    ("greedy+lam2", lambda a, p, l: greedy_congestion_matching(a, p, l, lam=2.0, **KW)),
    ("ILA", lambda a, p, l: hungarian_multi_av_matching(a, p, l, **KW)),
    ("ILA+lam1", lambda a, p, l: ila_congestion_matching(a, p, l, lam=1.0, **KW)),
    ("ILA+filter2.5", lambda a, p, l: ila_filter_matching(a, p, l, theta=2.5, **KW)),
]

print(f"{'config':>18} " + "".join(f"{n:>15}" for n, _ in VARIANTS), flush=True)
print("-" * (19 + 15 * len(VARIANTS)))

overall = {n: [] for n, _ in VARIANTS}
for label, na, npv in CONFIGS:
    means = {}
    for name, fn in VARIANTS:
        vals = []
        for seed in SEEDS:
            avs, pvs, lmin = instance(na, npv, seed)
            vals.append(towed(fn(avs, pvs, lmin)[0]))
        means[name] = st.mean(vals)
    base = means["greedy"]
    for name, _ in VARIANTS:
        overall[name].append(100 * means[name] / base)
    cells = "".join(f"{means[n]:>9.0f} ({100 * means[n] / base - 100:+5.2f}%)"
                    .rjust(15) for n, _ in VARIANTS)
    print(f"{label:>18} {cells}", flush=True)

print(f"\nmean relative to shipped greedy (100%):")
for name, _ in VARIANTS:
    print(f"  {name:<16}{st.mean(overall[name]):7.2f}%")
