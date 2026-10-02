"""How often do greedy and ILA return time-infeasible solutions?

Checked against the convention the implementation intends ("either side waits",
NOTES.md 2026-10-02). An earlier version of this audit used the wrong
convention and produced false positives.

Found on one instance: a PV handed from one AV to another at the same point,
where the clocks at the handover differ by more than tau. The downstream check
appears to use a stale PV clock. If that is common, every saving figure
reported with time constraints on is overstated.

Run:  PYTHONPATH=. venv/bin/python -u milp/audit_time_feasibility.py
Writes data/results/milp/time_feasibility_audit.csv
"""
from __future__ import annotations

import argparse, csv
from pathlib import Path

from core.data import generate_mock_data
from core.greedy_multi import greedy_multi_av_matching
from core.hungarian_multi import hungarian_multi_av_matching
from milp.check_feasible import check

ap = argparse.ArgumentParser()
ap.add_argument("--pvs", type=int, nargs="+", default=[20, 50, 100, 200])
ap.add_argument("--ratios", type=float, nargs="+", default=[0.1, 0.2, 0.4, 0.8])
ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
ap.add_argument("--tau", type=float, default=5.0)
ap.add_argument("--waiting", default="either",
                help="semantics to check against; the implementation intends 'either'")
a = ap.parse_args()

OUT = Path("data/results/milp/time_feasibility_audit.csv")
FIELDS = ["algorithm", "pv", "ratio", "av", "seed", "segments", "towed",
          "violations", "multi_segment_pvs", "towed_in_violating_pv"]
rows = []
print(f"{'alg':>7}{'PV':>5}{'ratio':>7}{'seed':>5} | {'segs':>5}{'towed':>7} | "
      f"{'violations':>11}{'multiseg PVs':>13}", flush=True)

for npv in a.pvs:
    for r in a.ratios:
        nav = max(1, round(npv * r))
        for seed in a.seeds:
            avs, pvs, l_min = generate_mock_data(
                num_av=nav, num_pv=npv, highway_length=100, av_capacity_range=(1, 3),
                min_trip_length=10, seed=seed, enable_time_constraints=True,
                av_speed_range=(0.8, 1.2), pv_speed_range=(0.8, 1.2), time_window=100.0)
            for name, fn in (("greedy", greedy_multi_av_matching), ("ILA", hungarian_multi_av_matching)):
                asg, _, _ = fn(avs, pvs, l_min, enable_time_constraints=True, time_tolerance=a.tau)
                probs = check(asg, avs, pvs, l_min, a.tau, waiting=a.waiting)
                per_pv = {}
                for x in asg:
                    per_pv.setdefault(x.pv.id, []).append(x)
                multi = sum(1 for v in per_pv.values() if len(v) > 1)
                bad_pv = {p.split()[1] for p in probs if p.startswith("PV ")}
                towed_bad = sum(x.dp - x.cp for x in asg if x.pv.id in bad_pv)
                row = dict(algorithm=name, pv=npv, ratio=r, av=nav, seed=seed,
                           segments=len(asg), towed=sum(x.dp - x.cp for x in asg),
                           violations=len(probs), multi_segment_pvs=multi,
                           towed_in_violating_pv=towed_bad)
                rows.append(row)
                print(f"{name:>7}{npv:5d}{r:7.2f}{seed:5d} | {len(asg):5d}{row['towed']:7d} | "
                      f"{len(probs):11d}{multi:13d}", flush=True)
with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
print("wrote", OUT)
