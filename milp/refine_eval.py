"""Score ILA + ruin-and-recreate refinement against the stored exact optima.

Re-creates each instance from the sweep CSVs (no MILP is re-solved; the stored
optimum or LP bound is reused), runs ILA, then core.refine.refine_ils, and
records both as a percentage of the reference.

Run:  PYTHONPATH=. venv/bin/python -u milp/refine_eval.py --source synthetic
      PYTHONPATH=. venv/bin/python -u milp/refine_eval.py --source m1
Writes data/results/milp/refine_<source>.csv  (resumable)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from core.data import generate_mock_data
from core.hungarian_multi import hungarian_multi_av_matching
from core.refine import refine_ils

ap = argparse.ArgumentParser()
ap.add_argument("--source", choices=["synthetic", "m1"], default="synthetic")
ap.add_argument("--iterations", type=int, default=600)
ap.add_argument("--time-budget", type=float, default=20.0)
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--refine-seed", type=int, default=1)
a = ap.parse_args()

SRC = {"synthetic": ["data/results/milp/lowratio_50.csv",
                     "data/results/milp/lowratio_100.csv"],
       "m1": ["data/results/milp/m1_lowratio_300.csv"]}[a.source]
OUT = Path("data/results/milp") / f"refine_{a.source}.csv"
FIELDS = ["source", "key", "ratio", "av", "pv", "proven", "optimum",
          "ila", "refined", "ila_pct", "refined_pct", "recovered_pp",
          "accepted", "iterations", "seconds"]
rows = list(csv.DictReader(OUT.open())) if OUT.exists() else []
done = {r["key"] for r in rows}

if a.source == "m1":
    from sumo.convert import load_corridor_trips, auto_time_unit, extract_instance
    corr, trips = load_corridor_trips("sumo/m1")
    tu = auto_time_unit(corr, trips)

towed = lambda asg: float(sum(s.dp - s.cp for s in asg))
print(f"{'key':>18}{'ratio':>7}{'AV':>4}{'PV':>5} {'ILA %':>8}{'refined %':>11}{'+pp':>7} {'sec':>6}", flush=True)

for path in SRC:
    try:
        src_rows = list(csv.DictReader(open(path)))
    except FileNotFoundError:
        continue
    for r in src_rows:
        if r["optimum"] in ("inf", "nan") or not r["ila_pct"]:
            continue
        if a.source == "synthetic":
            key = f"pv{r['pv']}_r{r['ratio']}_s{r['seed']}"
        else:
            key = f"t{float(r['t0']):.0f}_r{r['ratio']}"
        if key in done:
            continue
        if a.source == "synthetic":
            avs, pvs, l_min = generate_mock_data(
                num_av=int(r["av"]), num_pv=int(r["pv"]), highway_length=100,
                av_capacity_range=(1, 3), min_trip_length=10, seed=int(r["seed"]),
                enable_time_constraints=False)
        else:
            avs, pvs, l_min, _ = extract_instance(
                corr, trips, t0=float(r["t0"]), window_s=float(r["window_s"]),
                ratio=float(r["ratio"]), seed=42, time_unit_s=tu)
        h, _, _ = hungarian_multi_av_matching(avs, pvs, l_min, enable_time_constraints=False)
        ila = towed(h)
        segs, rep = refine_ils(h, avs, pvs, l_min, iterations=a.iterations,
                               seed=a.refine_seed, alpha=a.alpha, time_budget=a.time_budget)
        ref = float(r["optimum"])
        row = dict(source=a.source, key=key, ratio=r["ratio"], av=len(avs), pv=len(pvs),
                   proven=r["proven"], optimum=ref, ila=ila, refined=rep["end"],
                   ila_pct=round(100 * ila / ref, 3),
                   refined_pct=round(100 * rep["end"] / ref, 3),
                   recovered_pp=round(100 * (rep["end"] - ila) / ref, 3),
                   accepted=rep["accepted"], iterations=rep["iterations"],
                   seconds=rep["seconds"])
        rows.append(row)
        print(f"{key:>18}{float(r['ratio']):7.2f}{len(avs):4d}{len(pvs):5d} "
              f"{row['ila_pct']:7.2f}%{row['refined_pct']:10.2f}%{row['recovered_pp']:+7.2f} "
              f"{rep['seconds']:6.1f}", flush=True)
        with OUT.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
print("wrote", OUT)
