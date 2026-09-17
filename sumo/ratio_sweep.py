"""AV:PV ratio sweep on the converted M1 instance, random labelling.

Resumable: re-running with the same --seeds skips cells already in the CSV.

"Pure" run: no labelling rule, no calibration -- the placeholder demand and a
random AV/PV split, ratio varied.  One seed unless --seeds given.
Time constraints in model units (mean speed = 1, tau = 5 model units).

Run:  PYTHONPATH=. venv/bin/python -u sumo/ratio_sweep.py --ratios 0.2 0.4 0.6 0.8
Writes sumo/m1/ratio_sweep.csv
"""
import argparse, csv, time
from pathlib import Path
from sumo.convert import load_corridor_trips, label
from core.greedy_multi import greedy_multi_av_matching
from core.hungarian_multi import hungarian_multi_av_matching
from core.metrics import baseline_total_powered_distance, greedy_total_powered_distance

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", default="sumo/m1")
ap.add_argument("--ratios", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
ap.add_argument("--seeds", type=int, nargs="+", default=[42])
ap.add_argument("--tau", type=float, default=5.0)
a = ap.parse_args()

corr, trips = load_corridor_trips(a.scenario)
out = Path(a.scenario) / "ratio_sweep.csv"
# resume: keep existing rows and skip (seed, ratio) cells already complete (4 rows each)
rows = list(csv.DictReader(out.open())) if out.exists() else []
done = {k for k in {(int(x["seed"]), float(x["ratio"])) for x in rows}
        if sum(1 for x in rows if (int(x["seed"]), float(x["ratio"])) == k) == 4}
for seed in a.seeds:
    for r in a.ratios:
        if (seed, r) in done:
            print(f"seed {seed} ratio {r:.1f} already done, skipping", flush=True); continue
        # usable count is independent of the split, so size AV from a dummy split
        _, _, l_min, info = label(corr, trips, av_fraction=0.5, seed=seed, time_unit_s=0)
        n_usable = info["n_av"] + info["n_pv"]
        n_av = round(n_usable * r / (1 + r))
        avs, pvs, l_min, info = label(corr, trips, n_av=n_av, seed=seed, time_unit_s=0)
        base = baseline_total_powered_distance(avs, pvs)
        res = {}
        for name, fn in (("greedy", greedy_multi_av_matching), ("ILA", hungarian_multi_av_matching)):
            for ton in (False, True):
                t0 = time.perf_counter()
                asg, _, _ = fn(avs, pvs, l_min, enable_time_constraints=ton, time_tolerance=a.tau)
                dt = time.perf_counter() - t0
                sav = (base - greedy_total_powered_distance(base, asg)) / base * 100
                res[(name, ton)] = sav
                rows.append(dict(seed=seed, ratio=r, n_av=len(avs), n_pv=len(pvs), algorithm=name,
                                 time_on=ton, saving_percent=round(sav, 3), segments=len(asg),
                                 runtime_s=round(dt, 2)))
                print(f"seed {seed} ratio {r:.1f} AV {len(avs):4d} PV {len(pvs):4d}  {name:6} "
                      f"time={'ON ' if ton else 'OFF'} {sav:6.2f}%  ({dt:5.1f}s)", flush=True)
        print(f"   -> gap ILA-greedy: OFF {res[('ILA',False)]-res[('greedy',False)]:+.2f} pp, "
              f"ON {res[('ILA',True)]-res[('greedy',True)]:+.2f} pp", flush=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print("wrote", out)
