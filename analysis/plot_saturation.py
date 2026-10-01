"""Where the ILA-optimum gap comes from: unused AV capacity.

The towed distance of any solution equals the AV capacity-distance it uses, so
"% of the optimum" is the same thing as "capacity filled, relative to the best
possible fill".  This plot separates the two: how full the optimum manages to
run the AVs, and how full ILA manages to run them.

Ceilings are recomputed from the instance definition; no MILP is re-solved.
Data: data/results/milp/lowratio_*.csv (proven-optimal rows only).

Run:  PYTHONPATH=. venv/bin/python analysis/plot_saturation.py
"""
import csv, collections, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core.data import generate_mock_data

by = collections.defaultdict(list)
for pv_size, path in ((50, "data/results/milp/lowratio_50.csv"),
                      (100, "data/results/milp/lowratio_100.csv")):
    try:
        rows = list(csv.DictReader(open(path)))
    except FileNotFoundError:
        continue
    for r in rows:
        if r["proven"] != "True" or r["optimum"] in ("inf", "nan"):
            continue
        avs, _, _ = generate_mock_data(
            num_av=int(r["av"]), num_pv=int(r["pv"]), highway_length=100,
            av_capacity_range=(1, 3), min_trip_length=10, seed=int(r["seed"]),
            enable_time_constraints=False)
        ceiling = sum(a.capacity * (a.exit_point - a.entry_point) for a in avs)
        by[(pv_size, float(r["ratio"]))].append(
            (100 * float(r["optimum"]) / ceiling, 100 * float(r["ila"]) / ceiling))

fig, ax = plt.subplots(figsize=(7.6, 4.6))
for pv_size, colour in ((50, "#9ecae1"), (100, "#08519c")):
    ks = sorted(k for k in by if k[0] == pv_size)
    if not ks:
        continue
    x = [k[1] for k in ks]
    ax.plot(x, [st.mean(o for o, _ in by[k]) for k in ks], "o-", color=colour, lw=1.8, ms=6,
            label=f"optimum, PV = {pv_size}")
    ax.plot(x, [st.mean(i for _, i in by[k]) for k in ks], "^--", color=colour, lw=1.6, ms=6,
            alpha=.75, label=f"ILA, PV = {pv_size}")
    for k in ks:
        print(f"PV {pv_size:3d} ratio {k[1]:5.2f}  optimum fills {st.mean(o for o,_ in by[k]):5.1f}%  "
              f"ILA fills {st.mean(i for _,i in by[k]):5.1f}%  (n={len(by[k])})")

ratios = sorted({k[1] for k in by})
ax.set_xscale("log"); ax.set_xticks(ratios)
ax.set_xticklabels([f"{t:.0%}" for t in ratios]); ax.minorticks_off()
ax.set_xlabel("AV : PV ratio")
ax.set_ylabel("AV capacity-distance filled  (%)")
ax.set_title("The optimum keeps the AVs almost full at every ratio.\n"
             "ILA matches it while AVs are few, and leaves capacity idle once there are many.",
             fontsize=10)
ax.grid(alpha=.25); ax.legend(fontsize=9, loc="lower left")
fig.tight_layout()
out = "analysis/figures/av_saturation.png"
fig.savefig(out, dpi=160)
print("wrote", out)
