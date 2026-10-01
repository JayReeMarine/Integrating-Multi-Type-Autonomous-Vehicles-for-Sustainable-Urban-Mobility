"""How close ILA gets to the exact optimum, as a function of the AV:PV ratio.

Answers the question Eunus set on 18 Sep (drop greedy; measure ILA against the
MILP optimum, especially where AVs are scarce).  Data: data/results/milp/
lowratio_*.csv, produced by milp/lowratio_sweep.py.  Spatial problem only
(time constraints off) -- milp.exact does not model temporal coupling.

Run:  PYTHONPATH=. venv/bin/python analysis/plot_lowratio.py
"""
import csv, glob, statistics as st, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = []
for f in sorted(glob.glob("data/results/milp/lowratio_*.csv")):
    rows += [r for r in csv.DictReader(open(f)) if r["proven"] == "True"]
by = collections.defaultdict(list)
for r in rows:
    by[(int(r["pv"]), float(r["ratio"]))].append(r)

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
colours = {50: "#08519c", 100: "#d62728"}

a = ax[0]
for pv in sorted({k[0] for k in by}):
    ks = sorted(k for k in by if k[0] == pv)
    x = [k[1] for k in ks]
    m = [st.mean(float(r["ila_pct"]) for r in by[k]) for k in ks]
    e = [st.stdev([float(r["ila_pct"]) for r in by[k]]) if len(by[k]) > 1 else 0 for k in ks]
    n = min(len(by[k]) for k in ks)
    a.errorbar(x, m, yerr=e, fmt="o-", ms=6, capsize=3, color=colours.get(pv, "0.3"),
               label=f"PV = {pv}  ({n}+ seeds)")
a.axhline(100, color="0.6", lw=0.8, ls="--")
a.text(0.011, 100.15, "exact optimum", fontsize=8, color="0.4")
a.set_xscale("log")
ticks = sorted({k[1] for k in by})
a.set_xticks(ticks); a.set_xticklabels([f"{t:.0%}" for t in ticks], fontsize=8)
a.set_xlabel("AV : PV ratio"); a.set_ylabel("ILA as % of the exact optimum")
a.set_title("ILA is near-optimal exactly where AVs are scarce", fontsize=10)
a.legend(fontsize=8, loc="lower left"); a.grid(alpha=.25)

b = ax[1]
for pv in sorted({k[0] for k in by}):
    ks = sorted(k for k in by if k[0] == pv)
    x = [k[1] for k in ks]
    b.plot(x, [st.mean(float(r["opt_coverage"]) for r in by[k]) for k in ks], "o-",
           color=colours.get(pv, "0.3"), label=f"optimum, PV = {pv}")
    b.plot(x, [st.mean(float(r["ila_coverage"]) for r in by[k]) for k in ks], "^--",
           color=colours.get(pv, "0.3"), alpha=.55, label=f"ILA, PV = {pv}")
b.set_xscale("log")
b.set_xticks(ticks); b.set_xticklabels([f"{t:.0%}" for t in ticks], fontsize=8)
b.set_xlabel("AV : PV ratio"); b.set_ylabel("% of total PV distance towed")
b.set_title("…but when AVs are scarce there is little to win", fontsize=10)
b.legend(fontsize=8, loc="upper left"); b.grid(alpha=.25)

fig.suptitle("Scarce AVs saturate: the optimum fills every AV, so any sensible heuristic matches it. "
             "Spatial problem, time constraints off.", fontsize=9)
fig.tight_layout()
out = "analysis/figures/ila_vs_optimum_by_ratio.png"
fig.savefig(out, dpi=160)
print("wrote", out, f"({len(rows)} proven-optimal instances)")
