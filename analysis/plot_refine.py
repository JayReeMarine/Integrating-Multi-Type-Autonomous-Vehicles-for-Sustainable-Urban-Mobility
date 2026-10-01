"""ILA, and ILA plus ruin-and-recreate refinement, against the exact optimum.

Data: data/results/milp/refine_synthetic.csv, refine_m1.csv
(scored against the optima stored by milp/lowratio_sweep.py and
milp/m1_lowratio.py; no MILP is re-solved).  Spatial problem, time constraints
off.  Rows whose optimality was not proven are scored against the LP bound and
excluded here.

Run:  PYTHONPATH=. venv/bin/python analysis/plot_refine.py
"""
import csv, collections, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = [
    ("Synthetic uniform instances", "data/results/milp/refine_synthetic.csv", "pv"),
    ("M1 corridor, 5-minute windows", "data/results/milp/refine_m1.csv", None),
]

fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.5), sharey=True)

for panel, (title, path, split) in zip(ax, PANELS):
    try:
        rows = [r for r in csv.DictReader(open(path)) if r["proven"] == "True"]
    except FileNotFoundError:
        continue
    groups = collections.defaultdict(list)
    for r in rows:
        key = (int(r["pv"]) if split else 0, float(r["ratio"]))
        groups[key].append(r)
    for pv_size, colour, dash in ((50, "#9ecae1", ":"), (100, "#08519c", "-"), (0, "#08519c", "-")):
        ks = sorted(k for k in groups if k[0] == pv_size)
        if not ks:
            continue
        x = [k[1] for k in ks]
        ila = [st.mean(float(r["ila_pct"]) for r in groups[k]) for k in ks]
        ref = [st.mean(float(r["refined_pct"]) for r in groups[k]) for k in ks]
        tag = f"  (PV = {pv_size})" if pv_size else ""
        panel.plot(x, ila, "o" + dash, color=colour, lw=1.6, ms=6, alpha=.75,
                   label=f"ILA{tag}")
        panel.plot(x, ref, "^-", color="#d62728" if not pv_size else ("#fb6a4a" if pv_size == 50 else "#a50f15"),
                   lw=2.0, ms=7, label=f"ILA + refinement{tag}")
        panel.fill_between(x, ila, ref, color="#d62728", alpha=.07)
    panel.axhline(100, color="0.55", lw=0.9, ls="--")
    ticks = sorted({k[1] for k in groups})
    panel.set_xscale("log"); panel.set_xticks(ticks)
    panel.set_xticklabels([f"{t:.0%}" for t in ticks], fontsize=8); panel.minorticks_off()
    panel.set_xlabel("AV : PV ratio")
    panel.set_title(title, fontsize=10.5)
    panel.grid(alpha=.25)
    panel.legend(fontsize=8, loc="lower left")

ax[0].set_ylabel("% of the exact optimum")
ax[0].set_ylim(93, 100.6)
ax[0].text(0.0105, 100.15, "exact optimum", fontsize=8, color="0.4")
fig.suptitle("Ruin-and-recreate refinement closes most of the gap, and the most where ILA is weakest.\n"
             "Spatial problem, time constraints off; proven-optimal instances only.", fontsize=9)
fig.tight_layout()
out = "analysis/figures/refinement_vs_optimum.png"
fig.savefig(out, dpi=160)
print("wrote", out)
