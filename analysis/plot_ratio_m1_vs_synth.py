"""ILA - greedy gap vs AV:PV ratio: synthetic uniform data vs first M1 run.

Synthetic: stored pv_av_sweep (time constraints ON, 4-8 seeds per cell), the
same data as plot_gap_vs_ratio.py.  M1: sumo/m1/ratio_sweep.csv (one seed,
random labels, randomTrips placeholder demand).  Same algorithms, same tau = 5
model units; only the instance distribution differs.

Run:  PYTHONPATH=. venv/bin/python analysis/plot_ratio_m1_vs_synth.py
"""
import csv, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

load = lambda p: list(csv.DictReader(open(p)))
G = load("data/results/greedy/pv_av_sweep.csv")
H = load("data/results/hungarian/pv_av_sweep.csv")
key = lambda r: (r["scenario_type"], r["fixed_value"], r["num_av"], r["num_pv"], r["seed"])
Gd, Hd = {key(r): r for r in G}, {key(r): r for r in H}
cells = {}
for k in set(Gd) & set(Hd):
    c = (int(k[2]), int(k[3]))
    cells.setdefault(c, []).append((float(Gd[k]["saving_percent"]), float(Hd[k]["saving_percent"])))
syn = sorted((av / pv, st.mean(h - g for g, h in v), st.mean(g for g, _ in v),
              st.mean(h for _, h in v), av, pv, av + pv) for (av, pv), v in cells.items())

M = load("sumo/m1/ratio_sweep.csv")
m1 = {}
for r in M:
    m1.setdefault(float(r["ratio"]), {})[(r["algorithm"], r["time_on"] == "True")] = float(r["saving_percent"])
mr = sorted(m1)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))

# --- left: gap
a = ax[0]
a.scatter([s[0] for s in syn], [s[1] for s in syn], s=[s[6] / 12 for s in syn],
          color="#9ecae1", edgecolor="#3182bd", lw=0.6, label="synthetic uniform (per cell, size = #vehicles)", zorder=2)
by = {}
for s in syn:
    by.setdefault(round(s[0], 3), []).append(s[1])
a.plot(sorted(by), [st.mean(by[r]) for r in sorted(by)], "-", color="#3182bd", lw=1.2, alpha=.7, label="synthetic mean per ratio")
a.plot(mr, [m1[r][("ILA", True)] - m1[r][("greedy", True)] for r in mr], "o-", color="#d62728", ms=7, lw=1.8,
       label="M1 SUMO, time ON (1 seed, placeholder demand)", zorder=4)
a.plot(mr, [m1[r][("ILA", False)] - m1[r][("greedy", False)] for r in mr], "s--", color="#d62728", ms=5, lw=1, alpha=.5,
       label="M1 SUMO, time OFF", zorder=3)
a.axhline(0, color="0.6", lw=.8)
a.set_xscale("log"); a.set_xticks([0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]); a.set_xticklabels(["0.05", "0.1", "0.2", "0.4", "0.8", "1.6", "3.2"])
a.set_xlim(0.04, 12); a.set_ylim(-0.15, 2.0)
a.set_xlabel("|AV| / |PV|"); a.set_ylabel("ILA − greedy  (pp)")
a.set_title("Where ILA beats greedy: synthetic vs M1", fontsize=10)
a.legend(fontsize=7.5, loc="upper right"); a.grid(alpha=.25)

# --- right: saving level (largest synthetic cells only, closest in size to M1's 1429)
b = ax[1]
big = [s for s in syn if s[6] >= 1000]
b.plot([s[0] for s in big], [s[2] for s in big], "o", color="#9ecae1", mec="#3182bd", label="synthetic greedy (cells ≥ 1000 vehicles)")
b.plot([s[0] for s in big], [s[3] for s in big], "^", color="#3182bd", ms=5, label="synthetic ILA")
b.plot(mr, [m1[r][("greedy", True)] for r in mr], "o-", color="#f4a582", mec="#d62728", label="M1 greedy, time ON")
b.plot(mr, [m1[r][("ILA", True)] for r in mr], "^-", color="#d62728", ms=6, label="M1 ILA, time ON")
b.plot(mr, [m1[r][("ILA", False)] for r in mr], "--", color="#d62728", alpha=.4, label="M1 both, time OFF")
b.set_xscale("log"); b.set_xticks([0.1, 0.2, 0.4, 0.8, 1.6]); b.set_xticklabels(["0.1", "0.2", "0.4", "0.8", "1.6"])
b.set_xlim(0.08, 2)
b.set_xlabel("|AV| / |PV|"); b.set_ylabel("energy saving  (% of AV+PV distance)")
b.set_title("Saving level: M1 instances save more at every ratio", fontsize=10)
b.legend(fontsize=7.5, loc="lower right"); b.grid(alpha=.25)

fig.suptitle("Same algorithms, same τ = 5 model units — only the instance distribution differs. "
             "M1 = one seed on randomTrips placeholder demand.", fontsize=9)
fig.tight_layout()
out = "analysis/figures/ratio_m1_vs_synth.png"
fig.savefig(out, dpi=160)
print("wrote", out)
