"""How close ILA gets to the exact optimum, as a function of the AV:PV ratio.

Answers the question Eunus set on 18 Sep: drop greedy, and measure ILA against
the MILP optimum, especially where AVs are scarce.

Data (milp/lowratio_sweep.py, milp/m1_lowratio.py):
  synthetic uniform instances, PV = 50 and 100, 10 seeds per ratio
  M1 inbound corridor, 5-minute windows (sumo/convert.py D6), 12 windows
Spatial problem only -- milp.exact does not model temporal coupling.
Rows where optimality was not proven use the LP bound, so their ILA % is a
LOWER bound on the true ratio; they are drawn hollow.

Run:  PYTHONPATH=. venv/bin/python analysis/plot_lowratio.py
"""
import csv, statistics as st, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = [
    ("synthetic, PV = 50", "data/results/milp/lowratio_50.csv", "#9ecae1", "o"),
    ("synthetic, PV = 100", "data/results/milp/lowratio_100.csv", "#08519c", "s"),
    ("M1 corridor, 5-min windows", "data/results/milp/m1_lowratio_300.csv", "#d62728", "^"),
]

def load(path):
    """Rows grouped by ratio, split into proven-optimal and LP-bound.

    Rows whose bound is infinite (the LP relaxation itself timed out) carry no
    information and are dropped.
    """
    try:
        rows = list(csv.DictReader(open(path)))
    except FileNotFoundError:
        return {}, {}
    ok, lp = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        if not r["ila_pct"] or r["optimum"] in ("inf", "nan"):
            continue
        (ok if r["proven"] == "True" else lp)[float(r["ratio"])].append(r)
    return ok, lp

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.5))
all_ratios = set()

for label, path, colour, marker in SERIES:
    ok, lp = load(path)
    if not ok and not lp:
        continue
    rs = sorted(ok)
    all_ratios |= set(rs) | set(lp)
    if rs:
        m = [st.mean(float(r["ila_pct"]) for r in ok[k]) for k in rs]
        e = [st.stdev([float(r["ila_pct"]) for r in ok[k]]) if len(ok[k]) > 1 else 0 for k in rs]
        n = [len(ok[k]) for k in rs]
        ax[0].errorbar(rs, m, yerr=e, fmt=marker + "-", ms=6, capsize=3, color=colour,
                       lw=1.6, label=f"{label}  (n = {min(n)}–{max(n)} proven)")
    # ratios measured only against the LP bound: the true % is at least this
    lp_only = sorted(k for k in lp if k not in ok)
    if lp_only:
        y = [st.mean(float(r["ila_pct"]) for r in lp[k]) for k in lp_only]
        ax[0].plot(lp_only, y, marker, ms=9, mfc="none", mec=colour, mew=1.5, ls="none")
        for x, yy in zip(lp_only, y):
            ax[0].annotate("", xy=(x, min(yy + 1.6, 100.4)), xytext=(x, yy),
                           arrowprops=dict(arrowstyle="->", color=colour, lw=1.1))
    cov_src = {**{k: v for k, v in lp.items()}, **{k: v for k, v in ok.items()}}
    cs = sorted(cov_src)
    ax[1].plot(cs, [st.mean(float(r["opt_coverage"]) for r in cov_src[k]) for k in cs],
               marker + "-", color=colour, lw=1.6, label=f"optimum — {label}")

a = ax[0]
a.axhline(100, color="0.55", lw=0.9, ls="--")
a.text(0.0105, 100.1, "exact optimum", fontsize=8, color="0.4")
a.set_ylim(92, 100.9)
a.set_title("Where AVs are scarce, ILA is already optimal", fontsize=10.5)
a.set_ylabel("ILA as % of the exact optimum")
a.legend(fontsize=8, loc="lower left", framealpha=.92)

b = ax[1]
b.set_title("…because there is almost nothing to win there", fontsize=10.5)
b.set_ylabel("% of total PV distance towed by the optimum")
b.legend(fontsize=8, loc="upper left")

ticks = sorted(all_ratios)
for x in ax:
    x.set_xscale("log")
    x.set_xticks(ticks)
    x.set_xticklabels([f"{t:.0%}" if t < 0.1 else f"{t:.0%}" for t in ticks], fontsize=8)
    x.set_xlabel("AV : PV ratio")
    x.grid(alpha=.25)
    x.minorticks_off()

fig.suptitle("Scarce AVs run at full capacity, so the optimum has no freedom left and any sensible heuristic matches it.\n"
             "Spatial problem, time constraints off.  Hollow marker with arrow: optimality not proven, "
             "measured against the LP bound, so the true value is at least that high.", fontsize=8.5)
fig.tight_layout()
out = "analysis/figures/ila_vs_optimum_by_ratio.png"
fig.savefig(out, dpi=160)
print("wrote", out)
