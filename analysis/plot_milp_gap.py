"""How far are greedy and ILA from the exact optimum?

Reads data/results/milp/exact_comparison.csv (written by milp.collect_results)
and draws two panels:

  left   what fraction of the optimum each heuristic reaches, per instance
  right  where that shortfall comes from, split by value rather than by segment
         count: how much is lost because the algorithms cannot express certain
         segment shapes, and how much because they choose badly among the shapes
         they can

The split comes from solving each instance twice, once unrestricted and once
with every segment forced to run to the end of its overlap - the only shape
greedy and ILA ever commit (milp/restricted.py).
"""
from __future__ import annotations

import csv
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GREEDY, ILA, INK = "#08519c", "#d94801", "#333333"

rows = [r for r in csv.DictReader(open("data/results/milp/exact_comparison.csv"))
        if r["proven_optimal"] == "True"]
if not rows:
    raise SystemExit("no proven-optimal rows in exact_comparison.csv")

sizes, by_size = [], defaultdict(list)
for r in rows:
    key = (int(r["num_av"]), int(r["num_pv"]))
    if key not in by_size:
        sizes.append(key)
    by_size[key].append(r)
sizes.sort()

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.5),
                              gridspec_kw={"width_ratios": [1.55, 1]})

# ---- left: distance from the optimum -----------------------------------
labels = [f"{a}/{p}" for a, p in sizes]
for k, key in enumerate(sizes):
    g = [float(r["greedy_pct"]) for r in by_size[key]]
    h = [float(r["ila_pct"]) for r in by_size[key]]
    ax.scatter([k - 0.13] * len(g), g, s=42, color=GREEDY, zorder=3,
               label="greedy" if k == 0 else None)
    ax.scatter([k + 0.13] * len(h), h, s=42, color=ILA, marker="^", zorder=3,
               label="ILA" if k == 0 else None)

gm = st.mean(float(r["greedy_pct"]) for r in rows)
hm = st.mean(float(r["ila_pct"]) for r in rows)
ax.axhline(100, color=INK, lw=1.1, zorder=2)
# the two means differ by less than 0.3 pp, so label them together
ax.axhspan(min(gm, hm), max(gm, hm), color=INK, alpha=0.10, zorder=1)
ax.axhline(gm, color=GREEDY, lw=1, ls="--", alpha=0.55, zorder=1)
ax.axhline(hm, color=ILA, lw=1, ls="--", alpha=0.55, zorder=1)
ax.text(len(sizes) - 0.45, 100.3, "exact optimum", fontsize=8.5, color=INK,
        ha="right")
ax.text(-0.45, (gm + hm) / 2 - 1.4,
        f"means: greedy {gm:.1f}%, ILA {hm:.1f}%", fontsize=8.5, color=INK)

ax.set_xticks(range(len(sizes)))
ax.set_xticklabels(labels)
ax.set_xlabel("instance size  (|AV| / |PV|)")
ax.set_ylabel("% of the exact optimum")
ax.set_title("Neither heuristic gets close to optimal", fontsize=11)
ax.set_xlim(-0.6, len(sizes) - 0.4)
ax.legend(frameon=False, loc="lower right", fontsize=9)
ax.grid(axis="y", alpha=0.25, lw=0.6)
ax.spines[["top", "right"]].set_visible(False)

# ---- right: where the shortfall comes from, by value ------------------
res = defaultdict(list)
for r in csv.DictReader(open("data/results/milp/restricted.csv")):
    res[(int(r["num_av"]), int(r["num_pv"]))].append(r)

rsizes = [k for k in sizes if k in res]
rlabels = [f"{a}/{p}" for a, p in rsizes]
shape_loss, choice_loss = [], []
for key in rsizes:
    v = res[key]
    shape_loss.append(st.mean(100 - 100 * float(r["opt_suffix"]) / float(r["opt_any"]) for r in v))
    choice_loss.append(st.mean(100 - 100 * float(r["greedy"]) / float(r["opt_suffix"]) for r in v))

xs = range(len(rsizes))
ax2.bar(xs, choice_loss, color="#08519c", width=0.62, zorder=3,
        label="choosing badly among available moves")
ax2.bar(xs, shape_loss, bottom=choice_loss, color="#fdae6b", width=0.62, zorder=3,
        label="segment shapes it cannot express")
allr = [r for v in res.values() for r in v]
tot_shape = st.mean(100 - 100 * float(r["opt_suffix"]) / float(r["opt_any"]) for r in allr)
tot_choice = st.mean(100 - 100 * float(r["greedy"]) / float(r["opt_suffix"]) for r in allr)
ax2.text(0.98, 0.04,
         f"overall {tot_choice:.1f}% choice  +  {tot_shape:.1f}% shape",
         transform=ax2.transAxes, ha="right", va="bottom", fontsize=9, color=INK)
ax2.set_xticks(list(xs))
ax2.set_xticklabels(rlabels)
ax2.set_xlabel("instance size  (|AV| / |PV|)")
ax2.set_ylabel("greedy's shortfall from optimal (%)")
ax2.set_title("Almost all of it is choice, not expressiveness", fontsize=11)
ax2.legend(frameon=False, fontsize=8.5, loc="upper right")
ax2.set_ylim(top=max(a + b for a, b in zip(choice_loss, shape_loss)) * 1.42)
ax2.grid(axis="y", alpha=0.25, lw=0.6)
ax2.spines[["top", "right"]].set_visible(False)

fig.suptitle("Exact optimum vs. greedy and ILA - time-free setting, "
             f"{len(rows)} instances solved to proven optimality",
             fontsize=12, y=1.005)
fig.tight_layout()
fig.savefig("analysis/figures/milp_gap.png", dpi=190, bbox_inches="tight")
print(f"greedy mean {gm:.2f}%  ILA mean {hm:.2f}%  "
      f"| shortfall: choice {tot_choice:.2f}%, shape {tot_shape:.2f}%")
print("wrote analysis/figures/milp_gap.png")
