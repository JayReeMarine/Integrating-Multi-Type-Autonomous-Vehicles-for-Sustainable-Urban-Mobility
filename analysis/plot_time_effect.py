"""Where does ILA's advantage over greedy come from?

The paper attributes it to globally coordinated *spatial* assignment. This plot
compares the advantage with the temporal synchronisation constraint switched off
and on, at the paper's own configurations and at the ratio where the gap is
largest. Reads data/results/milp/time_effect.csv.
"""
from __future__ import annotations

import csv
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OFF, ON, INK = "#9ecae1", "#08519c", "#333333"

rows = list(csv.DictReader(open("data/results/milp/time_effect.csv")))
adv = defaultdict(list)
for r in rows:
    gain = 100 * (float(r["ila"]) - float(r["greedy"])) / float(r["greedy"])
    adv[(r["config"], r["time_constraints"] == "True")].append(gain)

configs, seen = [], set()
for r in rows:
    if r["config"] not in seen:
        seen.add(r["config"])
        configs.append((r["config"], float(r["ratio"])))

fig, ax = plt.subplots(figsize=(8.2, 4.9))
x = range(len(configs))
w = 0.36
off = [st.mean(adv[(c, False)]) for c, _ in configs]
on = [st.mean(adv[(c, True)]) for c, _ in configs]
off_e = [st.stdev(adv[(c, False)]) for c, _ in configs]
on_e = [st.stdev(adv[(c, True)]) for c, _ in configs]

ax.bar([i - w / 2 for i in x], off, w, yerr=off_e, capsize=3, color=OFF,
       label="temporal constraint OFF", zorder=3)
ax.bar([i + w / 2 for i in x], on, w, yerr=on_e, capsize=3, color=ON,
       label="temporal constraint ON", zorder=3)

for i, (a, b, ae, be) in enumerate(zip(off, on, off_e, on_e)):
    ax.text(i - w / 2, a + ae + 0.09, f"{a:.2f}", ha="center", fontsize=8.5, color=INK)
    ax.text(i + w / 2, b + be + 0.09, f"{b:.2f}", ha="center", fontsize=8.5, color=INK)

ax.axhline(0, color=INK, lw=0.9)
ax.set_xticks(list(x))
ax.set_xticklabels([f"{c}\n{a}:{p}" for (c, _), (a, p) in
                    zip(configs, [(50, 200), (80, 400), (160, 200), (320, 400)])],
                   fontsize=9)
ax.set_ylabel("ILA advantage over greedy\n(% of towed distance)")
ax.set_title("ILA's advantage over greedy is larger when the temporal\n"
             "constraint is active - in all four configurations", fontsize=11)
ax.legend(frameon=False, fontsize=9, loc="upper left",
          bbox_to_anchor=(0.0, -0.16), ncol=2)
ax.set_ylim(top=max(on) + max(on_e) + 0.45)
fig.text(0.5, -0.02,
         "Bars are means over 4 seeds; whiskers are $\\pm$1 s.d. The ON and OFF "
         "intervals overlap within each configuration,\nso the pattern rests on it "
         "holding in all four rather than on any single one. More seeds (item #5) "
         "would settle it.",
         ha="center", va="top", fontsize=7.8, color="#666666", linespacing=1.6)
ax.grid(axis="y", alpha=0.25, lw=0.6)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("analysis/figures/time_effect.png", dpi=190, bbox_inches="tight")
print("wrote analysis/figures/time_effect.png")
