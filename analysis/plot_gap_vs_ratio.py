"""ILA - greedy saving gap as a function of the AV/PV ratio.

Re-plots the existing pv_av_sweep results (no new experiments).
greedy and hungarian(ILA) were run on identical instances (same seeds
42-45, same parameters), so the comparison is paired per instance.
"""
import csv, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

load = lambda p: list(csv.DictReader(open(p)))
G_ROWS = load("data/results/greedy/pv_av_sweep.csv")
H_ROWS = load("data/results/hungarian/pv_av_sweep.csv")

key = lambda r: (r["scenario_type"], r["fixed_value"], r["num_av"], r["num_pv"], r["seed"])
G = {key(r): r for r in G_ROWS}
H = {key(r): r for r in H_ROWS}

# paired per-seed difference, grouped by (num_av, num_pv)
cells = {}
for k in set(G) & set(H):
    c = (int(k[2]), int(k[3]))
    cells.setdefault(c, []).append(
        float(H[k]["saving_percent"]) - float(G[k]["saving_percent"])
    )

pts = sorted(
    ((av / pv, st.mean(d), st.stdev(d) if len(d) > 1 else 0.0, av, pv)
     for (av, pv), d in cells.items()),
    key=lambda t: t[0],
)

fig, ax = plt.subplots(figsize=(7.2, 4.4))

# shade the range proposed in Revision Plan item 2 (PV 400 vs AV 100/50/25/10)
ax.axvspan(0.02, 0.26, color="#d94801", alpha=0.10, zorder=0)
ax.text(0.075, 1.38, "Revision Plan item 2\nproposed sweep\n(PV 400, AV 100-10)",
        ha="center", va="top", fontsize=8, color="#8c2d04")

x = [p[0] for p in pts]
y = [p[1] for p in pts]
e = [p[2] for p in pts]
ax.errorbar(x, y, yerr=e, fmt="o", ms=5, lw=0, elinewidth=1,
            capsize=2.5, color="#08519c", ecolor="#9ecae1", zorder=3)

# trend: mean gap per distinct ratio
by_ratio = {}
for r, m, _s, _a, _p in pts:
    by_ratio.setdefault(round(r, 4), []).append(m)
tr = sorted(by_ratio)
ax.plot(tr, [st.mean(by_ratio[r]) for r in tr], "-", lw=1.4,
        color="#08519c", alpha=0.55, zorder=2)

ax.axhline(0, color="0.65", lw=0.8, zorder=1)
ax.set_xscale("log")
ax.set_xticks([0.01, 0.03, 0.1, 0.3, 0.8, 1.6, 3.2, 6.4])
ax.set_xticklabels(["0.01", "0.03", "0.1", "0.3", "0.8", "1.6", "3.2", "6.4"])
ax.set_xlabel("Active-to-passive vehicle ratio  (|AV| / |PV|)")
ax.set_ylabel("ILA $-$ greedy   (percentage points)")
ax.set_title("Where ILA actually beats greedy\n"
             "paired difference in energy saving, existing 1D uniform data (4 seeds)",
             fontsize=10.5)

peak = max(pts, key=lambda p: p[1])
ax.annotate(f"peak {peak[1]:.2f} pp\n(AV {peak[3]} / PV {peak[4]})",
            xy=(peak[0], peak[1]), xytext=(1.5, 1.30), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.8, color="0.4"))

ax.grid(alpha=0.25, lw=0.6)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("gap_vs_ratio.png", dpi=190)
print(f"points: {len(pts)}   peak {peak[1]:.3f} pp at ratio {peak[0]:.2f}")
print("gap in proposed range (ratio <= 0.26): "
      f"{[f'{p[1]:.3f}' for p in pts if p[0] <= 0.26]}")
