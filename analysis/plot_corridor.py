"""Schematic of the M1 inbound corridor on the 100-unit grid.

Ramp positions, lane count per edge and the interchange-free stretch, drawn
from sumo/m1/corridor.json.  No simulation data involved.

Run:  PYTHONPATH=. venv/bin/python analysis/plot_corridor.py
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

c = json.load(open("sumo/m1/corridor.json"))
L, G = c["length_m"], c["grid"]
f = G / L

fig, ax = plt.subplots(figsize=(12, 4.4))

# lane count as a step band
for e in c["chain"]:
    ax.fill_between([e["from_m"] * f, e["to_m"] * f], 0, e["lanes"],
                    color="#d0d7de", step="post", linewidth=0)
ax.plot([0, G], [0, 0], color="k", lw=2)
ax.set_ylim(-3.2, 7.2)
ax.set_yticks([3, 4, 5, 6]); ax.set_ylabel("lanes")

# ramps
for r in c["ramps"]:
    x = r["grid"]
    if r["kind"] == "ON":
        ax.annotate("", xy=(x, 0), xytext=(x, -2.2),
                    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5))
        ax.text(x, -2.5, f"{x:.1f}", ha="center", va="top", fontsize=7, color="#d62728")
    else:
        ax.annotate("", xy=(x, -2.2), xytext=(x, 0),
                    arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5))
        name = r["name"].replace(" Off Ramp", "").replace(" Ramp Of", "")
        ax.text(x, 7.3, name, rotation=60, ha="left", va="bottom", fontsize=7, color="#1f77b4")
        ax.text(x, -2.5, f"{x:.1f}", ha="center", va="top", fontsize=7, color="#1f77b4")

# scarcity zone: longest single edge
longest = max(c["chain"], key=lambda e: e["to_m"] - e["from_m"])
a, b = longest["from_m"] * f, longest["to_m"] * f
ax.axvspan(a, b, color="#ffe680", alpha=.5, zorder=0)
ax.text((a + b) / 2, 1.2, f"no junction\n{(longest['to_m']-longest['from_m'])/1000:.1f} km",
        ha="center", fontsize=8)

ax.set_xlim(-1, 101)
ax.set_xlabel(f"grid units  (1 unit = {L/G:.0f} m;  0 = Springvale Rd, 100 = CityLink, {L/1000:.1f} km)")
ax.set_title("M1 Monash Freeway inbound — the 1D corridor as the matching algorithms see it\n"
             "red ↑ on-ramp (entry)   blue ↓ off-ramp (exit)   grey = lane count   yellow = no junction",
             fontsize=9, pad=70)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = "analysis/figures/m1_corridor.png"
fig.savefig(out, dpi=150)
print("wrote", out)
