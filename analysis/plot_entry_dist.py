"""Synthetic (uniform) vs SUMO-M1 instance structure.

Left column: synthetic generate_mock_data(80 AV, 400 PV, L=100, L_min=10,
seed 42, time on) -- the paper's main configuration.
Right column: M1 inbound corridor, randomTrips placeholder demand
(sumo/m1, 1 h).  Demand is NOT calibrated; only the network is real.
The shapes shown depend on the network (ramp positions), not on the demand
model, except the entry-time panel which is uniform in both by construction.

Run:  PYTHONPATH=. venv/bin/python analysis/plot_entry_dist.py
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core.data import generate_mock_data, L_MIN
from sumo.convert import load_corridor_trips

avs, pvs, _ = generate_mock_data(80, 400, 100, (2, 4), L_MIN, seed=42,
                                 enable_time_constraints=True,
                                 av_speed_range=(0.8, 1.2), pv_speed_range=(0.8, 1.2))
syn = avs + pvs
corr, trips = load_corridor_trips("sumo/m1")
L, G = corr["length_m"], corr["grid"]
m1 = [t.grid(L, G) for t in trips]
ramps = corr["ramps"]

fig, ax = plt.subplots(3, 2, figsize=(11, 8.5))
bins = range(0, 102, 2)

# row 0: entry position
ax[0, 0].hist([v.entry_point for v in syn], bins=bins, color="#888")
ax[0, 1].hist([e for e, _ in m1], bins=bins, color="#1f77b4")
for r in ramps:
    ax[0, 1].axvline(r["grid"], color="#d62728" if r["kind"] == "ON" else "#bbb",
                     lw=0.8, ls="--", alpha=.8)
ax[0, 0].set_title(f"Synthetic  (n={len(syn)}, 101 possible points)")
ax[0, 1].set_title(f"M1 inbound, SUMO  (n={len(m1)}, 9 possible points)")
ax[0, 0].set_ylabel("entry position\ncount")

# row 1: exit position
ax[1, 0].hist([v.exit_point for v in syn], bins=bins, color="#888")
ax[1, 1].hist([x for _, x in m1], bins=bins, color="#1f77b4")
for r in ramps:
    ax[1, 1].axvline(r["grid"], color="#d62728" if r["kind"] == "OFF" else "#bbb",
                     lw=0.8, ls="--", alpha=.8)
ax[1, 0].set_ylabel("exit position\ncount")

# row 2: trip length
lb = range(0, 102, 2)
ax[2, 0].hist([v.exit_point - v.entry_point for v in syn], bins=lb, color="#888")
ax[2, 1].hist([x - e for e, x in m1], bins=lb, color="#1f77b4")
for a in ax[2]:
    a.axvline(L_MIN, color="k", lw=1, ls=":")
    a.text(L_MIN + 1, a.get_ylim()[1] * .9, f"L_min={L_MIN}", fontsize=8)
ax[2, 0].set_ylabel("trip length\ncount")
short = sum(x - e < L_MIN for e, x in m1)
ax[2, 1].text(0.98, 0.9, f"{short}/{len(m1)} = {short/len(m1):.0%} below L_min",
              transform=ax[2, 1].transAxes, ha="right", fontsize=9)

for a in ax[:, 0]:
    a.set_xlim(0, 100)
for a in ax[:, 1]:
    a.set_xlim(0, 100)
ax[2, 0].set_xlabel("grid units (0-100)")
ax[2, 1].set_xlabel("grid units (0-100 = 20.5 km, Springvale Rd -> CityLink)")
fig.suptitle("Instance structure: uniform synthetic vs real M1 corridor "
             "(dashed = on-ramps red / off-ramps grey; demand = randomTrips placeholder)",
             fontsize=10)
fig.tight_layout()
out = "analysis/figures/entry_dist_synth_vs_m1.png"
fig.savefig(out, dpi=150)
print("wrote", out)
