"""SUMO (M1 corridor) -> ActiveVehicle / PassiveVehicle lists.

Pipeline position:  sumo -> routes.xml + fcd.xml -> [this] -> core matching.

What it does
  1. For every vehicle whose route uses the inbound main-line chain
     (sumo/m1/corridor.json), take entry/exit metres from the first/last
     chain edge of its route.
  2. From FCD samples on chain edges, fit position = a + b*t.  Slope b is the
     constant speed the 1D model assumes (D2); the residuals are the
     linearisation error the model ignores.  entry_time is the fitted time at
     the entry position.
  3. Project metres to the 100-unit grid (D5) and round to int, because the
     core models use integer points.
  4. Label AV / PV.  SUMO has no such concept; the rule is ours.  Only
     `random` is implemented today (placeholder until the labelling rule is
     decided).
  5. Optionally rescale time so that mean speed = 1 grid/time-unit, which
     matches the synthetic generator (speed 0.8-1.2, tau = 5).

Usage
  PYTHONPATH=. venv/bin/python sumo/convert.py sumo/m1 --summary
  from sumo.convert import load_corridor_trips, label
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from core.models import ActiveVehicle, PassiveVehicle
from core.data import L_MIN


@dataclass
class CorridorTrip:
    id: str
    depart_s: float          # SUMO depart (at ramp stub), for reference
    entry_m: float           # main-line entry, metres from corridor start
    exit_m: float
    entry_time_s: float      # fitted time at entry_m
    speed_mps: float         # fitted constant speed
    n_samples: int
    max_resid_s: float       # max |t_obs - t_fit| over FCD samples
    rms_resid_s: float

    def grid(self, length_m: float, grid: int) -> Tuple[int, int]:
        f = grid / length_m
        return round(self.entry_m * f), round(self.exit_m * f)


def _fit(ts: Sequence[float], xs: Sequence[float]) -> Tuple[float, float]:
    """Least squares x = a + b t; returns (a, b)."""
    n = len(ts)
    mt, mx = sum(ts) / n, sum(xs) / n
    sxx = sum((t - mt) ** 2 for t in ts)
    if sxx == 0:
        return mx, 0.0
    b = sum((t - mt) * (x - mx) for t, x in zip(ts, xs)) / sxx
    return mx - b * mt, b


def load_corridor_trips(scenario_dir: Path | str) -> Tuple[dict, List[CorridorTrip]]:
    d = Path(scenario_dir)
    corr = json.loads((d / "corridor.json").read_text())
    chain = {e["id"]: e for e in corr["chain"]}

    # route -> entry/exit metres
    span: Dict[str, Tuple[float, float, float]] = {}
    for v in ET.parse(d / "routes.xml").getroot().iter("vehicle"):
        el = [e for e in v.find("route").get("edges").split() if e in chain]
        if not el:
            continue
        span[v.get("id")] = (float(v.get("depart")),
                             chain[el[0]]["from_m"], chain[el[-1]]["to_m"])

    # FCD -> samples on chain edges
    samples: Dict[str, List[Tuple[float, float]]] = {vid: [] for vid in span}
    for _, ts in ET.iterparse(d / "fcd.xml", events=("end",)):
        if ts.tag != "timestep":
            continue
        t = float(ts.get("time"))
        for veh in ts:
            vid = veh.get("id")
            if vid not in samples:
                continue
            lane = veh.get("lane")
            if lane.startswith(":"):
                continue
            edge = lane.rsplit("_", 1)[0]
            if edge in chain:
                samples[vid].append((t, chain[edge]["from_m"] + float(veh.get("pos"))))
        ts.clear()

    trips: List[CorridorTrip] = []
    for vid, (dep, em, xm) in span.items():
        s = samples[vid]
        if len(s) < 2:
            continue  # too short on the main line to fit (kept count below)
        ts_, xs_ = zip(*s)
        a, b = _fit(ts_, xs_)
        if b <= 0:
            continue
        resid = [t - (x - a) / b for t, x in s]
        trips.append(CorridorTrip(
            id=vid, depart_s=dep, entry_m=em, exit_m=xm,
            entry_time_s=(em - a) / b, speed_mps=b, n_samples=len(s),
            max_resid_s=max(abs(r) for r in resid),
            rms_resid_s=(sum(r * r for r in resid) / len(resid)) ** 0.5,
        ))
    corr["_n_route_users"] = len(span)
    return corr, trips


def label(
    corr: dict,
    trips: List[CorridorTrip],
    *,
    n_av: Optional[int] = None,
    av_fraction: Optional[float] = None,
    capacity_range: Tuple[int, int] = (2, 4),
    seed: int = 42,
    rule: str = "random",
    l_min: int = L_MIN,
    time_unit_s: Optional[float] = None,
) -> Tuple[List[ActiveVehicle], List[PassiveVehicle], int, dict]:
    """Split trips into AV / PV lists in the core model format.

    time_unit_s: seconds per model time unit. None -> seconds (speed in
    grid/s). "auto" via time_unit_s=0 -> choose so mean speed = 1.0.
    Returns (avs, pvs, l_min, info).
    """
    if rule != "random":
        raise NotImplementedError("only rule='random' exists until the labelling rule is decided")
    L, G = corr["length_m"], corr["grid"]
    usable = [t for t in trips if (lambda g: g[1] - g[0] >= l_min)(t.grid(L, G))]
    dropped_short = len(trips) - len(usable)

    if time_unit_s == 0:
        mean_mps = statistics.fmean(t.speed_mps for t in usable)
        time_unit_s = (L / G) / mean_mps      # seconds to cover one grid unit
    tu = time_unit_s or 1.0

    rng = random.Random(seed)
    order = usable[:]
    rng.shuffle(order)
    if n_av is None:
        n_av = round(len(order) * (av_fraction if av_fraction is not None else 0.2))
    avs, pvs = [], []
    for k, t in enumerate(order):
        e, x = t.grid(L, G)
        speed = t.speed_mps * tu / (L / G)      # grid units per time unit
        entry_time = t.entry_time_s / tu
        if k < n_av:
            avs.append(ActiveVehicle(id=f"AV{t.id}", entry_point=e, exit_point=x,
                                     capacity=rng.randint(*capacity_range),
                                     entry_time=entry_time, speed=speed))
        else:
            pvs.append(PassiveVehicle(id=f"PV{t.id}", entry_point=e, exit_point=x,
                                      entry_time=entry_time, speed=speed))
    info = dict(n_trips=len(trips), dropped_short=dropped_short, n_av=len(avs),
                n_pv=len(pvs), time_unit_s=tu, rule=rule, seed=seed)
    return avs, pvs, l_min, info


def summary(corr: dict, trips: List[CorridorTrip]) -> dict:
    L, G = corr["length_m"], corr["grid"]
    grids = [t.grid(L, G) for t in trips]
    lens = [x - e for e, x in grids]
    ods = {g for g in grids}
    return dict(
        route_users=corr["_n_route_users"], fitted=len(trips),
        distinct_entry_points=len({e for e, _ in grids}),
        distinct_exit_points=len({x for _, x in grids}),
        distinct_od_pairs=len(ods),
        trips_shorter_than_lmin=sum(l < L_MIN for l in lens),
        trip_len_grid=dict(min=min(lens), median=statistics.median(lens), max=max(lens)),
        speed_kmh=dict(mean=statistics.fmean(t.speed_mps for t in trips) * 3.6,
                       min=min(t.speed_mps for t in trips) * 3.6,
                       max=max(t.speed_mps for t in trips) * 3.6),
        entry_time_s=dict(min=min(t.entry_time_s for t in trips),
                          max=max(t.entry_time_s for t in trips)),
        linearisation_resid_s=dict(
            max_over_vehicles=max(t.max_resid_s for t in trips),
            median_of_max=statistics.median(t.max_resid_s for t in trips),
            p90_of_max=sorted(t.max_resid_s for t in trips)[int(0.9 * len(trips))],
            share_max_gt_5s=sum(t.max_resid_s > 5 for t in trips) / len(trips),
            median_rms=statistics.median(t.rms_resid_s for t in trips)),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario_dir")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--dump", help="write trips json here")
    a = ap.parse_args()
    corr, trips = load_corridor_trips(a.scenario_dir)
    if a.summary:
        print(json.dumps(summary(corr, trips), indent=1))
    if a.dump:
        Path(a.dump).write_text(json.dumps([asdict(t) for t in trips], indent=0))
        print("wrote", a.dump)
