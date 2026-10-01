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

  6. Sub-instance extraction (D6): slice the hour into time windows so that
     instances are small enough for the exact MILP, while keeping the real
     ramp geometry (9 entry / 9 exit points) of the full corridor.

Usage
  PYTHONPATH=. venv/bin/python sumo/convert.py sumo/m1 --summary
  PYTHONPATH=. venv/bin/python sumo/convert.py sumo/m1 --windows 120 300 600
  from sumo.convert import load_corridor_trips, label, extract_instance
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

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


# ---------------------------------------------------------------------------
# D6 — sub-instance extraction
#
# The exact MILP (milp/exact.py) proves optimality at AV 15 / PV 30 in ~142 s
# and fails within 60 s at AV 20 / PV 40, while one simulated hour on the M1
# yields ~1750 fitted trips.  Slicing by entry time keeps the corridor geometry
# (same 9 entry / 9 exit points, same ramp spacing, same speed distribution)
# and only reduces how many vehicles are present at once.
# ---------------------------------------------------------------------------


class InstanceTooSmall(ValueError):
    """Raised when a window cannot supply the requested AV:PV ratio."""


def usable_trips(corr: dict, trips: Sequence[CorridorTrip],
                 *, l_min: int = L_MIN) -> List[CorridorTrip]:
    """Trips whose grid-projected length is at least l_min (same filter as label())."""
    L, G = corr["length_m"], corr["grid"]
    return [t for t in trips if (lambda g: g[1] - g[0] >= l_min)(t.grid(L, G))]


def auto_time_unit(corr: dict, trips: Sequence[CorridorTrip],
                   *, l_min: int = L_MIN) -> float:
    """Seconds per model time unit such that mean speed = 1 grid unit / time unit.

    Compute this ONCE on the whole scenario and pass it to every window.
    Letting each window auto-scale itself would give each a different time
    unit, so tau = 5 would mean a different number of seconds per window and
    the windows would not be comparable.
    """
    L, G = corr["length_m"], corr["grid"]
    u = usable_trips(corr, trips, l_min=l_min)
    if not u:
        raise InstanceTooSmall("no trips pass the l_min filter")
    return (L / G) / statistics.fmean(t.speed_mps for t in u)


def window_trips(trips: Sequence[CorridorTrip], t0: float,
                 window_s: float) -> List[CorridorTrip]:
    """Trips whose fitted main-line entry time falls in [t0, t0 + window_s)."""
    return [t for t in trips if t0 <= t.entry_time_s < t0 + window_s]


def iter_windows(trips: Sequence[CorridorTrip], window_s: float, *,
                 begin: Optional[float] = None, end: Optional[float] = None,
                 ) -> Iterator[Tuple[float, List[CorridorTrip]]]:
    """Yield (t0, trips_in_window) over consecutive windows covering the run.

    begin/end default to the first and last entry time seen.  The final window
    is yielded even if it is only partly inside [begin, end), so callers that
    care about equal exposure should drop it.
    """
    if not trips:
        return
    lo = min(t.entry_time_s for t in trips) if begin is None else begin
    hi = max(t.entry_time_s for t in trips) if end is None else end
    t0 = lo
    while t0 < hi:
        yield t0, window_trips(trips, t0, window_s)
        t0 += window_s


def resolve_counts(n_pool: int, *, ratio: Optional[float] = None,
                   n_av: Optional[int] = None, n_pv: Optional[int] = None,
                   exact_ratio: bool = True) -> Tuple[int, int]:
    """Decide (n_av, n_pv) from a pool of n_pool usable trips.

    ratio means |AV| / |PV| (so 0.02 is "2 AV per 100 PV").

    * n_av and n_pv given      -> used as is.
    * ratio and n_pv given     -> n_av = round(ratio * n_pv).
    * ratio only, exact_ratio  -> the largest k >= 1 with k + round(k / ratio)
                                  <= n_pool; hits the ratio closely and
                                  discards the remainder.
    * ratio only, not exact    -> split the whole pool, n_av =
                                  round(n_pool * ratio / (1 + ratio)).  Uses
                                  every vehicle but the realised ratio can be
                                  far off at small ratios.

    Raises InstanceTooSmall with the required pool size when it cannot be met.
    """
    if n_av is not None and n_pv is not None:
        if n_av + n_pv > n_pool:
            raise InstanceTooSmall(
                f"asked for {n_av} AV + {n_pv} PV = {n_av + n_pv} vehicles, "
                f"pool has {n_pool}")
        return n_av, n_pv
    if ratio is None:
        raise ValueError("give ratio, or both n_av and n_pv")
    if ratio <= 0:
        raise ValueError("ratio must be > 0")

    if n_pv is not None:
        k = max(1, round(ratio * n_pv))
        if k + n_pv > n_pool:
            raise InstanceTooSmall(
                f"ratio {ratio:g} with {n_pv} PV needs {k + n_pv} vehicles, "
                f"pool has {n_pool}")
        return k, n_pv

    if not exact_ratio:
        k = max(1, round(n_pool * ratio / (1 + ratio)))
        if k >= n_pool:
            raise InstanceTooSmall(
                f"ratio {ratio:g} leaves no PV in a pool of {n_pool}")
        return k, n_pool - k

    need = 1 + round(1 / ratio)
    if n_pool < need:
        raise InstanceTooSmall(
            f"ratio {ratio:g} needs at least {need} usable trips "
            f"(1 AV + {round(1 / ratio)} PV), pool has {n_pool}")
    k = 1
    while (k + 1) + round((k + 1) / ratio) <= n_pool:
        k += 1
    return k, round(k / ratio)


def extract_instance(
    corr: dict,
    trips: Sequence[CorridorTrip],
    *,
    t0: float,
    window_s: float,
    ratio: Optional[float] = None,
    n_av: Optional[int] = None,
    n_pv: Optional[int] = None,
    exact_ratio: bool = True,
    capacity_range: Tuple[int, int] = (2, 4),
    seed: int = 42,
    l_min: int = L_MIN,
    time_unit_s: Optional[float] = None,
    rebase_time: bool = True,
) -> Tuple[List[ActiveVehicle], List[PassiveVehicle], int, dict]:
    """One MILP-sized instance from the window [t0, t0 + window_s).

    time_unit_s should come from auto_time_unit() on the FULL trip list so that
    every window shares one time scale; passing None keeps real seconds.

    rebase_time subtracts t0 so entry times start near zero.  Matching only
    ever compares time differences (hungarian_multi.get_overlap_with_av tests
    abs(pv_time - av_time) against the tolerance), so a constant shift does not
    change any result; it just keeps the numbers small and windows comparable.

    Returns (avs, pvs, l_min, info) — the same shapes label() returns, so this
    goes straight into milp.exact.solve() and
    core.hungarian_multi.hungarian_multi_av_matching().
    """
    pool = usable_trips(corr, window_trips(trips, t0, window_s), l_min=l_min)
    k_av, k_pv = resolve_counts(len(pool), ratio=ratio, n_av=n_av, n_pv=n_pv,
                                exact_ratio=exact_ratio)

    rng = random.Random(seed)
    chosen = pool[:]
    rng.shuffle(chosen)
    chosen = chosen[:k_av + k_pv]
    if rebase_time:
        chosen = [replace(t, entry_time_s=t.entry_time_s - t0) for t in chosen]

    avs, pvs, lm, info = label(
        corr, chosen, n_av=k_av, capacity_range=capacity_range, seed=seed,
        l_min=l_min, time_unit_s=time_unit_s)
    info.update(window_t0=t0, window_s=window_s, pool=len(pool),
                requested_ratio=ratio, realised_ratio=len(avs) / len(pvs) if pvs else None,
                discarded_from_pool=len(pool) - (k_av + k_pv),
                rebase_time=rebase_time)
    return avs, pvs, lm, info


def window_stats(corr: dict, trips: Sequence[CorridorTrip], window_s: float, *,
                 l_min: int = L_MIN,
                 ratios: Sequence[float] = (0.01, 0.02, 0.05, 0.10, 0.20),
                 drop_last: bool = True) -> dict:
    """Per-window counts and which of `ratios` each window can supply."""
    L, G = corr["length_m"], corr["grid"]
    rows = []
    for t0, w in iter_windows(trips, window_s):
        u = usable_trips(corr, w, l_min=l_min)
        grids = [t.grid(L, G) for t in u]
        ok = []
        for r in ratios:
            try:
                resolve_counts(len(u), ratio=r)
                ok.append(r)
            except InstanceTooSmall:
                pass
        rows.append(dict(t0=t0, n=len(w), usable=len(u),
                         od_pairs=len({g for g in grids}),
                         entries=len({e for e, _ in grids}),
                         exits=len({x for _, x in grids}),
                         ratios_ok=ok))
    if drop_last and len(rows) > 1:
        rows = rows[:-1]          # last window is usually a partial slice
    med = lambda key: statistics.median([r[key] for r in rows]) if rows else 0
    return dict(
        window_s=window_s, n_windows=len(rows),
        vehicles=dict(median=med("n"), min=min((r["n"] for r in rows), default=0),
                      max=max((r["n"] for r in rows), default=0)),
        usable=dict(median=med("usable"), min=min((r["usable"] for r in rows), default=0),
                    max=max((r["usable"] for r in rows), default=0)),
        od_pairs=dict(median=med("od_pairs"), min=min((r["od_pairs"] for r in rows), default=0),
                      max=max((r["od_pairs"] for r in rows), default=0)),
        entries_median=med("entries"), exits_median=med("exits"),
        windows_supporting={str(r): sum(r in row["ratios_ok"] for row in rows)
                            for r in ratios},
        rows=rows,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario_dir")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--dump", help="write trips json here")
    ap.add_argument("--windows", nargs="+", type=float, metavar="SEC",
                    help="report sub-instance stats for these window lengths")
    ap.add_argument("--windows-json", help="write the window report here")
    a = ap.parse_args()
    corr, trips = load_corridor_trips(a.scenario_dir)
    if a.summary:
        print(json.dumps(summary(corr, trips), indent=1))
    if a.dump:
        Path(a.dump).write_text(json.dumps([asdict(t) for t in trips], indent=0))
        print("wrote", a.dump)
    if a.windows:
        rep = {str(int(w)): window_stats(corr, trips, w) for w in a.windows}
        for w, r in rep.items():
            print(f"\n--- window {int(w) // 60} min ({w} s), {r['n_windows']} windows ---")
            print(f"  vehicles/window  median {r['vehicles']['median']:.0f} "
                  f"({r['vehicles']['min']}-{r['vehicles']['max']})")
            print(f"  usable (>=L_min) median {r['usable']['median']:.0f} "
                  f"({r['usable']['min']}-{r['usable']['max']})")
            print(f"  OD pairs         median {r['od_pairs']['median']:.0f} "
                  f"({r['od_pairs']['min']}-{r['od_pairs']['max']})")
            print(f"  entry/exit pts   median {r['entries_median']:.0f} / {r['exits_median']:.0f}")
            print("  windows supporting ratio: " + ", ".join(
                f"{float(k):.0%}={v}/{r['n_windows']}"
                for k, v in r["windows_supporting"].items()))
        if a.windows_json:
            Path(a.windows_json).write_text(json.dumps(rep, indent=1))
            print("\nwrote", a.windows_json)
