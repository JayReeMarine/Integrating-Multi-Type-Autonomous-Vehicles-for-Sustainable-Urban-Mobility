"""Local-search refinement of a matching produced by greedy or ILA.

Why: against the exact optimum (see NOTES.md, 2026-10-01) ILA loses nothing when
AVs are scarce and 2-7 pp once AVs are plentiful, and the loss is entirely
*packing* -- the optimum keeps the AV capacity-distance ~100 % full, ILA leaves
some idle.  This module tries to recover that capacity without re-solving.

Scope: the spatial problem (no time constraints), matching milp.exact, so that
the result can be scored against a proven optimum.

Two things measured on 2026-10-01 shape the design:

* ILA stops when no feasible segment of length >= l_min remains, so the idle AV
  capacity it leaves sits entirely in runs *shorter* than l_min (measured: the
  longest idle run was 8 with l_min = 10). Nothing can simply be inserted.
* ILA always commits the whole residual overlap, so it uses few long segments
  (21 segments, mean length 36) where the optimum uses many short ones
  (53 segments, mean 15.5) and serves far more PVs (37 vs 21).

So the gap is not idle capacity to fill; it is capacity filled with the wrong
segments. The repair has to tear down and rebuild, which is what `refine_ils`
does (ruin and recreate): remove a contiguous stretch of the corridor from some
AVs, rebuild greedily, keep the result only if the total improved.

Moves
  fill    insert the longest feasible segment that is still available
  extend  lengthen an existing segment where capacity and PV coverage allow
  eject   drop one segment, re-fill greedily, keep only if the total improves
  ruin    (refine_ils) drop everything in a random road window, then rebuild
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from core.models import ActiveVehicle, PassiveVehicle
from core.hungarian_multi import SegmentAssignment


@dataclass
class _State:
    """Feasibility bookkeeping on the unit-interval grid [0, L)."""
    avs: Sequence[ActiveVehicle]
    pvs: Sequence[PassiveVehicle]
    l_min: int
    length: int
    av_load: Dict[str, List[int]] = field(default_factory=dict)   # PVs attached per interval
    pv_cover: Dict[str, List[int]] = field(default_factory=dict)  # 1 where towed
    segments: List[SegmentAssignment] = field(default_factory=list)

    @classmethod
    def build(cls, avs, pvs, l_min, assignments) -> "_State":
        length = max(max(v.exit_point for v in avs), max(v.exit_point for v in pvs))
        s = cls(avs=avs, pvs=pvs, l_min=l_min, length=length,
                av_load={a.id: [0] * length for a in avs},
                pv_cover={p.id: [0] * length for p in pvs},
                segments=list(assignments))
        for a in s.segments:
            s._apply(a, +1)
        return s

    def _apply(self, seg: SegmentAssignment, sign: int) -> None:
        load, cover = self.av_load[seg.av.id], self.pv_cover[seg.pv.id]
        for x in range(seg.cp, seg.dp):
            load[x] += sign
            cover[x] += sign

    def total(self) -> int:
        return sum(s.dp - s.cp for s in self.segments)

    def runs(self, av: ActiveVehicle, pv: PassiveVehicle) -> List[Tuple[int, int]]:
        """Maximal sub-intervals of the AV-PV overlap that are insertable."""
        cp, dp = max(av.entry_point, pv.entry_point), min(av.exit_point, pv.exit_point)
        if dp - cp < self.l_min:
            return []
        load, cover = self.av_load[av.id], self.pv_cover[pv.id]
        out, start = [], None
        for x in range(cp, dp):
            free = load[x] < av.capacity and cover[x] == 0
            if free and start is None:
                start = x
            elif not free and start is not None:
                if x - start >= self.l_min:
                    out.append((start, x))
                start = None
        if start is not None and dp - start >= self.l_min:
            out.append((start, dp))
        return out

    def candidates(self) -> List[Tuple[int, ActiveVehicle, PassiveVehicle, int, int]]:
        out = []
        for av in self.avs:
            for pv in self.pvs:
                for s, e in self.runs(av, pv):
                    out.append((e - s, av, pv, s, e))
        return out

    def best_insertion(self) -> Optional[Tuple[int, ActiveVehicle, PassiveVehicle, int, int]]:
        c = self.candidates()
        return max(c, key=lambda t: t[0]) if c else None

    def insert(self, av, pv, s, e) -> None:
        seg = SegmentAssignment(pv=pv, av=av, cp=s, dp=e)
        self.segments.append(seg)
        self._apply(seg, +1)

    def remove(self, seg: SegmentAssignment) -> None:
        self.segments.remove(seg)
        self._apply(seg, -1)


def _fill(st: _State, rng=None, alpha: float = float("inf")) -> int:
    """Insert feasible segments until none remain. Returns the gain.

    alpha controls the bias: inf picks the longest run (what ILA effectively
    does), 0 picks uniformly at random, and values in between sample with
    probability proportional to length**alpha. Longest-first rebuilds the same
    few long segments ILA already chose, so the randomised variants are what
    let the repair find the optimum's many-short-segments structure.
    """
    gain = 0
    while True:
        cand = st.candidates()
        if not cand:
            break
        if rng is None or alpha == float("inf"):
            pick = max(cand, key=lambda t: t[0])
        else:
            w = [c[0] ** alpha for c in cand]
            pick = rng.choices(cand, weights=w, k=1)[0]
        gain += pick[0]
        st.insert(*pick[1:])
    return gain


def _extend(st: _State) -> int:
    """Grow existing segments into adjacent free space."""
    gain = 0
    for seg in list(st.segments):
        load, cover = st.av_load[seg.av.id], st.pv_cover[seg.pv.id]
        lo = max(seg.av.entry_point, seg.pv.entry_point)
        hi = min(seg.av.exit_point, seg.pv.exit_point)
        new_cp = seg.cp
        while new_cp > lo and load[new_cp - 1] < seg.av.capacity and cover[new_cp - 1] == 0:
            new_cp -= 1
        new_dp = seg.dp
        while new_dp < hi and load[new_dp] < seg.av.capacity and cover[new_dp] == 0:
            new_dp += 1
        if new_cp == seg.cp and new_dp == seg.dp:
            continue
        st.remove(seg)
        st.insert(seg.av, seg.pv, new_cp, new_dp)
        gain += (new_dp - new_cp) - (seg.dp - seg.cp)
    return gain


def refine(
    assignments: Sequence[SegmentAssignment],
    avs: Sequence[ActiveVehicle],
    pvs: Sequence[PassiveVehicle],
    l_min: int,
    *,
    eject_rounds: int = 1,
    max_ejects: Optional[int] = None,
) -> Tuple[List[SegmentAssignment], dict]:
    """Return an improved assignment list and a report of what each move gained."""
    st = _State.build(avs, pvs, l_min, assignments)
    start = st.total()
    report = {"start": start, "fill": 0, "extend": 0, "eject": 0}

    while True:
        g = _fill(st) + _extend(st)
        report["fill"] += g
        if g == 0:
            break

    for _ in range(eject_rounds):
        round_gain = 0
        # shortest segments first: they are the cheapest to give up
        order = sorted(st.segments, key=lambda s: s.dp - s.cp)
        if max_ejects:
            order = order[:max_ejects]
        for seg in order:
            if seg not in st.segments:
                continue
            before = st.total()
            snapshot = list(st.segments)
            st.remove(seg)
            _fill(st)
            _extend(st)
            if st.total() <= before:                      # undo
                for s in list(st.segments):
                    st.remove(s)
                for s in snapshot:
                    st.insert(s.av, s.pv, s.cp, s.dp)
            else:
                round_gain += st.total() - before
        report["eject"] += round_gain
        if round_gain == 0:
            break

    report["end"] = st.total()
    report["gain"] = report["end"] - start
    return st.segments, report


# ---------------------------------------------------------------- ruin & recreate

def refine_ils(
    assignments: Sequence[SegmentAssignment],
    avs: Sequence[ActiveVehicle],
    pvs: Sequence[PassiveVehicle],
    l_min: int,
    *,
    iterations: int = 300,
    seed: int = 0,
    window_frac: Tuple[float, float] = (0.15, 0.5),
    av_frac: Tuple[float, float] = (0.3, 1.0),
    alpha: float = 1.0,
    time_budget: Optional[float] = None,
) -> Tuple[List[SegmentAssignment], dict]:
    """Iterated local search: ruin a road window on some AVs, rebuild, keep if better.

    Greedy rebuild inserts the longest feasible run first; a run is limited by
    capacity and by what the PV already has, so rebuilt segments may be partial
    -- the degree of freedom ILA never uses.
    """
    import random
    import time as _time

    rng = random.Random(seed)
    st = _State.build(avs, pvs, l_min, assignments)
    _fill(st); _extend(st)
    best = list(st.segments)
    best_total = st.total()
    start_total = sum(s.dp - s.cp for s in assignments)
    report = {"start": start_total, "after_fill": best_total, "accepted": 0, "iterations": 0}
    t0 = _time.perf_counter()

    for it in range(iterations):
        if time_budget is not None and _time.perf_counter() - t0 > time_budget:
            break
        report["iterations"] = it + 1
        span = int(st.length * rng.uniform(*window_frac))
        lo = rng.randint(0, max(0, st.length - span))
        hi = lo + span
        victims = rng.sample(list(avs), max(1, int(len(avs) * rng.uniform(*av_frac))))
        vids = {a.id for a in victims}

        for seg in [s for s in st.segments if s.av.id in vids and s.cp < hi and s.dp > lo]:
            st.remove(seg)
        _fill(st, rng, alpha); _extend(st)

        if st.total() > best_total:
            best_total = st.total()
            best = list(st.segments)
            report["accepted"] += 1
        else:                                   # restart from the incumbent
            for s in list(st.segments):
                st.remove(s)
            for s in best:
                st.insert(s.av, s.pv, s.cp, s.dp)

    report["end"] = best_total
    report["gain"] = best_total - start_total
    report["seconds"] = round(_time.perf_counter() - t0, 2)
    return best, report
