"""Congestion-aware variants of greedy and ILA.

The exact-solver diagnosis (milp/diagnose.py) showed that the optimum uses about
74% more towing segments, each about 37% shorter, and serves more PVs from the
same capacity. Both shipped algorithms score a candidate segment by its length
alone, so they commit long tows that hold an AV's capacity over a long stretch
of road and shut other PVs out of it. Neither prices that exclusivity.

These variants keep the two algorithms exactly as they are except for the value
they assign to a candidate segment. For AV *i* at position *x* let

    contest(i, x)  PVs still uncovered whose residual route could be towed by
                   AV i across x
    free(i, x)     capacity AV i still has at x

and define the scarcity of that position as contest / free. A candidate segment
is then worth

    adjusted = saving / (1 + lam * mean scarcity over [cp, dp))

so a tow across contested road is worth less than the same distance across empty
road. `lam=0` reproduces the original algorithms exactly, which the tests rely
on.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from core.models import ActiveVehicle, PassiveVehicle
from core.hungarian_multi import (
    AVCapacityState,
    PVRoutingState,
    SegmentAssignment,
    _hungarian_min_cost,
    DEFAULT_TIME_TOLERANCE,
)

Candidate = Tuple[float, float, ActiveVehicle, PassiveVehicle, int, int, float, float]


def _scarcity_map(avs: Sequence[ActiveVehicle], pvs: Sequence[PassiveVehicle],
                  pv_states: Dict[str, PVRoutingState],
                  av_states: Dict[str, AVCapacityState], l_min: int,
                  *, enable_time_constraints: bool,
                  time_tolerance: float) -> Dict[str, Dict[int, float]]:
    """For each AV, how contested each position on its route currently is."""
    contest: Dict[str, Dict[int, int]] = {av.id: {} for av in avs}

    for av in avs:
        counts = contest[av.id]
        for pv in pvs:
            state = pv_states[pv.id]
            if state.is_fully_covered:
                continue
            overlap = state.get_overlap_with_av(
                av, enable_time_constraints=enable_time_constraints,
                time_tolerance=time_tolerance)
            if overlap is None:
                continue
            cp, dp, _, _ = overlap
            if dp - cp < l_min:
                continue
            for x in range(cp, dp):
                counts[x] = counts.get(x, 0) + 1

    scarcity: Dict[str, Dict[int, float]] = {}
    for av in avs:
        state = av_states[av.id]
        m: Dict[int, float] = {}
        for x, c in contest[av.id].items():
            free = av.capacity - state.get_capacity_at_point(x)
            m[x] = c / free if free > 0 else float(c)
        scarcity[av.id] = m
    return scarcity


def _adjusted(saving: int, av: ActiveVehicle, cp: int, dp: int,
              scarcity: Dict[str, Dict[int, float]], lam: float) -> float:
    """Discount a candidate by how contested the road it occupies is."""
    if lam <= 0 or dp <= cp:
        return float(saving)
    m = scarcity.get(av.id, {})
    mean = sum(m.get(x, 0.0) for x in range(cp, dp)) / (dp - cp)
    return saving / (1.0 + lam * mean)


def _candidates(avs, pvs, pv_states, av_states, l_min, scarcity, lam,
                *, enable_time_constraints, time_tolerance) -> List[Candidate]:
    """Feasible (AV, PV) segments this round, with raw and adjusted value."""
    out: List[Candidate] = []
    for av in avs:
        av_state = av_states[av.id]
        for pv in pvs:
            state = pv_states[pv.id]
            if state.is_fully_covered:
                continue
            overlap = state.get_overlap_with_av(
                av, enable_time_constraints=enable_time_constraints,
                time_tolerance=time_tolerance)
            if overlap is None:
                continue
            cp, dp, ct, dt = overlap
            saving = dp - cp
            if saving < l_min or not av_state.can_accommodate_segment(cp, dp):
                continue
            out.append((_adjusted(saving, av, cp, dp, scarcity, lam),
                        float(saving), av, pv, cp, dp, ct, dt))
    return out


def greedy_congestion_matching(
    avs: List[ActiveVehicle], pvs: List[PassiveVehicle], l_min: int, *,
    lam: float = 1.0, enable_time_constraints: bool = False,
    time_tolerance: float = DEFAULT_TIME_TOLERANCE,
) -> Tuple[List[SegmentAssignment], float, Dict[str, List[SegmentAssignment]]]:
    """Greedy, but ranking candidates by congestion-adjusted value."""
    pv_states = {pv.id: PVRoutingState(pv=pv) for pv in pvs}
    av_states = {av.id: AVCapacityState(av=av) for av in avs}
    assignments: List[SegmentAssignment] = []
    per_pv: Dict[str, List[SegmentAssignment]] = {pv.id: [] for pv in pvs}
    total = 0.0

    while True:
        scarcity = _scarcity_map(avs, pvs, pv_states, av_states, l_min,
                                 enable_time_constraints=enable_time_constraints,
                                 time_tolerance=time_tolerance)
        cands = _candidates(avs, pvs, pv_states, av_states, l_min, scarcity, lam,
                            enable_time_constraints=enable_time_constraints,
                            time_tolerance=time_tolerance)
        if not cands:
            break
        # highest adjusted value; ties broken deterministically
        cands.sort(key=lambda c: (-c[0], c[2].id, c[3].id, c[4]))
        _, saving, av, pv, cp, dp, ct, dt = cands[0]

        a = SegmentAssignment(pv=pv, av=av, cp=cp, dp=dp,
                              coupling_time=ct, decoupling_time=dt)
        assignments.append(a)
        per_pv[pv.id].append(a)
        total += saving
        pv_states[pv.id].mark_segment_covered(cp, dp, ct, dt, av.speed)
        av_states[av.id].add_assignment(cp, dp)

    return assignments, total, per_pv


def ila_congestion_matching(
    avs: List[ActiveVehicle], pvs: List[PassiveVehicle], l_min: int, *,
    lam: float = 1.0, enable_time_constraints: bool = False,
    time_tolerance: float = DEFAULT_TIME_TOLERANCE,
) -> Tuple[List[SegmentAssignment], float, Dict[str, List[SegmentAssignment]]]:
    """ILA, but with congestion priced into the LSAP cost matrix.

    Structure is unchanged: expand each AV into capacity slots, solve one LSAP
    per round, confirm the returned pairs against point-wise capacity, repeat.
    Only the matrix entries differ - they carry the adjusted value rather than
    the raw segment length, so a round trades off contested road against empty
    road instead of maximising distance blindly.
    """
    if not avs or not pvs:
        return [], 0.0, {pv.id: [] for pv in pvs}

    pv_states = {pv.id: PVRoutingState(pv=pv) for pv in pvs}
    av_states = {av.id: AVCapacityState(av=av) for av in avs}
    assignments: List[SegmentAssignment] = []
    per_pv: Dict[str, List[SegmentAssignment]] = {pv.id: [] for pv in pvs}
    total = 0.0

    slots: List[ActiveVehicle] = [av for av in avs for _ in range(av.capacity)]
    pv_index = {pv.id: j for j, pv in enumerate(pvs)}
    max_rounds = len(avs) * len(pvs) * 2

    for _ in range(max_rounds):
        scarcity = _scarcity_map(avs, pvs, pv_states, av_states, l_min,
                                 enable_time_constraints=enable_time_constraints,
                                 time_tolerance=time_tolerance)
        cands = _candidates(avs, pvs, pv_states, av_states, l_min, scarcity, lam,
                            enable_time_constraints=enable_time_constraints,
                            time_tolerance=time_tolerance)
        if not cands:
            break

        by_pair = {(av.id, pv.id): c for c in cands
                   for av, pv in ((c[2], c[3]),)}
        best = max(c[0] for c in cands)

        n = len(slots) + len(pvs)
        BIG_M = best * 1000.0 + 1.0
        UNMATCHED = best + 1.0
        cost = [[UNMATCHED] * n for _ in range(n)]
        for s, av in enumerate(slots):
            for pv in pvs:
                c = by_pair.get((av.id, pv.id))
                cost[s][pv_index[pv.id]] = (best - c[0] + 1.0) if c else BIG_M
        for s in range(len(slots), n):
            for j in range(len(pvs), n):
                cost[s][j] = 0.0

        row_to_col = _hungarian_min_cost(cost)

        chosen: List[Candidate] = []
        for s in range(min(len(row_to_col), len(slots))):
            j = row_to_col[s]
            if j < 0 or j >= len(pvs):
                continue
            c = by_pair.get((slots[s].id, pvs[j].id))
            if c is not None:
                chosen.append(c)

        chosen.sort(key=lambda c: (c[3].id, c[4]))
        applied = 0
        used_pvs: set = set()
        for _adj, saving, av, pv, cp, dp, ct, dt in chosen:
            if pv.id in used_pvs:
                continue
            state = pv_states[pv.id]
            if not any(cp >= s0 and dp <= e0
                       for s0, e0, _t in state.uncovered_segments):
                continue
            if not av_states[av.id].can_accommodate_segment(cp, dp):
                continue
            used_pvs.add(pv.id)
            a = SegmentAssignment(pv=pv, av=av, cp=cp, dp=dp,
                                  coupling_time=ct, decoupling_time=dt)
            assignments.append(a)
            per_pv[pv.id].append(a)
            total += saving
            state.mark_segment_covered(cp, dp, ct, dt, av.speed)
            av_states[av.id].add_assignment(cp, dp)
            applied += 1

        if applied == 0:
            break

    return assignments, total, per_pv


def ila_filter_matching(
    avs: List[ActiveVehicle], pvs: List[PassiveVehicle], l_min: int, *,
    theta: float = 1.5, enable_time_constraints: bool = False,
    time_tolerance: float = DEFAULT_TIME_TOLERANCE,
) -> Tuple[List[SegmentAssignment], float, Dict[str, List[SegmentAssignment]]]:
    """ILA with congestion used as a filter rather than as a discount.

    `ila_congestion_matching` discounts contested segments in the cost matrix,
    which changes what the LSAP is maximising: it optimises adjusted score and
    can trade real towed distance away to avoid congestion. Measurements showed
    that makes ILA worse at every weight tried, while the same signal helps
    greedy, which only uses it to rank.

    This variant keeps the LSAP objective as raw towed distance and instead
    withholds candidates whose mean scarcity exceeds `theta` from the current
    round. They stay eligible later, once earlier commits have resolved some of
    the contention. A round that would otherwise be empty falls back to the
    unfiltered candidate set, so filtering can delay a commit but never prevent
    the algorithm from terminating with the same coverage it would have reached.

    `theta=inf` reproduces the original ILA.
    """
    if not avs or not pvs:
        return [], 0.0, {pv.id: [] for pv in pvs}

    pv_states = {pv.id: PVRoutingState(pv=pv) for pv in pvs}
    av_states = {av.id: AVCapacityState(av=av) for av in avs}
    assignments: List[SegmentAssignment] = []
    per_pv: Dict[str, List[SegmentAssignment]] = {pv.id: [] for pv in pvs}
    total = 0.0

    slots: List[ActiveVehicle] = [av for av in avs for _ in range(av.capacity)]
    pv_index = {pv.id: j for j, pv in enumerate(pvs)}

    for _ in range(len(avs) * len(pvs) * 2):
        scarcity = _scarcity_map(avs, pvs, pv_states, av_states, l_min,
                                 enable_time_constraints=enable_time_constraints,
                                 time_tolerance=time_tolerance)
        cands = _candidates(avs, pvs, pv_states, av_states, l_min, scarcity, 0.0,
                            enable_time_constraints=enable_time_constraints,
                            time_tolerance=time_tolerance)
        if not cands:
            break

        def mean_scarcity(av: ActiveVehicle, cp: int, dp: int) -> float:
            m = scarcity.get(av.id, {})
            return sum(m.get(x, 0.0) for x in range(cp, dp)) / max(1, dp - cp)

        kept = [c for c in cands if mean_scarcity(c[2], c[4], c[5]) <= theta]
        if not kept:                      # never stall on account of the filter
            kept = cands

        by_pair = {(c[2].id, c[3].id): c for c in kept}
        best = max(c[1] for c in kept)    # raw saving, not adjusted

        n = len(slots) + len(pvs)
        BIG_M = best * 1000.0 + 1.0
        UNMATCHED = best + 1.0
        cost = [[UNMATCHED] * n for _ in range(n)]
        for s, av in enumerate(slots):
            for pv in pvs:
                c = by_pair.get((av.id, pv.id))
                cost[s][pv_index[pv.id]] = (best - c[1] + 1.0) if c else BIG_M
        for s in range(len(slots), n):
            for j in range(len(pvs), n):
                cost[s][j] = 0.0

        row_to_col = _hungarian_min_cost(cost)

        chosen: List[Candidate] = []
        for s in range(min(len(row_to_col), len(slots))):
            j = row_to_col[s]
            if 0 <= j < len(pvs):
                c = by_pair.get((slots[s].id, pvs[j].id))
                if c is not None:
                    chosen.append(c)

        chosen.sort(key=lambda c: (c[3].id, c[4]))
        applied = 0
        used_pvs: set = set()
        for _adj, saving, av, pv, cp, dp, ct, dt in chosen:
            if pv.id in used_pvs:
                continue
            state = pv_states[pv.id]
            if not any(cp >= s0 and dp <= e0
                       for s0, e0, _t in state.uncovered_segments):
                continue
            if not av_states[av.id].can_accommodate_segment(cp, dp):
                continue
            used_pvs.add(pv.id)
            a = SegmentAssignment(pv=pv, av=av, cp=cp, dp=dp,
                                  coupling_time=ct, decoupling_time=dt)
            assignments.append(a)
            per_pv[pv.id].append(a)
            total += saving
            state.mark_segment_covered(cp, dp, ct, dt, av.speed)
            av_states[av.id].add_assignment(cp, dp)
            applied += 1

        if applied == 0:
            break

    return assignments, total, per_pv
