"""Independent feasibility check for a set of towing assignments.

Written to settle a disagreement: on one instance the heuristics reported a
higher towed distance than milp.exact_time's proven optimum, which is
impossible unless either the MILP is too restrictive or the heuristic solution
is infeasible. This checker re-derives the PV clocks from scratch and tests
every constraint, so it does not share code (or assumptions) with either side.

Semantics checked, as stated in the paper and implemented in
core/hungarian_multi.py:
  * a tow runs at least l_min
  * an AV carries at most C_i PVs at any point
  * a PV is towed by at most one AV at any point
  * a tow may start only where the PV's clock and the AV's clock agree within
    tau; while towed, the PV rides the AV, so its clock becomes the AV's
"""
from __future__ import annotations

from typing import List, Sequence, Tuple


def check(assignments, avs, pvs, l_min: int, tau: float,
          waiting: str = "either") -> List[str]:
    """`waiting` must match the convention the solution was produced under.

    "either"  whoever arrives first waits; the PV couples at
              max(t_pv, t_av) and then rides at the AV's speed.
              This is what core/{greedy,hungarian}_multi.py implement.
    "none"    the PV's clock snaps to the AV's at coupling.
    """
    problems: List[str] = []
    by_pv = {}
    for a in assignments:
        by_pv.setdefault(a.pv.id, []).append(a)

    # length, AV capacity, PV exclusivity
    for a in assignments:
        if a.dp - a.cp < l_min:
            problems.append(f"segment {a.pv.id}<-{a.av.id} [{a.cp},{a.dp}) shorter than l_min")
        if a.cp < max(a.av.entry_point, a.pv.entry_point) or a.dp > min(a.av.exit_point, a.pv.exit_point):
            problems.append(f"segment {a.pv.id}<-{a.av.id} [{a.cp},{a.dp}) outside the shared overlap")
    span = max(max(v.exit_point for v in avs), max(v.exit_point for v in pvs))
    for av in avs:
        for x in range(span):
            n = sum(1 for a in assignments if a.av.id == av.id and a.cp <= x < a.dp)
            if n > av.capacity:
                problems.append(f"AV {av.id} carries {n} > capacity {av.capacity} at {x}")
                break
    for pv in pvs:
        for x in range(span):
            n = sum(1 for a in by_pv.get(pv.id, []) if a.cp <= x < a.dp)
            if n > 1:
                problems.append(f"PV {pv.id} towed by {n} AVs at {x}")
                break

    # clocks, rebuilt from scratch
    for pv in pvs:
        segs = sorted(by_pv.get(pv.id, []), key=lambda a: a.cp)
        t = pv.entry_time
        x = pv.entry_point
        for a in segs:
            t += (a.cp - x) / pv.speed          # free running up to the coupling point
            x = a.cp
            av_t = a.av.time_at_point(a.cp)
            if av_t is None:
                problems.append(f"AV {a.av.id} is not on the road at {a.cp}")
                continue
            if abs(t - av_t) > tau + 1e-9:
                problems.append(
                    f"PV {pv.id} meets AV {a.av.id} at {a.cp} with |{t:.3f}-{av_t:.3f}|"
                    f" = {abs(t-av_t):.3f} > tau {tau}")
            if waiting == "none":
                t = a.av.time_at_point(a.dp)    # clock snaps to the AV's
            else:
                t = max(t, av_t) + (a.dp - a.cp) / a.av.speed   # first to arrive waits
            x = a.dp
    return problems


def describe(assignments) -> str:
    return "  " + "\n  ".join(
        f"{a.pv.id} <- {a.av.id}  [{a.cp}, {a.dp})" for a in sorted(assignments, key=lambda a: (a.pv.id, a.cp)))
