"""Exact solver for the time-free dynamic platoon formation problem.

Reviewer item #4 asks for a comparison against exact optimal solutions on small
instances. This module formulates the problem as a mixed-integer linear program
and solves it with HiGHS via `scipy.optimize.milp`.

Scope: the temporal synchronisation constraint of Section 2.2 is *state
dependent* (towing a PV changes its arrival time on later segments), which is not
expressible as a linear constraint. This solver therefore covers the spatial
problem only, i.e. the setting the codebase calls
`enable_time_constraints=False`. That is enough to answer whether the
formulation leaves room for coordination to beat a myopic choice.

Formulation. Positions are integers, so a towing segment is a set of unit
intervals. For every AV-PV pair (i, j) with a non-empty overlap
[cp_ij, dp_ij) and every position x in that overlap:

    z[i,j,x] = 1  iff  PV j is towed by AV i over the unit interval [x, x+1)
    s[i,j,x] = 1  marks a position where a towing segment starts

    maximise  sum z[i,j,x]                                    (towed distance)

    s.t.  sum_j z[i,j,x] <= C_i        for all i, x           (point-wise capacity)
          sum_i z[i,j,x] <= 1          for all j, x           (PV non-overlap)
          z[i,j,x] - z[i,j,x-1] <= s[i,j,x]                   (a rise starts a segment)
          s[i,j,x] <= z[i,j,y]  for y in [x, x+L_min)         (segments reach L_min)
          s[i,j,x] = 0  where x + L_min > dp_ij               (no segment would fit)

Dropping the integrality requirement yields the LP relaxation, whose value is an
upper bound on the optimum. `solve()` returns it alongside, so a run that hits
the time limit still bounds how much any algorithm could possibly gain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import LinearConstraint, Bounds, milp
from scipy.sparse import coo_matrix

from core.models import ActiveVehicle, PassiveVehicle


@dataclass
class ExactResult:
    """Outcome of one exact solve."""
    optimal: Optional[float]      # best integral objective found (towed distance)
    upper_bound: float            # LP relaxation value; optimum cannot exceed it
    status: str                   # solver status message
    proven_optimal: bool          # True when the MILP solved to optimality
    n_binary: int
    n_constraints: int
    seconds: float

    @property
    def reference(self) -> float:
        """Value to compare heuristics against: the optimum when proven, else the bound."""
        return self.optimal if self.proven_optimal else self.upper_bound


def _overlap(av: ActiveVehicle, pv: PassiveVehicle) -> Tuple[int, int]:
    """Maximum realisable overlap [cp, dp) of an AV-PV pair (Section 2.2)."""
    cp = max(av.entry_point, pv.entry_point)
    dp = min(av.exit_point, pv.exit_point)
    return cp, dp


def build(avs: Sequence[ActiveVehicle], pvs: Sequence[PassiveVehicle], l_min: int,
          *, suffix_only: bool = False, full_only: bool = False):
    """Build the MILP. Returns (c, A, lb, ub, integrality, n_z, index maps).

    With ``suffix_only=True`` every towing segment must run to the end of the
    pair's overlap. That is exactly the shape both heuristics can produce: they
    commit ``s = max(e_i, l_j)``, ``e = dp_ij``, so the start can be pushed back
    by earlier towing but the end never is. Solving under this restriction
    separates the part of the optimality gap that comes from choosing badly
    among available moves from the part that needs segment shapes the algorithms
    cannot express at all.

    The restricted optimum is an upper bound on what the heuristics could reach:
    it may start a segment late anywhere, whereas they can only do so as a
    consequence of an earlier commit on the same PV.
    """
    # ---- variables -------------------------------------------------------
    z_index: Dict[Tuple[int, int, int], int] = {}
    pair_span: Dict[Tuple[int, int], Tuple[int, int]] = {}

    for i, av in enumerate(avs):
        for j, pv in enumerate(pvs):
            cp, dp = _overlap(av, pv)
            if dp - cp < l_min:
                continue                      # pair can never carry a segment
            pair_span[(i, j)] = (cp, dp)
            for x in range(cp, dp):
                z_index[(i, j, x)] = len(z_index)

    n_z = len(z_index)
    s_index: Dict[Tuple[int, int, int], int] = {
        key: n_z + n for n, key in enumerate(z_index)
    }
    n_vars = n_z + len(s_index)

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    lb: List[float] = []
    ub: List[float] = []
    row = 0

    def add_row(entries, low, high):
        nonlocal row
        for col, val in entries:
            rows.append(row); cols.append(col); vals.append(val)
        lb.append(low); ub.append(high); row += 1

    # ---- point-wise capacity: sum_j z[i,j,x] <= C_i ----------------------
    for i, av in enumerate(avs):
        for x in range(av.entry_point, av.exit_point):
            entries = [(z_index[(i, j, x)], 1.0)
                       for j in range(len(pvs)) if (i, j, x) in z_index]
            if entries:
                add_row(entries, -np.inf, float(av.capacity))

    # ---- PV non-overlap: sum_i z[i,j,x] <= 1 -----------------------------
    for j, pv in enumerate(pvs):
        for x in range(pv.entry_point, pv.exit_point):
            entries = [(z_index[(i, j, x)], 1.0)
                       for i in range(len(avs)) if (i, j, x) in z_index]
            if len(entries) > 1:
                add_row(entries, -np.inf, 1.0)

    # ---- segment structure ------------------------------------------------
    for (i, j), (cp, dp) in pair_span.items():
        for x in range(cp, dp):
            zx, sx = z_index[(i, j, x)], s_index[(i, j, x)]

            # a rise in z must be marked by a start
            if x == cp:
                add_row([(zx, 1.0), (sx, -1.0)], -np.inf, 0.0)
            else:
                zprev = z_index[(i, j, x - 1)]
                add_row([(zx, 1.0), (zprev, -1.0), (sx, -1.0)], -np.inf, 0.0)

            # full-only: a pair is towed over its whole overlap or not at all
            if full_only and x > cp:
                add_row([(zx, 1.0), (z_index[(i, j, cp)], -1.0)], 0.0, 0.0)

            # suffix-only: towing at x forces towing at x+1, so runs reach dp
            if (suffix_only or full_only) and x + 1 < dp:
                add_row([(zx, 1.0), (z_index[(i, j, x + 1)], -1.0)], -np.inf, 0.0)

            # a start must be followed by at least l_min towed units
            if x + l_min > dp:
                add_row([(sx, 1.0)], 0.0, 0.0)          # no room: forbid start
            else:
                for y in range(x, x + l_min):
                    add_row([(sx, 1.0), (z_index[(i, j, y)], -1.0)], -np.inf, 0.0)

    A = coo_matrix((vals, (rows, cols)), shape=(row, n_vars)).tocsr()
    c = np.zeros(n_vars)
    c[:n_z] = -1.0                                  # maximise towed distance
    integrality = np.ones(n_vars)
    return c, A, np.array(lb), np.array(ub), integrality, n_z, z_index, s_index


def solve(avs: Sequence[ActiveVehicle], pvs: Sequence[PassiveVehicle],
          l_min: int, *, time_limit: float = 600.0,
          suffix_only: bool = False, full_only: bool = False) -> ExactResult:
    """Solve one instance exactly (or return the LP bound if the limit is hit)."""
    import time as _time

    c, A, lb, ub, integrality, n_z, z_index, _ = build(
        avs, pvs, l_min, suffix_only=suffix_only, full_only=full_only)
    constraints = LinearConstraint(A, lb, ub)
    bounds = Bounds(0, 1)

    # LP relaxation first: cheap, and an upper bound even if the MILP times out.
    t0 = _time.perf_counter()
    lp = milp(c=c, constraints=constraints, bounds=bounds,
              integrality=np.zeros_like(integrality),
              options={"time_limit": time_limit})
    upper = -lp.fun if lp.success and lp.fun is not None else float("inf")

    res = milp(c=c, constraints=constraints, bounds=bounds,
               integrality=integrality,
               options={"time_limit": time_limit, "presolve": True})
    seconds = _time.perf_counter() - t0

    proven = bool(res.status == 0)
    value = -res.fun if res.fun is not None else None
    return ExactResult(optimal=value, upper_bound=upper, status=res.message,
                       proven_optimal=proven, n_binary=A.shape[1],
                       n_constraints=A.shape[0], seconds=seconds)


def extract_segments(avs: Sequence[ActiveVehicle], pvs: Sequence[PassiveVehicle],
                     l_min: int, *, time_limit: float = 600.0):
    """Solve, then report the towing segments of the optimal solution.

    Returns (segments, n_partial), where each segment is
    (av_index, pv_index, start, end, is_partial). A segment is *partial* when it
    is strictly inside the pair's maximal overlap, i.e. a solution the current
    algorithms cannot produce: both greedy and ILA always commit the whole
    residual overlap. Counting these separates the part of the optimality gap
    that is reachable by better combination choices from the part that needs the
    formulation of reviewer item #9 to be widened.
    """
    c, A, lb, ub, integrality, n_z, z_index, _ = build(avs, pvs, l_min)
    res = milp(c=c, constraints=LinearConstraint(A, lb, ub), bounds=Bounds(0, 1),
               integrality=integrality,
               options={"time_limit": time_limit, "presolve": True})
    if res.x is None:
        return [], 0

    chosen = {key for key, col in z_index.items() if res.x[col] > 0.5}
    segments = []
    for i, av in enumerate(avs):
        for j, pv in enumerate(pvs):
            xs = sorted(x for (a, b, x) in chosen if a == i and b == j)
            if not xs:
                continue
            cp, dp = _overlap(av, pv)
            start = prev = xs[0]
            for x in xs[1:] + [None]:
                if x is None or x != prev + 1:
                    end = prev + 1
                    segments.append((i, j, start, end, (start > cp or end < dp)))
                    if x is not None:
                        start = x
                if x is not None:
                    prev = x
    return segments, sum(1 for s in segments if s[4])
