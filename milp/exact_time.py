"""Exact optimum WITH temporal synchronisation.

The 2026-09 notes recorded that a time-constrained MILP is "non-linear, because
towing changes later arrival times". That is true of the schedule but not of the
model: a PV's clock is a *linear* function of the towing decisions.

Let t[j,x] be the time at which PV j reaches point x. AV clocks are unaffected
by towing, so t_i(x) is a *constant*. A towed PV rides on its AV, so while it is
towed its clock is that AV's clock; otherwise it advances at its own speed:

    z[i,j,x] = 1          ->  t[j,x+1] = t_i(x+1)        (a constant)
    sum_i z[i,j,x] = 0    ->  t[j,x+1] = t[j,x] + 1/v_j

Both are linear once written with an indicator (big-M). This matches
core/hungarian_multi.py, which sets the PV's downstream segment start time to
`decoupling_time`, the AV's time at the decoupling point -- the PV's clock is
*snapped* to the AV's on coupling, not merely advanced at the AV's rate.

(An earlier version of this file advanced the PV clock at the AV's rate while
carrying the coupling discrepancy forward. That is a different, more permissive
problem, and it showed: on 1 of 12 validation instances the heuristics beat its
"optimum", which is impossible for a correct model. Kept here as a note because
the symptom is the only cheap way to catch this class of error.)

The tolerance is checked only where a tow starts -- once coupled the two move
together -- which is exactly what core/hungarian_multi.py does
(`abs(pv_time_at_cp - av_time_at_cp) > time_tolerance` rejects the pair). So

    |t[j,x] - t_i(x)| <= tau + M (1 - s[i,j,x])

with s the start indicator already present in milp.exact. Big-M is sized from
the instance rather than guessed.

Scope and caution
  * This solves the same temporal semantics the heuristics implement: tolerance
    at the coupling point, PV clock updated downstream of a tow.
  * Big-M weakens the LP relaxation, so expect this to be far slower than the
    spatial MILP and to need smaller instances.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from core.models import ActiveVehicle, PassiveVehicle
from milp.exact import ExactResult, _overlap


WAITING_MODES = ("either", "pv_only", "none")


def build_time(avs: Sequence[ActiveVehicle], pvs: Sequence[PassiveVehicle],
               l_min: int, tau: float, waiting: str = "either"):
    """Variables: z[i,j,x], s[i,j,x] (binary), t[j,x], w[i,j,x] (continuous).

    `waiting` selects the semantics, because the project has three coherent
    candidates and the implementation is a mix of them (see NOTES.md,
    2026-10-02):

      "either"   whoever arrives first waits; the PV's clock is lifted to
                 max(t_pv, t_av) at coupling.  This is what
                 core/{greedy,hungarian}_multi.py do, so it is the default and
                 the only setting under which a heuristic-vs-optimum comparison
                 is apples to apples.
      "pv_only"  the AV never waits: coupling needs the PV to be there no later
                 than the AV, and an early PV waits.  Physically the sane one,
                 because the AV's own timetable then stays exact.
      "none"     no waiting at all: the PV's clock snaps to the AV's.  The
                 strictest variant.

    Caveat that applies to all three: like the heuristics, this model does not
    propagate an AV's wait into the AV's own timetable, so a waiting AV still
    serves its other passengers on the unshifted schedule.  Fixing that is
    option (c) in the notes and is not implemented.
    """
    if waiting not in WAITING_MODES:
        raise ValueError(f"waiting must be one of {WAITING_MODES}")
    z_index: Dict[Tuple[int, int, int], int] = {}
    pair_span: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for i, av in enumerate(avs):
        for j, pv in enumerate(pvs):
            cp, dp = _overlap(av, pv)
            if dp - cp < l_min:
                continue
            pair_span[(i, j)] = (cp, dp)
            for x in range(cp, dp):
                z_index[(i, j, x)] = len(z_index)
    n_z = len(z_index)
    s_index = {key: n_z + n for n, key in enumerate(z_index)}
    n_bin = 2 * n_z

    # one clock variable per PV per point of its own route (inclusive of exit)
    t_index: Dict[Tuple[int, int], int] = {}
    for j, pv in enumerate(pvs):
        for x in range(pv.entry_point, pv.exit_point + 1):
            t_index[(j, x)] = n_bin + len(t_index)
    n_t = len(t_index)
    # one "lift" per candidate coupling: how long the PV's clock jumps forward
    # when it waits for, or is waited for by, the AV
    w_index: Dict[Tuple[int, int, int], int] = {}
    if waiting != "none":
        for key in z_index:
            w_index[key] = n_bin + n_t + len(w_index)
    n_vars = n_bin + n_t + len(w_index)

    # big-M: the widest clock discrepancy the instance can produce
    horizon = max(max(v.entry_time for v in list(avs) + list(pvs)), 0.0)
    slowest = min(min(v.speed for v in avs), min(v.speed for v in pvs))
    span = max(max(v.exit_point for v in avs), max(v.exit_point for v in pvs))
    big_m = horizon + span / slowest + tau + 1.0

    rows: List[int] = []; cols: List[int] = []; vals: List[float] = []
    lo: List[float] = []; hi: List[float] = []
    row = 0

    def add(entries, low, high):
        nonlocal row
        for col, val in entries:
            rows.append(row); cols.append(col); vals.append(val)
        lo.append(low); hi.append(high); row += 1

    # capacity
    for i, av in enumerate(avs):
        for x in range(av.entry_point, av.exit_point):
            e = [(z_index[(i, j, x)], 1.0) for j in range(len(pvs)) if (i, j, x) in z_index]
            if e:
                add(e, -np.inf, float(av.capacity))
    # one tower per PV per point
    for j, pv in enumerate(pvs):
        for x in range(pv.entry_point, pv.exit_point):
            e = [(z_index[(i, j, x)], 1.0) for i in range(len(avs)) if (i, j, x) in z_index]
            if len(e) > 1:
                add(e, -np.inf, 1.0)
    # segment structure
    for (i, j), (cp, dp) in pair_span.items():
        for x in range(cp, dp):
            zx, sx = z_index[(i, j, x)], s_index[(i, j, x)]
            if x == cp:
                add([(zx, 1.0), (sx, -1.0)], -np.inf, 0.0)
            else:
                add([(zx, 1.0), (z_index[(i, j, x - 1)], -1.0), (sx, -1.0)], -np.inf, 0.0)
            if x + l_min > dp:
                add([(sx, 1.0)], 0.0, 0.0)
            else:
                for y in range(x, x + l_min):
                    add([(sx, 1.0), (z_index[(i, j, y)], -1.0)], -np.inf, 0.0)

    # clock: start value, then one indicator pair per unit interval
    for j, pv in enumerate(pvs):
        add([(t_index[(j, pv.entry_point)], 1.0)], pv.entry_time, pv.entry_time)
        for x in range(pv.entry_point, pv.exit_point):
            towers = [i for i in range(len(avs)) if (i, j, x) in z_index]
            t_next, t_now = t_index[(j, x + 1)], t_index[(j, x)]

            # free running: enforced only when no AV tows this interval
            free_z = [(z_index[(i, j, x)], -big_m) for i in towers]
            add([(t_next, 1.0), (t_now, -1.0)] + free_z, -np.inf, 1.0 / pv.speed)
            add([(t_next, -1.0), (t_now, 1.0)] + free_z, -np.inf, -1.0 / pv.speed)

            # towed: ride at the AV's speed, plus any wait taken at a coupling
            for i in towers:
                zx = z_index[(i, j, x)]
                step = 1.0 / avs[i].speed
                if waiting == "none":
                    # the PV's clock is the AV's clock
                    av_next = avs[i].time_at_point(x + 1)
                    if av_next is None:
                        continue
                    add([(t_next, 1.0), (zx, big_m)], -np.inf, av_next + big_m)
                    add([(t_next, -1.0), (zx, big_m)], -np.inf, -av_next + big_m)
                else:
                    wx = w_index[(i, j, x)]
                    #  t_next = t_now + w + step, active only when z = 1:
                    #    t_next - t_now - w + M z <= step + M
                    add([(t_next, 1.0), (t_now, -1.0), (wx, -1.0), (zx, big_m)],
                        -np.inf, step + big_m)
                    add([(t_next, -1.0), (t_now, 1.0), (wx, 1.0), (zx, big_m)],
                        -np.inf, -step + big_m)

    # the wait is zero except at a coupling, where it lifts the PV to the AV
    for (i, j), (cp, dp) in pair_span.items():
        if waiting == "none":
            continue
        for x in range(cp, dp):
            wx, sx = w_index[(i, j, x)], s_index[(i, j, x)]
            av_t = avs[i].time_at_point(x)
            # the lift can never exceed tau: coupling requires the clocks to
            # already agree within tau, so bounding w by tau rather than by M
            # keeps the LP relaxation usable.
            add([(wx, 1.0), (sx, -tau)], -np.inf, 0.0)            # w <= tau * s
            if av_t is not None:
                #  w >= t_av(x) - t[j,x]  when s = 1
                add([(wx, -1.0), (t_index[(j, x)], -1.0), (sx, big_m)],
                    -np.inf, big_m - av_t)

    # tolerance at the coupling point only
    for (i, j), (cp, dp) in pair_span.items():
        av = avs[i]
        for x in range(cp, dp):
            av_t = av.time_at_point(x)
            if av_t is None:
                continue
            sx, tx = s_index[(i, j, x)], t_index[(j, x)]
            #  t[j,x] - av_t <= tau + M(1-s)
            if waiting != "pv_only":
                add([(tx, 1.0), (sx, big_m)], -np.inf, tau + big_m + av_t)
            else:
                # the AV never waits: the PV may not arrive after it
                add([(tx, 1.0), (sx, big_m)], -np.inf, big_m + av_t)
            #  av_t - t[j,x] <= tau + M(1-s)
            add([(tx, -1.0), (sx, big_m)], -np.inf, tau + big_m - av_t)

    A = coo_matrix((vals, (rows, cols)), shape=(row, n_vars)).tocsr()
    c = np.zeros(n_vars); c[:n_z] = -1.0
    integrality = np.zeros(n_vars); integrality[:n_bin] = 1.0
    v_lb = np.zeros(n_vars); v_ub = np.ones(n_vars)
    if w_index:
        v_ub[n_bin + n_t:] = tau               # a lift is at most the tolerance

    # Per-variable clock windows. Leaving the clocks in [0, M] makes the LP
    # relaxation hopeless under the waiting conventions: fractional couplings
    # buy fractional waits and the bound drifts towards the spatial optimum
    # (measured: LP 240.9 against a true optimum of 63 on a 5x10 instance).
    # A PV cannot reach x before travelling at the fastest speed available, and
    # cannot be later than free-running at its own speed plus one tolerance per
    # coupling it could have made.
    # A towed PV travels at its AV's speed, which may be slower than its own, so
    # the window has to admit riding the slowest vehicle for the whole distance.
    # Each coupling can additionally shift the clock by at most tau.
    fastest = max(max(v.speed for v in avs), max(v.speed for v in pvs))
    slowest = min(min(v.speed for v in avs), min(v.speed for v in pvs))
    for j, pv in enumerate(pvs):
        for x in range(pv.entry_point, pv.exit_point + 1):
            d = x - pv.entry_point
            col = t_index[(j, x)]
            shifts = (d // max(l_min, 1) + 1) * tau
            v_lb[col] = max(0.0, pv.entry_time + d / fastest - shifts)
            v_ub[col] = pv.entry_time + d / slowest + shifts
    return c, A, np.array(lo), np.array(hi), integrality, v_lb, v_ub, n_z, n_bin


def solve_time(avs, pvs, l_min, *, tau: float, time_limit: float = 600.0,
               waiting: str = "either") -> ExactResult:
    import time as _time
    c, A, lo, hi, integrality, v_lb, v_ub, n_z, n_bin = build_time(
        avs, pvs, l_min, tau, waiting=waiting)
    cons = LinearConstraint(A, lo, hi)
    bounds = Bounds(v_lb, v_ub)
    t0 = _time.perf_counter()
    lp = milp(c=c, constraints=cons, bounds=bounds,
              integrality=np.zeros_like(integrality), options={"time_limit": time_limit})
    upper = -lp.fun if lp.success and lp.fun is not None else float("inf")
    res = milp(c=c, constraints=cons, bounds=bounds, integrality=integrality,
               options={"time_limit": time_limit, "presolve": True})
    secs = _time.perf_counter() - t0
    return ExactResult(optimal=(-res.fun if res.fun is not None else None),
                       upper_bound=upper, status=res.message,
                       proven_optimal=bool(res.status == 0),
                       n_binary=n_bin, n_constraints=A.shape[0], seconds=secs)
