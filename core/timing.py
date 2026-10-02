"""One place that knows how a towed PV's clock behaves.

Both matchers used to check a candidate coupling only at its own coupling point,
against the PV's clock *at that moment*. That misses the case where a tow
assigned later sits **upstream** of one assigned earlier: the new tow shifts the
PV's clock, and the already-committed downstream coupling silently stops
satisfying the tolerance. Greedy picks the longest candidate first, so this
happens whenever a long downstream tow is chosen before a shorter upstream one.

Measured before the fix: about 1.1 % of segments and 2 % of towed distance sat
in PVs with at least one broken coupling (NOTES.md, 2026-10-02).

Convention ("either"): whoever arrives first waits, so a tow starts at
max(pv_clock, av_clock) and the PV then rides at the AV's speed. This is what
the matchers already implemented at a single coupling; `simulate` applies it to
a PV's whole journey at once.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from core.models import ActiveVehicle, PassiveVehicle

# a tow of this PV: (cp, dp, av)
Tow = Tuple[int, int, ActiveVehicle]


def simulate(pv: PassiveVehicle, tows: Sequence[Tow]) -> List[Tuple[int, int, ActiveVehicle, float, float]]:
    """Replay the PV's journey; returns (cp, dp, av, coupling_time, decoupling_time)."""
    out = []
    t, x = pv.entry_time, pv.entry_point
    for cp, dp, av in sorted(tows, key=lambda s: s[0]):
        t += (cp - x) / pv.speed                       # free running up to the tow
        av_t = av.time_at_point(cp)
        coupling = t if av_t is None else max(t, av_t)  # the first to arrive waits
        decoupling = coupling + (dp - cp) / av.speed
        out.append((cp, dp, av, coupling, decoupling))
        t, x = decoupling, dp
    return out


def violations(pv: PassiveVehicle, tows: Sequence[Tow], tolerance: float) -> List[str]:
    """Couplings whose clocks disagree by more than the tolerance, after replay."""
    bad = []
    for cp, dp, av, coupling, _ in simulate(pv, tows):
        av_t = av.time_at_point(cp)
        if av_t is None:
            bad.append(f"{av.id} is not on the road at {cp}")
            continue
        # the PV's own clock on arrival is the coupling time minus any wait it
        # did for the AV, so compare the pre-wait clocks
        pv_t = coupling if coupling <= av_t else coupling
        if abs(pv_t - av_t) > tolerance + 1e-9:
            bad.append(f"{pv.id} meets {av.id} at {cp}: |{pv_t:.3f} - {av_t:.3f}| > {tolerance}")
    return bad


def is_feasible(pv: PassiveVehicle, tows: Sequence[Tow], tolerance: float) -> bool:
    """True when every coupling in `tows` still satisfies the tolerance."""
    t, x = pv.entry_time, pv.entry_point
    for cp, dp, av in sorted(tows, key=lambda s: s[0]):
        t += (cp - x) / pv.speed
        av_t = av.time_at_point(cp)
        if av_t is None or abs(t - av_t) > tolerance + 1e-9:
            return False
        t = max(t, av_t) + (dp - cp) / av.speed
        x = dp
    return True


def times(pv: PassiveVehicle, tows: Sequence[Tow]) -> dict:
    """Coupling and decoupling times keyed by (cp, dp), for recording on assignments."""
    return {(cp, dp): (c, d) for cp, dp, _, c, d in simulate(pv, tows)}
