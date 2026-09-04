# Working notes — SUMO pipeline

Track: reviewer weakness #1 (realistic data) for the 1D dynamic platoon
formation paper. Parallel track (MILP, weakness #4) is being worked in a
separate session; this file and `sumo/` are the boundary of this one.
Root `NOTES.md`, `core/`, `experiments/`, `analysis/`, `visualization/`
are read-only from here.

---

## 2026-09-04 — Installation

### Homebrew is a dead end. Do not retry it.

`brew tap dlr-ts/sumo && brew install sumo` fails on this machine, and the
failure is not fixable by retrying or by trusting the tap:

```
Error: dlr-ts/sumo/sumo: undefined method 'cxxstdlib_check' for class ... Sumo
```

Three separate problems, all verified by reading
`/opt/homebrew/Library/Taps/dlr-ts/homebrew-sumo`:

1. **The formula does not load.** `Formula/sumo.rb:53` calls
   `cxxstdlib_check :skip`, a Homebrew DSL method that no longer exists in
   Homebrew 6.0.21. All 24 versioned formulae in the tap carry the same line,
   so pinning an older SUMO version does not help.
2. **The tap is stale.** Its newest formula is SUMO **1.20.0** (2024).
   Current upstream is **1.27.1**.
3. **No bottle for this machine.** The bottle block covers
   `arm64_sonoma`/`ventura`/`monterey` only. This machine is macOS **15.7.9
   (Sequoia)**, arm64, so even a loadable formula would build from source,
   pulling cmake + fox + mesa + X11 + xerces-c + proj.

`brew trust dlr-ts/sumo` did succeed; it is the *formula* that is broken, not
the trust setting. The earlier "untrusted tap" error was a separate,
now-resolved issue.

### What was installed instead: the PyPI distribution

```bash
venv/bin/pip install -r sumo/requirements-sumo.txt
```

`eclipse-sumo` ships prebuilt macOS binaries as a `py3-none-macosx_14_0_arm64`
wheel. The `py3-none` tag means it carries no CPython version constraint, so
**Python 3.14 is fine** — this was the main risk and it did not materialise.

Verified working:

| | |
|---|---|
| SUMO | 1.27.1 |
| Binaries in `venv/bin` | `sumo`, `sumo-gui`, `netconvert`, `duarouter`, `polyconvert` |
| Build features | Proj, **GDAL**, GUI, SWIG, Eigen, Parquet, JuPedSim |
| Python tools | full `tools/` tree incl. `randomTrips.py`, `routeSampler.py`, `osmWebWizard.py` |
| `SUMO_HOME` | `venv/lib/python3.14/site-packages/sumo` |

GDAL and Proj are compiled in, which matters for the OSM import step.

**Environment impact: none.** A `pip install --dry-run` report before
installing showed exactly four new packages (`eclipse-sumo`, `sumolib`,
`traci`, `sumo-data`) and **no version change to numpy, scipy, pandas or
matplotlib**. The 48/48 reproduction guarantee from 2026-09-04 is therefore
intact, and the shared `venv/` is safe for the parallel MILP session.

Pins live in `sumo/requirements-sumo.txt`, deliberately *not* in the root
`requirements-lock.txt`, which describes the environment the published
results were reproduced in.

**`SUMO_HOME` is not required for normal use.** Verified 2026-09-04: with
`SUMO_HOME` unset, `venv/bin/sumo` runs to completion and `sumolib` / `traci`
import and read a net file. The pip package resolves its own data directory.

It is still worth exporting, because some tools reference `$SUMO_HOME`
explicitly — the `tools/` scripts are addressed by that path, and OSM import
uses the typemaps under `data/typemap/` (`osmNetconvert.typ.xml` and friends
are present in the package):

```bash
export SUMO_HOME="$PWD/venv/lib/python3.14/site-packages/sumo"
```

Not currently set in `~/.zshrc`. Scenario scripts in `sumo/` should set it
themselves rather than depending on the shell profile.

### Smoke test

`sumo/smoketest/` — two 1 km edges, 600 veh/h for 200 s, 300 s simulated.
`netconvert` built the net and `sumo` ran to completion (34 vehicles).
Generated `net.xml` / `fcd.xml` / `tripinfo.xml` are gitignored; the three
input XMLs are committed so the test can be re-run.

---

## 2026-09-04 — Output formats

### `--fcd-output` (per-vehicle, per-timestep)

```xml
<timestep time="30.00">
  <vehicle id="f0.0" x="866.36" y="-4.80" angle="90.00"
           type="DEFAULT_VEHTYPE" speed="28.36" pos="866.36"
           lane="e0_0" slope="0.00"/>
```

`pos` is the distance in metres along the *lane*, and `lane` identifies the
edge. Together they are the primitive the 1D projection needs: 1D coordinate
= (cumulative arc length of the corridor up to that edge) + `pos`.

Cost: ~145 bytes per vehicle per second. The smoke test produced 363 KB for
34 vehicles over 300 s. Extrapolating to 400 vehicles over a 1 h peak gives
roughly 200 MB — workable, but `--fcd-output.period` should be used if the
scenario grows. *(Inference from the observed rate; not measured at scale.)*

### `--tripinfo-output` (one row per completed trip)

```xml
<tripinfo id="f0.0" depart="0.00" departLane="e0_0" departPos="5.10"
          departSpeed="27.80" arrival="70.00" arrivalLane="e1_0"
          arrivalPos="1000.00" duration="70.00" routeLength="1995.00"
          waitingTime="0.00" timeLoss="1.78" speedFactor="1.06" .../>
```

Note `speedFactor` — SUMO already draws a per-vehicle speed multiplier, so
speeds are non-uniform even before congestion effects.

**Which to use.** `tripinfo` alone is not enough: it gives entry/exit of the
*whole trip*, not entry/exit of the chosen corridor, and no intermediate
timing. The projection needs FCD. `tripinfo` is still useful as a
cross-check and for `timeLoss`.

---

## Design decisions taken so far

### D1 — Use a freeway, one direction, not an urban arterial

Original plan said "pick an arterial". Changed to a **single direction of a
freeway** (Melbourne: Monash M1 or Eastern Fwy), because:

- It matches the model. `generate_mock_data()`'s parameter is literally
  `highway_length`, and physical towing is a weak story on a signalised
  arterial with turning traffic.
- The 1D coordinate is well defined. Arterials branch and have one-way pairs,
  which makes a single monotone arc-length coordinate awkward. A freeway is
  a clean chain of edges.
- **It helps the reviewer response rather than hurting it.** Entry and exit
  can only happen at ramps, so positions become discrete and clustered —
  which is the sharpest available contrast with uniform sampling.

One direction only: `core/models.py` enforces `entry_point < exit_point`, and
towing between opposing directions is physically meaningless.

### D2 — Keep the constant-speed model for the first iteration

`ActiveVehicle.time_at_point()` is linear: `entry_time + (x - entry) / speed`.
SUMO gives time-varying speed. Two options:

- **(a)** Extract only `(entry_pos, exit_pos, entry_time, effective_speed)`
  where `effective_speed = corridor_distance / corridor_duration`, leaving
  the model and both algorithms untouched. **Chosen for iteration 1.**
- **(b)** Extend the model to consume the real FCD position-time trace.
  More faithful, but it changes the feasibility predicate that the MILP in
  the other session is being written against. Deferred.

(a) carries an obligation: **measure the linearisation error.** Compare the
real FCD arrival time at corridor midpoints against the linear prediction,
and compare that error to `DEFAULT_TIME_TOLERANCE = 5.0` in
`core/greedy_multi.py`. If the error is comparable to or larger than the
tolerance, the time constraint stops meaning anything and (b) becomes
necessary. This is a cheap measurement and should be done in the first
scenario, not later.

### D3 — The pipeline must control |AV|, |PV|, capacity and corridor length independently

Reason in the risk section below. This is a hard requirement on the converter,
not a nice-to-have.

---

## Assessment: what SUMO can and cannot deliver

Recorded because it changes what the fortnightly update to Aamir should
contain, and because the Revision Plan currently over-promises here.

### The 1D projection discards most of what SUMO computes

SUMO models car-following, lane changing, signal control, ramp merging, route
choice and network topology. What survives the projection into
`ActiveVehicle` / `PassiveVehicle` is **four numbers per vehicle**:

```
entry_point, exit_point, entry_time, speed
```

So SUMO's entire contribution reduces to: *the joint distribution of those
four numbers stops being uniform and independent.* Whether that moves the
ILA–greedy gap is then a question about sensitivity to distribution shape.

### Evidence says the gap is controlled by something else

The 2026-09-04 re-analysis (root `NOTES.md`) found the gap is an inverted-U in
the AV:PV ratio, peaking at 1.26 pp around 0.8. That ratio is a **free
parameter I choose**, identically available in synthetic and in SUMO data.
SUMO does not set it. So a larger gap measured on SUMO data at ratio 0.8 is
not attributable to realism.

Correction to the earlier framing: the quantity that actually governs
contention is not `|AV|/|PV|` but total towing supply against demand,
roughly `Σ(capacity × AV route length) / Σ(PV route length)`. The earlier
sweeps held the capacity range fixed, which made the two proportional by
accident. SUMO route-length distributions differ from the synthetic ones, so
**ratio 0.8 in SUMO is not the same contention level as ratio 0.8 in
synthetic data.** Sweeps must be defined on the normalised supply/demand
quantity.

### A mechanism that pushes the gap the *wrong* way

Real freeway traffic clusters in space and time. Vehicles co-located on the
same segment at the same moment are automatically time-synchronised, so the
`time_tolerance` filter — which rejects a large share of pairs in uniform
synthetic data — will reject far fewer. More feasible pairs means less
contention, which means a **smaller** gap.

This is the opposite of Revision Plan item #2, which predicts realism widens
the gap via demand concentration. Both mechanisms are plausible; which
dominates is an empirical question and is cheap to test.
*(Inference, not verified.)*

### The mechanism that could still vindicate the plan

In the synthetic generator `entry`, `exit` and `speed` are drawn
**independently**. On a real freeway they are correlated: long trips
concentrate on particular ramp pairs, speed couples to time of day and
position, entry times follow a peak profile. Correlation can create *local*
scarcity pockets — a specific ramp pair in a specific window with one viable
AV — that independent uniform sampling never produces. That is exactly the
"only one viable partner" mechanism the plan relies on.

Not dismissible, and cheap to check. Worth two weeks; not worth two months.

### The one question the SUMO track has to answer

> **At matched supply/demand contention, does the SUMO instance distribution
> produce a larger ILA–greedy gap than the synthetic distribution?**

Two curves on one axis. Anything else cannot separate realism's contribution
from the choice of operating point. Hence requirement D3.

### The go/no-go instrument is the MILP, not SUMO

Aamir's question (2026-08-21) is whether ILA has enough headroom over greedy
to justify a much longer project. **SUMO cannot answer that**, because it
reports the gap on one dataset, not whether a larger gap is attainable at all.

How far greedy sits from optimal is a property of the *problem*, not of the
data's realism, and it is decisive:

| MILP result at peak contention | Reading | Consequence |
|---|---|---|
| greedy ≈ 99%+ of optimal | almost no headroom exists | no amount of realism produces a large gap; the framing has to change |
| greedy ≈ 90% of optimal | headroom exists, ILA captures little of it | ILA is the weak part; improving it is justified, and realism may plausibly help |

These imply opposite responses, and the test **runs today on the existing
synthetic data** — no need to wait for SUMO.

Practical consequences:

- The MILP session's first deliverable should be one number: greedy's
  optimality gap at the peak-contention operating point.
- The SUMO track is the **credibility deliverable** — mandatory, since
  weakness #1 was raised by R1, R2 and the meta-review and will block
  resubmission on its own — but it is not the decision instrument. Build the
  minimum viable pipeline (one freeway, one direction, one peak hour) and do
  not gild it.
- The fortnightly update to Aamir needs both tracks to be meaningful.

---

## Open question blocking the pipeline design

**How is demand generated?**

`randomTrips.py` samples origin and destination edges at random. If the
scenario is built that way, only the road network is real; the demand is
still random, and Revision Plan item #2's premise — "SUMO data concentrates
demand on popular routes" — does not hold. Reviewer weakness #1 would then
be addressed only cosmetically.

Alternatives to check: real count data (VicRoads / DTP Victoria open traffic
volumes — *existence and licence not yet verified*) fed through
`routeSampler.py`, `dfrouter` or `flowrouter`, or at minimum a realistic
temporal demand profile with ramp-weighted OD.

Jay met Mushfiq before this session; **what he advised on demand generation
determines which way the pipeline goes** and how much weight the
correlation-structure argument above can carry.

*(The `randomTrips.py` default behaviour above is stated from documentation
familiarity and has not yet been confirmed by reading the script in
`$SUMO_HOME/tools/`. Confirm before relying on it.)*

---

## Next

- [ ] Confirm with Mushfiq's advice which demand-generation route to take
- [ ] Read `$SUMO_HOME/tools/randomTrips.py` and confirm its default sampling
- [ ] Pick the corridor; extract the OSM extent and import with `netconvert`
- [ ] First scenario, one peak hour, and measure:
      (a) the entry-time distribution — is it actually non-uniform?
      (b) the D2 linearisation error against `time_tolerance = 5.0`
      (c) how often the time filter rejects a pair, versus synthetic data
- [ ] Converter: FCD → `ActiveVehicle` / `PassiveVehicle` lists matching
      `core/data.py::generate_mock_data()`, with |AV|, |PV|, capacity and
      corridor length as independent knobs (D3). AV/PV labelling is ours to
      define; SUMO has no such concept.
