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

**How is demand generated?** — body replaced with the 2026-09-17 survey.
Mushfiq gave no specific guidance ("try it yourself"). Below are the results of
reading (1) the `randomTrips.py` code, (2) the Victorian DTP open-data portal,
(3) the `routeSampler.py` code, with verified facts separated from inference.
Basis for question #1 at the 18 Sep meeting with Eunus.

### 1. `randomTrips.py` — confirmed behaviour (verified: code read directly, SUMO 1.27.1)

File: `venv/lib/python3.14/site-packages/sumo/tools/randomTrips.py` (1096
lines). This replaces the earlier "documentation familiarity, not confirmed"
caveat.

- **Default origin/destination sampling is uniform over edges.** No weighting
  by length, lanes or speed: `--length` defaults False (L111), `--lanes` False
  (L113), `--speed-exponent` 0.0 (L117–118), `--random-factor` 1.0 (L125),
  `typeFactors` default 1.0 (L301). So `edge_probability` (L479–541) returns 1
  for every eligible edge and `RandomEdgeGenerator.get` (L391–394) samples by
  cumulative weight. The 3.62 km main-line edge and a 60 m ramp stub have the
  same origin probability.
- **`--fringe-factor max`** (L127; applied L517–523): sets the probability of
  non-fringe edges to 0 (L522–523). Every trip starts/ends at a fringe edge (no
  incoming connection = origin / no outgoing = destination), but **uniformly
  among fringe edges.**
- **`--weights-prefix P`** (L56–58; `LoadedProps` L544–552; applied L564–569):
  reads `P.src.xml`/`P.dst.xml`/`P.via.xml` and **replaces the probability
  function entirely.** Edges not in the file get weight 0
  (`defaultdict(lambda: 0)`, L547). Format is
  `<edgedata><interval><edge id value/>` — identical to what
  `--weights-output-prefix` exports (L396–411), so the defaults can be dumped
  and edited.
- **OD pairs are drawn independently** (`get_trip` L422–448): origin (L427) and
  destination (L429) are sampled separately and accepted if the **Euclidean
  straight-line distance** from origin from-node to destination to-node is at
  least `--min-distance` (L441–443). Not route length. **Weights change only the
  marginal distributions; randomTrips cannot produce a joint OD structure.**
- **`--period`** (L188–191; L747–748): given N values, [begin,end] is split into
  N intervals with a different period each → a stepped peak profile is
  possible. Default `[1.0]` = one vehicle per second (L224–225). Without
  `--random-depart` (L203) departures are equally spaced within an interval.

**Cause of the 72.1 exit spike (verified on `m1.net.xml`, 2026-09-17).**
The first guess in the 2026-09-17 section ("randomTrips picks many fringe edges
behind that ramp") does not hold — there is 1 sink-fringe edge behind 72.1
(5.6, 15.5, 31.2, 86.8 have 2). Real cause: the network contains the outbound
carriageway and 13 of the 26 sink-fringe edges are on the outbound side. Since
`--fringe-factor max` samples uniformly among fringe edges, about half of the
inbound-origin trips draw an outbound-side destination. Those trips leave the
inbound main line at an off-ramp from which the outbound main line is reachable
and U-turn. 700/1212 = 58% matches. → The smoke savings 64.66/64.68% sat on an
artificial overlap of 700 vehicles exiting at one point. *(Cross-checked in the
build session: 518 of the 708 vehicles exiting at 72.1 did continue onto the
outbound main line; four off-ramps can reach outbound — 20.9, 37.4, 64.8, 72.1 —
and shortest-path routing concentrates them on 72.1.)* **Fix:** list only
inbound-side fringe edges in `.src.xml`/`.dst.xml` via `--weights-prefix`
(everything else is automatically 0). **Applied 2026-09-17 in
`sumo/m1/make_weights.py`; see the 2026-09-17 section for the corrected
numbers.**

### 2. Real traffic-count data (verified: portal pages opened 2026-09-17)

Portal `discover.data.vic.gov.au`, search "traffic volume" → 8 hits, 4 from
DTP. All **CC BY 4.0**, all **point counts, no OD** (as expected).
`opendata.transport.vic.gov.au` hosts the actual files.

| Dataset | URL (discover.data.vic.gov.au/dataset/…) | Spatial unit | Temporal unit | OD | Corridor coverage |
|---|---|---|---|---|---|
| **TIRTL Traffic Counts and Classification** | `tirtl-traffic-counts` | site (infrared detector), per direction, Austroads class and speed bins | **15 min**, monthly ZIPs (2025-11 to 2026-09), updated daily | none | **partial — below** |
| Telemetry Traffic Counts and Classification | `telemetry-traffic-counts-and-classification` | site, 15 min | 15 min, monthly ZIPs | none | **0 sites** (all 55 sites are rural) |
| Traffic Signal Volume Data | `traffic-signal-volume-data` | SCATS signalised intersections, loop detectors per lane | 15 min, 2014– | none | unchecked (site list not opened; TIRTL already has ramp detectors, so lower priority) |
| Historical Annual Average Daily Traffic Volume | `historical-annual-average-daily-traffic-volume` | declared-road segments, per direction | **AADT only**, 2001–2019, one GeoJSON per year | none | unchecked (2019 file 32 MB, not opened) |

**TIRTL coverage of the corridor (verified: `tirtl_sites.csv` 31 KB downloaded
and projected onto the inbound main line. Projection computed directly from the
UTM 55 parameters in net.xml, max error 0.01 m against 200 OSM nodes. Result
`sumo/osm/tirtl_on_corridor.json`, gitignored.)**

- **Grid 2.0–39.5 (km 0.4–8.1)**: 16 inbound main-line detectors at ~500 m
  spacing ("M1 Inbound - CH 15860 … 23810"). Springvale (5.6) to Warrigal (37.4)
  = the eastern 40% is densely covered.
- **Grid 61.1–62.2**: 1 main-line detector ("Before High Street Bridge Inbound")
  + **4 High Street on-ramp detectors** (Top/Mid/Before_Stop/After_Stop = either
  side of the ramp-metering stop line). Immediately upstream of our ON ramp at
  62.9 → this ramp matches for certain.
- Further ramp detectors: Stephensons (27.3), Stanley (29.5), Atkinson (35.2)
  Inbound. Our OSM on-ramps were unnamed; this data names them. **But which OSM
  ramp is which needs projection onto the ramp edges themselves (not done,
  ~30 min).**
- **Grid 39.5–61.1 (Warrigal–High St, 4.4 km): no detectors.**
- **Grid 62.2–100 (Burke Rd, Toorak, CityLink, 7.8 km): no detectors.** CityLink
  is a Transurban toll road, so presumably absent from state data (inference).
  **The former spike point 72.1 and the whole CityLink section are outside the
  measured range.**

**Further verification required:** (a) whether the AADT 2019 GeoJSON contains
the Burke Rd–Toorak Monash main-line segment (toll-road exclusion not confirmed
on the page; even if present it is a daily average, no peak profile), (b) the
actual size and column layout of a TIRTL monthly ZIP (not downloaded), (c)
whether the Traffic Signal site list includes M1 ramp-metering signals.

### 3. `routeSampler.py` — confirmed behaviour (verified: code read directly)

File: same path, `tools/routeSampler.py` (1527 lines).

- **Input:** `-r/--route-files` **candidate route file is mandatory** (L50–51
  `required=True`); counts via at least one of `-d/--edgedata-files` (per edge,
  L56), `-t/--turn-files` (per turn, L52), `-O/--od-files`
  (edgeRelation/tazRelation OD, L58). Default edge-count attribute `entered`
  (L62) = what a detector counts.
- **Algorithm** (`sampleRoutes` L1183–1252): pick a detector with remaining count
  at random (L1207) → pick **uniformly** one candidate route passing it (L1208)
  → add one vehicle and decrement every detector on that route by 1
  (L1215–1216) → repeat until all counts are filled or no route is usable.
  `--weighted` (L1204) samples routes by their probability. `--optimize` (L739)
  runs a posterior LP to reduce GEH mismatch. `--geh-ok` default 5 (L141).
- **Meaning: routeSampler does not estimate OD.** The OD structure comes from
  (a) which routes exist in the candidate file and (b) a bias towards long
  routes that pass many detectors (`--minimize-vehicles` L138 explicitly
  strengthens this). **The trip-length distribution is a product of the method,
  not of the data.** The "18% below L_min" of the 2026-09-17 section (28%
  before the 72.1 fix) will change under this method, but the new value is not
  a measurement either. This must be stated when it goes into the paper.

**Minimal path from point counts to demand (prose, no code):**
① Candidate routes: one explicit route per (entry, exit) pair with entry < exit
among the 9 inbound entries × 9 exits (≈40 routes) — enumerating instead of
randomTrips gives full control of the OD set and rules out the 72.1 problem at
source. ② Counts: from a TIRTL monthly ZIP take one chosen morning peak hour for
the 16 inbound main-line sites + the High St ramp site, aggregate 15 min → 1 h,
map each site to its main-line/ramp edge via the grid position in
`tirtl_on_corridor.json`, and write
`<edgeData><interval begin end><edge id entered=N/>` XML.
③ `routeSampler -r candidates.rou.xml -d counts.xml -o result.rou.xml`, with
`--optimize full` if needed. ④ Feed the resulting route file to `sumo`. The
western 60% with no measurements is filled by routeSampler from candidate
routes and random choice — that section must be flagged **uncalibrated** in
the results. GEH values via `--mismatch-output` (L75) can be reported as
calibration quality.

### 4. Options for the 18 Sep meeting

| | Cost | What it buys |
|---|---|---|
| **(i) Keep the randomTrips placeholder** (+ inbound-only weights fix) | fix 30 min, done. No extra data | pipe works; positions are real ramps (16 points), speeds from car-following. **Times and OD still uniform** → reviewer #1's "uniformly sampled … times" remains |
| **(ii) Partial calibration with TIRTL counts via routeSampler** | 2–3 days: download one monthly ZIP, map sites → edges, enumerate candidate routes, run, check GEH | eastern 40% main line + High St ramp **calibrated to government 15-min detector counts**; peak-hour profile is measured. Western 60%, OD and trip lengths remain method products — **must be labelled "partial calibration" honestly** |
| **(iii) Full calibration of the whole corridor** | weeks + data we do not have (CityLink counts are Transurban's, OD needs a travel survey) | defensible whole-corridor demand. **Not possible** with current open data |

**Recommendation: (ii) for the first paper iteration.** Reasons: (i) does not
answer "times" among reviewer #1's three words ("positions, times, and
speeds"). (ii) buys, for 2–3 days, the sentence "calibrated to DTP TIRTL
detector counts (GEH < 5 at N sites)", which answers the meta-review directly.
The coverage gap and the absence of OD estimation are stated as limitations,
and stated limitations rarely reject a paper. (iii) is outside this paper's
scope and belongs to the 2D paper.

**(i) + the fix was applied immediately** — the structure measurements that do
not depend on the demand model (D2 error, feasible-pair share) and the D6 MILP
comparison did not need to wait for (ii), and the 72.1 bias contaminated them
too. Done 2026-09-17.

### 5. The toll boundary — a limit of both (i) and (ii) (raised by Jay, 2026-09-17)

Jay: "Toorak Rd is the last free exit before the CityLink toll; I take it myself
to avoid CityLink." **Verified in `monash-m1.osm`:** all 41 CityLink ways carry
`toll=yes`, and so do the Toorak→CityLink on-ramp (`Toorak-Citylink In Ramp
On` variants) and the Yarra Boulevard Off Ramp. So on the inbound carriageway
**the last free exit is 72.1 Toorak, and 86.8 Yarra Blvd is already inside the
tolled section.**

Nothing in the simulation knows about tolls — neither `duarouter` nor
`inbound.dst.xml` (13 edges, all weight 1.0). Consequently:

- the first run (700 at 72.1) over-represented Toorak for the **wrong reason**
  (outbound-destination U-turns, §1);
- the re-run (244 at 72.1, applied by the other session) treats Toorak like any
  other exit and therefore **under-represents** it. Its largest exit is now
  Yarra Blvd at 493, a topology artefact (two sink edges → double weight) that
  points the wrong way relative to reality (inside the toll).
- Grid 62–100 has no TIRTL sensor, so **option (ii) cannot calibrate this
  either.**

Note also the direct count on the *current* `routes.xml`: of the 244 vehicles
leaving at Toorak, 100% have inbound-side destinations. The earlier claim that
the original 700 were outbound-destination U-turns rests on the topology
argument (Toorak is the only inbound off-ramp that reaches the outbound
carriageway) plus the 58% arithmetic; the original routes file was overwritten
before a direct count could be made, so that magnitude is inference.

**Possible responses (unverified, cheapest first):** (a) raise the Toorak sink
weight in `inbound.dst.xml` and justify it from the AADT 2019 GeoJSON — the
drop in daily volume between the Monash main line just before Toorak and the
CityLink main line just after it is the toll-avoidance share (32 MB, not opened;
whether tolled segments appear in AADT at all is unconfirmed). (b) Give toll
edges a travel-time penalty via `duarouter --weight-files` so the router avoids
them — both the option and whether netconvert carries the OSM `toll` tag into
`net.xml` need checking. (c) In the paper: "toll boundary at Toorak Rd not
modelled; western section uncalibrated."

Recommendation (ii) stands, but **the limitations wording must name the toll
boundary.** A Melbourne-based reviewer will see it immediately.

**Link to meeting question #3 (physical unit of τ):** under (ii) entry times
come from real-second detector counts, so reading τ in real seconds is the
natural choice. Under (i) 5 model units is easier to defend. The demand
decision partly determines the τ decision.

---

## Next

- [x] ~~Confirm with Mushfiq's advice~~ — no guidance given; decided ourselves (§4 above)
- [x] ~~Read `randomTrips.py` and confirm its default sampling~~ — §1
- [x] ~~Pick the corridor; extract OSM; import with `netconvert`~~ — D4/D5
- [x] ~~First scenario, one peak hour, measure (a)(b)(c)~~ — 2026-09-17 section
- [x] ~~Converter~~ — `sumo/convert.py`
- [x] ~~Apply inbound-only src/dst via `--weights-prefix` and re-run~~ —
      `sumo/m1/make_weights.py`; after re-run: 1800/1800 use the main line,
      OD 44, 18% below L_min, exit spike gone (max exit point 493 at 86.8,
      which has two sink edges)
- [ ] After the meeting, if (ii) is adopted: download one TIRTL monthly ZIP and
      check size/columns → map sites → edges (from `tirtl_on_corridor.json`;
      re-project ramp detectors onto ramp edges) → enumerate candidate routes →
      routeSampler → report GEH
- [ ] Check whether the AADT 2019 GeoJSON has the Burke Rd–Toorak main-line
      segment (to fill at least the western main-line total)
- [ ] **Toll boundary (§5):** from AADT, volume just before vs just after Toorak
      → basis for a Toorak sink weight; or check whether `duarouter
      --weight-files` can penalise `toll=yes` edges
- [ ] Draft paper wording: "demand partially calibrated to DTP TIRTL 15-min
      counts (16 mainline + 1 ramp site, eastern 40% of corridor); western
      section and OD structure uncalibrated" — limitations first
---
## 2026-09-04 — Corridor fixed, OSM import

### D4 — Corridor: M1 Monash Freeway, inbound (towards the CBD)

Jay specified "CBD ~ Clayton". Better than the original proposal
(EastLink → Burnley): it is the Monash commuter axis, so the paper can motivate
it as a real commuting corridor rather than "an arbitrary freeway". Tunnels
excluded for the first iteration.

**Data acquisition.** Overpass API, bbox `(-37.93, 145.00, -37.81, 145.17)`,
filter `highway=motorway|motorway_link`. Result `sumo/osm/monash-m1.osm`,
305 KB, 341 ways / 1811 nodes. Query kept in `sumo/osm/query.overpassql`.
The western bbox edge at 145.00 effectively cuts off the Burnley tunnel (only
2 tunnel ways remain).

**Conversion.**

```bash
netconvert --osm-files monash-m1.osm \
  --type-files "$SUMO_HOME/data/typemap/osmNetconvert.typ.xml" \
  --output-file m1.net.xml \
  --geometry.remove --ramps.guess --junctions.join \
  --remove-edges.isolated --keep-edges.by-vclass passenger \
  --output.street-names true --output.original-names true
```

`--output.street-names true` is mandatory. Without it every edge name is
dropped and main line cannot be told from ramps (the first attempt produced
zero main-line edges for exactly this reason).

### Corridor structure

| | inbound (to CBD) | outbound (to Clayton) |
|---|---|---|
| Longest connected chain | **31/31 edges (unbroken)** | 36/40 (4 breaks) |
| Length | **20.53 km** | 22.60 km |
| Ramp junctions | 16 (ON 8 / OFF 8) | 16 |

**Inbound adopted.** Fully connected and matches the morning-peak direction.

Lane count grows 3 → 4 → 5 → 6 towards the CBD. This is realism the synthetic
data lacks and could be tied to the AV capacity model.

### Verification (done in code, not by eye in netedit)

- **0 dead-end ramps.** Every ramp attached to the corridor connects onward.
  No OSM import quality issue.
- **8.96–12.58 km is a single edge (3.62 km).** No junction at all, between
  Warrigal Rd and Burke Rd. This is a real interchange-free stretch, not
  missing data. **A naturally occurring scarcity zone** — a structure uniform
  synthetic data cannot produce.

### Ramp positions (converted to the 100-unit grid) — key result

```
 5.6 OFF Springvale Rd    43.6 ON  Monash Fwy
12.5 ON  Monash Fwy       62.9 ON  Monash Fwy
15.5 OFF Blackburn Rd     64.8 OFF Burke Rd
20.3 ON  Monash Fwy       67.9 ON  Monash Fwy
20.9 OFF Forster Rd       72.1 OFF Monash In-Toorak
25.6 ON  Monash Fwy       75.5 ON  Toorak-Citylink
31.2 OFF Huntingdale Rd   86.8 OFF Yarra Boulevard
37.4 OFF Warrigal Rd      92.1 ON  CityLink
```

**The synthetic data spreads positions uniformly over the 101 integer points
0–100. The real M1 has entry/exit at only 16 points.** This is the first
substantive difference confirmed by adopting SUMO, and a partial
counter-example to the scepticism earlier in this file that "the 1D projection
passes through only four numbers" — the marginal distributions of
`entry_point`/`exit_point` really do change a lot.

(Whether this widens the ILA–greedy gap remains a separate question. The gap is
a function of contention, and contention is still a parameter we set.)

### D5 — Units: keep the 100-unit grid

20.53 km / 100 = **1 unit ≈ 205 m**. So `L_MIN = 10 units ≈ 2.05 km` as the
minimum towing distance, which is physically reasonable.

Why not switch to metres: the existing 48 combinations must stay on the same
axis for "synthetic curve vs SUMO curve overlaid at equal contention" to be
meaningful. Changing units makes that comparison impossible.

Note: `ActiveVehicle.entry_point` is an `int`, so the converter has to
quantise to 205 m steps. Ramp spacing is 1–3 km, so quantisation loss is
negligible in practice.

### Remaining risks

- **Ramp metering.** The M1 is a managed motorway with ramp signals, which the
  OSM import does not reproduce. Ignore in the first iteration, but a reviewer
  may raise it when the realism of entry patterns is discussed.
- **On-ramp names lost.** Off-ramps keep their names (Blackburn, Warrigal, …)
  but most on-ramps are tagged only `Monash Freeway On Ramp`, so the feeder
  road is unknown. Harmless while only positions are needed.
- **CityLink toll section included.** The western end of the corridor (beyond
  75.5 units) is CityLink. Tolls affect real route choice; randomTrips ignores
  them.

### Next

- [ ] Decide demand generation (still the blocker) — check Mushfiq's advice
- [ ] Run the first scenario and measure the D2 linearisation error

---

## 2026-09-04 — The MILP results change the SUMO track's question

The parallel session produced exact-solver results (root `NOTES.md`,
`data/results/milp/`). They answer the claim earlier in this file that "the
go/no-go instrument is the MILP", so they are reflected here. **What follows is
a reading of the MILP session's results from the SUMO track's point of view;
the original analysis is in the root `NOTES.md`.**

> *Added 2026-09-17:* the "coin flip" reading below was written on the
> time-free results only. Time-on results at paper scale (root `NOTES.md`,
> "Two regimes") later showed the shipped ILA ahead of greedy by +1.3–2.5 pp in
> all four configurations. The "heuristic vs optimum" question stays valid;
> "the ILA–greedy gap is not worth measuring" does not.

### Result summary (as read)

| | mean % of optimum |
|---|---|
| greedy | 93.33% |
| ILA | 93.09% |

ILA 9 wins / 8 losses / 4 ties. A coin flip, marginally behind on average.
Of the remaining ~7%, only 0.65% is lost to the algorithms' expressiveness
limit; the other ~6% is choosing badly among available options.

Scope: AV 4–15 / PV 8–30, **spatial problem without time constraints only**.
Different scale from the existing sweep (AV 10–320 / PV 50–800), so
generalisation to large instances is not yet supported.

### This is the second row of the earlier table

Of the two branches set up in this file's "go/no-go instrument is the MILP"
section, this is **"there is headroom and ILA is not taking it"**. The ceiling
is not low; both heuristics are mediocre. Against the largest observed
ILA–greedy gap of 1.26 pp the real headroom is ~7 pp, so there is genuine room
for a better algorithm.

### The SUMO track's question changes

The question set earlier:

> ~~At matched contention, do SUMO distributions widen the ILA–greedy gap?~~

**Dropped.** If ILA and greedy are a coin flip, the gap between them is not a
quantity worth measuring. Replacement question:

> **Does realistic instance structure increase or decrease the headroom to the
> optimum (~7%)?**

That is, the comparison moves from "ILA vs greedy" to "heuristic vs optimum".
If the M1 structure confirmed today — entry/exit at 16 points rather than 101,
a 3.62 km stretch with no choices at 8.96–12.58 km — raises contention, the
headroom grows and a better algorithm is worth more. This question cannot be
answered without SUMO, and only with the MILP as the reference.

### D6 — The converter must be able to extract small instances

The MILP takes 142 s at AV 15 / PV 30 and fails to prove optimality within 60 s
at AV 20 / PV 40. A one-hour SUMO peak scenario is far larger.

So besides exporting a whole scenario, the converter **must be able to extract
small sub-instances (AV ~10 / PV ~25) from the real corridor**, e.g. by
splitting the peak hour into 5-minute windows. That keeps the M1's real
structure (16 ramp points, the scarcity zone, lane-count changes) while
bringing the size within MILP reach.

Without this the new question above cannot be answered — there would be no
way to compute the optimum baseline.

### Caution

The MILP handles only the spatial problem with `enable_time_constraints=False`.
One of the biggest contributions of SUMO data is a realistic time distribution,
and on that axis there is no optimum baseline yet. A time-on comparison needs a
MILP extension, which is non-linear (towing changes later arrival times) and
not simple. It is safer to align the first iteration on the spatial problem so
that comparability is secured.

---

## 2026-09-17 — First M1 run, converter, structure measurements (pipeline passes end to end)

Purpose: before the 18 Sep meeting with Eunus, confirm the pipe works all the
way through and put numbers on the structural differences that hold regardless
of the demand model. **Demand is a placeholder** (`randomTrips`, uncalibrated).
The saving figures below are evidence that the pipe works, not results.

### What was built

| File | Role |
|---|---|
| `sumo/m1/corridor.py` → `corridor.json` | Reproducible script for the inbound main-line chain + ramp positions (previously computed inline, no file). Output matches D4/D5 exactly: 31 edges, 20.53 km, 16 ramps (ON 8 / OFF 8), identical grid positions |
| `sumo/m1/make_weights.py` → `inbound.src.xml`, `inbound.dst.xml` | Restricts randomTrips origins/destinations to the 13 + 13 fringe edges on the inbound side (see *Correction* below) |
| `sumo/m1/trips.xml`, `routes.xml`, `m1.sumocfg`, `tripinfo.xml`, `fcd.xml` | `randomTrips.py -b 0 -e 3600 --period 2 --weights-prefix inbound --seed 42 --validate`; `sumo` runs in ~4 s, all 1800 vehicles arrive. FCD at 10 s intervals (1 s would be hundreds of MB) |
| `sumo/convert.py` | `load_corridor_trips()` — main-line entry/exit metres from the route, least-squares `x = a + b·t` on FCD samples on the main line → speed b, `entry_time`, residuals. `label()` — AV/PV split (only `random` so far), capacity (2,4), drops trips below L_min, time-unit rescaling option. `summary()` — structure metrics |
| `sumo/m1/structure.json` | The measurements below |
| `sumo/ratio_sweep.py` → `sumo/m1/ratio_sweep.csv` | AV:PV ratio sweep 0.2–0.8, greedy vs ILA, time OFF/ON |
| `sumo/smoke_match.py` → `sumo/m1/smoke_match.json` | Smoke run of greedy / ILA on the converted instance (time OFF/ON, model units vs real seconds) |
| `analysis/plot_corridor.py` → `analysis/figures/m1_corridor.png` | Corridor schematic: 16 ramps, lane count per edge, the 3.6 km no-junction stretch |
| `analysis/plot_entry_dist.py` → `analysis/figures/entry_dist_synth_vs_m1.png` | Synthetic (80/400, seed 42) vs M1: entry, exit and trip-length histograms with ramp positions as dashed lines |

### Correction — the 72.1 exit spike in the first run (found by the demand-survey chat)

The first run used `--fringe-factor max` on the whole network. 708 of 1212
main-line vehicles exited at 72.1 (Toorak). My first explanation ("randomTrips
picks many fringe edges behind that ramp") was **wrong**: that ramp has one
sink edge. The real cause, verified here: `m1.net.xml` contains both
carriageways, half of the 26 fringe sinks are on the outbound side, and
randomTrips samples destinations uniformly over fringe edges. Trips bound for an
outbound-side sink leave the inbound main line at the first off-ramp from which
the router can reach the outbound carriageway — 518 of the 708 vehicles exiting
at 72.1 continued onto the outbound main line (U-turn trips). Four off-ramps
can reach outbound (20.9, 37.4, 64.8, 72.1); shortest-path routing concentrates
them on 72.1.

Fix: `make_weights.py` writes weight files listing only fringe sources that
reach the inbound chain, and fringe sinks reachable from it, without touching
the outbound chain; `--weights-prefix inbound` gives every unlisted edge weight
0. Everything below is from the corrected run. The numbers of the first run
(OD 43, 28% below L_min, 64.66/64.68% saving) are superseded.

### Structure measurements — hold regardless of the demand model

| | synthetic | M1 inbound |
|---|---|---|
| Vehicles using the main line | — | 1800 / 1800 → 1749 fitted (51 had < 2 FCD samples on the main line) |
| Possible entry points | 101 | **9** (corridor start + 8 ON) |
| Possible exit points | 101 | **9** (8 OFF + corridor end) |
| OD pairs actually occurring | thousands | **44** |
| Trips below L_min = 10 (≈ 2.05 km) | 0 by design | **320 / 1749 = 18%** (adjacent ramps are 0.6 km apart in places) |
| Speed | 0.8–1.2 (arbitrary) | mean 88 km/h, range 15–120 |

Residual demand artefact: exits are still uniform over sink *edges*, so a ramp
with two sink edges (Yarra Blvd, 86.8) gets twice the share (493 vehicles).
Placeholder, not calibrated.

### D2 (constant-speed assumption) error — measured

These are residuals after fitting the best constant speed, so they are a
**lower bound on the error of any constant-speed model**. FCD sampled at 10 s.

| Trip length | n | median of per-vehicle max residual | share exceeding τ = 5 s |
|---|---|---|---|
| < 4 km | 710 | 0.56 s | 0% |
| 4–10 km | 435 | 2.5 s | 36% |
| 10–15 km | 322 | 13.9 s | 95% |
| > 15 km | 282 | 21.9 s | 100% |

Reading: the error accumulates with trip length (lane count 3→6, merges,
deceleration before exits). **If "τ = 5 s" is read as real seconds, the
constant-speed model exceeds the tolerance for almost every trip longer than
10 km on the M1.** So revision item #10 (temporal synchronisation semantics) is
a model-choice problem, not only a definitions problem. Options: (a) larger τ,
(b) piecewise speeds, (c) read τ as 5 model time units rather than 5 s — see
below.

### Time-unit problem (newly surfaced)

The synthetic generator uses speed ≈ 1 grid unit per time unit, τ = 5, entry
window 100. With 1 grid unit = 205 m and the M1 mean speed, one model time
unit = **8.43 s**. Hence:

- reading the paper's τ = 5 in **model units** gives **≈ 42 s** of real time
- reading it as **5 real seconds** roughly halves the saving (see smoke run)

The paper writes "τ = 5 s" but never fixes the physical scale of the grid, so
both readings are self-consistent. **Decision needed at the meeting.**
`convert.label(time_unit_s=0)` gives model units (mean speed normalised to 1);
`time_unit_s=1` gives real seconds.

### Smoke run (AV 286 / PV 1143, ratio 0.25, `av_fraction=0.2 seed=42`, capacity 2–4)

| | time OFF | time ON, τ = 5 model units (≈ 42 s) | time ON, τ = 5 real s |
|---|---|---|---|
| greedy | 60.00% (101 s) | 50.43% | 24.10% |
| ILA | 60.02% (1.6 s) | **51.67% (+1.24 pp)** | 24.16% |

How to read this: (i) the pipe works. (ii) 60% is above the synthetic 25–52%
range; the instance is 3.6× the paper's 80/400 and demand is uniform over
inbound-side fringe edges, so this says nothing about realistic saving yet.
(iii) ILA +1.24 pp with time ON points the same way as "ILA's advantage comes
from time constraints" in the root `NOTES.md`, but **one sample, placeholder
demand** — nothing yet. (iv) greedy 46–105 s vs ILA ≈ 2 s — the #3 runtime
unfairness is unchanged and gets worse with instance size.

### Ratio sweep — first look (`sumo/ratio_sweep.py`, one seed, random labels, placeholder demand)

`--ratios 0.2 0.4 0.6 0.8`, 1429 usable vehicles split so that |AV|/|PV| = ratio,
capacity 2–4, τ = 5 model units. Output `sumo/m1/ratio_sweep.csv`.

| ratio | AV | PV | time OFF (greedy = ILA) | greedy ON | ILA ON | ILA − greedy | PV distance covered (ILA ON) |
|---|---|---|---|---|---|---|---|
| 0.2 | 238 | 1191 | 50.10% | 43.69% | 44.28% | **+0.60** | 53% |
| 0.4 | 408 | 1021 | 71.23% | 58.96% | 60.82% | **+1.86** | 85% |
| 0.6 | 536 | 893 | 62.28% | 58.82% | 59.85% | **+1.03** | 96% |
| 0.8 | 635 | 794 | 55.65% | 54.20% | 54.69% | **+0.49** | 98% |

Reading:
1. With time constraints OFF the gap is exactly 0.00 at every ratio; it opens
   only with time ON. Same as the synthetic finding in the root `NOTES.md`.
2. The gap is an inverted U with its peak at 0.4 here, versus 0.8 on synthetic
   data. Inference: the last column shows PV coverage already at 96% by ratio
   0.6 — nothing left to compete for, so the algorithms converge. M1 trips
   enter at 9 points, so overlap is high and saturation comes earlier.
3. The paper's saving metric divides by AV + PV distance, so it peaks at 0.4
   and then falls as the AV share of the denominator grows (PV share of the
   baseline: 83% → 56%). PV coverage is monotone (53% → 98%). Which metric to
   report is worth raising at the meeting.
Figure: `analysis/plot_ratio_m1_vs_synth.py` → `analysis/figures/ratio_m1_vs_synth.png`
(M1 overlaid on the stored synthetic pv_av sweep). Against synthetic cells of
comparable size: M1 gap is larger at 0.2 (+0.60 vs +0.17–0.40) and 0.4 (+1.86
vs +1.12–1.25), smaller at 0.8 (+0.49 vs +1.26). Saving level is 10–15 pp higher
on M1 at every ratio (concentrated entries → more overlap). Same algorithms;
nothing was "improved" — only the instance distribution changed. Synthetic cells
at the same ratio already differ by up to 1.25 pp among themselves, so one M1
seed cannot confirm or refute Revision Plan item #2; it is consistent with it.
4. Direction and shape (time-only advantage, inverted U) agree with the
   synthetic sweep and are somewhat trustworthy; magnitude and peak position
   are one seed on placeholder demand and are not.

### Verification checklist (other chat or Jay, 10 minutes)

- [ ] `PYTHONPATH=. venv/bin/python sumo/m1/corridor.py` → 31 edges / 20.53 km / 16 ramps, grid positions identical to the D5 table
- [ ] `PYTHONPATH=. venv/bin/python sumo/m1/make_weights.py` → 13 sources / 13 sinks
- [ ] `PYTHONPATH=. venv/bin/python sumo/convert.py sumo/m1 --summary` → fitted 1749, OD 44, short 320
- [ ] Every `label()` output satisfies `entry_point < exit_point`, `0 ≤ … ≤ 100`, `exit − entry ≥ 10`, `speed > 0`
- [ ] With `time_unit_s=0`, mean speed ≈ 1.00
- [ ] Re-running `analysis/plot_entry_dist.py` reproduces the figure; no single exit point above ~500

### Decisions needed (promoted to meeting questions)

1. Demand generation — keep the placeholder vs real counts (see the other chat's survey)
2. AV/PV labelling rule — random fraction / vehicle class / other
3. **Physical unit of τ** — 5 real seconds vs 5 model units (≈ 42 s); handling of the constant-speed drift on long trips
4. The 18% below L_min — exclude (current) vs include in the baseline denominator

### Next

- [ ] Implement `label(rule=…)` once the labelling rule is fixed
- [ ] Ratio sweep 0.2–1.6 (meaningful only after labelling is fixed)
- [ ] D6: extract MILP-sized instances by splitting into 5-minute windows
- [ ] Demand: `routeSampler` route depending on the other chat's survey
