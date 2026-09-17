# SUMO primer — for this repository

Based on what is actually installed and verified in this project (SUMO 1.27.1,
pip distribution). Not a general tutorial; only the parts our pipeline needs.

---

## 1. What SUMO is

**Simulation of Urban MObility** — an open-source traffic simulator written by
the German Aerospace Center (DLR) and maintained by the Eclipse Foundation.
EPL-2.0.

Two defining properties:

**(1) It is microscopic.**
Unlike macroscopic models that treat traffic as a fluid with averages, it
**tracks every vehicle as an individual object** with its own position, speed,
acceleration and destination. Our problem (an AV tows a specific PV from a
specific point) is inherently per-vehicle, so microscopic is a must.

**(2) It is time-discrete.**
Default step 1 s (changeable with `--step-length`). At every step it computes
the next speed of every vehicle with a car-following model (default: Krauss)
and updates positions. Hence the output is "every second, every vehicle's
position" — that is FCD.

**It is deterministic.** Same input + same `--seed` = identical output. This
fits the reproducibility our `analysis/reproduce_check.py` demands.

---

## 2. Structure: four concepts and you are done

What makes SUMO look hard is not the syntax but "why are there so many
files"; there are only four roles.

| Role | File | Meaning |
|---|---|---|
| **Network** | `net.xml` | what the roads look like |
| **Demand** | `*.rou.xml` | who goes when, from where to where |
| **Config** | `*.sumocfg` | binds the two and sets options |
| **Output** | `fcd.xml`, `tripinfo.xml` | results |

Pipeline:

```
  nodes.xml + edges.xml  ──netconvert──►  net.xml
  (or an OSM .osm file)                       │
                                              ▼
  trips.xml ──duarouter──► routes.rou.xml ──► sumo ──► fcd.xml
                                                       tripinfo.xml
```

**Never write `net.xml` by hand.** It is always a product of `netconvert`.
(To edit by hand use the `netedit` GUI.)

---

## 3. Syntax — all XML

The files below exist in `sumo/smoketest/` and are verified to run.

### 3.1 Nodes (junctions)

```xml
<nodes>
  <node id="n0" x="0.0"    y="0.0"/>
  <node id="n1" x="1000.0" y="0.0"/>
  <node id="n2" x="2000.0" y="0.0"/>
</nodes>
```

`x`, `y` are planar Cartesian coordinates in metres. The three above lie on
the x-axis 1 km apart.

### 3.2 Edges (roads)

```xml
<edges>
  <edge id="e0" from="n0" to="n1" numLanes="2" speed="27.8"/>
  <edge id="e1" from="n1" to="n2" numLanes="2" speed="27.8"/>
</edges>
```

**Edges are directed.** `from`→`to` only. A two-way road is two edges. This
is why our decision "one freeway direction only" is natural in SUMO.

`speed` is in **metres per second**. 27.8 m/s ≈ 100 km/h. The most common
beginner mistake — beware.

### 3.3 Compile

```bash
netconvert -n nodes.xml -e edges.xml -o net.xml
```

`netconvert` generates lane connections, internal junction links, priorities
and signals from the nodes/edges and writes `net.xml`. The same tool handles
OSM:

```bash
netconvert --osm-files melbourne.osm -o net.xml \
           --type-files "$SUMO_HOME/data/typemap/osmNetconvert.typ.xml"
```

### 3.4 Demand — four ways

**(a) `<trip>` — origin and destination only; a router computes the path**

```xml
<trip id="t0" depart="0.00" from="e0" to="e1"/>
```

Must be converted to `.rou.xml` by `duarouter` before `sumo` accepts it.

**(b) `<vehicle>` + `<route>` — explicit path**

```xml
<route id="r0" edges="e0 e1"/>
<vehicle id="v0" route="r0" depart="0.00"/>
```

**(c) `<flow>` — generate many vehicles automatically (used in the smoke test)**

```xml
<routes>
  <flow id="f0" from="e0" to="e1" begin="0" end="200" vehsPerHour="600"/>
</routes>
```

Inserts vehicles at 600 per hour during 0–200 s. Generated ids are `f0.0`,
`f0.1`, … (which is why the smoke test produced 34 vehicles).

**(d) `<vType>` — vehicle type. The most important one for us**

```xml
<vType id="AV" length="12.0" maxSpeed="30.0" accel="1.5" decel="4.0"
       carFollowModel="Krauss" color="1,0,0"/>
<vType id="PV" length="4.5"  maxSpeed="35.0" accel="2.6" decel="4.5"
       color="0,0,1"/>

<flow id="fAV" type="AV" from="e0" to="e1" begin="0" end="3600"
      vehsPerHour="100"/>
```

**SUMO has no notion of AV/PV.** We create the label with `vType` and read it
back from the `type` attribute in the FCD output. This is the cleanest way for
the converter to tell AV from PV — better than parsing id strings.

### 3.5 Config file (`.sumocfg`)

Command-line options bundled as XML. Functionally equivalent, but better for
reproducibility.

```xml
<configuration>
  <input>
    <net-file value="net.xml"/>
    <route-files value="flows.xml"/>
  </input>
  <output>
    <fcd-output value="fcd.xml"/>
    <tripinfo-output value="tripinfo.xml"/>
  </output>
  <time>
    <begin value="0"/>
    <end value="3600"/>
  </time>
  <processing>
    <step-length value="1.0"/>
  </processing>
</configuration>
```

```bash
sumo -c scenario.sumocfg
```

`sumo` writes the configuration it received into the header of every output
file (visible at the top of our `fcd.xml`). Useful for after-the-fact tracing.

### 3.6 Unit conventions (memorise)

| Quantity | Unit |
|---|---|
| speed | **m/s** |
| time | s |
| distance | m |
| acceleration | m/s² |
| coordinates | m (UTM projection when imported from OSM) |

---

## 4. Reading the output

### `--fcd-output` — Floating Car Data

Every step, every vehicle's state:

```xml
<timestep time="30.00">
  <vehicle id="f0.0" x="866.36" y="-4.80" angle="90.00"
           type="DEFAULT_VEHTYPE" speed="28.36" pos="866.36"
           lane="e0_0" slope="0.00"/>
```

Fields that matter to us:

- `pos` — **distance travelled from the start of the lane (m)**. Raw material
  for the 1D coordinate.
- `lane` — `<edge id>_<lane index>`. `e0_0` is lane 0 of edge `e0`.
- `type` — the `vType` id. To be used for AV/PV.
- `speed` — actual instantaneous speed (m/s). Not constant.

1D coordinate = (cumulative corridor length up to that edge) + `pos`

Cost: about 145 bytes per vehicle-second. The smoke test's 34 vehicles ×
300 s gave 363 KB. For larger runs increase the sampling interval with an
option such as `--fcd-output.period 5`.

### `--tripinfo-output` — one line per trip

```xml
<tripinfo id="f0.0" depart="0.00" arrival="70.00" duration="70.00"
          routeLength="1995.00" waitingTime="0.00" timeLoss="1.78"
          speedFactor="1.06" .../>
```

Note `speedFactor` — SUMO draws a random speed multiplier per vehicle. So
speeds are already non-uniform even without congestion.

**Caution:** `tripinfo` gives the departure/arrival of the *whole trip*, not
entry/exit on *our corridor*. The 1D projection needs FCD.

---

## 5. Setup (already done in this repository)

```bash
# install (done)
venv/bin/pip install -r sumo/requirements-sumo.txt

# needed only for the tools/ scripts
export SUMO_HOME="$PWD/venv/lib/python3.14/site-packages/sumo"
```

`sumo` / `netconvert` / `sumolib` / `traci` work without `SUMO_HOME`
(verified 2026-09-04). It is needed only to call `$SUMO_HOME/tools/*.py` by
path.

Do not use the Homebrew path — see `NOTES-sumo.md` for why.

### Minimal example to try yourself

```bash
cd sumo/smoketest
export SUMO_HOME="$(cd ../.. && pwd)/venv/lib/python3.14/site-packages/sumo"
../../venv/bin/netconvert -n nodes.xml -e edges.xml -o net.xml
../../venv/bin/sumo -n net.xml -r flows.xml \
    --fcd-output fcd.xml --tripinfo-output tripinfo.xml --end 300
```

---

## 6. GUI — works (after installing XQuartz)

SUMO ships two GUI tools, both in `venv/bin`.

| Tool | Use |
|---|---|
| `sumo-gui` | watch the simulation, play/pause, follow individual vehicles |
| `netedit` | network editor; inspect and fix OSM import results |

The SUMO GUI is built on the FOX toolkit and **needs an X11 server**. The pip
wheel bundles the X11 *client* libraries
(`@loader_path/../../sumo_data/.libs/libX11.6.dylib`), but client libraries
do not provide a server. XQuartz is that server.

**Status (2026-09-04): XQuartz installed, confirmed working.**

Note: right after installation `DISPLAY` is not injected into the shell
automatically (until log out / log in). Until then set it explicitly:

```bash
open -a XQuartz                       # start the X server (once)
DISPLAY=:0 venv/bin/sumo-gui -c scenario.sumocfg
```

After `open -a XQuartz` the server is on `:0` and the socket appears at
`/tmp/.X11-unix/X0`. After one log out / log in `DISPLAY` is set automatically
and the prefix is no longer needed.

### When to use it

**Not needed for the pipeline.** `netconvert` → `sumo` → FCD → converter is
entirely command line; running experiments never needs the GUI.

**Genuinely useful for validating the OSM import.** Raw OSM is messy and the
import commonly produces:

- broken ramp connections
- lane counts that differ from reality
- the freeway cut in the middle
- vehicles trapped at the clipped boundary

Hard to catch from the XML alone; checking visually in `netedit` is much
faster.

## 7. The flow we actually use in this project

```
1. OSM extract      Melbourne freeway section → melbourne.osm
2. netconvert       --osm-files + osmNetconvert.typ.xml → net.xml
3. (validate)       check ramps and lanes in netedit        ← needs XQuartz
4. demand           randomTrips.py or real-count based       ← undecided
5. vType            attach AV / PV labels here
6. run sumo         --fcd-output
7. converter        FCD → ActiveVehicle / PassiveVehicle
8. matching         existing greedy / ILA
```

Step 4 is currently undecided, and it feeds back into step 1 (the OSM extract
extent) — real detector data requires the detector locations to be inside the
network.

*(Status 2026-09-17: steps 1, 2, 6, 7, 8 exist — see `NOTES-sumo.md`. Step 5
is a random split in `convert.label()` for now, not a `vType`; step 4 remains
the open decision.)*
