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

---

## 2026-09-04 — 코리도어 확정 및 OSM 임포트

### D4 — 코리도어: M1 Monash Freeway, inbound (CBD 방향)

Jay가 "CBD ~ Clayton"을 지정. 원래 제안(EastLink→Burnley)보다 나은 선택 —
Monash 통근 축이라 "임의로 고른 프리웨이"가 아니라 실제 통근 코리도어로
동기부여할 수 있다. 터널은 제외하기로 결정(1차 반복).

**데이터 취득.** Overpass API, bbox `(-37.93, 145.00, -37.81, 145.17)`,
필터 `highway=motorway|motorway_link`. 결과 `sumo/osm/monash-m1.osm`,
305 KB, way 341개 / node 1811개. 쿼리는 `sumo/osm/query.overpassql`에 보존.
bbox 서쪽 경계 145.00이 Burnley 터널을 사실상 잘라냈다(터널 way 2개만 잔존).

**변환.**

```bash
netconvert --osm-files monash-m1.osm \
  --type-files "$SUMO_HOME/data/typemap/osmNetconvert.typ.xml" \
  --output-file m1.net.xml \
  --geometry.remove --ramps.guess --junctions.join \
  --remove-edges.isolated --keep-edges.by-vclass passenger \
  --output.street-names true --output.original-names true
```

`--output.street-names true`가 필수다. 없으면 엣지 이름이 전부 버려져서
본선/램프를 구분할 수 없다(처음 시도에서 이것 때문에 본선 엣지 0개가 나왔다).

### 코리도어 구조

| | inbound (CBD 방향) | outbound (Clayton 방향) |
|---|---|---|
| 최장 연결 체인 | **31/31 엣지 (끊김 없음)** | 36/40 (4개 끊김) |
| 길이 | **20.53 km** | 22.60 km |
| 램프 접속점 | 16 (ON 8 / OFF 8) | 16 |

**inbound 채택.** 완전 연결이고 오전 첨두 방향과 일치한다.

차선 수가 CBD 방향으로 3 → 4 → 5 → 6으로 증가한다. 합성 데이터에 없는
현실성이며, AV 용량 모델과 연결지을 여지가 있다.

### 검증 결과 (코드로 수행, netedit 육안 검사 아님)

- **막다른 램프 0개.** 코리도어에 붙은 모든 램프가 도로망으로 이어진다.
  OSM 임포트 품질 문제 없음.
- **8.96–12.58 km 구간이 단일 엣지(3.62 km).** 갈림길이 하나도 없다.
  Warrigal Rd와 Burke Rd 사이. 실제로 나들목이 없는 구간이며, 데이터
  결손이 아니다. **자연 발생한 희소성 구간**으로, 균등 합성 데이터가
  만들어낼 수 없는 구조다.

### 램프 위치 (100단위 격자 환산) — 핵심 결과

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

**합성 데이터는 0–100의 101개 정수 지점에 균등하게 위치를 뿌렸다.
실제 M1은 진입/진출이 16개 지점에만 존재한다.** 이것이 SUMO 도입으로
확인된 첫 번째 실질적 차이이며, `NOTES-sumo.md` 앞부분의 "1D 투영이
숫자 4개만 통과시킨다"는 회의론에 대한 부분적 반례이기도 하다 —
`entry_point`/`exit_point`의 주변 분포가 실제로 크게 달라진다.

(다만 이것이 ILA–greedy 격차를 벌리는지는 별개 문제로 남아 있다.
격차는 경합 수준의 함수이고, 경합은 여전히 우리가 정하는 파라미터다.)

### D5 — 단위: 100단위 격자 유지

20.53 km / 100 = **1단위 ≈ 205 m**. 따라서 `L_MIN = 10단위 ≈ 2.05 km`
최소 견인 거리로, 물리적으로 타당하다.

미터 단위로 전환하지 않는 이유: 기존 48개 조합과 같은 축을 유지해야
"합성 곡선 vs SUMO 곡선을 같은 경합 수준에서 겹쳐 그리기"가 성립한다.
단위를 바꾸면 그 비교가 불가능해진다.

주의: `ActiveVehicle.entry_point`는 `int`이므로 변환기는 205 m 단위로
양자화해야 한다. 램프 간격이 1–3 km이므로 양자화 손실은 실질적으로 없다.

### 남은 위험

- **Ramp metering.** M1은 램프 신호 제어가 있는 managed motorway인데
  OSM 임포트는 재현하지 못한다. 1차에서는 무시하되, 진입 패턴의 현실성을
  논할 때 리뷰어가 짚을 수 있다.
- **On-ramp 이름 소실.** off-ramp는 이름이 살아 있으나(Blackburn, Warrigal
  등) on-ramp는 대부분 `Monash Freeway On Ramp`로만 태깅되어 유입 도로를
  알 수 없다. 위치만 필요한 현 단계에서는 무해하다.
- **CityLink 유료 구간 포함.** 코리도어 서쪽 끝(75.5단위 이후)이 CityLink다.
  통행료가 실제 경로 선택에 영향을 주지만 randomTrips는 이를 무시한다.

### 다음

- [ ] 수요 생성 방식 결정 (여전히 블로커) — Mushfiq 자문 내용 확인
- [ ] 첫 시나리오 실행 후 D2 선형화 오차 측정

---

## 2026-09-04 — MILP 결과가 SUMO 트랙의 질문을 바꿈

병렬 세션이 exact solver 결과를 냈다(루트 `NOTES.md`,
`data/results/milp/`). 이 파일 앞부분의 "go/no-go 도구는 MILP"라는 주장에
대한 답이 나왔으므로 여기 반영한다. **아래는 MILP 세션의 결과를 읽고
SUMO 트랙 관점에서 해석한 것이며, 원 분석은 루트 `NOTES.md` 참조.**

### 결과 요약 (읽은 내용)

| | 최적해 대비 평균 |
|---|---|
| greedy | 93.33% |
| ILA | 93.09% |

ILA 9승 / 8패 / 4무. 동전 던지기이며 평균으로는 근소하게 열세.
남은 ~7% 중 알고리즘의 표현력 한계로 잃는 것은 0.65%뿐이고, 나머지 ~6%는
가용한 선택지 중 잘못 고른 결과다.

범위: AV 4–15 / PV 8–30, **시간 제약 없는 공간 문제만**. 기존 스윕
(AV 10–320 / PV 50–800)과 규모가 다르므로 큰 인스턴스로의 일반화는
아직 근거 없음.

### 앞부분 표의 두 번째 행에 해당한다

이 파일의 "go/no-go 도구는 MILP" 절에서 세운 두 갈래 중, **"천장은 있는데
ILA가 못 먹고 있다"** 쪽이다. 천장이 낮아서 못 하는 게 아니라 두 휴리스틱이
모두 평범한 것이다. 관측된 최대 ILA–greedy 격차 1.26 pp에 대해 실제 여지는
~7 pp이므로, 더 나은 알고리즘이 들어갈 자리는 실재한다.

### SUMO 트랙의 질문이 바뀐다

앞부분에서 세운 질문:

> ~~경합 수준을 맞췄을 때 SUMO 분포가 ILA–greedy 격차를 키우는가?~~

**폐기.** ILA와 greedy가 동전 던지기인 이상, 둘 사이 격차는 측정할 가치가
있는 양이 아니다. 대체 질문:

> **현실적인 인스턴스 구조가 최적해 대비 여지(~7%)를 키우는가, 줄이는가?**

즉 비교 대상이 "ILA vs greedy"에서 "휴리스틱 vs 최적해"로 이동한다.
오늘 확인된 M1 구조 — 진입/진출이 101개가 아닌 16개 지점, 8.96–12.58 km의
선택지 없는 3.62 km 구간 — 이 경합을 키운다면 여지가 커지고, 더 나은
알고리즘의 가치가 커진다. 이 질문은 SUMO 없이는 답할 수 없고, MILP를
기준선으로 써야만 답할 수 있다.

### D6 — 변환기는 작은 인스턴스를 뽑아낼 수 있어야 한다

MILP는 AV 15 / PV 30에서 142초, AV 20 / PV 40은 60초 내 최적성 증명 실패.
SUMO 첨두 1시간 시나리오는 이보다 훨씬 크다.

따라서 변환기는 **전체 시나리오를 통째로 내보내는 것 외에, 실제 코리도어에서
작은 부분 인스턴스(AV ~10 / PV ~25)를 추출할 수 있어야 한다.** 예: 첨두
1시간을 5분 시간창으로 분할. 이렇게 하면 M1의 실제 구조(16개 램프 지점,
희소 구간, 차선 수 변화)는 보존되면서 규모만 MILP 사정권으로 내려온다.

이것이 없으면 위의 새 질문에 답할 수 없다 — 최적해 기준선을 계산할
방법이 없기 때문이다.

### 주의

MILP는 `enable_time_constraints=False`인 공간 문제만 다룬다. SUMO 데이터의
가장 큰 기여 중 하나가 현실적인 시간 분포인데, 그 축에서는 최적해 기준선이
아직 없다. 시간 제약이 켜진 비교를 하려면 MILP 쪽 확장이 필요하고, 그것은
비선형(견인이 이후 도착 시각을 바꿈)이라 간단하지 않다. 1차 반복은 공간
문제로 맞추어 비교 가능성을 확보하는 편이 안전하다.
