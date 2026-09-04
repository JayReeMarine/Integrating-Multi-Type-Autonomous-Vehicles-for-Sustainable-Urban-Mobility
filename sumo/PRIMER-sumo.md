# SUMO 입문 정리 — 이 저장소 기준

이 프로젝트에서 실제로 설치·검증된 것(SUMO 1.27.1, pip 배포판)을 기준으로
정리한 것. 일반적인 튜토리얼이 아니라 우리 파이프라인에 필요한 부분만.

---

## 1. SUMO가 뭔가

**Simulation of Urban MObility** — 독일항공우주센터(DLR)가 만들고 Eclipse
재단이 관리하는 오픈소스 교통 시뮬레이터. EPL-2.0.

핵심 성격 두 가지:

**(1) 마이크로스코픽(microscopic)이다.**
교통류를 유체처럼 평균으로 다루는 매크로스코픽 모델과 달리, **차량 한 대
한 대를 개별 객체로 추적**한다. 각 차량은 자기 위치·속도·가속도·목적지를
갖는다. 우리 문제(AV가 특정 PV를 특정 지점에서 견인)는 개별 차량 단위여야
하므로 마이크로스코픽이 필수다.

**(2) 시간 이산(time-discrete)이다.**
기본 1초 스텝(`--step-length`로 변경 가능). 매 스텝마다 모든 차량에 대해
차량추종모델(기본: Krauss)로 다음 속도를 계산하고 위치를 갱신한다.
그래서 출력이 "매 초, 모든 차량의 위치" 형태로 나온다 — 이게 FCD다.

**결정론적이다.** 같은 입력 + 같은 `--seed` = 완전히 같은 출력. 우리
`analysis/reproduce_check.py`가 요구하는 재현성과 잘 맞는다.

---

## 2. 구조: 개념 4개면 끝

SUMO를 어렵게 만드는 건 문법이 아니라 "파일이 왜 이렇게 많은가"인데,
역할은 4가지뿐이다.

| 역할 | 파일 | 뜻 |
|---|---|---|
| **네트워크** | `net.xml` | 도로가 어떻게 생겼나 |
| **수요(demand)** | `*.rou.xml` | 누가, 언제, 어디서 어디로 가나 |
| **설정** | `*.sumocfg` | 위 둘을 묶고 옵션 지정 |
| **출력** | `fcd.xml`, `tripinfo.xml` | 결과 |

파이프라인:

```
  nodes.xml + edges.xml  ──netconvert──►  net.xml
  (또는 OSM .osm 파일)                        │
                                              ▼
  trips.xml ──duarouter──► routes.rou.xml ──► sumo ──► fcd.xml
                                                       tripinfo.xml
```

**`net.xml`은 절대 손으로 쓰지 않는다.** 항상 `netconvert`의 생성물이다.
(사람이 편집하려면 `netedit` GUI를 쓴다.)

---

## 3. 문법 — 전부 XML

아래는 `sumo/smoketest/`에 실제로 있고 동작이 검증된 파일들이다.

### 3.1 노드 (교차점)

```xml
<nodes>
  <node id="n0" x="0.0"    y="0.0"/>
  <node id="n1" x="1000.0" y="0.0"/>
  <node id="n2" x="2000.0" y="0.0"/>
</nodes>
```

`x`, `y`는 평면 직교 좌표(미터). 위 셋은 x축 위에 1 km 간격으로 놓인 점이다.

### 3.2 엣지 (도로)

```xml
<edges>
  <edge id="e0" from="n0" to="n1" numLanes="2" speed="27.8"/>
  <edge id="e1" from="n1" to="n2" numLanes="2" speed="27.8"/>
</edges>
```

**엣지는 방향이 있다.** `from`→`to` 한 방향만. 양방향 도로는 엣지 2개다.
우리가 "프리웨이 한 방향만 쓴다"는 결정이 SUMO에서 자연스러운 이유가 이것.

`speed`는 **초당 미터(m/s)**다. 27.8 m/s ≈ 100 km/h. 초보자가 가장 많이
틀리는 지점이니 주의.

### 3.3 컴파일

```bash
netconvert -n nodes.xml -e edges.xml -o net.xml
```

`netconvert`가 노드/엣지로부터 차선 연결, 교차로 내부 링크, 우선순위,
신호를 자동 생성해 `net.xml`을 만든다. OSM을 쓸 때도 같은 도구다:

```bash
netconvert --osm-files melbourne.osm -o net.xml \
           --type-files "$SUMO_HOME/data/typemap/osmNetconvert.typ.xml"
```

### 3.4 수요 — 4가지 방식

**(a) `<trip>` — 출발지·목적지만. 경로는 라우터가 계산**

```xml
<trip id="t0" depart="0.00" from="e0" to="e1"/>
```

`duarouter`로 실제 경로를 계산해 `.rou.xml`로 바꿔야 `sumo`가 먹는다.

**(b) `<vehicle>` + `<route>` — 경로를 명시**

```xml
<route id="r0" edges="e0 e1"/>
<vehicle id="v0" route="r0" depart="0.00"/>
```

**(c) `<flow>` — 여러 대를 자동 생성 (스모크 테스트에서 쓴 것)**

```xml
<routes>
  <flow id="f0" from="e0" to="e1" begin="0" end="200" vehsPerHour="600"/>
</routes>
```

0~200초 동안 시간당 600대 비율로 차량을 투입. 생성된 차량 id는
`f0.0`, `f0.1`, … 이 된다. (스모크 테스트에서 34대가 나온 이유.)

**(d) `<vType>` — 차종 정의. 우리에게 가장 중요하다**

```xml
<vType id="AV" length="12.0" maxSpeed="30.0" accel="1.5" decel="4.0"
       carFollowModel="Krauss" color="1,0,0"/>
<vType id="PV" length="4.5"  maxSpeed="35.0" accel="2.6" decel="4.5"
       color="0,0,1"/>

<flow id="fAV" type="AV" from="e0" to="e1" begin="0" end="3600"
      vehsPerHour="100"/>
```

**SUMO에는 AV/PV 개념이 없다.** 우리가 `vType`으로 라벨을 만들어 붙이고,
FCD 출력의 `type` 속성으로 다시 읽어내는 것이다. 이게 변환기에서
AV/PV를 구분하는 가장 깔끔한 방법이다 — id 문자열 파싱보다 낫다.

### 3.5 설정 파일 (`.sumocfg`)

커맨드라인 옵션을 XML로 묶은 것. 기능적으로 동등하지만 재현성이 좋다.

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

실제로 `sumo`는 실행할 때마다 자기가 받은 설정을 출력 파일 머리에
그대로 기록한다(우리 `fcd.xml` 상단에서 확인됨). 사후 추적에 유용하다.

### 3.6 단위 규약 (외워둘 것)

| 양 | 단위 |
|---|---|
| 속도 | **m/s** |
| 시간 | 초 |
| 거리 | m |
| 가속도 | m/s² |
| 좌표 | m (OSM 임포트 시 UTM 투영) |

---

## 4. 출력 읽기

### `--fcd-output` — Floating Car Data

매 스텝, 모든 차량의 상태:

```xml
<timestep time="30.00">
  <vehicle id="f0.0" x="866.36" y="-4.80" angle="90.00"
           type="DEFAULT_VEHTYPE" speed="28.36" pos="866.36"
           lane="e0_0" slope="0.00"/>
```

우리에게 중요한 필드:

- `pos` — **레인 시작점부터의 진행거리(m)**. 1D 좌표의 재료.
- `lane` — `<엣지id>_<레인번호>`. `e0_0`은 엣지 `e0`의 0번 레인.
- `type` — `vType` id. AV/PV 구분에 쓸 것.
- `speed` — 그 순간의 실제 속도(m/s). 등속이 아님.

1D 좌표 = (코리도어에서 해당 엣지까지의 누적 길이) + `pos`

비용: 차량·초당 약 145 바이트. 스모크 테스트는 34대 × 300초에 363 KB.
커지면 `--fcd-output.period 5` 같은 옵션으로 샘플링 간격을 늘린다.

### `--tripinfo-output` — 통행 1건당 1줄

```xml
<tripinfo id="f0.0" depart="0.00" arrival="70.00" duration="70.00"
          routeLength="1995.00" waitingTime="0.00" timeLoss="1.78"
          speedFactor="1.06" .../>
```

`speedFactor`가 눈에 띈다 — SUMO가 차량마다 속도 배율을 무작위로 뽑는다.
즉 정체가 없어도 속도는 이미 비균등하다.

**주의:** `tripinfo`는 *통행 전체*의 출발/도착이지 *우리 코리도어*의
진입/진출이 아니다. 1D 투영에는 FCD가 필요하다.

---

## 5. 세팅 (이 저장소에서는 이미 완료)

```bash
# 설치 (완료됨)
venv/bin/pip install -r sumo/requirements-sumo.txt

# tools/ 스크립트를 쓸 때만 필요
export SUMO_HOME="$PWD/venv/lib/python3.14/site-packages/sumo"
```

`sumo` / `netconvert` / `sumolib` / `traci`는 `SUMO_HOME` 없이도 동작한다
(2026-09-04 검증). `$SUMO_HOME/tools/*.py`를 경로로 부를 때만 필요하다.

Homebrew 경로는 쓰지 말 것 — 이유는 `NOTES-sumo.md` 참조.

### 직접 해볼 최소 예제

```bash
cd sumo/smoketest
export SUMO_HOME="$(cd ../.. && pwd)/venv/lib/python3.14/site-packages/sumo"
../../venv/bin/netconvert -n nodes.xml -e edges.xml -o net.xml
../../venv/bin/sumo -n net.xml -r flows.xml \
    --fcd-output fcd.xml --tripinfo-output tripinfo.xml --end 300
```

---

## 6. GUI — 동작함 (XQuartz 설치 후)

SUMO는 GUI 도구 두 개를 함께 설치한다. 둘 다 `venv/bin`에 있다.

| 도구 | 용도 |
|---|---|
| `sumo-gui` | 시뮬레이션을 눈으로 보며 재생/일시정지, 차량 개별 추적 |
| `netedit` | 네트워크 편집기. OSM 임포트 결과 검사·수정 |

SUMO GUI는 FOX 툴킷 기반이라 **X11 서버가 필요**하다. pip 휠이 X11
*클라이언트* 라이브러리는 번들로 갖고 있지만
(`@loader_path/../../sumo_data/.libs/libX11.6.dylib`), 클라이언트
라이브러리는 서버를 제공하지 않는다. XQuartz가 그 서버다.

**상태 (2026-09-04): XQuartz 설치됨, 동작 확인됨.**

주의: 설치 직후 `DISPLAY`가 셸에 자동 주입되지 않는다(로그아웃/로그인
전까지). 그때까지는 명시적으로 지정하면 된다:

```bash
open -a XQuartz                       # X 서버 기동 (한 번)
DISPLAY=:0 venv/bin/sumo-gui -c scenario.sumocfg
```

`open -a XQuartz` 후 서버는 `:0`에 뜨고 소켓은 `/tmp/.X11-unix/X0`에
생긴다. 로그아웃/로그인을 한 번 하면 `DISPLAY`가 자동으로 잡히므로
그 뒤로는 접두어가 필요 없다.

### 언제 쓰나

**파이프라인에는 불필요하다.** `netconvert` → `sumo` → FCD → 변환기는
전부 커맨드라인이고, 실험 실행에 GUI는 한 번도 필요하지 않다.

**OSM 임포트 검증에는 필수적으로 유용하다.** OSM 원본은 지저분해서
임포트 후 흔히 이런 문제가 생긴다:

- 램프 연결이 끊어짐
- 차선 수가 실제와 다름
- 프리웨이가 중간에 끊김
- 잘라낸 경계에서 차량이 갇힘

XML만 보고 잡기 어렵다. `netedit`으로 눈으로 확인하는 편이 훨씬 빠르다.

## 7. 우리 프로젝트에서 실제로 쓸 흐름

```
1. OSM 추출        멜버른 프리웨이 구간 → melbourne.osm
2. netconvert      --osm-files + osmNetconvert.typ.xml → net.xml
3. (검증)          netedit으로 램프·차선 확인          ← XQuartz 필요
4. 수요 생성       randomTrips.py 또는 실제 카운트 기반  ← 미결정
5. vType 정의      AV / PV 라벨을 여기서 붙임
6. sumo 실행       --fcd-output
7. 변환기          FCD → ActiveVehicle / PassiveVehicle
8. 매칭 실행       기존 greedy / ILA
```

현재 4번이 미결정 상태이고, 그것이 1번(OSM 추출 범위)까지 되돌려
결정한다 — 실제 검지기 데이터를 쓰려면 검지기 위치가 네트워크 안에
포함되어야 하기 때문이다.
