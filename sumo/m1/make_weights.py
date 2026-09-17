"""Restrict randomTrips origins/destinations to the inbound side of the network.

Why: m1.net.xml contains both carriageways. With --fringe-factor max,
randomTrips samples destinations uniformly over all fringe sinks, half of which
are on the outbound side; those trips leave the inbound main line at the first
off-ramp that can reach the outbound carriageway (mostly 72.1 Toorak) and
U-turn. That produced the 700-vehicle exit spike at 72.1 in the first run.

Writes sumo/m1/inbound.src.xml and inbound.dst.xml for `--weights-prefix
sumo/m1/inbound`. Edges not listed get weight 0.
  sources : fringe source edges that reach the inbound chain without touching
            the outbound chain
  sinks   : fringe sink edges reachable from the inbound chain without touching
            the outbound chain

Run:  PYTHONPATH=. venv/bin/python sumo/m1/make_weights.py
"""
import json
from pathlib import Path
import sumolib

HERE = Path(__file__).resolve().parent
net = sumolib.net.readNet(str(HERE.parent / "osm" / "m1.net.xml"))
corr = json.loads((HERE / "corridor.json").read_text())
inb = {e["id"] for e in corr["chain"]}
motorway = {e.getID() for e in net.getEdges()
            if e.getType() == "highway.motorway" and not e.getID().startswith(":")}
outb = motorway - inb


def reach(start, step, target, forbidden):
    seen, front = {start.getID()}, [start]
    while front:
        nxt = []
        for e in front:
            for s in step(e):
                sid = s.getID()
                if sid in forbidden:
                    continue
                if sid in target:
                    return True
                if sid not in seen:
                    seen.add(sid); nxt.append(s)
        front = nxt
    return False


real = [e for e in net.getEdges() if not e.getID().startswith(":")]
sources = [e for e in real if not e.getIncoming()
           and (e.getID() in inb or reach(e, lambda x: x.getOutgoing(), inb, outb))]
sinks = [e for e in real if not e.getOutgoing()
         and (e.getID() in inb or reach(e, lambda x: x.getIncoming(), inb, outb))]


def write(path, edges, end=3600):
    lines = [f'    <edge id="{e.getID()}" value="1.0"/>' for e in edges]
    path.write_text('<edgedata>\n  <interval begin="0" end="%d">\n%s\n  </interval>\n</edgedata>\n'
                    % (end, "\n".join(lines)))


write(HERE / "inbound.src.xml", sources)
write(HERE / "inbound.dst.xml", sinks)
print(f"inbound-side fringe sources: {len(sources)}  sinks: {len(sinks)}")
for e in sources: print("  src", e.getID(), e.getName() or "(unnamed)")
for e in sinks:   print("  dst", e.getID(), e.getName() or "(unnamed)")
