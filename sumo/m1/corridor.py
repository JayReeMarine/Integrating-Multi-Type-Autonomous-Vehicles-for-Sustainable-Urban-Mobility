"""Extract the inbound (CBD-bound) M1 main-line chain and its ramp positions.

Reproduces the numbers recorded in NOTES-sumo.md (D4/D5): 31 edges, 20.53 km,
16 ramp junctions. Writes sumo/m1/corridor.json for convert.py.

Run:  PYTHONPATH=. venv/bin/python sumo/m1/corridor.py
"""
import json
import sys
from pathlib import Path

import sumolib

NET = Path(__file__).resolve().parent.parent / "osm" / "m1.net.xml"
OUT = Path(__file__).resolve().parent / "corridor.json"
GRID = 100  # corridor length in grid units (D5)


def main():
    net = sumolib.net.readNet(str(NET))
    motorway = [e for e in net.getEdges()
                if not e.getID().startswith(":") and e.getType() == "highway.motorway"]
    mset = {e.getID() for e in motorway}

    # successor within main line (motorway -> motorway)
    def nxt(e):
        return [s for s in e.getOutgoing() if s.getID() in mset]

    def prv(e):
        return [p for p in e.getIncoming() if p.getID() in mset]

    heads = [e for e in motorway if not prv(e)]
    chains = []
    for h in heads:
        chain, e = [h], h
        while True:
            n = nxt(e)
            if len(n) != 1:
                break
            e = n[0]
            if e in chain:
                break
            chain.append(e)
        chains.append(chain)

    def length(ch):
        return sum(e.getLength() for e in ch)

    def lon(e):
        # projected x grows eastward; good enough for direction, no pyproj needed
        return e.getFromNode().getCoord()[0]

    # inbound = heads east, ends west (x decreases)
    inbound = [c for c in chains if lon(c[-1]) < lon(c[0])]
    chain = max(inbound, key=length)
    total = length(chain)
    print(f"inbound chain: {len(chain)} edges, {total/1000:.2f} km, "
          f"{len(chains)} chains total")

    # cumulative position of every node along the chain
    pos_m, cum = {}, 0.0
    pos_m[chain[0].getFromNode().getID()] = 0.0
    for e in chain:
        cum += e.getLength()
        pos_m[e.getToNode().getID()] = cum

    cset = {e.getID() for e in chain}
    ramps = []
    for e in chain:
        for node, kind in ((e.getFromNode(), "ON"), (e.getToNode(), "OFF")):
            if kind == "ON":
                for inc in node.getIncoming():
                    if inc.getID() not in cset and not inc.getID().startswith(":"):
                        ramps.append(dict(kind="ON", edge=inc.getID(), name=inc.getName(),
                                          node=node.getID(), m=pos_m[node.getID()]))
            else:
                for out in node.getOutgoing():
                    if out.getID() not in cset and not out.getID().startswith(":"):
                        ramps.append(dict(kind="OFF", edge=out.getID(), name=out.getName(),
                                          node=node.getID(), m=pos_m[node.getID()]))
    # de-duplicate by (kind, node)
    seen, uniq = set(), []
    for r in ramps:
        k = (r["kind"], r["node"])
        if k not in seen:
            seen.add(k); uniq.append(r)
    ramps = sorted(uniq, key=lambda r: r["m"])
    for r in ramps:
        r["grid"] = round(r["m"] / total * GRID, 1)
    print(f"ramp junctions: {len(ramps)} "
          f"(ON {sum(r['kind']=='ON' for r in ramps)} / OFF {sum(r['kind']=='OFF' for r in ramps)})")
    for r in ramps:
        print(f"  {r['grid']:5.1f} {r['kind']:3} {r['name'] or '(unnamed)'}  [{r['edge']}]")

    out = dict(
        net=str(NET.relative_to(NET.parents[2])),
        length_m=total, grid=GRID,
        chain=[dict(id=e.getID(), from_m=pos_m[e.getFromNode().getID()],
                    to_m=pos_m[e.getToNode().getID()], lanes=e.getLaneNumber())
               for e in chain],
        ramps=ramps,
    )
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
