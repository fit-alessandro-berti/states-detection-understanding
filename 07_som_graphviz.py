#!/usr/bin/env python3
"""
Render SOM state matrix to Graphviz (neato) with colored cells and local transitions.

Colors: Understock=red, Overstock=blue, Normal=white, unknown=gray.
Nodes are positioned on a grid using the row/col coords from the JSON.
Only transitions between adjacent cells (Chebyshev distance 1) are drawn.
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Dict, Tuple


COLOR_MAP = {"Understock": "lightblue", "Overstock": "lightcoral", "Normal": "white"}
NODE_SPACING = 2.0  # increase distance between nodes


def load_state_matrix(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def chebyshev_adjacent(c1: Tuple[int, int], c2: Tuple[int, int]) -> bool:
    dr = abs(c1[0] - c2[0])
    dc = abs(c1[1] - c2[1])
    return (dr, dc) != (0, 0) and max(dr, dc) == 1


def main():
    p = argparse.ArgumentParser(description="Convert SOM state matrix JSON to Graphviz (neato) format.")
    p.add_argument("--input", type=str, default="som_state_matrix.json", help="Path to som_state_matrix.json")
    p.add_argument("--output", type=str, default="som_state_matrix.gv", help="Where to write the Graphviz file")
    args = p.parse_args()

    data = load_state_matrix(args.input)
    states = data["states"]

    # Build maps for quick lookup
    id_to_coord: Dict[int, Tuple[int, int]] = {int(s["id"]): tuple(s["coord"]) for s in states}

    lines = []
    lines.append('digraph G {')
    lines.append('  graph [layout=neato, splines=true, overlap=false];')
    lines.append('  node [shape=box, style="filled,rounded", fontsize=10, width=0.6, height=0.6, fixedsize=true];')
    lines.append('  edge [fontsize=9, arrowsize=0.7];')

    for s in states:
        sid = int(s["id"])
        r, c = id_to_coord[sid]
        status = s.get("majority_status") or "Unknown"
        color = COLOR_MAP.get(status, "gray")
        label = f"({r},{c})\\n{status}\\n{int(s['count'])}"
        x = float(c) * NODE_SPACING
        y = float(-r) * NODE_SPACING  # flip rows to have row 0 on top
        lines.append(f'  n{sid} [label="{label}", fillcolor="{color}", pos="{x},{y}!"];')

    # Edges for adjacent transitions
    for s in states:
        sid = int(s["id"])
        from_coord = id_to_coord[sid]
        trans: Dict[str, int] = s.get("transitions_to_other_states", {})
        for dst_str, count in trans.items():
            dst = int(dst_str)
            if count <= 0:
                continue
            to_coord = id_to_coord.get(dst)
            if to_coord is None:
                continue
            if not chebyshev_adjacent(from_coord, to_coord):
                continue
            lines.append(
                f'  n{sid} -> n{dst} [taillabel="  {count}  ", labeldistance=1.0, labelangle=0, labelfontsize=9];'
            )

    lines.append('}')

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote Graphviz file to {args.output}")


if __name__ == "__main__":
    main()
