#!/usr/bin/env python3
"""
Extract a compact SOM (3x3) matrix with visit frequencies and transition counts.
==============================================================================

Reads the global SOM assignments and meta, then reports:
- `frequency_matrix`: 3x3 matrix of how many events landed on each neuron.
- `states`: per-state counts plus outgoing transitions to other states.

Usage (defaults match the other scripts):
python 06_som_state_matrix.py \
  --assignments som_global_assignments.dump \
  --meta som_global_meta.dump \
  --output som_state_matrix.json
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


def load_dump(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def infer_window_size(vectors: List[List[List[float]]], assignments: List[List[int]]) -> int:
    """Infer the sliding window size that produced the assignments."""
    offsets = set()
    for v, a in zip(vectors, assignments):
        if v and a:
            offsets.add(len(v) - len(a) + 1)
    if not offsets:
        raise ValueError("Could not infer window size (no non-empty traces).")
    if len(offsets) != 1:
        raise ValueError(f"Inconsistent window sizes detected: {sorted(offsets)}")
    return offsets.pop()


def count_frequencies_and_status(
    assignments: List[List[int]],
    vectors: List[List[List[float]]],
    n_states: int,
    window_size: int,
) -> Tuple[List[int], List[Counter]]:
    """Counts events per SOM state and tracks stock-status counts per state."""
    freq = [0 for _ in range(n_states)]
    status_counts: List[Counter] = [Counter() for _ in range(n_states)]
    offset = window_size - 1  # window index -> last event index

    for seq_assign, seq_vec in zip(assignments, vectors):
        if not seq_assign:
            continue
        for j, state in enumerate(seq_assign):
            event_idx = j + offset
            if event_idx >= len(seq_vec):
                raise IndexError("Assignment index exceeds available events.")
            status = seq_vec[event_idx][-1]  # last element is the stock status string
            freq[state] += 1
            status_counts[state][status] += 1
    return freq, status_counts


def count_transitions(assignments: List[List[int]], n_states: int) -> List[List[int]]:
    transitions = [[0 for _ in range(n_states)] for _ in range(n_states)]
    for seq in assignments:
        for a, b in zip(seq[:-1], seq[1:]):
            transitions[a][b] += 1
    return transitions


def main():
    parser = argparse.ArgumentParser(description="Build SOM frequency matrix and transition counts.")
    parser.add_argument("--assignments", type=str, default="som_global_assignments.dump", help="Path to SOM assignments dump.")
    parser.add_argument("--meta", type=str, default="som_global_meta.dump", help="Path to SOM meta dump (rows/cols info).")
    parser.add_argument("--vectors", type=str, default="vectors.dump", help="Path to original vectors with stock/status labels.")
    parser.add_argument("--output", type=str, default="som_state_matrix.json", help="Where to store the resulting JSON.")
    args = parser.parse_args()

    assignments = load_dump(args.assignments)
    meta = load_dump(args.meta)
    vectors = load_dump(args.vectors)

    rows = int(meta["rows"])
    cols = int(meta["cols"])
    n_states = rows * cols
    coord_map = meta.get("coord_map") or [(r, c) for r in range(rows) for c in range(cols)]

    window_size = infer_window_size(vectors, assignments)
    freq, status_counts = count_frequencies_and_status(assignments, vectors, n_states, window_size)
    transitions = count_transitions(assignments, n_states)

    output_states = []
    for idx, coord in enumerate(coord_map):
        transitions_to_others = {str(j): transitions[idx][j] for j in range(n_states) if j != idx and transitions[idx][j] > 0}
        status_dist: Dict[str, int] = {str(k): int(v) for k, v in status_counts[idx].items()}
        majority_status = max(status_counts[idx].items(), key=lambda kv: kv[1])[0] if status_counts[idx] else None
        output_states.append(
            {
                "id": idx,
                "coord": [int(coord[0]), int(coord[1])],
                "count": int(freq[idx]),
                "majority_status": majority_status,
                "status_counts": status_dist,
                "transitions_to_other_states": transitions_to_others,
            }
        )

    result = {
        "grid": {"rows": rows, "cols": cols},
        "window_size": window_size,
        "frequency_matrix": [freq[i * cols:(i + 1) * cols] for i in range(rows)],
        "states": output_states,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote SOM matrix and transition stats to {args.output}")


if __name__ == "__main__":
    main()
