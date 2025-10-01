#!/usr/bin/env python3
"""
Classification / clustering quality metrics for SOM assignments (string labels safe)
====================================================================================

This script evaluates the quality of a single global SOM classification by
comparing neuron assignments to true **state** labels from `vectors.dump`.
It is robust to **string state labels** (e.g., 'Normal', 'Understock', 'Overstock').

Adds:
- Optional `--classes` to control the display/ordering of states (default:
  Normal Understock Overstock). Metrics work regardless of order.

Metrics
-------
Supervised (need labels): micro/macro/balanced accuracy, macro/weighted P/R/F1,
confusion matrix, purity, homogeneity/completeness/V, NMI (sqrt variant), ARI,
VI (variation of information).

Unsupervised + sequential: weighted cluster entropy (+normalized),
adjacency of consecutive assignments (same/neighboring neuron), and
stock variance explained by neuron (η²).

Usage
-----
python classification_quality_metrics.py \
  --assign som_global_assignments.dump \
  --vectors vectors.dump \
  --meta som_global_meta.dump \
  --classes Normal Understock Overstock

More detailed information:

# Metrics for Evaluating the SOM “Classification”

Below is a plain-English + formula guide to every metric the script reports. Where helpful, I’ll note the **range**, **what “good” looks like**, **why it’s useful**, and **gotchas**.

---

## Setup & Notation

* You trained **one SOM** on all window vectors and then assigned each event to a **neuron** (cluster).
* You also have a **ground-truth state** for each event: one of `Normal`, `Understock`, `Overstock`.

Let:

* (n) = total number of labeled events used for evaluation.
* (C) = number of clusters (neurons).
* (S) = number of states (here, (S=3)).
* (n_{ij}) = number of samples whose **predicted cluster** is (i) and **true state** is (j).
  The **contingency matrix** is ( \mathbf{N} = [n_{ij}] ) of shape (C \times S).
* Row/column sums: ( n_{i\cdot} = \sum_j n_{ij} ), ( n_{\cdot j} = \sum_i n_{ij} ), ( n=\sum_{ij} n_{ij} ).

For **classification metrics** (accuracy, precision, recall, F1, confusion matrix), clusters are first mapped to **predicted states** via **majority vote**:

* For cluster (i), assign it the state ( \arg\max_j n_{ij} ).
  Then each event in cluster (i) is predicted as that state.

> **Note**: Majority mapping collapses clusters to classes; great for readability, but it can overstate performance if a class is very dominant.

---

## Supervised “Classifier” Metrics (state-aware)

These treat the cluster→state majority mapping as a classifier.

### 1) Micro Accuracy (a.k.a. overall accuracy)

**Definition:**
[
\text{Acc}_\text{micro} = \frac{\text{# correct predictions}}{n}.
]

* **Range:** ([0,1]). Higher is better.
* **Pros:** Simple, intuitive.
* **Cons:** Sensitive to **class imbalance** (dominant class drives the score).

### 2) Macro Accuracy & Balanced Accuracy

**Macro Accuracy**: average of per-class accuracies (equivalently macro recall):
[
\text{Acc}*\text{macro} = \frac{1}{S}\sum*{j=1}^S
\frac{\text{# correctly predicted state } j}{\text{# true state } j}.
]

* **Range:** ([0,1]). Higher is better.
* **Pros:** Treats each class equally; robust to imbalance.
* **Balanced Accuracy** is **identical** to macro recall for multi-class classification.

### 3) Precision / Recall / F1 (Macro & Weighted)

For each class (j):

* ( \text{Precision}_j = \frac{\text{TP}_j}{\text{TP}_j+\text{FP}_j} )
* ( \text{Recall}_j = \frac{\text{TP}_j}{\text{TP}_j+\text{FN}_j} )
* ( \text{F1}_j = \frac{2,\text{Precision}_j,\text{Recall}_j}{\text{Precision}_j+\text{Recall}_j} ), with 0 if denom=0.

**Macro** P/R/F1 = mean over classes.
**Weighted** P/R/F1 = average over classes weighted by class prevalence (n_{\cdot j}/n).

* **Use macro** to compare performance across classes fairly.
* **Use weighted** to reflect the dataset’s class distribution.

### 4) Confusion Matrix (states × predicted states)

A standard (S \times S) confusion matrix after majority mapping.

* **Interpretation:** Rows = true states, Columns = predicted states.
* Off-diagonal mass indicates confusions (e.g., `Understock` → `Normal`).

---

## Clustering Quality (state-aware but mapping-free)

These use the raw contingency matrix ( \mathbf{N} ) and do **not** rely on the cluster→state mapping.

Let ( P = \mathbf{N}/n ), ( p_c ) = cluster distribution (row sums of (P)), ( p_s ) = state distribution (column sums).

### 5) Purity

[
\text{Purity} = \frac{1}{n}\sum_{i=1}^C \max_j n_{ij}.
]

* **Range:** ([0,1]). Higher is better.
* **Meaning:** Each cluster is “pure” if most of its members share the same state.
* **Caveat:** More clusters generally increase purity; compare at fixed (C).

### 6) Homogeneity, Completeness, and V-measure

* **Homogeneity:** each cluster contains only members of a single class.
  [
  \text{hom} = \frac{I(\text{clusters};\text{states})}{H(\text{states})}
  ]

* **Completeness:** all members of a given class are assigned to the same cluster.
  [
  \text{com} = \frac{I(\text{clusters};\text{states})}{H(\text{clusters})}
  ]

* **V-measure:** harmonic mean of homogeneity & completeness.

* **Range:** ([0,1]). Higher is better.

* **Pros:** Balance cluster purity and class completeness.

### 7) Normalized Mutual Information (NMI)

[
\text{NMI} = \frac{I(\text{clusters};\text{states})}{\sqrt{H(\text{clusters}),H(\text{states})}}.
]

* **Range:** ([0,1]). Higher is better.
* **Interpretation:** Symmetric association between clusters and states.

### 8) Adjusted Rand Index (ARI)

Measures agreement between cluster labels and state labels, **adjusted for chance**.

* **Range:** typically ([-1,1]).
  0 ≈ random; 1 = perfect agreement; negative = worse than random.
* **Pros:** Chance-corrected; popular for clustering evaluation.

### 9) Variation of Information (VI)

[
\text{VI} = H(\text{clusters}) + H(\text{states}) - 2,I(\text{clusters};\text{states}).
]

* **Range:** ([0,\infty)). **Lower is better.**
* **Interpretation:** A proper metric on partitions; 0 iff partitions are identical.

---

## Unsupervised & Sequential Diagnostics

These look at properties of the cluster assignments themselves and how they align with the process order.

### 10) Weighted Cluster Entropy (and normalized)

For each cluster (i), consider the **state distribution inside that cluster**:
[
H_i = -\sum_j p(j|i)\log p(j|i),
\quad
p(j|i)=\frac{n_{ij}}{n_{i\cdot}}.
]
Aggregate:
[
H_\text{weighted} = \sum_{i=1}^C \frac{n_{i\cdot}}{n},H_i,
\qquad
H_\text{norm} = \frac{H_\text{weighted}}{\log S}.
]

* **Range:** (H_\text{norm}\in[0,1]). **Lower is better** (clusters are purer).

### 11) Adjacency of Consecutive Assignments (temporal smoothness)

We measure how often consecutive events in the same **trace** land on the **same or neighboring neuron** on the SOM grid.

* Neighbor definition: **Chebyshev distance** ≤ 1 on the 2D grid, i.e., same cell or any of its 8 neighbors.
* Reported:

  * **Adjacency (all transitions)**: fraction of consecutive pairs that are adjacent.
  * **Adjacency (same-state only)**: same but restricted to transitions where the ground-truth state didn’t change.
  * **Adjacency (diff-state only)**: restricted to transitions where the ground-truth state changed.

**Interpretation:**

* High **adjacency (same-state)** → the SOM preserves local continuity within a state.
* Lower **adjacency (diff-state)** → state changes tend to jump further on the map (desirable if you want states to be separated).

### 12) Stock η² (variance explained by neuron)

Treat **neuron ID** as a categorical predictor for the continuous `stock` value.
Compute ANOVA-style **effect size**:
[
\eta^2 = \frac{\text{SS}*\text{between}}{\text{SS}*\text{total}},
]
where (\text{SS}*\text{between}) is the between-neuron sum of squares and (\text{SS}*\text{total}) is the total sum of squares.

* **Range:** ([0,1]). Higher means neurons explain more variance in stock.
* **Use:** Does the map capture meaningful structure in the stock signal?

---

## Confusion Matrix (clusters × states)

This is the raw (C \times S) table ( \mathbf{N} ) before majority mapping:

* Row (i) = a SOM neuron.
* Column (j) = a true state (`Normal`, `Understock`, `Overstock`).
* Entry (n_{ij}) = count assigned to neuron (i) with true state (j).

**Reading tips:**

* A **pure** cluster has almost all mass in one column.
* **Row sums** show cluster size; **column sums** reflect class prevalence.
* Use it to spot systematic confusions (e.g., `Understock` leaking into `Normal`).

---

## Practical Tips & Gotchas

* **Imbalance:** If `Normal` dominates, micro accuracy and weighted F1 may look high even when minority states suffer. Check **macro** metrics, **homogeneity/completeness**, and the **confusion matrix**.
* **Purity vs. #clusters:** Purity rises with more clusters. Compare purity across methods at the **same grid size**.
* **Sequence awareness:** High **adjacency (same-state)** with low **adjacency (diff-state)** suggests the map respects process dynamics: steady periods stay local; transitions jump.
* **η² (stock):** If this is near 0, neurons don’t separate `stock` well; consider different window sizes, PCA dimensions, or a different grid size.
* **Majority mapping:** It is a **post-hoc classifier**; excellent for interpretability but can hide multi-modal clusters. Always pair it with NMI/ARI/VI and the contingency matrix.

---

## When to Prefer What

* **Single headline number?** Micro accuracy (easy) or NMI (cluster–state agreement).
* **Fair to minority states?** Macro accuracy and macro F1.
* **“Are my clusters purer or more complete?”** Homogeneity (purity-like) and completeness.
* **Do clusters respect process order?** Adjacency metrics.
* **Do clusters align with numeric signals?** Stock η².
"""
from __future__ import annotations
import argparse
import math
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np

# -----------------------------
# I/O
# -----------------------------

def load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)

# -----------------------------
# Helpers
# -----------------------------

def factor_grid(k: int) -> Tuple[int, int]:
    """Heuristic: find (rows, cols) with rows*cols >= k and |rows-cols| minimal."""
    r = int(np.floor(np.sqrt(max(1, k))))
    best = (max(1, r), int(np.ceil(k / max(1, r))))
    best_gap = abs(best[0] - best[1])
    for rr in range(max(1, r - 3), r + 4):
        cc = int(np.ceil(k / rr))
        if rr * cc >= k and abs(rr - cc) < best_gap:
            best = (rr, cc)
            best_gap = abs(rr - cc)
    return best


def chebyshev_neighbor(id1: int, id2: int, rows: int, cols: int) -> bool:
    r1, c1 = divmod(int(id1), cols)
    r2, c2 = divmod(int(id2), cols)
    return max(abs(r1 - r2), abs(c1 - c2)) <= 1

# -----------------------------
# Contingency utilities (string-safe)
# -----------------------------

def contingency_matrix(labels_true: np.ndarray, labels_pred: np.ndarray,
                       class_order: Optional[List[str]] = None) -> Tuple[np.ndarray, Dict[int,int], Dict[object,int], List[object]]:
    """Return C[cluster, state], maps, and ordered list of state labels.

    - labels_true: array-like of ground-truth state labels (strings OK)
    - labels_pred: array-like of cluster ids (ints)
    - class_order: optional explicit ordering of state labels for columns
    """
    # Determine state label set and order
    if class_order is None or len(class_order) == 0:
        states_ordered = list(np.unique(labels_true))
    else:
        # Keep only labels that actually occur to avoid empty columns at the end
        present = set(labels_true.tolist())
        states_ordered = [s for s in class_order if s in present]
        # Append any extra labels not in class_order (rare)
        for s in np.unique(labels_true):
            if s not in present:  # always False; kept for clarity
                continue
            if s not in states_ordered:
                states_ordered.append(s)

    clusters = np.unique(labels_pred)
    s_map = {lab: i for i, lab in enumerate(states_ordered)}        # label -> col idx
    c_map = {int(cid): i for i, cid in enumerate(clusters)}         # cluster id -> row idx
    C = np.zeros((len(clusters), len(states_ordered)), dtype=np.int64)
    for yt, yp in zip(labels_true, labels_pred):
        C[c_map[int(yp)], s_map[yt]] += 1
    return C, c_map, s_map, states_ordered

# -----------------------------
# Supervised metrics
# -----------------------------

def majority_mapping(C: np.ndarray, states_ordered: List[object]) -> Dict[int, object]:
    """Map cluster-row index -> majority state *label*."""
    mapping: Dict[int, object] = {}
    for ci in range(C.shape[0]):
        if C[ci].sum() == 0:
            mapping[ci] = states_ordered[0]
        else:
            mapping[ci] = states_ordered[int(np.argmax(C[ci]))]
    return mapping


def apply_mapping(y_pred_clusters: np.ndarray, c_map: Dict[int,int], mapping_on_states: Dict[int,object]) -> np.ndarray:
    """Convert cluster ids to predicted *state labels* via mapping_on_states."""
    inv_c_map = {k: v for k, v in c_map.items()}  # cluster_id -> row idx
    out = [mapping_on_states[inv_c_map[int(cid)]] for cid in y_pred_clusters]
    return np.array(out, dtype=object)


def accuracy_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float,float,float]:
    micro = float(np.mean(y_true == y_pred)) if y_true.size else 0.0
    classes = np.unique(y_true)
    accs = []
    for c in classes:
        mask = (y_true == c)
        if mask.any():
            accs.append(float(np.mean(y_pred[mask] == c)))
    macro = float(np.mean(accs)) if accs else 0.0
    balanced = macro
    return micro, macro, balanced


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float,float,float,float,float,float]:
    classes, counts = np.unique(y_true, return_counts=True)
    weights = counts / counts.sum()
    precs, recs, f1s = [], [], []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        precs.append(prec); recs.append(rec); f1s.append(f1)
    macro_p = float(np.mean(precs)) if precs else 0.0
    macro_r = float(np.mean(recs)) if recs else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    weighted_p = float(np.sum(weights * np.array(precs))) if precs else 0.0
    weighted_r = float(np.sum(weights * np.array(recs))) if recs else 0.0
    weighted_f1 = float(np.sum(weights * np.array(f1s))) if f1s else 0.0
    return macro_p, macro_r, macro_f1, weighted_p, weighted_r, weighted_f1


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log(p))) if p.size else 0.0


def mutual_information(C: np.ndarray) -> Tuple[float,float,float]:
    n = np.sum(C)
    if n == 0:
        return 0.0, 0.0, 0.0
    P = C / n
    pc = P.sum(axis=1, keepdims=True)
    ps = P.sum(axis=0, keepdims=True)
    Hc = entropy(pc.ravel())
    Hs = entropy(ps.ravel())
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(P > 0, P / (pc @ ps), 1.0)
        MI = float(np.sum(np.where(P > 0, P * np.log(ratio), 0.0)))
    return MI, Hc, Hs


def homogeneity_completeness_v(C: np.ndarray) -> Tuple[float,float,float]:
    MI, Hc, Hs = mutual_information(C)
    hom = 1.0 if Hs == 0 else MI / Hs
    com = 1.0 if Hc == 0 else MI / Hc
    if hom + com == 0:
        V = 0.0
    else:
        beta = 1.0
        V = (1 + beta) * hom * com / (beta * hom + com)
    return float(hom), float(com), float(V)


def nmi_ari(C: np.ndarray) -> Tuple[float,float]:
    MI, Hc, Hs = mutual_information(C)
    nmi = 0.0 if (Hc == 0 or Hs == 0) else MI / math.sqrt(Hc * Hs)
    nij = C
    ai = nij.sum(axis=1)
    bj = nij.sum(axis=0)
    n = nij.sum()
    def comb2(x):
        return x * (x - 1) // 2
    sum_comb_c = np.sum(comb2(nij))
    sum_comb_a = np.sum(comb2(ai))
    sum_comb_b = np.sum(comb2(bj))
    expected_index = sum_comb_a * sum_comb_b / max(1, comb2(n))
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    denom = max_index - expected_index
    ari = 0.0 if denom == 0 else (sum_comb_c - expected_index) / denom
    return float(nmi), float(ari)


def variation_of_information(C: np.ndarray) -> float:
    MI, Hc, Hs = mutual_information(C)
    return float(Hc + Hs - 2 * MI)

# -----------------------------
# Unsupervised + sequence metrics
# -----------------------------

def weighted_cluster_entropy(C: np.ndarray) -> Tuple[float,float]:
    n = C.sum()
    if n == 0:
        return 0.0, 0.0
    weights = C.sum(axis=1) / n
    ent = []
    for i in range(C.shape[0]):
        p = C[i] / max(1, C[i].sum())
        ent.append(entropy(p))
    H = float(np.sum(weights * np.array(ent)))
    Hn = H / math.log(max(2, C.shape[1])) if C.shape[1] > 1 else 0.0
    return H, Hn


def adjacency_metrics(assign_traces: List[List[int]], state_traces: List[List[object]], rows: int, cols: int):
    tot = 0
    good_any = 0
    good_same_state = 0
    tot_same_state = 0
    good_diff_state = 0
    tot_diff_state = 0
    for a, s in zip(assign_traces, state_traces):
        m = min(len(a), len(s))
        for i in range(m - 1):
            tot += 1
            nbh = chebyshev_neighbor(a[i], a[i+1], rows, cols)
            if nbh:
                good_any += 1
            if s[i] == s[i+1]:
                tot_same_state += 1
                if nbh:
                    good_same_state += 1
            else:
                tot_diff_state += 1
                if nbh:
                    good_diff_state += 1
    return {
        'adjacency_all': (good_any / tot) if tot else float('nan'),
        'adjacency_same_state': (good_same_state / tot_same_state) if tot_same_state else float('nan'),
        'adjacency_diff_state': (good_diff_state / tot_diff_state) if tot_diff_state else float('nan'),
    }


def stocks_eta_squared(assign_traces: List[List[int]], stock_traces: List[List[float]]) -> float:
    y = []
    g = []
    for a, st in zip(assign_traces, stock_traces):
        m = min(len(a), len(st))
        y.extend(st[:m])
        g.extend(a[:m])
    if not y:
        return float('nan')
    y = np.asarray(y, dtype=np.float64)
    g = np.asarray(g, dtype=np.int64)
    grand_mean = y.mean()
    ss_between = 0.0
    ss_total = float(np.sum((y - grand_mean) ** 2))
    for gid in np.unique(g):
        mask = (g == gid)
        if mask.any():
            ss_between += mask.sum() * float(y[mask].mean() - grand_mean) ** 2
    eta2 = ss_between / ss_total if ss_total > 0 else float('nan')
    return float(eta2)

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description='Evaluate SOM classification quality with scientific metrics (string labels safe)')
    ap.add_argument('--assign', type=str, default='som_global_assignments.dump', help='Assignments (list of traces of neuron ids)')
    ap.add_argument('--vectors', type=str, default='vectors.dump', help='Original event vectors (for state, stock)')
    ap.add_argument('--meta', type=str, default='som_global_meta.dump', help='Optional SOM meta to get rows/cols')
    ap.add_argument('--classes', nargs='*', default=['Normal','Understock','Overstock'], help='Optional display/ordering of state labels')
    args = ap.parse_args()

    assignments: List[List[int]] = load_pickle(args.assign)
    vectors = load_pickle(args.vectors)

    # Grid shape
    rows = cols = None
    try:
        meta = load_pickle(args.meta)
        rows = int(meta.get('rows'))
        cols = int(meta.get('cols'))
    except Exception:
        pass
    k_neurons = max((max(a) if a else -1) for a in assignments) + 1
    if rows is None or cols is None:
        rows, cols = factor_grid(k_neurons)

    # Build aligned sequences of (cluster id, state, stock)
    flat_clusters: List[int] = []
    flat_states: List[object] = []
    flat_stocks: List[float] = []

    state_traces: List[List[object]] = []
    stock_traces: List[List[float]] = []

    for a, t in zip(assignments, vectors):
        m = min(len(a), len(t))
        if m <= 0:
            state_traces.append([])
            stock_traces.append([])
            continue
        states_i = [t[j][-1] for j in range(m)]                 # string labels
        stocks_i = [float(t[j][-2]) for j in range(m)]          # numeric
        clusters_i = a[:m]
        state_traces.append(states_i)
        stock_traces.append(stocks_i)
        flat_states.extend(states_i)
        flat_stocks.extend(stocks_i)
        flat_clusters.extend(clusters_i)

    y_true = np.array(flat_states, dtype=object)
    y_pred_clusters = np.asarray(flat_clusters, dtype=np.int64)

    # Contingency matrix (clusters x states)
    C, c_map, s_map, states_ordered = contingency_matrix(y_true, y_pred_clusters, class_order=args.classes)

    # Majority mapping cluster -> state label
    map_c2s = majority_mapping(C, states_ordered)
    y_pred_states = apply_mapping(y_pred_clusters, c_map, map_c2s)

    # Accuracies
    micro_acc, macro_acc, balanced_acc = accuracy_scores(y_true, y_pred_states)

    # Precision/Recall/F1
    macro_p, macro_r, macro_f1, weighted_p, weighted_r, weighted_f1 = precision_recall_f1(y_true, y_pred_states)

    # Purity
    purity = float(np.sum(np.max(C, axis=1)) / max(1, C.sum()))

    # Homogeneity/Completeness/V
    hom, com, V = homogeneity_completeness_v(C)

    # NMI & ARI & VI
    nmi, ari = nmi_ari(C)
    vi = variation_of_information(C)

    # Weighted cluster entropy (and normalized)
    Hc_w, Hc_w_norm = weighted_cluster_entropy(C)

    # Transition adjacency metrics
    adj = adjacency_metrics(assignments, state_traces, rows=rows, cols=cols)

    # Stock R^2 (η²) by clusters
    eta2_stock = stocks_eta_squared(assignments, stock_traces)

    # ---- Report ----
    print("=== DATA ===")
    print(f"Traces: {len(assignments)} | Samples used: {y_true.size} | Neurons (K): {k_neurons} | Grid: {rows}x{cols}")

    print("\n=== SUPERVISED METRICS (using state as ground truth) ===")
    print(f"Micro Accuracy      : {micro_acc:.4f}")
    print(f"Macro Accuracy      : {macro_acc:.4f}")
    print(f"Balanced Accuracy   : {balanced_acc:.4f}")
    print(f"Macro Precision/Recall/F1 : {macro_p:.4f} / {macro_r:.4f} / {macro_f1:.4f}")
    print(f"Weighted Precision/Recall/F1 : {weighted_p:.4f} / {weighted_r:.4f} / {weighted_f1:.4f}")
    print(f"Purity              : {purity:.4f}")
    print(f"Homogeneity         : {hom:.4f}")
    print(f"Completeness        : {com:.4f}")
    print(f"V-measure           : {V:.4f}")
    print(f"NMI (sqrt)          : {nmi:.4f}")
    print(f"ARI                 : {ari:.4f}")
    print(f"VI (lower is better): {vi:.4f}")

    print("\n=== UNSUPERVISED / SEQUENTIAL DIAGNOSTICS ===")
    print(f"Weighted cluster entropy        : {Hc_w:.4f}")
    print(f"Weighted cluster entropy (norm) : {Hc_w_norm:.4f}")
    print(f"Adjacency (all transitions)     : {adj['adjacency_all']:.4f}")
    print(f"Adjacency (same-state only)     : {adj['adjacency_same_state']:.4f}")
    print(f"Adjacency (diff-state only)     : {adj['adjacency_diff_state']:.4f}")
    print(f"Stock η^2 (explained by neuron) : {eta2_stock:.4f}")

    # Confusion matrix (clusters x states) with provided state order
    print("\n=== CONFUSION MATRIX (clusters x states) ===")
    header = "state\\cluster | " + " ".join(f"{str(s):>12}" for s in states_ordered)
    print(header)
    # keep rows in ascending cluster-id order
    cluster_ids_sorted = [cid for cid,_ in sorted(c_map.items(), key=lambda kv: kv[1])]
    for row_idx, cid in enumerate(cluster_ids_sorted):
        row = " ".join(f"{int(x):>12}" for x in C[row_idx])
        print(f"{int(cid):>14} | {row}")

if __name__ == '__main__':
    main()
