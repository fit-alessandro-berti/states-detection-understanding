#!/usr/bin/env python3
"""
Sliding-window + PCA (per-trace) with elbow-based model selection.

This script loads the original `vectors.dump` (list of traces -> list of event vectors),
removes the last two elements from each event vector (stock, state),
selects an "optimal" sliding window length via an elbow rule on PCA reconstruction error,
then selects the PCA dimensionality (k) via another elbow rule, and finally
projects each trace's windows **separately** (fit PCA per trace) so that the saved
`vectors_internal.dump` is a **list of lists of lists**:

    internal_vectors[trace_index][window_index][component_index]  # floats

Where each `internal_vectors[trace_index]` is the list of PCA embeddings for the
windows extracted from that trace. Short traces that cannot form a single window
are represented by an empty list to keep indices aligned with the input.

Usage example
-------------
python pca_sliding_window_elbow_grouped.py \
  --input vectors.dump \
  --output-internal vectors_internal.dump \
  --min-window 2 \
  --max-window 20 \
  --evr-threshold 0.95 \
  --max-components 50 \
  --max-windows-per-size 100000

Notes
-----
* We use per-trace PCA for the final embeddings to respect the requirement that
  "each intermediate list (trace) should be transformed and windowed separately".
* For elbow curves (window-size and k), we aggregate reconstruction MSE across
  traces via a window-count-weighted mean.
* We use scikit-learn PCA when available; otherwise we fall back to SVD.
"""
from __future__ import annotations
import argparse
import math
import os
import pickle
import random
from typing import Iterable, List, Sequence, Tuple

import numpy as np

# Optional sklearn PCA
try:
    from sklearn.decomposition import PCA as SKPCA  # type: ignore
    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    _HAVE_SKLEARN = False

# -----------------------------
# I/O and preprocessing
# -----------------------------

def load_vectors_dump(path: str) -> List[List[List[float]]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise ValueError("vectors.dump must be a list of traces")
    return data


def drop_stock_and_state(traces: List[List[List[float]]]) -> List[List[List[float]]]:
    """Drop the last two values (stock, state) from each event vector, preserving structure."""
    processed: List[List[List[float]]] = []
    for t in traces:
        new_t: List[List[float]] = []
        for vec in t:
            if len(vec) < 2:
                raise ValueError("Each event vector must have at least 2 elements to drop (stock, state)")
            new_t.append(vec[:-1])
        processed.append(new_t)
    return processed


def save_dump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

# -----------------------------
# Windowing
# -----------------------------

def windows_for_trace(trace: List[List[float]], window_size: int) -> np.ndarray:
    """Return a matrix of shape (num_windows, window_size * d) for a single trace.
    If the trace is too short, return an empty (0, window_size*d) array.
    """
    T = len(trace)
    if T < window_size:
        # Determine d from trace (if empty, d=0)
        d = len(trace[0]) if T > 0 else 0
        return np.empty((0, window_size * d), dtype=np.float64)
    M = np.asarray(trace, dtype=np.float64)  # (T, d)
    d = M.shape[1]
    Ws = [M[i:i+window_size].reshape(window_size * d) for i in range(T - window_size + 1)]
    return np.vstack(Ws).astype(np.float64)


def build_windows_grouped(
    traces: List[List[List[float]]],
    window_size: int,
) -> List[np.ndarray]:
    """Return per-trace window matrices. Each entry is (n_i, window_size*d) or (0, ...)."""
    mats: List[np.ndarray] = []
    for t in traces:
        mats.append(windows_for_trace(t, window_size))
    return mats

# -----------------------------
# PCA helpers
# -----------------------------

def pca_fit_full(X: np.ndarray, n_components: int | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center X and fit PCA up to n_components. Returns (components, mean, evr)."""
    if X.size == 0:
        raise ValueError("Empty matrix provided to PCA.")
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    if _HAVE_SKLEARN:
        pca = SKPCA(n_components=n_components, svd_solver="auto", random_state=17)
        pca.fit(Xc)
        return pca.components_.copy(), mean.ravel(), pca.explained_variance_ratio_.copy()
    # SVD fallback
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    total_var = (S ** 2).sum()
    evr = (S ** 2) / (total_var + 1e-12)
    if n_components is None:
        k = S.shape[0]
    else:
        k = int(min(n_components, S.shape[0]))
    return Vt[:k, :], mean.ravel(), evr[:k]


def pca_project(X: np.ndarray, components: np.ndarray, mean: np.ndarray, k: int) -> np.ndarray:
    """Project X onto first k principal components and return the embedding (Z)."""
    k = int(min(k, components.shape[0]))
    Xc = X - mean
    W = components[:k, :]  # (k, D)
    Z = Xc @ W.T
    return Z


def reconstruction_mse(X: np.ndarray, components: np.ndarray, mean: np.ndarray, k: int) -> float:
    k = int(min(k, components.shape[0]))
    Xc = X - mean
    W = components[:k, :]  # (k, D)
    Z = Xc @ W.T           # (n, k)
    Xc_rec = Z @ W         # (n, D)
    X_rec = Xc_rec + mean
    diff = (X - X_rec)
    return float(np.mean(diff * diff))


def k_for_evr(evr: np.ndarray, threshold: float) -> int:
    c = np.cumsum(evr)
    k = int(np.searchsorted(c, threshold, side="left") + 1)
    return min(max(k, 1), len(evr))

# -----------------------------
# Elbow / knee detection
# -----------------------------

def elbow_x_by_max_distance(x: Sequence[float], y: Sequence[float]) -> float:
    """Return x at maximum distance to the line from (x[0], y[0]) to (x[-1], y[-1]) after min-max scaling."""
    if len(x) != len(y) or len(x) < 3:
        return float(x[max(0, len(x)//2)])
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_n = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-12)
    x1, y1 = x_n[0], y_n[0]
    x2, y2 = x_n[-1], y_n[-1]
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    denom = math.hypot(a, b) + 1e-12
    d = np.abs(a * x_n + b * y_n + c) / denom
    idx = int(np.argmax(d))
    return float(x[idx])

# -----------------------------
# Model selection (aggregate across traces)
# -----------------------------

def select_window_size(
    traces_internal: List[List[List[float]]],
    min_w: int,
    max_w: int,
    evr_threshold: float,
    max_windows_per_trace: int | None,
    rng: random.Random,
) -> Tuple[int, List[Tuple[int, float]]]:
    """Pick window size via elbow on weighted-mean MSE across traces.

    For each w, for each trace with windows:
      - Fit PCA (full rank)
      - Compute k95 from EVR threshold
      - Compute reconstruction MSE using k95
    Aggregate MSE with weights = number of windows in that trace (cap via subsampling if requested).
    """
    # Realistic upper bound: cannot exceed the longest trace length
    max_len = max((len(t) for t in traces_internal), default=0)
    if max_len < min_w:
        raise ValueError(f"All traces are shorter than min_w={min_w} (max trace length is {max_len}).")

    W_vals = list(range(min_w, min(max_w, max_len) + 1))
    if len(W_vals) < 2:
        raise ValueError("Not enough window sizes to run elbow selection.")

    curve: List[Tuple[int, float]] = []
    for w in W_vals:
        mats = build_windows_grouped(traces_internal, w)
        total_weight = 0
        weighted_mse_sum = 0.0
        for X in mats:
            if X.shape[0] == 0:
                continue
            # Optional subsampling per trace for speed
            if max_windows_per_trace is not None and X.shape[0] > max_windows_per_trace:
                idx = rng.sample(range(X.shape[0]), k=max_windows_per_trace)
                X = X[idx]
            comps, mean, evr = pca_fit_full(X, n_components=None)
            k95 = k_for_evr(evr, evr_threshold)
            mse = reconstruction_mse(X, comps, mean, k95)
            weighted_mse_sum += mse * X.shape[0]
            total_weight += X.shape[0]
        if total_weight == 0:
            # No trace had windows for this w; skip
            continue
        curve.append((w, weighted_mse_sum / total_weight))

    ws = [w for (w, _) in curve]
    mses = [m for (_, m) in curve]
    chosen_w = int(elbow_x_by_max_distance(ws, mses))
    return chosen_w, curve


def select_pca_k_for_window(
    traces_internal: List[List[List[float]]],
    window_size: int,
    max_components: int,
    max_windows_per_trace: int | None,
    rng: random.Random,
) -> Tuple[int, List[Tuple[int, float]]]:
    """Pick PCA components via elbow on weighted-mean MSE across traces for fixed window size.

    For each trace:
      - Fit full PCA once; then compute MSE(k) for k=1..K_t (trace rank).
    Aggregate the MSE(k) across traces (weights = number of windows used per trace).
    """
    mats = build_windows_grouped(traces_internal, window_size)

    # Precompute per-trace PCA and MSE curves
    per_trace_curves: List[Tuple[np.ndarray, np.ndarray, int]] = []  # (ks, mses, weight)
    K_global = 0

    for X in mats:
        if X.shape[0] == 0:
            continue
        if max_windows_per_trace is not None and X.shape[0] > max_windows_per_trace:
            idx = rng.sample(range(X.shape[0]), k=max_windows_per_trace)
            X = X[idx]
        n, D = X.shape
        K_t = int(min(max_components, n, D))
        if K_t < 1:
            continue
        comps, mean, _ = pca_fit_full(X, n_components=K_t)
        ks = np.arange(1, comps.shape[0] + 1, dtype=int)
        mses = np.array([reconstruction_mse(X, comps, mean, int(k)) for k in ks], dtype=float)
        per_trace_curves.append((ks, mses, X.shape[0]))
        K_global = max(K_global, int(ks.max()))

    if not per_trace_curves:
        raise ValueError("No windows available to select PCA components at the chosen window size.")

    # Aggregate into a single curve up to K_global
    agg_ks = np.arange(1, K_global + 1, dtype=int)
    agg_mses = np.zeros_like(agg_ks, dtype=float)
    agg_weights = np.zeros_like(agg_ks, dtype=float)

    for ks, mses, w in per_trace_curves:
        for i, k in enumerate(ks):
            agg_mses[k-1] += mses[i] * w
            agg_weights[k-1] += w

    # Guard against positions with zero weight (e.g., traces too small for high k)
    valid = agg_weights > 0
    agg_mses[valid] = agg_mses[valid] / agg_weights[valid]
    # For any invalid positions (should be rare), fill via forward fill
    for i in range(len(agg_mses)):
        if not valid[i]:
            agg_mses[i] = agg_mses[i-1] if i > 0 else agg_mses[valid][0]

    chosen_k = int(elbow_x_by_max_distance(list(agg_ks), list(agg_mses)))
    curve = list(zip(agg_ks.tolist(), agg_mses.tolist()))
    return chosen_k, curve

# -----------------------------
# Final per-trace embeddings
# -----------------------------

def compute_embeddings_per_trace(
    traces_internal: List[List[List[float]]],
    window_size: int,
    k_components: int,
) -> List[List[List[float]]]:
    """For each trace, fit PCA on its own windows and project to k_components.
    Returns nested python lists (float64) with shape [n_traces][n_windows_i][k].
    Traces shorter than the window produce an empty list to preserve alignment.
    """
    grouped_windows = build_windows_grouped(traces_internal, window_size)
    output: List[List[List[float]]] = []
    for X in grouped_windows:
        if X.shape[0] == 0:
            output.append([])
            continue
        n, D = X.shape
        K_t = int(min(k_components, n, D))
        comps, mean, _ = pca_fit_full(X, n_components=K_t)
        Z = pca_project(X, comps, mean, K_t)  # (n, K_t)
        output.append(Z.astype(np.float64).tolist())
    return output

# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Sliding-window + PCA (per-trace) with elbow selection")
    parser.add_argument("--input", type=str, default="vectors.dump", help="Path to input vectors.dump")
    parser.add_argument("--output-internal", type=str, default="vectors_internal.dump", help="Where to write the final per-trace PCA window embeddings")
    parser.add_argument("--min-window", type=int, default=2, help="Minimum sliding window size to test")
    parser.add_argument("--max-window", type=int, default=20, help="Maximum sliding window size to test (capped by max trace length)")
    parser.add_argument("--evr-threshold", type=float, default=0.95, help="Explained-variance threshold for k at window selection phase (0-1)")
    parser.add_argument("--max-components", type=int, default=50, help="Upper bound on PCA components to consider at the k-elbow step")
    parser.add_argument("--max-windows-per-trace", type=int, default=100000, help="If >0, subsample at most this many windows per trace when computing elbows")
    parser.add_argument("--seed", type=int, default=17, help="Random seed for any subsampling")

    args = parser.parse_args()

    rng = random.Random(args.seed)
    max_wpt = None if args.max_windows_per_trace <= 0 else int(args.max_windows_per_trace)

    # 1) Load + preprocess (drop stock, state)
    traces = load_vectors_dump(args.input)
    traces_internal = drop_stock_and_state(traces)

    # 2) Window-size selection (aggregate elbow over traces)
    chosen_w, w_curve = select_window_size(
        traces_internal,
        min_w=args.min_window,
        max_w=args.max_window,
        evr_threshold=args.evr_threshold,
        max_windows_per_trace=max_wpt,
        rng=rng,
    )

    # 3) PCA components selection at chosen window size (aggregate elbow over traces)
    chosen_k, k_curve = select_pca_k_for_window(
        traces_internal,
        window_size=chosen_w,
        max_components=args.max_components,
        max_windows_per_trace=max_wpt,
        rng=rng,
    )

    # 4) Final per-trace embeddings (fit PCA per trace, then project)
    final_internal = compute_embeddings_per_trace(
        traces_internal=traces_internal,
        window_size=chosen_w,
        k_components=chosen_k,
    )

    # 5) Persist and report
    save_dump(final_internal, args.output_internal)

    print("=== Window-size selection (weighted mean MSE) ===")
    print("w\tMSE")
    for w, mse in w_curve:
        print(f"{w}\t{mse:.8f}")

    print("\nChosen sliding window size:", chosen_w)

    print("\n=== PCA components selection (weighted mean MSE) for w = {} ===".format(chosen_w))
    print("k\tMSE")
    for k, mse in k_curve:
        print(f"{k}\t{mse:.8f}")

    print("\nChosen number of PCA components:", chosen_k)

    print("\nSUMMARY:")
    print(f"SLIDING_WINDOW_SIZE={chosen_w}")
    print(f"PCA_COMPONENTS={chosen_k}")
    print(f"INTERNAL_VECTORS_DUMP={os.path.abspath(args.output_internal)}")


if __name__ == "__main__":
    main()
