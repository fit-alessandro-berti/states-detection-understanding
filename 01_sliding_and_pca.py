"""
What this script does
---------------------
1) Loads a pickled file "vectors.dump" (list of traces; each trace is a list of event vectors).
2) Preprocesses each event vector to drop the last two elements (stock, state), leaving only the
   one-hot activity part. Saves these preprocessed traces to a new dump file.
3) Builds windowed datasets for a range of sliding-window sizes.
4) For each window size, fits PCA and computes reconstruction error using a fixed
   explained-variance threshold (e.g., 95%). It then picks the window size using an
   "elbow" (knee) detection heuristic.
5) With the chosen window size, runs PCA for k = 1..K and picks the number of components
   using the same elbow heuristic on the reconstruction-error curve.
6) Prints the chosen window size and PCA components, and writes the final preprocessed
   list of traces ("internal vectors") to a new dump file.

Notes
-----
- This script works with just NumPy. If scikit-learn is available, it uses it for PCA; otherwise it
  falls back to a small PCA implementation via SVD.
- Reconstruction error: mean squared error (MSE) between original and reconstructed windowed data.
- The elbow detection uses the maximum perpendicular distance to the line between the first and last
  points of the (x, y) curve after normalizing x and y to [0, 1].

Usage
-----
python pca_sliding_window_elbow.py \
    --input vectors.dump \
    --output-internal vectors_internal.dump \
    --min-window 2 \
    --max-window 20 \
    --evr-threshold 0.95 \
    --max-components 50 \
    --max-windows-per-size 100000

"""
from __future__ import annotations
import argparse
import math
import os
import pickle
import random
from typing import List, Sequence, Tuple

import numpy as np

# Try to import scikit-learn PCA; if unavailable, we will use a lightweight SVD-based fallback.
try:
    from sklearn.decomposition import PCA as SKPCA  # type: ignore
    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    _HAVE_SKLEARN = False


# -----------------------------
# Data loading & preprocessing
# -----------------------------

def load_vectors_dump(path: str) -> List[List[List[float]]]:
    """Load the original vectors.dump created by your transformation script.

    Expected shape: list of traces; each trace is a list of event vectors; each event vector is a list of numbers.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise ValueError("vectors.dump must be a list of traces")
    return data


def drop_stock_and_state(traces: List[List[List[float]]]) -> List[List[List[float]]]:
    """Drop the last two elements from every per-event vector (stock, state),
    returning only the one-hot activity part. Leaves the nested structure intact.
    """
    processed: List[List[List[float]]] = []
    for trace in traces:
        new_trace = []
        for vec in trace:
            if len(vec) < 2:
                raise ValueError("Each event vector must have at least 2 elements to drop (stock, state)")
            new_trace.append(vec[:-2])
        if new_trace:  # keep only non-empty traces
            processed.append(new_trace)
    return processed


def save_vectors_dump(traces: List[List[List[float]]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(traces, f, protocol=pickle.HIGHEST_PROTOCOL)


# -----------------------------
# Windowing utilities
# -----------------------------

def build_windowed_matrix(
    traces: List[List[List[float]]],
    window_size: int,
    max_windows: int | None = None,
    rng: random.Random | None = None,
) -> np.ndarray:
    """Create a 2D array X of shape (n_windows, window_size * dim) by concatenating contiguous
    windows from the sequence of per-event vectors in each trace.

    Args:
        traces: list of traces, each trace is list of event vectors (length d).
        window_size: sliding window length (must be >= 2 recommended).
        max_windows: if provided, randomly subsample at most this many windows across all traces.
        rng: Python random.Random used for subsampling (if needed).

    Returns:
        X: np.ndarray with dtype float32 of shape (n_windows, window_size * d)
    """
    assert window_size >= 1
    windows: List[np.ndarray] = []
    for trace in traces:
        T = len(trace)
        if T < window_size:
            continue
        # shape (T, d)
        mat = np.asarray(trace, dtype=np.float32)
        d = mat.shape[1]
        # Generate windowed slices by concatenating
        for start in range(0, T - window_size + 1):
            w = mat[start : start + window_size]  # (window_size, d)
            windows.append(w.reshape(window_size * d))

    if not windows:
        raise ValueError(
            f"No windows could be formed for window_size={window_size}. "
            "Make sure at least one trace has length >= window_size."
        )

    if max_windows is not None and len(windows) > max_windows:
        rng = rng or random.Random(17)
        idx = rng.sample(range(len(windows)), k=max_windows)
        windows = [windows[i] for i in idx]

    X = np.vstack(windows).astype(np.float32)
    return X


# -----------------------------
# PCA + reconstruction error
# -----------------------------

def _pca_fit_transform(
    X: np.ndarray,
    n_components: int | None = None,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Center X, fit PCA, return (X_centered, (components, mean, explained_variance_ratio)).

    - If scikit-learn is available, use it and return its components_.
    - Otherwise, do a thin SVD on centered X. components shape will be (n_components, n_features).
    """
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean

    if _HAVE_SKLEARN:
        # Let sklearn choose full rank if n_components is None
        pca = SKPCA(n_components=n_components, svd_solver="randomized" if n_components and n_components < min(X.shape) else "auto", random_state=17)
        pca.fit(Xc)
        components = pca.components_.copy()
        evr = pca.explained_variance_ratio_.copy()
        return Xc, (components, mean.ravel(), evr)

    # Fallback: numpy SVD
    # Xc = U S Vt, where rows are samples, cols are features
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    total_var = (S ** 2).sum()
    ev = (S ** 2) / total_var

    if n_components is None:
        k = S.shape[0]
    else:
        k = int(min(n_components, S.shape[0]))

    components = Vt[:k, :]
    evr = ev[:k]
    return Xc, (components, mean.ravel(), evr)


def _pca_project_reconstruct(
    Xc: np.ndarray,
    pca_params: Tuple[np.ndarray, np.ndarray, np.ndarray],
    k: int,
) -> np.ndarray:
    """Reconstruct X from the first k components given centered data Xc.

    Args:
        Xc: centered data (X - mean)
        pca_params: (components, mean, evr)
        k: number of components to use for reconstruction
    Returns:
        X_recon: reconstructed data (un-centered)
    """
    components, mean, _ = pca_params
    k = int(min(k, components.shape[0]))
    W = components[:k, :]  # (k, D)
    Z = Xc @ W.T           # (n, k)
    Xc_rec = Z @ W         # (n, D)
    X_rec = Xc_rec + mean
    return X_rec


def reconstruction_mse(X: np.ndarray, X_rec: np.ndarray) -> float:
    diff = X.astype(np.float64) - X_rec.astype(np.float64)
    return float(np.mean(diff * diff))


def components_for_evr_threshold(evr: np.ndarray, threshold: float) -> int:
    """Return the smallest k such that cumulative explained variance ratio >= threshold.
    If even all components are below the threshold, return len(evr).
    """
    c = np.cumsum(evr)
    k = int(np.searchsorted(c, threshold, side="left") + 1)
    return min(max(k, 1), len(evr))


# -----------------------------
# Elbow detection (knee point)
# -----------------------------

def elbow_x_by_max_distance(x: Sequence[float], y: Sequence[float]) -> float:
    """Pick x at the maximum perpendicular distance to the line from first to last point.

    Both x and y are normalized to [0, 1] before computing distances to avoid scale bias.
    Returns the original x value (not normalized) at the elbow.
    """
    if len(x) != len(y) or len(x) < 3:
        # Need at least 3 points to define a meaningful elbow
        return x[int(len(x) / 2)]

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Normalize to [0, 1]
    x_n = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-12)

    x1, y1 = x_n[0], y_n[0]
    x2, y2 = x_n[-1], y_n[-1]

    # Line coefficients for ax + by + c = 0
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    denom = math.hypot(a, b) + 1e-12

    # Perpendicular distance from each (x_i, y_i) to the line
    d = np.abs(a * x_n + b * y_n + c) / denom
    idx = int(np.argmax(d))
    return float(x[idx])


# -----------------------------
# Main model-selection routine
# -----------------------------

def pick_window_size(
    traces_internal: List[List[List[float]]],
    min_w: int,
    max_w: int,
    evr_threshold: float,
    max_windows_per_size: int,
    rng: random.Random,
) -> Tuple[int, List[Tuple[int, float, int]]]:
    """Choose window size using elbow on reconstruction error across window sizes.

    For each window size w in [min_w, max_w]:
      - Build X_w by concatenating windows of length w.
      - Fit PCA (full rank), compute k* = k(95% EVR) and reconstruction error using first k* components.

    The elbow is computed on the curve (w, MSE_w). Returns chosen w and a list of (w, mse, k95).
    """
    results: List[Tuple[int, float, int]] = []
    # Determine actual max_w given data
    max_trace_len = max((len(t) for t in traces_internal), default=0)
    if max_trace_len < min_w:
        raise ValueError(f"All traces are shorter than min_w={min_w} (max trace length is {max_trace_len}).")

    W_values = list(range(min_w, min(max_w, max_trace_len) + 1))
    if len(W_values) < 2:
        raise ValueError("Not enough valid window sizes to perform elbow selection.")

    for w in W_values:
        X = build_windowed_matrix(traces_internal, w, max_windows=max_windows_per_size, rng=rng)
        # Fit PCA with all possible components; we'll select k for EVR threshold.
        Xc, p = _pca_fit_transform(X, n_components=None)
        components, mean, evr = p
        k95 = components_for_evr_threshold(evr, evr_threshold)
        X_rec = _pca_project_reconstruct(Xc, p, k95)
        mse = reconstruction_mse(X, X_rec)
        results.append((w, mse, k95))

    # Elbow on (w, mse)
    ws = [r[0] for r in results]
    mses = [r[1] for r in results]
    chosen_w = int(elbow_x_by_max_distance(ws, mses))
    return chosen_w, results


def pick_pca_components_for_window(
    traces_internal: List[List[List[float]]],
    window_size: int,
    max_components: int,
    max_windows: int,
    rng: random.Random,
) -> Tuple[int, List[Tuple[int, float]]]:
    """For a fixed window size, compute reconstruction MSE for k = 1..K and pick k by elbow.

    Returns the chosen k and the list of (k, mse).
    """
    X = build_windowed_matrix(traces_internal, window_size, max_windows=max_windows, rng=rng)

    # Fit a PCA once up to the max feasible components; reuse to reconstruct for each k
    # Max feasible is limited by rank = min(n_samples, n_features)
    n_samples, n_features = X.shape
    K_feasible = min(n_samples, n_features)
    K = int(min(max_components, K_feasible))
    if K < 2:
        raise ValueError(f"Not enough rank to evaluate components (K={K}).")

    Xc, p = _pca_fit_transform(X, n_components=K)

    curve: List[Tuple[int, float]] = []
    for k in range(1, K + 1):
        X_rec = _pca_project_reconstruct(Xc, p, k)
        mse = reconstruction_mse(X, X_rec)
        curve.append((k, mse))

    ks = [c[0] for c in curve]
    mses = [c[1] for c in curve]
    chosen_k = int(elbow_x_by_max_distance(ks, mses))
    return chosen_k, curve


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Sliding-window + PCA elbow selection")
    parser.add_argument("--input", type=str, default="vectors.dump", help="Path to input vectors.dump")
    parser.add_argument("--output-internal", type=str, default="vectors_internal.dump", help="Path to write preprocessed internal vectors dump")
    parser.add_argument("--min-window", type=int, default=2, help="Minimum sliding window size to test")
    parser.add_argument("--max-window", type=int, default=20, help="Maximum sliding window size to test (capped by max trace length)")
    parser.add_argument("--evr-threshold", type=float, default=0.95, help="Explained variance threshold for selecting k per window size (0-1)")
    parser.add_argument("--max-components", type=int, default=50, help="Upper bound on PCA components to test at the final step")
    parser.add_argument("--max-windows-per-size", type=int, default=100000, help="If >0, subsample at most this many windows per window size for speed")
    parser.add_argument("--seed", type=int, default=17, help="Random seed used for subsampling windows")

    args = parser.parse_args()

    rng = random.Random(args.seed)

    # 1) Load
    traces = load_vectors_dump(args.input)

    # 2) Preprocess to drop stock & state
    traces_internal = drop_stock_and_state(traces)

    # 3) Save internal vectors for future reuse
    #save_vectors_dump(traces_internal, args.output_internal)

    # 4) Window size selection via elbow on reconstruction error (at EVR threshold)
    chosen_w, window_results = pick_window_size(
        traces_internal,
        min_w=args.min_window,
        max_w=args.max_window,
        evr_threshold=args.evr_threshold,
        max_windows_per_size=max(1, args.max_windows_per_size),
        rng=rng,
    )

    # 5) For chosen window, pick PCA components by elbow on k -> MSE
    chosen_k, k_curve = pick_pca_components_for_window(
        traces_internal,
        window_size=chosen_w,
        max_components=args.max_components,
        max_windows=max(1, args.max_windows_per_size),
        rng=rng,
    )

    # 6) Pretty print results
    print("=== Window-size selection (EVR-threshold = {:.2f}) ===".format(args.evr_threshold))
    print("w\tMSE\tk_for_threshold")
    for (w, mse, k95) in window_results:
        print(f"{w}\t{mse:.6f}\t{k95}")

    print("\nChosen sliding window size:", chosen_w)

    print("\n=== PCA components selection for w = {} ===".format(chosen_w))
    print("k\tMSE")
    for (k, mse) in k_curve:
        print(f"{k}\t{mse:.6f}")

    print("\nChosen number of PCA components:", chosen_k)

    # Build the full windowed matrix with the optimal window size (no subsampling)
    X_final = build_windowed_matrix(traces_internal, chosen_w, max_windows=None, rng=rng)

    # Fit PCA with the chosen number of components and project to get embeddings (long floats)
    Xc_final, p_final = _pca_fit_transform(X_final, n_components=chosen_k)
    components, mean, _ = p_final
    W = components[:chosen_k, :]  # (k, D)
    Z = Xc_final @ W.T  # (n_windows, k) float64

    # Save as a simple list-of-lists (each row is a PCA window embedding)
    save_vectors_dump(Z.tolist(), args.output_internal)

    # 7) Final summary lines (easy to grep)
    print("\nSUMMARY:")
    print(f"SLIDING_WINDOW_SIZE={chosen_w}")
    print(f"PCA_COMPONENTS={chosen_k}")
    print(f"INTERNAL_VECTORS_DUMP={os.path.abspath(args.output_internal)}")


if __name__ == "__main__":
    main()
