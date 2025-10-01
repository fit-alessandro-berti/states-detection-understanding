#!/usr/bin/env python3
"""
Train a Self-Organizing Map (SOM) **per trace** on `vectors_internal.dump`,
searching over grid sizes using a quality objective that combines:
  1) **Neuron-usage entropy** (higher is better; normalized to [0,1])
  2) **Adjacency consistency** along the trace (fraction of consecutive windows
     mapped to the same or neighboring neurons; higher is better)

The script picks the grid size that maximizes a weighted score:
    score = alpha * adjacency_consistency + (1 - alpha) * entropy_norm

It prints the chosen grid per trace with metrics and stores a structured
result file with per-trace assignments, metrics, and SOM codebooks.

Input shape (from your previous pipeline):
    vectors_internal.dump  ->  List[ Trace ]
    Trace -> List[ WindowVector(float) ] with dimension k (PCA components)

Output:
    som_results.dump -> List[ dict ] (aligned by trace index), each dict contains:
        {
            'rows': int,
            'cols': int,
            'alpha': float,
            'adjacency_consistency': float,
            'entropy': float,
            'entropy_norm': float,
            'combined_score': float,
            'quantization_error': float,
            'assignments': List[ [r, c] ],  # BMU grid coords per window
            'codebook': List[ List[float] ], # (rows*cols, k)
            'k': int, # input dimensionality
            'n_samples': int
        }
    For traces shorter than 2 windows, metrics will be computed where meaningful
    and adjacency_consistency defaults to 1.0 if there is <= 1 transition.

Usage
-----
python som_per_trace_gridsearch.py \
  --input vectors_internal.dump \
  --output som_results.dump \
  --min-side 3 --max-side 12 \
  --alpha 0.7 \
  --epochs 20 \
  --lr 0.5 --sigma auto \
  --neighbor-mode 8 \
  --max-samples-per-trace 20000 \
  --seed 17

Notes
-----
* This is a minimal, dependency-free SOM (online learning, Gaussian neighborhood,
  exponential decay schedules). If you have MiniSom or sklearn-som, you can swap
  it in easily.
* Features are z-scored **per trace** to make training stable across traces.
* Grid search tries square maps from min_side to max_side inclusive. You can
  also pass explicit sizes via --sizes 4x6 6x8 ... to override.
"""
from __future__ import annotations
import argparse
import math
import os
import pickle
import random
from typing import List, Optional, Sequence, Tuple

import numpy as np

# -----------------------------
# Utility I/O
# -----------------------------

def load_dump(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_dump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

# -----------------------------
# SOM implementation (rectangular grid)
# -----------------------------
class SOM:
    def __init__(self, rows: int, cols: int, dim: int, lr: float = 0.5, sigma: Optional[float] = None,
                 epochs: int = 20, seed: int = 17):
        assert rows > 0 and cols > 0
        self.rows = rows
        self.cols = cols
        self.dim = dim
        self.lr0 = float(lr)
        self.sigma0 = float(max(rows, cols) / 2.0) if (sigma is None or sigma == 'auto') else float(sigma)
        self.epochs = int(max(1, epochs))
        self.rng = np.random.default_rng(seed)
        # codebook weights (rows*cols, dim)
        self.W = None  # type: Optional[np.ndarray]
        # precompute neuron grid coordinates and pairwise squared distances
        coords = np.stack(np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij'), axis=-1)  # (rows, cols, 2)
        self.coords = coords.reshape(rows * cols, 2).astype(np.float64)
        diffs = self.coords[:, None, :] - self.coords[None, :, :]  # (M, M, 2)
        self.grid_dist2 = np.sum(diffs * diffs, axis=-1)  # (M, M)

    def _init_weights(self, X: np.ndarray) -> None:
        n, d = X.shape
        assert d == self.dim
        # init by sampling from data with small noise
        idx = self.rng.choice(n, size=self.rows * self.cols, replace=True)
        W = X[idx].copy()
        W += 1e-6 * self.rng.standard_normal(W.shape)
        self.W = W

    def _bmu_index(self, x: np.ndarray) -> int:
        # return index of best matching unit by squared Euclidean distance
        dif = self.W - x  # (M, d)
        return int(np.argmin(np.einsum('md,md->m', dif, dif)))

    def fit(self, X: np.ndarray) -> None:
        if X.size == 0:
            raise ValueError('Empty data passed to SOM.fit')
        n, d = X.shape
        if self.W is None:
            self._init_weights(X)
        M = self.rows * self.cols
        total_iters = max(1, self.epochs * n)
        # time constants for decay
        tau = total_iters / math.log(self.sigma0 + 1e-12)
        for t in range(total_iters):
            # exponential decay schedules
            lr_t = self.lr0 * math.exp(-t / total_iters)
            sigma_t = max(1e-3, self.sigma0 * math.exp(-t / tau))
            # sample a data point
            i = self.rng.integers(0, n)
            x = X[i]
            # BMU
            b = self._bmu_index(x)
            # Neighborhood (Gaussian on grid distance)
            g = np.exp(-self.grid_dist2[b] / (2.0 * sigma_t * sigma_t))  # (M,)
            # Update all neurons (online rule)
            # W_m <- W_m + lr * g_m * (x - W_m)
            self.W += lr_t * g[:, None] * (x[None, :] - self.W)

    def transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (bmu_indices, bmu_coords) for each row in X."""
        diffs = X[:, None, :] - self.W[None, :, :]  # (n, M, d)
        dist2 = np.einsum('nmd,nmd->nm', diffs, diffs)
        bmu_idx = np.argmin(dist2, axis=1)
        bmu_coords = self.coords[bmu_idx]
        return bmu_idx, bmu_coords

    def quantization_error(self, X: np.ndarray) -> float:
        diffs = X[:, None, :] - self.W[None, :, :]
        dist2 = np.einsum('nmd,nmd->nm', diffs, diffs)
        min_dist2 = np.min(dist2, axis=1)
        return float(np.mean(min_dist2))

# -----------------------------
# Metrics
# -----------------------------

def neuron_entropy(bmu_idx: np.ndarray, n_units: int) -> Tuple[float, float]:
    """Return (H, H_norm). H in nats; H_norm = H / log(n_units) in [0,1]."""
    if bmu_idx.size == 0 or n_units <= 1:
        return 0.0, 0.0
    counts = np.bincount(bmu_idx, minlength=n_units).astype(float)
    p = counts / max(1, counts.sum())
    # avoid log(0)
    p = np.clip(p, 1e-12, 1.0)
    H = float(-np.sum(p * np.log(p)))
    H_norm = H / math.log(n_units)
    return H, H_norm


def adjacency_consistency(bmu_coords: np.ndarray, neighbor_mode: int = 8) -> float:
    """Proportion of consecutive pairs mapped to the same or neighboring neurons.
       neighbor_mode=4 -> use Manhattan distance <= 1 (4-neighborhood)
       neighbor_mode=8 -> use Chebyshev distance <= 1 (8-neighborhood)
    """
    n = bmu_coords.shape[0]
    if n <= 1:
        return 1.0
    dif = np.abs(np.diff(bmu_coords, axis=0))  # (n-1, 2)
    if neighbor_mode == 4:
        d = np.sum(dif, axis=1)  # Manhattan
        good = (d <= 1)
    else:  # 8-neighborhood (default)
        d = np.max(dif, axis=1)  # Chebyshev
        good = (d <= 1)
    return float(np.mean(good))

# -----------------------------
# Standardization
# -----------------------------

def zscore_per_trace(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Xz, mean, std). std is clipped to avoid divide-by-zero."""
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Xz = (X - mu) / sd
    return Xz, mu.ravel(), sd.ravel()

# -----------------------------
# Grid search per trace
# -----------------------------

def parse_sizes(args) -> List[Tuple[int, int]]:
    if args.sizes:
        sizes = []
        for s in args.sizes:
            if 'x' not in s:
                raise ValueError(f"Invalid --sizes entry '{s}'. Use RxC, e.g., 6x8")
            r, c = s.lower().split('x')
            sizes.append((int(r), int(c)))
        return sizes
    else:
        return [(s, s) for s in range(args.min_side, args.max_side + 1)]


def grid_search_trace(
    X: np.ndarray,
    sizes: Sequence[Tuple[int, int]],
    alpha: float,
    epochs: int,
    lr: float,
    sigma: Optional[float],
    neighbor_mode: int,
    seed: int,
) -> Tuple[dict, List[dict]]:
    """Train one SOM per candidate size, compute metrics, return best result and all candidates.
    Returns (best, candidates)."""
    n, k = X.shape
    rng = np.random.default_rng(seed)
    candidates: List[dict] = []
    for (r, c) in sizes:
        som = SOM(r, c, k, lr=lr, sigma=sigma, epochs=epochs, seed=int(rng.integers(0, 2**31-1)))
        som.fit(X)
        bmu_idx, bmu_coords = som.transform(X)
        qe = som.quantization_error(X)
        H, Hn = neuron_entropy(bmu_idx, r * c)
        adj = adjacency_consistency(bmu_coords, neighbor_mode=neighbor_mode)
        score = alpha * adj + (1.0 - alpha) * Hn
        cand = {
            'rows': r,
            'cols': c,
            'alpha': alpha,
            'adjacency_consistency': adj,
            'entropy': H,
            'entropy_norm': Hn,
            'combined_score': score,
            'quantization_error': qe,
            'assignments': bmu_coords.astype(int).tolist(),
            'codebook': som.W.astype(float).tolist(),
            'k': int(k),
            'n_samples': int(n),
        }
        candidates.append(cand)

    # Pick the best by combined score, then by higher adjacency, then lower QE, then higher entropy
    def sort_key(c):
        return (-c['combined_score'], -c['adjacency_consistency'], c['quantization_error'], -c['entropy_norm'])

    candidates.sort(key=sort_key)
    return candidates[0], candidates

# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description='Per-trace SOM grid search with entropy + adjacency criteria')
    ap.add_argument('--input', type=str, default='vectors_internal.dump', help='Path to vectors_internal.dump')
    ap.add_argument('--output', type=str, default='som_results.dump', help='Where to store per-trace SOM results')
    ap.add_argument('--min-side', type=int, default=3, help='Minimum side for square grids (ignored if --sizes is set)')
    ap.add_argument('--max-side', type=int, default=12, help='Maximum side for square grids (ignored if --sizes is set)')
    ap.add_argument('--sizes', nargs='*', help="Explicit list of grid sizes like '4x6 6x8 8x8' (overrides min/max)")
    ap.add_argument('--alpha', type=float, default=0.7, help='Weight for adjacency in combined score (0..1)')
    ap.add_argument('--epochs', type=int, default=20, help='Training epochs (passes) per candidate size')
    ap.add_argument('--lr', type=float, default=0.5, help='Initial learning rate for SOM')
    ap.add_argument('--sigma', type=str, default='auto', help="Initial neighborhood radius; 'auto' -> max(rows, cols)/2")
    ap.add_argument('--neighbor-mode', type=int, default=8, choices=[4,8], help='Adjacency neighborhood: 4 or 8')
    ap.add_argument('--max-samples-per-trace', type=int, default=20000, help='Subsample each trace to at most this many windows for speed (<=0 means no cap)')
    ap.add_argument('--seed', type=int, default=17, help='Random seed')

    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    data = load_dump(args.input)
    if not isinstance(data, list):
        raise ValueError('Expected a list of traces in vectors_internal.dump')

    sizes = parse_sizes(args)

    results: List[dict] = []

    for ti, trace in enumerate(data):
        X = np.asarray(trace, dtype=np.float64)
        print(f"\n--- Trace {ti} ---")
        if X.ndim != 2 or X.shape[0] == 0:
            print('Empty or invalid trace; skipping')
            results.append({
                'rows': 0, 'cols': 0, 'alpha': args.alpha,
                'adjacency_consistency': float('nan'),
                'entropy': float('nan'), 'entropy_norm': float('nan'),
                'combined_score': float('nan'), 'quantization_error': float('nan'),
                'assignments': [], 'codebook': [],
                'k': int(X.shape[1]) if X.ndim == 2 else 0,
                'n_samples': int(X.shape[0]) if X.ndim == 2 else 0,
            })
            continue

        # Optional subsampling for speed
        if args.max_samples_per_trace > 0 and X.shape[0] > args.max_samples_per_trace:
            idx = rng.choice(X.shape[0], size=args.max_samples_per_trace, replace=False)
            X = X[idx]

        # Standardize per trace
        Xz, mu, sd = zscore_per_trace(X)

        best, candidates = grid_search_trace(
            Xz, sizes, alpha=args.alpha, epochs=args.epochs, lr=args.lr,
            sigma=(None if args.sigma == 'auto' else float(args.sigma)),
            neighbor_mode=args.neighbor_mode, seed=int(rng.integers(0, 2**31-1))
        )

        print(f"Chosen grid: {best['rows']}x{best['cols']}")
        print(f"  n_samples={best['n_samples']}, k={best['k']}")
        print(f"  adjacency={best['adjacency_consistency']:.4f}, entropy_norm={best['entropy_norm']:.4f}, QE={best['quantization_error']:.6f}")
        print(f"  combined_score={best['combined_score']:.4f} (alpha={best['alpha']})")

        # Store the best
        # Note: We keep standardized-space codebook. If you need original scale, invert with mu/sd.
        best['trace_index'] = ti
        results.append(best)

    save_dump(results, args.output)

    print("\nSUMMARY:")
    print(f"OUTPUT={os.path.abspath(args.output)}")

if __name__ == '__main__':
    main()
