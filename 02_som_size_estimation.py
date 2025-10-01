#!/usr/bin/env python3
"""
FAST per-trace SOM grid search (NumPy-only) — **stability-fixed**
-----------------------------------------------------------------
This replaces the previous fast script with numerically stable batch updates to
eliminate overflows/NaNs you saw. Key fixes:
  • **Stable batch update**:
        W ← W + η · ( (GᵀX / (∑G + ε)) − W )
    instead of W ← W + η · (GᵀX − diag(∑G)W). This bounds the update size.
  • **Sigma / LR schedules** safe for any epoch count; sigma floored (≥0.5).
  • **Float32 training + clipping** of codebook weights to ±W_CLIP (default 10) since inputs are z-scored.
  • Robust BMU/QE computations and NaN guards.

Usage (unchanged):
python som_per_trace_gridsearch_fast.py \
  --input vectors_internal.dump \
  --output som_results.dump \
  --min-side 3 --max-side 12 \
  --alpha 0.7 \
  --epochs 20 \
  --batch-size 1024 \
  --lr 0.5 --sigma auto \
  --neighbor-mode 8 \
  --max-samples-per-trace 20000 \
  --jobs 0 \
  --seed 17
"""
from __future__ import annotations
import argparse
import math
import os
import pickle
from typing import List, Optional, Sequence, Tuple

import numpy as np
import multiprocessing as mp

# -----------------------------
# I/O utils
# -----------------------------

def load_dump(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_dump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

# -----------------------------
# Metrics
# -----------------------------

def neuron_entropy(bmu_idx: np.ndarray, n_units: int) -> Tuple[float, float]:
    if bmu_idx.size == 0 or n_units <= 1:
        return 0.0, 0.0
    counts = np.bincount(bmu_idx, minlength=n_units).astype(np.float64)
    s = counts.sum()
    if s <= 0:
        return 0.0, 0.0
    p = counts / s
    p = np.clip(p, 1e-12, 1.0)
    H = float(-np.sum(p * np.log(p)))
    Hn = H / math.log(n_units)
    return H, Hn


def adjacency_consistency(bmu_coords: np.ndarray, neighbor_mode: int = 8) -> float:
    n = bmu_coords.shape[0]
    if n <= 1:
        return 1.0
    dif = np.abs(np.diff(bmu_coords, axis=0))  # (n-1, 2)
    if neighbor_mode == 4:
        d = np.sum(dif, axis=1)  # Manhattan distance
        good = (d <= 1)
    else:  # 8-neighborhood
        d = np.max(dif, axis=1)  # Chebyshev distance
        good = (d <= 1)
    return float(np.mean(good))

# -----------------------------
# Standardization
# -----------------------------

def zscore_per_trace(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Xz = (X - mu) / sd
    return Xz, mu.ravel(), sd.ravel()

# -----------------------------
# Fast SOM (batch updates, stability-fixed)
# -----------------------------
class FastSOM:
    def __init__(self, rows: int, cols: int, dim: int, lr: float, sigma: Optional[float], epochs: int, seed: int,
                 w_clip: float = 10.0):
        assert rows > 0 and cols > 0
        self.rows = int(rows)
        self.cols = int(cols)
        self.dim = int(dim)
        self.lr0 = float(lr)
        self.sigma0 = float(max(rows, cols) / 2.0) if (sigma is None) else float(sigma)
        self.epochs = int(max(1, epochs))
        self.rng = np.random.default_rng(seed)
        self.W = None  # (M, d), float32
        self.W_CLIP = float(w_clip)
        # Precompute grid geometry
        coords = np.stack(np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij'), axis=-1)
        self.coords = coords.reshape(rows * cols, 2).astype(np.int32)  # (M, 2)
        diffs = self.coords[:, None, :].astype(np.float32) - self.coords[None, :, :].astype(np.float32)
        self.grid_dist2 = np.sum(diffs * diffs, axis=-1).astype(np.float32)  # (M, M)

    def _init_weights(self, X: np.ndarray) -> None:
        n, d = X.shape
        idx = self.rng.integers(0, n, size=self.rows * self.cols)
        W = X[idx].astype(np.float32, copy=True)
        W += 1e-3 * self.rng.standard_normal(W.shape, dtype=np.float32)
        self.W = np.clip(W, -self.W_CLIP, self.W_CLIP)

    @staticmethod
    def _bmu_batch(X: np.ndarray, W: np.ndarray) -> np.ndarray:
        # Compute BMUs for a batch X against all neurons W using ||x-w||^2 trick.
        X = X.astype(np.float32, copy=False)
        W = W.astype(np.float32, copy=False)
        x_norm = np.sum(X * X, axis=1, dtype=np.float32)  # (B,)
        w_norm = np.sum(W * W, axis=1, dtype=np.float32)  # (M,)
        # dist2 = x_norm[:,None] + w_norm[None,:] - 2 X W^T
        prod = X @ W.T  # (B, M)
        dist2 = x_norm[:, None] + w_norm[None, :] - 2.0 * prod
        # Guard against tiny negative due to FP error
        dist2 = np.maximum(dist2, 0.0)
        return np.argmin(dist2, axis=1)

    def _kernel(self, sigma: float) -> np.ndarray:
        # Gaussian neighborhood kernel for current sigma: (M, M) float32
        return np.exp(-self.grid_dist2 / (2.0 * (sigma * sigma))).astype(np.float32)

    def fit_batch(self, X: np.ndarray, batch_size: int = 1024) -> None:
        if X.size == 0:
            raise ValueError('Empty data passed to SOM.fit_batch')
        X = np.clip(X.astype(np.float32, copy=False), -self.W_CLIP, self.W_CLIP)
        n, d = X.shape
        if self.W is None:
            self._init_weights(X)
        M = self.rows * self.cols
        eps = np.float32(1e-6)

        # Epoch-based schedules (safe when epochs==1)
        for e in range(self.epochs):
            lr_e = self.lr0 * math.exp(-e / max(1, self.epochs - 1))
            sigma_e = max(0.5, self.sigma0 * math.exp(-e / max(1.0, self.epochs - 1)))
            K = self._kernel(sigma_e)  # (M, M)
            # Shuffle once per epoch
            idx_all = np.arange(n)
            self.rng.shuffle(idx_all)
            # Batches
            for start in range(0, n, batch_size):
                stop = min(start + batch_size, n)
                Xb = X[idx_all[start:stop]]  # (B, d)
                if Xb.size == 0:
                    continue
                # BMUs for the batch
                b = self._bmu_batch(Xb, self.W)  # (B,)
                # Neighborhood weights for BMUs
                G = K[b, :]  # (B, M)
                sum_g = np.sum(G, axis=0, dtype=np.float32) + eps  # (M,)
                GX = (G.T @ Xb).astype(np.float32)                 # (M, d)
                W_target = GX / sum_g[:, None]                     # (M, d)
                # Stable convex-combination update
                self.W += lr_e * (W_target - self.W)
                # Clip to keep numbers in a safe dynamic range
                np.clip(self.W, -self.W_CLIP, self.W_CLIP, out=self.W)

    def transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.clip(X.astype(np.float32, copy=False), -self.W_CLIP, self.W_CLIP)
        b = self._bmu_batch(X, self.W)
        return b, self.coords[b]

    def quantization_error(self, X: np.ndarray) -> float:
        X = np.clip(X.astype(np.float32, copy=False), -self.W_CLIP, self.W_CLIP)
        x_norm = np.sum(X * X, axis=1, dtype=np.float32)
        w_norm = np.sum(self.W * self.W, axis=1, dtype=np.float32)
        prod = X @ self.W.T
        dist2 = x_norm[:, None] + w_norm[None, :] - 2.0 * prod
        dist2 = np.maximum(dist2, 0.0)
        qe = np.min(dist2, axis=1)
        return float(np.mean(qe, dtype=np.float64))

# -----------------------------
# Grid search per trace (single-process)
# -----------------------------

def parse_sizes(args) -> List[Tuple[int, int]]:
    if args.sizes:
        out: List[Tuple[int,int]] = []
        for s in args.sizes:
            if 'x' not in s:
                raise ValueError(f"Invalid --sizes entry '{s}'. Use RxC, e.g., 6x8")
            r, c = s.lower().split('x')
            out.append((int(r), int(c)))
        return out
    else:
        return [(s, s) for s in range(args.min_side, args.max_side + 1)]


def grid_search_trace(
    Xz: np.ndarray,
    sizes: Sequence[Tuple[int, int]],
    alpha: float,
    epochs: int,
    batch_size: int,
    lr: float,
    sigma: Optional[float],
    neighbor_mode: int,
    seed: int,
) -> Tuple[dict, List[dict]]:
    n, k = Xz.shape
    rng = np.random.default_rng(seed)
    candidates: List[dict] = []
    for (r, c) in sizes:
        som = FastSOM(r, c, k, lr=lr, sigma=sigma, epochs=epochs, seed=int(rng.integers(0, 2**31-1)))
        som.fit_batch(Xz, batch_size=batch_size)
        bmu_idx, bmu_coords = som.transform(Xz)
        qe = som.quantization_error(Xz)
        H, Hn = neuron_entropy(bmu_idx, r * c)
        adj = adjacency_consistency(bmu_coords, neighbor_mode=neighbor_mode)
        score = alpha * adj + (1.0 - alpha) * Hn
        # Guard against NaNs/Infs in metrics
        if not np.isfinite([qe, H, Hn, adj, score]).all():
            qe = float('inf')
            score = -float('inf')
            Hn = 0.0
            adj = 0.0
        cand = {
            'rows': r,
            'cols': c,
            'alpha': alpha,
            'adjacency_consistency': float(adj),
            'entropy': float(H),
            'entropy_norm': float(Hn),
            'combined_score': float(score),
            'quantization_error': float(qe),
            'assignments': bmu_coords.astype(int).tolist(),
            'codebook': som.W.astype(np.float64).tolist(),
            'k': int(k),
            'n_samples': int(n),
        }
        candidates.append(cand)

    candidates.sort(key=lambda c: (-c['combined_score'], -c['adjacency_consistency'], c['quantization_error'], -c['entropy_norm']))
    return candidates[0], candidates

# -----------------------------
# Worker for multiprocessing
# -----------------------------

def _process_trace(args_tuple):
    (ti, trace, sizes, alpha, epochs, batch_size, lr, sigma, neighbor_mode, max_samples, base_seed) = args_tuple
    try:
        X = np.asarray(trace, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] == 0:
            return ti, {
                'rows': 0, 'cols': 0, 'alpha': alpha,
                'adjacency_consistency': float('nan'),
                'entropy': float('nan'), 'entropy_norm': float('nan'),
                'combined_score': float('nan'), 'quantization_error': float('nan'),
                'assignments': [], 'codebook': [],
                'k': int(X.shape[1]) if X.ndim == 2 else 0,
                'n_samples': int(X.shape[0]) if X.ndim == 2 else 0,
                'trace_index': ti,
            }
        rng = np.random.default_rng(base_seed + ti)
        if max_samples > 0 and X.shape[0] > max_samples:
            idx = rng.choice(X.shape[0], size=max_samples, replace=False)
            X = X[idx]
        Xz, _, _ = zscore_per_trace(X)
        best, _ = grid_search_trace(
            Xz.astype(np.float32, copy=False),
            sizes, alpha=alpha, epochs=epochs, batch_size=batch_size, lr=lr,
            sigma=(None if isinstance(sigma, str) and sigma == 'auto' else float(sigma)),
            neighbor_mode=neighbor_mode, seed=int(rng.integers(0, 2**31-1))
        )
        best['trace_index'] = ti
        return ti, best
    except Exception as e:
        return ti, {
            'rows': -1, 'cols': -1, 'alpha': alpha,
            'adjacency_consistency': float('nan'), 'entropy': float('nan'), 'entropy_norm': float('nan'),
            'combined_score': float('nan'), 'quantization_error': float('nan'),
            'assignments': [], 'codebook': [], 'k': 0, 'n_samples': 0,
            'error': str(e), 'trace_index': ti,
        }

# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description='FAST per-trace SOM grid search (NumPy-only, stability-fixed)')
    ap.add_argument('--input', type=str, default='vectors_internal.dump', help='Path to vectors_internal.dump')
    ap.add_argument('--output', type=str, default='som_results.dump', help='Where to store per-trace SOM results')
    ap.add_argument('--min-side', type=int, default=3, help='Minimum side for square grids (ignored if --sizes is set)')
    ap.add_argument('--max-side', type=int, default=12, help='Maximum side for square grids (ignored if --sizes is set)')
    ap.add_argument('--sizes', nargs='*', help="Explicit list of grid sizes like '4x6 6x8 8x8' (overrides min/max)")
    ap.add_argument('--alpha', type=float, default=0.7, help='Weight for adjacency in combined score (0..1)')
    ap.add_argument('--epochs', type=int, default=20, help='Training epochs per candidate size')
    ap.add_argument('--batch-size', type=int, default=1024, help='Batch size for vectorized training')
    ap.add_argument('--lr', type=float, default=0.5, help='Initial learning rate')
    ap.add_argument('--sigma', type=str, default='auto', help="Initial neighborhood radius; 'auto' -> max(rows, cols)/2")
    ap.add_argument('--neighbor-mode', type=int, default=8, choices=[4,8], help='Adjacency neighborhood: 4 or 8')
    ap.add_argument('--max-samples-per-trace', type=int, default=20000, help='Subsample each trace to at most this many windows (<=0 means no cap)')
    ap.add_argument('--jobs', type=int, default=0, help='Parallel worker processes (0 -> use os.cpu_count(), 1 -> no parallelism)')
    ap.add_argument('--seed', type=int, default=17, help='Base random seed')

    args = ap.parse_args()

    data = load_dump(args.input)
    if not isinstance(data, list):
        raise ValueError('Expected a list of traces in vectors_internal.dump')

    # Determine grid sizes
    if args.sizes:
        sizes = []
        for s in args.sizes:
            r, c = s.lower().split('x')
            sizes.append((int(r), int(c)))
    else:
        sizes = [(s, s) for s in range(args.min_side, args.max_side + 1)]

    jobs = os.cpu_count() if args.jobs == 0 else max(1, args.jobs)

    tasks = [
        (ti, trace, sizes, args.alpha, args.epochs, max(1, args.batch_size), args.lr, args.sigma,
         args.neighbor_mode, max(0, args.max_samples_per_trace), args.seed)
        for ti, trace in enumerate(data)
    ]

    results: List[dict] = [None] * len(tasks)  # type: ignore

    if jobs == 1:
        for t in tasks:
            ti, res = _process_trace(t)
            results[ti] = res
    else:
        with mp.get_context('spawn').Pool(processes=jobs) as pool:
            for ti, res in pool.imap_unordered(_process_trace, tasks, chunksize=1):
                results[ti] = res

    save_dump(results, args.output)

    # Console summary
    for res in results:
        ti = res.get('trace_index', -1)
        print(f"\n--- Trace {ti} ---")
        if 'error' in res:
            print('ERROR:', res['error'])
            continue
        print(f"Chosen grid: {res['rows']}x{res['cols']}")
        print(f"  n_samples={res['n_samples']}, k={res['k']}")
        print(f"  adjacency={res['adjacency_consistency']:.4f}, entropy_norm={res['entropy_norm']:.4f}, QE={res['quantization_error']:.6f}")
        print(f"  combined_score={res['combined_score']:.4f} (alpha={res['alpha']})")

    print("\nSUMMARY:")
    print(f"OUTPUT={os.path.abspath(args.output)}")

if __name__ == '__main__':
    main()
