#!/usr/bin/env python3
"""
Train ONE SOM on all traces and assign each point to a neuron (per trace).
==========================================================================

Input  : vectors_internal.dump  -> List[Trace], Trace -> List[WindowVector(float)]
Output : som_global_assignments.dump -> List[List[int]] with shape [n_traces][len(trace_i)]
         where each inner list contains BMU **IDs** (id = row*cols + col) for that trace.

Features
--------
- Supports traces with **different lengths** and **variable dimensionality** by
  global z-scoring over present dims and zero-padding to the max dimension.
- Uses the same **stable, fast batch SOM** used previously (convex-combination update).
- Saves a `som_global_meta.dump` file with grid info, codebook and preprocessing stats.

Example
-------
python som_global_assign.py \
  --input vectors_internal.dump \
  --output som_global_assignments.dump \
  --meta-output som_global_meta.dump \
  --rows 12 --cols 12 \
  --epochs 30 --batch-size 2048 --lr 0.4 --sigma auto \
  --max-samples-per-trace 50000 --seed 17

The assignments file looks like: [[0, 0, 1, 1, 2, ...], [5, 5, 6, ...], ...]
"""
from __future__ import annotations
import argparse
import os
import pickle
import math
from typing import List, Tuple, Optional

import numpy as np

# -----------------------------
# I/O
# -----------------------------

def load_dump(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_dump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

# -----------------------------
# Global stats over variable-dim traces
# -----------------------------

def infer_kmax(traces: List[List[List[float]]]) -> int:
    kmax = 0
    for t in traces:
        if len(t) == 0:
            continue
        kmax = max(kmax, len(t[0]))
    return kmax


def global_mean_std(traces: List[List[List[float]]], kmax: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute global mean/std per dimension over all traces, only over dimensions
    actually present in each trace (no padding)."""
    sums = np.zeros(kmax, dtype=np.float64)
    sumsqs = np.zeros(kmax, dtype=np.float64)
    counts = np.zeros(kmax, dtype=np.int64)

    for t in traces:
        if not t:
            continue
        X = np.asarray(t, dtype=np.float64)
        d = X.shape[1]
        sums[:d] += X.sum(axis=0)
        sumsqs[:d] += np.sum(X * X, axis=0)
        counts[:d] += X.shape[0]

    mu = np.zeros(kmax, dtype=np.float64)
    sd = np.ones(kmax, dtype=np.float64)
    mask = counts > 0
    mu[mask] = sums[mask] / counts[mask]
    var = np.maximum(0.0, (sumsqs[mask] / counts[mask]) - mu[mask] ** 2)
    sd[mask] = np.sqrt(np.maximum(var, 1e-12))
    return mu, sd


def standardize_and_pad(
    traces: List[List[List[float]]], mu: np.ndarray, sd: np.ndarray, kmax: int
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for t in traces:
        if not t:
            out.append(np.empty((0, kmax), dtype=np.float32))
            continue
        X = np.asarray(t, dtype=np.float64)
        d = X.shape[1]
        Xz = (X - mu[:d]) / sd[:d]
        # pad to kmax with zeros (represents standardized mean for missing dims)
        if d < kmax:
            pad = np.zeros((Xz.shape[0], kmax - d), dtype=np.float64)
            Xz = np.concatenate([Xz, pad], axis=1)
        out.append(Xz.astype(np.float32))
    return out

# -----------------------------
# Stable fast SOM (batch)
# -----------------------------
class FastSOM:
    def __init__(self, rows: int, cols: int, dim: int, lr: float, sigma: Optional[float], epochs: int, seed: int,
                 w_clip: float = 10.0):
        self.rows = int(rows)
        self.cols = int(cols)
        self.dim = int(dim)
        self.lr0 = float(lr)
        self.sigma0 = float(max(rows, cols) / 2.0) if (sigma is None) else float(sigma)
        self.epochs = int(max(1, epochs))
        self.rng = np.random.default_rng(seed)
        self.W = None  # (M, d) float32
        self.W_CLIP = float(w_clip)
        coords = np.stack(np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij'), axis=-1)
        self.coords = coords.reshape(rows * cols, 2).astype(np.int32)
        diffs = self.coords[:, None, :].astype(np.float32) - self.coords[None, :, :].astype(np.float32)
        self.grid_dist2 = np.sum(diffs * diffs, axis=-1).astype(np.float32)

    def _init_weights(self, X: np.ndarray) -> None:
        n, d = X.shape
        idx = self.rng.integers(0, n, size=self.rows * self.cols)
        W = X[idx].astype(np.float32, copy=True)
        W += 1e-3 * self.rng.standard_normal(W.shape, dtype=np.float32)
        self.W = np.clip(W, -self.W_CLIP, self.W_CLIP)

    @staticmethod
    def _bmu_batch(X: np.ndarray, W: np.ndarray) -> np.ndarray:
        X = X.astype(np.float32, copy=False)
        W = W.astype(np.float32, copy=False)
        x_norm = np.sum(X * X, axis=1, dtype=np.float32)
        w_norm = np.sum(W * W, axis=1, dtype=np.float32)
        prod = X @ W.T
        dist2 = x_norm[:, None] + w_norm[None, :] - 2.0 * prod
        dist2 = np.maximum(dist2, 0.0)
        return np.argmin(dist2, axis=1)

    def _kernel(self, sigma: float) -> np.ndarray:
        return np.exp(-self.grid_dist2 / (2.0 * (sigma * sigma))).astype(np.float32)

    def fit(self, X: np.ndarray, batch_size: int = 2048) -> None:
        if X.size == 0:
            raise ValueError('Empty data passed to SOM.fit')
        X = np.clip(X.astype(np.float32, copy=False), -self.W_CLIP, self.W_CLIP)
        n, d = X.shape
        if self.W is None:
            self._init_weights(X)
        eps = np.float32(1e-6)
        for e in range(self.epochs):
            lr_e = self.lr0 * math.exp(-e / max(1, self.epochs - 1))
            sigma_e = max(0.5, self.sigma0 * math.exp(-e / max(1.0, self.epochs - 1)))
            K = self._kernel(sigma_e)
            idx = np.arange(n)
            self.rng.shuffle(idx)
            for s in range(0, n, batch_size):
                Xb = X[idx[s: s + batch_size]]
                if Xb.size == 0:
                    continue
                b = self._bmu_batch(Xb, self.W)
                G = K[b, :]
                sum_g = np.sum(G, axis=0, dtype=np.float32) + eps
                GX = (G.T @ Xb).astype(np.float32)
                W_target = GX / sum_g[:, None]
                self.W += lr_e * (W_target - self.W)
                np.clip(self.W, -self.W_CLIP, self.W_CLIP, out=self.W)

    def transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.clip(X.astype(np.float32, copy=False), -self.W_CLIP, self.W_CLIP)
        b = self._bmu_batch(X, self.W)
        return b, self.coords[b]

# -----------------------------
# Training + assignment
# -----------------------------

def bmu_id_from_coords(coords: np.ndarray, rows: int, cols: int) -> np.ndarray:
    # id = r * cols + c
    return (coords[:, 0] * cols + coords[:, 1]).astype(np.int64)


def main():
    p = argparse.ArgumentParser(description='Train ONE SOM on all traces and assign BMU ids per trace')
    p.add_argument('--input', type=str, default='vectors_internal.dump', help='Path to vectors_internal.dump')
    p.add_argument('--output', type=str, default='som_global_assignments.dump', help='Where to store per-trace BMU id sequences')
    p.add_argument('--meta-output', type=str, default='som_global_meta.dump', help='Optional: where to store SOM codebook & preprocessing info')
    p.add_argument('--rows', type=int, default=3, help='SOM rows')
    p.add_argument('--cols', type=int, default=3, help='SOM cols')
    p.add_argument('--epochs', type=int, default=30, help='Training epochs')
    p.add_argument('--batch-size', type=int, default=2048, help='Batch size for SOM training')
    p.add_argument('--lr', type=float, default=0.4, help='Initial learning rate')
    p.add_argument('--sigma', type=str, default='auto', help="Initial neighborhood radius; 'auto' -> max(rows, cols)/2")
    p.add_argument('--max-samples-per-trace', type=int, default=100000, help='Cap samples per trace for training (<=0 means no cap)')
    p.add_argument('--seed', type=int, default=17, help='Random seed')
    args = p.parse_args()

    data = load_dump(args.input)
    if not isinstance(data, list):
        raise ValueError('Expected a list of traces in vectors_internal.dump')

    # Determine common dimensionality and global z-score stats
    kmax = infer_kmax(data)
    if kmax == 0:
        raise ValueError('All traces are empty.')
    mu, sd = global_mean_std(data, kmax)

    # Standardize (per global mu/sd) and pad
    traces_std = standardize_and_pad(data, mu, sd, kmax)

    # Build training matrix (optionally subsample per trace)
    rng = np.random.default_rng(args.seed)
    X_list = []
    for X in traces_std:
        if args.max_samples_per_trace <= 0 or X.shape[0] <= args.max_samples_per_trace:
            X_list.append(X)
        else:
            idx = rng.choice(X.shape[0], size=args.max_samples_per_trace, replace=False)
            X_list.append(X[idx])
    X_train = np.vstack([x for x in X_list if x.size > 0]).astype(np.float32)

    # Train SOM
    som = FastSOM(args.rows, args.cols, kmax,
                  lr=args.lr,
                  sigma=(None if args.sigma == 'auto' else float(args.sigma)),
                  epochs=args.epochs,
                  seed=args.seed)
    som.fit(X_train, batch_size=args.batch_size)

    # Assign BMUs per trace (full, not subsampled)
    assignments: List[List[int]] = []
    for X in traces_std:
        if X.size == 0:
            assignments.append([])
            continue
        _, coords = som.transform(X)
        ids = bmu_id_from_coords(coords, args.rows, args.cols)
        assignments.append(ids.tolist())

    save_dump(assignments, args.output)

    # Optional meta
    meta = {
        'rows': args.rows,
        'cols': args.cols,
        'k': int(kmax),
        'mu': mu.astype(float).tolist(),
        'sd': sd.astype(float).tolist(),
        'codebook': som.W.astype(np.float64).tolist(),
        'coord_map': [(int(r), int(c)) for r in range(args.rows) for c in range(args.cols)],
        'bmu_id_formula': 'id = row * cols + col'
    }
    save_dump(meta, args.meta_output)

    print('Saved assignments to:', os.path.abspath(args.output))
    print('Saved meta to       :', os.path.abspath(args.meta_output))

if __name__ == '__main__':
    main()
