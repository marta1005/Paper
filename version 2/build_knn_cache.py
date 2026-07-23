#!/usr/bin/env python3
"""
build_knn_cache.py — One-time preprocessing: compute k-NN graphs per simulation.

Saves per-simulation edge index and edge attributes to cache/.
Run once before training. Supports parallel processing.

Usage:
  python build_knn_cache.py                          # all splits, k=8
  python build_knn_cache.py --split train --k 8
  python build_knn_cache.py --split test  --k 8
  python build_knn_cache.py --workers 8              # parallel over sims

Runtime estimate: ~1.5s per sim → ~10 min for all 468 sims with --workers 8.
"""
import argparse
import time
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from multiprocessing import Pool
import functools

DATA_DIR  = Path(__file__).parent / 'data'
CACHE_DIR = Path(__file__).parent / 'cache'
N_P       = 260_774


def build_sim_edges(args):
    """Build kNN edges for a single simulation. Called by worker processes."""
    idx, split, k, X_path = args
    try:
        X_mmap = np.load(str(X_path), mmap_mode='r')
        n_sims = 312 if split == 'train' else 156
        X_sim  = np.asarray(X_mmap.reshape(n_sims, N_P, 9)[idx])  # [N_P, 9]

        coords  = X_sim[:, :3].astype(np.float64)   # x, y, z
        normals = X_sim[:, 3:6].astype(np.float32)  # nx, ny, nz

        tree          = cKDTree(coords)
        dists, inds   = tree.query(coords, k=k + 1, workers=1)  # k+1: includes self
        dists, inds   = dists[:, 1:], inds[:, 1:]               # exclude self

        N   = N_P
        src = np.repeat(np.arange(N, dtype=np.int32), k)
        dst = inds.ravel().astype(np.int32)
        d   = dists.ravel().astype(np.float32)

        dx = (coords[dst, 0] - coords[src, 0]).astype(np.float32)
        dy = (coords[dst, 1] - coords[src, 1]).astype(np.float32)
        dz = (coords[dst, 2] - coords[src, 2]).astype(np.float32)
        n_dot = (normals[src] * normals[dst]).sum(axis=1).astype(np.float32)

        edge_index = np.stack([src, dst], axis=0)                      # [2, N_P*k]
        edge_attr  = np.column_stack([dx, dy, dz, d, n_dot])           # [N_P*k, 5]

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(str(CACHE_DIR / f'edges_{split}_s{idx:03d}_k{k}.npy'),    edge_index)
        np.save(str(CACHE_DIR / f'edge_attr_{split}_s{idx:03d}_k{k}.npy'), edge_attr)
        return idx, True, None
    except Exception as e:
        return idx, False, str(e)


def build_split(split, k, workers):
    n_sims   = 312 if split == 'train' else 156
    x_fname  = 'X_train.npy' if split == 'train' else 'X_test.npy'
    X_path   = DATA_DIR / x_fname

    # Find which sims still need to be built
    pending = [
        i for i in range(n_sims)
        if not (CACHE_DIR / f'edges_{split}_s{i:03d}_k{k}.npy').exists()
    ]
    if not pending:
        print(f"  [{split}] All {n_sims} sims already cached.")
        return

    print(f"  [{split}] Building kNN (k={k}) for {len(pending)}/{n_sims} sims "
          f"using {workers} workers...")
    t0   = time.time()
    args = [(i, split, k, X_path) for i in pending]

    if workers == 1:
        results = [build_sim_edges(a) for a in args]
    else:
        with Pool(workers) as pool:
            results = pool.map(build_sim_edges, args)

    ok = sum(1 for _, s, _ in results if s)
    fail = [(i, e) for i, s, e in results if not s]
    dt = time.time() - t0
    print(f"  [{split}] Done: {ok} ok, {len(fail)} failed in {dt:.0f}s "
          f"({dt/len(pending):.1f}s/sim)")
    for i, e in fail:
        print(f"    FAIL sim {i}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',   choices=['train', 'test', 'all'], default='all')
    parser.add_argument('--k',       type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    splits = ['train', 'test'] if args.split == 'all' else [args.split]
    for s in splits:
        build_split(s, args.k, args.workers)


if __name__ == '__main__':
    main()
