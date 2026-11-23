"""Descriptor matching: Euclidean nearest neighbors, Lowe ratio, mutual check."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Match:
    query_idx: int
    train_idx: int
    distance: float


def match_features(desc1: np.ndarray, desc2: np.ndarray,
                   ratio: float = 0.78, mutual: bool = True,
                   block_size: int = 512) -> list[Match]:
    if len(desc1) < 1 or len(desc2) < 2:
        return []
    if not (0 < ratio < 1):
        raise ValueError('ratio must be in (0,1)')
    a = np.asarray(desc1, dtype=np.float32)
    b = np.asarray(desc2, dtype=np.float32)
    if a.shape[1] != b.shape[1]:
        raise ValueError('Descriptor dimensions do not match')
    # L2^2 = ||a||^2 + ||b||^2 - 2*a.b, bounded memory per block.
    b_sq = np.sum(b*b, axis=1)
    candidates = []
    for start in range(0, len(a), block_size):
        block = a[start:start+block_size]
        d2 = np.maximum(np.sum(block*block, axis=1)[:,None]
                        + b_sq[None,:] - 2*block@b.T, 0.)
        top2 = np.argpartition(d2, 1, axis=1)[:, :2]
        chosen = np.take_along_axis(d2, top2, axis=1)
        order = np.argsort(chosen, axis=1)
        top2 = np.take_along_axis(top2, order, axis=1)
        chosen = np.take_along_axis(chosen, order, axis=1)
        ok = (chosen[:,0] < ratio**2 * np.maximum(chosen[:,1], 1e-12))
        for local in np.nonzero(ok)[0]:
            candidates.append(Match(start+int(local), int(top2[local,0]),
                                    float(np.sqrt(chosen[local,0]))))
    if not mutual or not candidates:
        return sorted(candidates, key=lambda x:x.distance)
    # Reverse nearest neighbors with chunking, no ratio rule for this check.
    best_dist = np.full(len(b), np.inf)
    reverse_nearest = np.full(len(b), -1, dtype=np.int32)
    a_sq = np.sum(a*a, axis=1)
    for start in range(0, len(b), block_size):
        block = b[start:start+block_size]
        d2 = np.maximum(np.sum(block*block,axis=1)[:,None]
                        + a_sq[None,:] - 2*block@a.T, 0.)
        idx = np.argmin(d2, axis=1)
        reverse_nearest[start:start+len(block)] = idx
        best_dist[start:start+len(block)] = d2[np.arange(len(block)),idx]
    return sorted([m for m in candidates if reverse_nearest[m.train_idx] == m.query_idx],
                  key=lambda m:m.distance)
