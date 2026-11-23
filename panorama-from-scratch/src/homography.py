"""DLT homography estimation with Hartley coordinate normalization."""
from __future__ import annotations
import numpy as np


def _normalize(points):
    center = points.mean(axis=0)
    shifted = points-center
    dist = np.mean(np.linalg.norm(shifted,axis=1))
    if dist < 1e-9:
        raise ValueError('Degenerate point configuration')
    s = np.sqrt(2.0) / dist
    T = np.array([[s,0,-s*center[0]], [0,s,-s*center[1]], [0,0,1]], dtype=np.float64)
    p = (points-center)*s
    return p, T


def estimate_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Find H such that dst ~ H @ src; requires >=4 non-collinear pairs."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.ndim != 2 or dst.shape != src.shape or src.shape[1] != 2 or len(src)<4:
        raise ValueError('src and dst must have identical (N,2) shapes, N>=4')
    sp, Ts = _normalize(src)
    dp, Td = _normalize(dst)
    n = len(src)
    A = np.zeros((2*n, 9), dtype=np.float64)
    x, y, u, v = sp[:,0], sp[:,1], dp[:,0], dp[:,1]
    A[::2,0:3] = np.column_stack((-x,-y,-np.ones(n)))
    A[::2,6:9] = np.column_stack((u*x,u*y,u))
    A[1::2,3:6] = np.column_stack((-x,-y,-np.ones(n)))
    A[1::2,6:9] = np.column_stack((v*x,v*y,v))
    # full_matrices=True yields nullspace even for the 8x9 minimal problem.
    _, svals, vt = np.linalg.svd(A, full_matrices=True)
    if svals[0] < 1e-12 or (n > 4 and svals[-2] / svals[0] < 1e-12):
        # When over-determined, poorly conditioned point sets are unusable.
        raise ValueError('Degenerate homography fit')
    Hnorm = vt[-1].reshape(3,3)
    H = np.linalg.inv(Td) @ Hnorm @ Ts
    if abs(H[2,2]) > 1e-10:
        H /= H[2,2]
    else:
        H /= max(np.linalg.norm(H),1e-12)
    if not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-12:
        raise ValueError('Singular homography')
    return H


def transform_points(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack((points,np.ones(len(points))))
    q = (H @ homogeneous.T).T
    with np.errstate(divide='ignore',invalid='ignore'):
        return q[:,:2] / q[:,2,None]


def reprojection_errors(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    projected = transform_points(H, src)
    return np.linalg.norm(projected - dst, axis=1)
