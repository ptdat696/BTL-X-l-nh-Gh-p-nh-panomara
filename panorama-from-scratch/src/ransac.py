"""Own RANSAC loop for homography: 4-sample DLT, consensus and refinement."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

from .homography import estimate_homography, reprojection_errors


@dataclass
class RansacResult:
    H: np.ndarray
    inliers: np.ndarray
    mean_error: float
    iterations: int


def _sample_is_degenerate(points):
    # Reject samples containing nearly-collinear triples.
    for a,b,c in ((0,1,2),(0,1,3),(0,2,3),(1,2,3)):
        p,q,r = points[[a,b,c]]
        area2 = abs((q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0]))
        scale2 = np.max(np.ptp(points,axis=0))**2
        if area2 < max(1e-8, 0.001*scale2):
            return True
    return False


def ransac_homography(src, dst, threshold=3.5, max_iterations=3000,
                      confidence=0.995, seed=42) -> RansacResult:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or len(src) < 4:
        raise ValueError('Need >=4 matched source/destination points')
    rng = np.random.default_rng(seed)
    n = len(src)
    best_count = 0
    best_mean = float('inf')
    best_mask = None
    best_H = None
    dynamic_limit = max_iterations
    trials = 0
    for trial in range(max_iterations):
        if trial >= dynamic_limit:
            break
        trials += 1
        sample = rng.choice(n,4,replace=False)
        if _sample_is_degenerate(src[sample]) or _sample_is_degenerate(dst[sample]):
            continue
        try:
            H = estimate_homography(src[sample],dst[sample])
        except (ValueError, np.linalg.LinAlgError):
            continue
        errors = reprojection_errors(H,src,dst)
        inliers = np.isfinite(errors) & (errors < threshold)
        count = int(inliers.sum())
        mean = float(np.mean(errors[inliers])) if count else float('inf')
        if count > best_count or (count == best_count and mean < best_mean):
            best_count, best_mean = count, mean
            best_mask, best_H = inliers, H
            if count >= 4:
                p_good = count/n
                p_failure = max(1e-12, 1.0-p_good**4)
                if p_failure < 1:
                    dynamic_limit = min(dynamic_limit, max(1,math.ceil(
                        math.log(1-confidence)/math.log(p_failure))))
    if best_count < 4 or best_mask is None:
        raise ValueError('RANSAC failed: fewer than 4 geometrically consistent matches')
    # Re-estimate H with all inliers, iteratively updating the consensus.
    for _ in range(3):
        try:
            refined = estimate_homography(src[best_mask], dst[best_mask])
        except (ValueError,np.linalg.LinAlgError):
            break
        errs = reprojection_errors(refined,src,dst)
        mask = np.isfinite(errs) & (errs < threshold)
        if mask.sum() < 4:
            break
        best_H = refined
        if np.array_equal(mask,best_mask):
            break
        best_mask = mask
    errs = reprojection_errors(best_H,src,dst)
    return RansacResult(best_H,best_mask,
                        float(np.mean(errs[best_mask])),trials)
