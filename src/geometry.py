"""Feature matching, DLT homography and RANSAC. No OpenCV algorithms."""

import math
import numpy as np


def match_features(desc1, desc2, ratio=0.78):
    """Return (index_in_image_1, index_in_image_2) after Lowe ratio + mutual check."""
    if len(desc1) == 0 or len(desc2) < 2:
        return []
    a, b = desc1.astype(np.float32), desc2.astype(np.float32)
    # Squared Euclidean distance between every pair of 128-D descriptors.
    d = np.maximum((a * a).sum(1)[:, None] +
                   (b * b).sum(1)[None, :] - 2 * a @ b.T, 0)
    two = np.argpartition(d, 1, axis=1)[:, :2]
    two = np.take_along_axis(two, np.argsort(np.take_along_axis(d, two, 1), 1), 1)
    reverse = np.argmin(d, axis=0)
    return [(i, int(j)) for i, (j, k) in enumerate(two)
            if d[i, j] < ratio ** 2 * max(float(d[i, k]), 1e-12)
            and reverse[j] == i]


def _normalize(points):
    center = points.mean(axis=0)
    distance = np.linalg.norm(points - center, axis=1).mean()
    if distance < 1e-10:
        raise ValueError('Các điểm trùng nhau, không tính được Homography.')
    s = np.sqrt(2) / distance
    T = np.array([[s, 0, -s * center[0]],
                  [0, s, -s * center[1]], [0, 0, 1]])
    return (points - center) * s, T


def homography(src, dst):
    """Normalized DLT: find H such that destination ~ H @ source."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or len(src) < 4:
        raise ValueError('Cần ít nhất 4 cặp điểm tương ứng.')
    p, T1 = _normalize(src)
    q, T2 = _normalize(dst)
    x, y = p.T
    u, v = q.T
    A = np.zeros((2 * len(src), 9))
    A[::2, :3] = np.column_stack((-x, -y, -np.ones(len(src))))
    A[::2, 6:] = np.column_stack((u * x, u * y, u))
    A[1::2, 3:6] = np.column_stack((-x, -y, -np.ones(len(src))))
    A[1::2, 6:] = np.column_stack((v * x, v * y, v))
    _, _, vt = np.linalg.svd(A, full_matrices=True)
    H = np.linalg.inv(T2) @ vt[-1].reshape(3, 3) @ T1
    if abs(H[2, 2]) < 1e-10:
        raise ValueError('Homography không hợp lệ.')
    H /= H[2, 2]
    if not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-12:
        raise ValueError('Homography bị suy biến.')
    return H


def project(H, points):
    points = np.asarray(points, float)
    p = np.column_stack((points, np.ones(len(points)))) @ H.T
    with np.errstate(divide='ignore', invalid='ignore'):
        return p[:, :2] / p[:, 2, None]


def _noncollinear(points):
    # No three of the four selected points should be approximately collinear.
    size = max(1, float(np.ptp(points, axis=0).max()) ** 2)
    for a, b, c in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
        p, q, r = points[[a, b, c]]
        area = abs((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
        if area < 0.001 * size:
            return False
    return True


def ransac(src, dst, threshold=3.5, max_trials=1200, seed=42):
    """Sample 4 point pairs, count inliers, and refit H from the best group."""
    if len(src) < 4:
        raise ValueError('Không đủ 4 điểm cho RANSAC.')
    rng = np.random.default_rng(seed)
    best = np.zeros(len(src), bool)
    best_H = None
    limit = max_trials
    for trial in range(max_trials):
        if trial >= limit:
            break
        ids = rng.choice(len(src), 4, replace=False)
        if not (_noncollinear(src[ids]) and _noncollinear(dst[ids])):
            continue
        try:
            H = homography(src[ids], dst[ids])
            error = np.linalg.norm(project(H, src) - dst, axis=1)
            mask = np.isfinite(error) & (error < threshold)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if mask.sum() > best.sum():
            best, best_H = mask, H
            # Stop early when the probability of a better sample is very low.
            p = (mask.sum() / len(src)) ** 4
            if p >= 1:
                limit = 1
            elif p > 0:
                limit = min(limit, max(1, math.ceil(math.log(0.005) / math.log(1 - p))))
    if best.sum() < 6:
        raise ValueError('Không đủ điểm khớp chính xác để ghép ảnh.')
    for _ in range(2):
        H = homography(src[best], dst[best])
        error = np.linalg.norm(project(H, src) - dst, axis=1)
        mask = np.isfinite(error) & (error < threshold)
        if mask.sum() < 6:
            break
        best_H = H
        if np.array_equal(best, mask):
            break
        best = mask
    return best_H, best
