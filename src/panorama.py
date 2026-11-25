"""Panorama pipeline: SIFT -> matching -> RANSAC -> ordering -> warping."""

from dataclasses import dataclass
from time import perf_counter
import numpy as np

from .sift import SIFT
from .image_utils import resize_bilinear
from .geometry import match_features, ransac
from .warping import composite_images, crop_to_valid


@dataclass
class Result:
    panorama: np.ndarray
    order: list[int]             # Indices of original input images, left to right
    keypoint_counts: list[int]
    inlier_counts: list[int]
    seconds: float


def _pair(i, j, keypoints, descriptors):
    """Return (H i->j, matches, inliers), or None if images do not overlap."""
    matches = match_features(descriptors[i], descriptors[j])
    if len(matches) < 6:
        return None
    src = np.array([[keypoints[i][a].x, keypoints[i][a].y] for a, b in matches])
    dst = np.array([[keypoints[j][b].x, keypoints[j][b].y] for a, b in matches])
    try:
        H, inliers = ransac(src, dst, seed=42 + i * 31 + j)
    except (ValueError, np.linalg.LinAlgError):
        return None
    return H, matches, inliers


def _find_order(images, keypoints, pairs):
    """Infer horizontal positions from shifts of matched keypoints (least squares)."""
    n = len(images)
    A, b = [], []
    for (i, j), (_, matches, inliers) in pairs.items():
        good = [match for match, ok in zip(matches, inliers) if ok]
        shift = np.median([keypoints[i][a].x / images[i].shape[1] -
                           keypoints[j][c].x / images[j].shape[1]
                           for a, c in good])
        weight = np.sqrt(len(good))
        row = np.zeros(n)
        row[i], row[j] = -weight, weight
        A.append(row[1:])        # Fix image 0 at position 0.
        b.append(weight * shift)
    if not A or np.linalg.matrix_rank(A) < n - 1:
        raise ValueError('Không tìm được đủ vùng chồng lấp giữa các ảnh.')
    positions = np.r_[0, np.linalg.lstsq(A, b, rcond=None)[0]]
    order = sorted(range(n), key=lambda i: positions[i])
    if any((min(i, j), max(i, j)) not in pairs
           for i, j in zip(order[:-1], order[1:])):
        raise ValueError('Không xác định được thứ tự ảnh. Hãy tăng vùng chồng lấp.')
    return order


def stitch(images, max_width=700, max_features=900):
    """Stitch 2-4 BGR images. All core image-processing steps are self-coded."""
    if not 2 <= len(images) <= 4:
        raise ValueError('Hãy chọn từ 2 đến 4 ảnh.')
    start = perf_counter()
    small = []
    for image in images:
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Đầu vào phải là ảnh màu.')
        h, w = image.shape[:2]
        if min(h, w) < 32:
            raise ValueError('Ảnh quá nhỏ.')
        if w > max_width:
            image = resize_bilinear(image, max_width, round(h * max_width / w))
        small.append(image)

    sift = SIFT(max_features=max_features, double_image=False)
    keypoints, descriptors = [], []
    for image in small:
        kp, desc = sift.detect_and_compute(image)
        keypoints.append(kp)
        descriptors.append(desc)

    pairs = {}
    for i in range(len(small) - 1):
        for j in range(i + 1, len(small)):
            pair = _pair(i, j, keypoints, descriptors)
            if pair is not None:
                pairs[(i, j)] = pair

    order = _find_order(small, keypoints, pairs) 
    chosen = []
    for i, j in zip(order[:-1], order[1:]):
        pair = pairs.get((min(i, j), max(i, j)))
        if pair is None:
            raise ValueError('Hai ảnh liền kề không có đủ vùng chồng lấp.')
        H, matches, inliers = pair
        chosen.append((H if i < j else np.linalg.inv(H), int(inliers.sum())))

    # Convert adjacent homographies to the coordinate system of the middle image.
    mid = len(order) // 2
    transforms = [np.eye(3) for _ in order]
    for i in range(mid - 1, -1, -1):
        transforms[i] = transforms[i + 1] @ chosen[i][0]
    for i in range(mid + 1, len(order)):
        transforms[i] = transforms[i - 1] @ np.linalg.inv(chosen[i - 1][0])

    panorama, mask = composite_images([small[i] for i in order], transforms,
                                     feather=35, max_megapixels=25)
    panorama, _ = crop_to_valid(panorama, mask)
    return Result(panorama, order, [len(keypoints[i]) for i in order],
                  [count for H, count in chosen], perf_counter() - start)
