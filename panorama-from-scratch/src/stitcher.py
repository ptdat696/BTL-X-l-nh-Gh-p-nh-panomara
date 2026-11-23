"""End-to-end image stitching with solely custom classical algorithms."""
from __future__ import annotations
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from .sift import SIFT, Keypoint
from .matcher import Match, match_features
from .ransac import ransac_homography
from .warping import composite_images, crop_to_valid


@dataclass
class StitchSettings:
    max_features: int = 1600
    contrast_threshold: float = 0.04
    ratio: float = 0.78
    mutual: bool = True
    ransac_threshold: float = 3.5
    ransac_iterations: int = 2500
    double_image: bool = True
    feather: float = 35.
    max_megapixels: float = 35.
    crop: bool = True
    seed: int = 42


@dataclass
class PairResult:
    source_idx: int
    dest_idx: int
    H: np.ndarray
    matches: list[Match]
    inliers: np.ndarray
    reprojection_error: float
    trials: int


@dataclass
class StitchResult:
    panorama: np.ndarray
    pairs: list[PairResult]
    keypoints: list[list[Keypoint]]
    descriptors: list[np.ndarray]
    transforms: list[np.ndarray]
    timings: dict[str,float]
    valid_mask: np.ndarray


def stitch(images: list[np.ndarray], settings: StitchSettings | None=None) -> StitchResult:
    """Stitch 2..N images supplied in physical left-to-right order.

    Consecutive frames should overlap. Reference frame = middle image.
    """
    settings = settings or StitchSettings()
    if not 2 <= len(images) <= 12:
        raise ValueError('Provide 2 to 12 overlapping images in left-to-right order')
    for i,im in enumerate(images):
        if im is None or im.ndim!=3 or im.shape[2]!=3 or min(im.shape[:2])<32:
            raise ValueError(f'Image {i} must be a 3-channel BGR image at least 32px')
    t0 = perf_counter()
    sift = SIFT(contrast_threshold=settings.contrast_threshold,
                max_features=settings.max_features,double_image=settings.double_image)
    keypoints, desc = [], []
    for im in images:
        k,d = sift.detect_and_compute(im)
        keypoints.append(k)
        desc.append(d)
    t1 = perf_counter()
    pairs = []
    for idx in range(len(images)-1):
        matches = match_features(desc[idx],desc[idx+1],settings.ratio,settings.mutual)
        if len(matches)<4:
            raise ValueError(f'Only {len(matches)} reliable matches between images '
                             f'{idx+1} and {idx+2} (need >=4). '
                             'Increase overlapping content or adjust --ratio/--contrast.')
        src = np.array([[keypoints[idx][m.query_idx].x,keypoints[idx][m.query_idx].y]
                        for m in matches],dtype=np.float64)
        dst = np.array([[keypoints[idx+1][m.train_idx].x,keypoints[idx+1][m.train_idx].y]
                        for m in matches],dtype=np.float64)
        try:
            res = ransac_homography(src,dst,threshold=settings.ransac_threshold,
                                    max_iterations=settings.ransac_iterations,
                                    seed=settings.seed+idx)
        except ValueError as exc:
            raise ValueError(f'Could not align images {idx+1} and {idx+2}: {exc}') from exc
        if int(res.inliers.sum()) < 6:
            raise ValueError(f'Insufficient geometric support between images {idx+1} '
                             f'and {idx+2}: {int(res.inliers.sum())} inliers')
        pairs.append(PairResult(idx,idx+1,res.H,matches,res.inliers,res.mean_error,res.iterations))
    t2 = perf_counter()
    ref = len(images)//2
    transforms: list[np.ndarray] = [np.eye(3) for _ in images]
    for i in range(ref-1,-1,-1):
        transforms[i] = transforms[i+1] @ pairs[i].H
        transforms[i] /= transforms[i][2,2]
    for i in range(ref+1,len(images)):
        transforms[i] = transforms[i-1] @ np.linalg.inv(pairs[i-1].H)
        transforms[i] /= transforms[i][2,2]
    pano,mask = composite_images(images,transforms,feather=settings.feather,
                                 max_megapixels=settings.max_megapixels)
    if settings.crop:
        pano,mask = crop_to_valid(pano,mask)
    t3 = perf_counter()
    return StitchResult(pano,pairs,keypoints,desc,transforms,
                        {'sift':t1-t0,'matching_ransac':t2-t1,
                         'warping_blending':t3-t2,'total':t3-t0},mask)
