"""Inverse homography warping + in-source feather confidence, no cv2.warpPerspective."""
from __future__ import annotations
import numpy as np

from .geometry import project as transform_points
from .image_utils import bilinear_sample


def transformed_bounds(image: np.ndarray, H: np.ndarray) -> np.ndarray:
    h,w = image.shape[:2]
    corners = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float64)
    return transform_points(H,corners)


def panorama_canvas(images, transforms, max_megapixels: float = 35.0):
    all_corners = np.concatenate([transformed_bounds(im,H)
                                  for im,H in zip(images,transforms)],axis=0)
    if not np.all(np.isfinite(all_corners)):
        raise ValueError('Warped image corners are not finite')
    lower = np.floor(all_corners.min(axis=0))
    upper = np.ceil(all_corners.max(axis=0))
    width, height = (upper-lower+1).astype(int)
    if width <= 0 or height <= 0 or width*height > max_megapixels*1e6:
        raise ValueError(f'Unsafe panorama canvas {width}x{height}. '
                         'Check image overlaps and estimated homographies.')
    offset = np.array([[1,0,-lower[0]], [0,1,-lower[1]], [0,0,1]],dtype=np.float64)
    return int(width),int(height),[offset@H for H in transforms]


def composite_images(images, transforms, feather: float = 35.,
                     max_megapixels=35., row_chunk=160):
    """Warp into shared canvas in row chunks and blend overlap with weights.

    A pixel inside one source is sampled using *inverse* H, with NumPy
    bilinear interpolation. In overlap, output = sum(weight*color)/sum(weight).
    Returns (panorama BGR uint8, union-valid mask).
    """
    if len(images) != len(transforms) or not images:
        raise ValueError('Number of images and transforms must match and be nonzero')
    width,height,Hs = panorama_canvas(images,transforms,max_megapixels)
    sum_color = np.zeros((height,width,3),dtype=np.float32)
    sum_weights = np.zeros((height,width),dtype=np.float32)
    for image,H in zip(images,Hs):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Expected 3-channel BGR images')
        h,w = image.shape[:2]
        try:
            inv = np.linalg.inv(H)
        except np.linalg.LinAlgError as e:
            raise ValueError('Cannot invert image homography') from e
        corners = transformed_bounds(image,H)
        min_x = max(0,int(np.floor(np.min(corners[:,0]))))
        max_x = min(width,int(np.ceil(np.max(corners[:,0])))+1)
        min_y = max(0,int(np.floor(np.min(corners[:,1]))))
        max_y = min(height,int(np.ceil(np.max(corners[:,1])))+1)
        if max_x<=min_x or max_y<=min_y:
            continue
        for y0 in range(min_y,max_y,row_chunk):
            y1 = min(y0+row_chunk,max_y)
            yy,xx = np.mgrid[y0:y1,min_x:max_x]
            denom = inv[2,0]*xx + inv[2,1]*yy + inv[2,2]
            safe = np.abs(denom) > 1e-10
            with np.errstate(divide='ignore',invalid='ignore'):
                xs = np.where(safe,(inv[0,0]*xx+inv[0,1]*yy+inv[0,2])/denom,-1)
                ys = np.where(safe,(inv[1,0]*xx+inv[1,1]*yy+inv[1,2])/denom,-1)
            valid = safe & (xs >= 0) & (xs <= w-1) & (ys >= 0) & (ys <= h-1)
            if not np.any(valid):
                continue
            sampled = bilinear_sample(image, np.where(valid,xs,0), np.where(valid,ys,0))
            edge_distance = np.minimum.reduce((xs,ys,w-1-xs,h-1-ys))
            if feather <= 0:
                weights = valid.astype(np.float32)
            else:
                weights = np.where(valid, np.clip(edge_distance+1,1,feather),0).astype(np.float32)
            sum_color[y0:y1,min_x:max_x] += sampled*weights[...,None]
            sum_weights[y0:y1,min_x:max_x] += weights
    mask = sum_weights > 0
    output = np.zeros_like(sum_color,dtype=np.uint8)
    if mask.any():
        output[mask] = np.clip(sum_color[mask]/sum_weights[mask,None],0,255).astype(np.uint8)
    return output, mask


def crop_to_valid(image,mask):
    ys,xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError('No valid warped pixels in panorama')
    x0,x1,y0,y1 = int(xs.min()),int(xs.max()+1),int(ys.min()),int(ys.max()+1)
    return image[y0:y1,x0:x1],mask[y0:y1,x0:x1]
