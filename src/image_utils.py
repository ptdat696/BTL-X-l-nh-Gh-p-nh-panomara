"""Basic numerical image operations implemented directly from arrays.

OpenCV is intentionally NOT imported here: Gaussian blur, resizing,
interpolation and grayscale conversion are implemented with NumPy arithmetic.
"""
from __future__ import annotations

import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    """Return grayscale float32 image in [0,1], from uint8 BGR input."""
    if image.ndim == 2:
        gray = image.astype(np.float32)
    elif image.ndim == 3 and image.shape[2] >= 3:
        bgr = image[:, :, :3].astype(np.float32)
        gray = 0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1] + 0.299 * bgr[:, :, 2]
    else:
        raise ValueError('Expected gray or BGR image')
    if image.dtype == np.uint8:
        gray /= 255.0
    return gray


def gaussian_kernel_1d(sigma: float) -> np.ndarray:
    if sigma <= 1e-8:
        return np.array([1.0], dtype=np.float32)
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Own separable Gaussian convolution with reflect-edge padding."""
    kernel = gaussian_kernel_1d(sigma)
    radius = len(kernel) // 2
    if not radius:
        return image.astype(np.float32, copy=True)
    image = image.astype(np.float32, copy=False)
    padded = np.pad(image, ((0, 0), (radius, radius)), mode='reflect')
    horizontal = np.zeros_like(image)
    for i, coeff in enumerate(kernel):
        horizontal += coeff * padded[:, i:i+image.shape[1]]
    padded = np.pad(horizontal, ((radius, radius), (0, 0)), mode='reflect')
    out = np.zeros_like(image)
    for i, coeff in enumerate(kernel):
        out += coeff * padded[i:i+image.shape[0], :]
    return out


def resize_bilinear(image: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
    """Own pixel-center bilinear resize; works for 2D/3D images."""
    if new_width <= 0 or new_height <= 0:
        raise ValueError('Target dimensions must be positive')
    h, w = image.shape[:2]
    if (h, w) == (new_height, new_width):
        return image.copy()
    xs = (np.arange(new_width, dtype=np.float64) + 0.5) * w / new_width - 0.5
    ys = (np.arange(new_height, dtype=np.float64) + 0.5) * h / new_height - 0.5
    xx, yy = np.meshgrid(np.clip(xs, 0, w-1), np.clip(ys, 0, h-1))
    return bilinear_sample(image, xx, yy, clamp=True).astype(image.dtype)


def bilinear_sample(image: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                    clamp: bool = False) -> np.ndarray:
    """Own bilinear interpolation at floating point array coordinates."""
    h, w = image.shape[:2]
    if clamp:
        xs = np.clip(xs, 0, w - 1)
        ys = np.clip(ys, 0, h - 1)
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    dx = np.clip(xs - x0, 0, 1)
    dy = np.clip(ys - y0, 0, 1)
    if image.ndim == 3:
        dx = dx[..., None]
        dy = dy[..., None]
    image = image.astype(np.float32, copy=False)
    return (image[y0, x0] * (1 - dx) * (1 - dy) +
            image[y0, x1] * dx * (1 - dy) +
            image[y1, x0] * (1 - dx) * dy +
            image[y1, x1] * dx * dy)


def image_gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Own centered-difference gradients (dx,dy)."""
    dx = np.zeros_like(image, dtype=np.float32)
    dy = np.zeros_like(image, dtype=np.float32)
    dx[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2
    dy[1:-1, :] = (image[2:, :] - image[:-2, :]) / 2
    dx[:, 0] = image[:, 1] - image[:, 0]
    dx[:, -1] = image[:, -1] - image[:, -2]
    dy[0, :] = image[1, :] - image[0, :]
    dy[-1, :] = image[-1, :] - image[-2, :]
    return dx, dy
