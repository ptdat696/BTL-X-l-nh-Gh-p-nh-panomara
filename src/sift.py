"""Educational, entirely self-implemented SIFT feature detector/descriptor.

No cv2.SIFT_create, filtering, resizing, gradients or feature APIs.
Stages: Gaussian scale space -> DoG -> scale-space extrema localization ->
orientation histogram -> 4x4x8 gradient-histogram descriptors.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .image_utils import gaussian_blur, image_gradients, resize_bilinear, to_gray


@dataclass(frozen=True)
class Keypoint:
    x: float                    # original image pixels
    y: float
    size: float                 # original image feature diameter
    angle: float                # radians, image coordinates
    response: float
    octave: int
    layer: float


class SIFT:
    def __init__(self, n_layers: int = 3, sigma: float = 1.6,
                 contrast_threshold: float = 0.04, edge_threshold: float = 10.,
                 max_features: int = 2000, double_image: bool = True,
                 max_octaves: int = 6):
        self.n_layers = n_layers
        self.sigma = sigma
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.max_features = max_features
        self.double_image = double_image
        self.max_octaves = max_octaves
        if n_layers < 2:
            raise ValueError('n_layers must be at least 2')

    def _pyramids(self, gray: np.ndarray):
        scale = 2 if self.double_image else 1
        if scale == 2:
            base = resize_bilinear(gray, gray.shape[1] * 2, gray.shape[0] * 2)
        else:
            base = gray.copy()
        # Input assumed to have approximately 0.5 pixel of prior smoothing.
        prior_sigma = 0.5 * scale
        base = gaussian_blur(base, math.sqrt(max(self.sigma**2 - prior_sigma**2, 0.01)))
        n_oct = min(self.max_octaves, int(math.floor(math.log2(min(base.shape[:2])))) - 4)
        n_oct = max(1, n_oct)
        k = 2 ** (1 / self.n_layers)
        sigma_increments = [0.0]
        for layer in range(1, self.n_layers + 3):
            prev_sigma = self.sigma * k ** (layer - 1)
            total_sigma = self.sigma * k ** layer
            sigma_increments.append(math.sqrt(total_sigma**2 - prev_sigma**2))
        gauss, dogs = [], []
        for octave in range(n_oct):
            if min(base.shape) < 24:
                break
            levels = [base]
            for inc in sigma_increments[1:]:
                levels.append(gaussian_blur(levels[-1], inc))
            gauss.append(levels)
            dogs.append([levels[i + 1] - levels[i] for i in range(len(levels) - 1)])
            base = levels[self.n_layers][::2, ::2].copy()
        return gauss, dogs

    def _candidates(self, dogs):
        """Boolean 3x3x3 DoG extrema test, vectorized per octave/layer."""
        threshold = 0.5 * self.contrast_threshold / self.n_layers
        for octave, dog_octave in enumerate(dogs):
            for layer in range(1, self.n_layers + 1):
                center = dog_octave[layer]
                h, w = center.shape
                border = 5
                if h < 2 * border + 1 or w < 2 * border + 1:
                    continue
                p = center[border:h-border, border:w-border]
                high = p > threshold
                low = p < -threshold
                for dl in (-1, 0, 1):
                    lev = dog_octave[layer + dl]
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dl == dy == dx == 0:
                                continue
                            neigh = lev[border+dy:h-border+dy, border+dx:w-border+dx]
                            high &= p > neigh
                            low &= p < neigh
                ys, xs = np.nonzero(high | low)
                for x, y in zip(xs + border, ys + border):
                    yield octave, layer, int(x), int(y)

    @staticmethod
    def _derivatives(dog_octave, layer, x, y):
        center = dog_octave[layer]
        dx = 0.5 * (center[y, x+1] - center[y, x-1])
        dy = 0.5 * (center[y+1, x] - center[y-1, x])
        ds = 0.5 * (dog_octave[layer+1][y,x] - dog_octave[layer-1][y,x])
        dxx = center[y,x+1] + center[y,x-1] - 2*center[y,x]
        dyy = center[y+1,x] + center[y-1,x] - 2*center[y,x]
        dss = dog_octave[layer+1][y,x] + dog_octave[layer-1][y,x] - 2*center[y,x]
        dxy = 0.25 * (center[y+1,x+1] - center[y+1,x-1] - center[y-1,x+1] + center[y-1,x-1])
        dxs = 0.25 * (dog_octave[layer+1][y,x+1] - dog_octave[layer+1][y,x-1]
                      - dog_octave[layer-1][y,x+1] + dog_octave[layer-1][y,x-1])
        dys = 0.25 * (dog_octave[layer+1][y+1,x] - dog_octave[layer+1][y-1,x]
                      - dog_octave[layer-1][y+1,x] + dog_octave[layer-1][y-1,x])
        gradient = np.array([dx, dy, ds], dtype=np.float64)
        hessian = np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
        return gradient, hessian

    def _localize(self, dog_octave, layer, x, y):
        n = self.n_layers
        height, width = dog_octave[0].shape
        for _ in range(5):
            if not (1 <= layer <= n and 5 <= x < width-5 and 5 <= y < height-5):
                return None
            grad, hess = self._derivatives(dog_octave, layer, x, y)
            try:
                offset = -np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                return None
            if not np.all(np.isfinite(offset)) or np.max(np.abs(offset)) > 2.0:
                return None
            if np.max(np.abs(offset)) < 0.5:
                break
            x += int(np.round(offset[0]))
            y += int(np.round(offset[1]))
            layer += int(np.round(offset[2]))
        else:
            return None
        if np.max(np.abs(offset)) >= 1.5:
            return None
        contrast = float(dog_octave[layer][y,x] + 0.5 * grad @ offset)
        if abs(contrast) * n < self.contrast_threshold:
            return None
        dxx, dxy, dyy = hess[0,0], hess[0,1], hess[1,1]
        trace = dxx + dyy
        det = dxx * dyy - dxy*dxy
        if det <= 0 or trace*trace*self.edge_threshold >= (
                (self.edge_threshold+1)**2 * det):
            return None
        return x + offset[0], y + offset[1], layer + offset[2], abs(contrast)

    @staticmethod
    def _orientations(dx, dy, x, y, sigma):
        """36-bin smoothed orientation histogram + 0.8 peak interpolation."""
        radius = max(1, int(round(4.5 * sigma)))
        cx, cy = int(round(x)), int(round(y))
        h, w = dx.shape
        x0, x1 = max(1, cx-radius), min(w-2, cx+radius)
        y0, y1 = max(1, cy-radius), min(h-2, cy+radius)
        if x1 < x0 or y1 < y0:
            return []
        xs = np.arange(x0, x1+1)
        ys = np.arange(y0, y1+1)
        xx, yy = np.meshgrid(xs, ys)
        r2 = (xx-x)**2 + (yy-y)**2
        gxx = dx[yy,xx]
        gyy = dy[yy,xx]
        mag = np.sqrt(gxx*gxx + gyy*gyy)
        weight = np.exp(-r2 / (2 * (1.5*sigma)**2))
        angle = np.mod(np.arctan2(gyy, gxx), 2*np.pi)
        bin_float = angle * 36 / (2*np.pi)
        bins0 = np.floor(bin_float).astype(np.int32) % 36
        frac = bin_float - np.floor(bin_float)
        hist = np.zeros(36, dtype=np.float64)
        np.add.at(hist, bins0.ravel(), ((1-frac)*mag*weight).ravel())
        np.add.at(hist, ((bins0+1)%36).ravel(), (frac*mag*weight).ravel())
        for _ in range(6):
            hist = (np.roll(hist, 1) + 4*hist + np.roll(hist, -1))/6
        peak = float(np.max(hist))
        if peak < 1e-10:
            return []
        peaks = []
        for i in range(36):
            before, now, after = hist[(i-1)%36], hist[i], hist[(i+1)%36]
            if now >= 0.8 * peak and now > before and now > after:
                denom = before - 2*now + after
                shift = 0.5*(before-after)/denom if abs(denom)>1e-12 else 0.
                peaks.append(((i + shift) * (2*np.pi/36)) % (2*np.pi))
        return peaks

    @staticmethod
    def _descriptor(dx, dy, x, y, sigma, angle):
        """Spatial+orientation trilinear histogram accumulation, 4x4x8=128."""
        hist_width = 3.0 * sigma
        radius = int(round(hist_width * math.sqrt(2) * 2.5))
        cx, cy = int(round(x)), int(round(y))
        h, w = dx.shape
        x0, x1 = max(1, cx-radius), min(w-2, cx+radius)
        y0, y1 = max(1, cy-radius), min(h-2, cy+radius)
        if x1 < x0 or y1 < y0:
            return None
        xx, yy = np.meshgrid(np.arange(x0,x1+1), np.arange(y0,y1+1))
        rx, ry = xx-x, yy-y
        c, s = math.cos(angle), math.sin(angle)
        col = (c*rx + s*ry)/hist_width + 1.5
        row = (-s*rx + c*ry)/hist_width + 1.5
        valid = (row > -1) & (row < 4) & (col > -1) & (col < 4)
        if not np.any(valid):
            return None
        row, col = row[valid], col[valid]
        gx, gy = dx[yy[valid], xx[valid]], dy[yy[valid], xx[valid]]
        magnitudes = np.sqrt(gx*gx + gy*gy)
        relangle = np.mod(np.arctan2(gy,gx) - angle, 2*np.pi) * 8/(2*np.pi)
        wgt = np.exp(-((row-1.5)**2 + (col-1.5)**2)/8.0) * magnitudes
        r0 = np.floor(row).astype(np.int32)
        c0 = np.floor(col).astype(np.int32)
        a0 = np.floor(relangle).astype(np.int32)
        rf, cf, af = row-r0, col-c0, relangle-a0
        hist = np.zeros((4,4,8), dtype=np.float64)
        for dr in (0,1):
            rr = r0 + dr
            wr = rf if dr else 1-rf
            for dc in (0,1):
                cc = c0 + dc
                wc = cf if dc else 1-cf
                ok = (rr >= 0)&(rr < 4)&(cc >= 0)&(cc < 4)
                if not np.any(ok):
                    continue
                for da in (0,1):
                    aa = (a0+da) % 8
                    wa = af if da else 1-af
                    np.add.at(hist, (rr[ok], cc[ok], aa[ok]),
                              (wr*wc*wa*wgt)[ok])
        desc = hist.ravel()
        norm = np.linalg.norm(desc)
        if norm < 1e-10:
            return None
        desc /= norm
        np.minimum(desc, 0.2, out=desc)
        desc /= max(np.linalg.norm(desc), 1e-12)
        return desc.astype(np.float32)

    def detect_and_compute(self, image: np.ndarray):
        gray = to_gray(image)
        if min(gray.shape) < 32:
            raise ValueError('Image must have a minimum side of at least 32 pixels')
        gauss, dogs = self._pyramids(gray)
        # Compute gradients once per Gaussian level.
        gradients = [[image_gradients(level) for level in octave] for octave in gauss]
        records = []
        for octidx, layer, x, y in self._candidates(dogs):
            result = self._localize(dogs[octidx], layer, x, y)
            if result is None:
                continue
            xf, yf, layerf, response = result
            layer_int = int(np.clip(round(layerf), 0, len(gauss[octidx])-1))
            sigma_local = self.sigma * 2 ** (layerf / self.n_layers)
            dx, dy = gradients[octidx][layer_int]
            angles = self._orientations(dx, dy, xf, yf, sigma_local)
            image_scale = 2.0**octidx / (2 if self.double_image else 1)
            for angle in angles:
                desc = self._descriptor(dx, dy, xf, yf, sigma_local, angle)
                if desc is not None:
                    kp = Keypoint(float(xf*image_scale), float(yf*image_scale),
                                  float(2*sigma_local*image_scale), float(angle),
                                  response, octidx, float(layerf))
                    records.append((kp, desc))
        records.sort(key=lambda r: r[0].response, reverse=True)
        if self.max_features > 0:
            records = records[:self.max_features]
        if not records:
            return [], np.empty((0,128), dtype=np.float32)
        return [r[0] for r in records], np.stack([r[1] for r in records])
