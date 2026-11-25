"""Quick tests: python -m unittest discover -s tests -v"""

from pathlib import Path
import unittest
import cv2
import numpy as np

from src.geometry import homography, project, ransac, match_features
from src.panorama import stitch

DATA = Path(__file__).resolve().parents[1] / 'data' / '4_images'
IMAGES = [cv2.imread(str(DATA / f'img_{i}.png')) for i in range(1, 5)]


class TestGeometry(unittest.TestCase):
    def test_homography(self):
        points = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [4, 8]], float)
        true_H = np.array([[1, .1, 12], [.05, 1.1, 9], [.0003, 0, 1]])
        found = homography(points, project(true_H, points))
        self.assertLess(np.max(abs(project(found, points) - project(true_H, points))), 1e-5)

    def test_matching(self):
        rng = np.random.default_rng(3)
        a = rng.random((20, 128), dtype=np.float32)
        b = np.concatenate((a.copy(), rng.random((15, 128), dtype=np.float32)))
        matches = match_features(a, b)
        self.assertEqual(len(matches), 20)
        self.assertTrue(all(i == j for i, j in matches))

    def test_ransac_rejects_outliers(self):
        rng = np.random.default_rng(4)
        points = rng.uniform(10, 400, (90, 2))
        true_H = np.array([[1.01, .01, 40], [-.02, .99, 15], [0, 0, 1]])
        dst = project(true_H, points) + rng.normal(0, .1, points.shape)
        dst[:20] = rng.uniform(10, 400, (20, 2))
        H, mask = ransac(points, dst)
        self.assertGreaterEqual(mask.sum(), 68)
        self.assertLess(np.median(np.linalg.norm(project(H, points[20:]) - dst[20:], axis=1)), 1.)


class TestPanorama(unittest.TestCase):
    def test_two_images(self):
        result = stitch(IMAGES[:2], auto_order=False)
        self.assertEqual(result.order, [0, 1])
        self.assertEqual(len(result.inlier_counts), 1)
        self.assertGreater(result.panorama.shape[1], 500)

    def test_three_images_reversed(self):
        result = stitch(IMAGES[:3][::-1], auto_order=True)
        self.assertEqual(result.order, [2, 1, 0])
        self.assertEqual(len(result.inlier_counts), 2)

    def test_four_images_shuffled(self):
        order_in = [2, 0, 3, 1]
        result = stitch([IMAGES[i] for i in order_in], auto_order=True)
        self.assertEqual(result.order, [1, 3, 0, 2])
        self.assertEqual(len(result.inlier_counts), 3)
        self.assertTrue(all(x >= 6 for x in result.inlier_counts))

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            stitch([IMAGES[0]])
        with self.assertRaises(ValueError):
            stitch([None, IMAGES[0]])


if __name__ == '__main__':
    unittest.main()
