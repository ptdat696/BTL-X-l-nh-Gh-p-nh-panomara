"""Unit tests: python -m unittest discover -s tests -v"""
import unittest
import numpy as np
from src.image_utils import gaussian_kernel_1d, gaussian_blur, resize_bilinear, to_gray
from src.homography import estimate_homography, transform_points
from src.ransac import ransac_homography
from src.matcher import match_features
from src.warping import composite_images


class TestCoreMath(unittest.TestCase):
    def test_gray(self):
        im=np.zeros((2,2,3),dtype=np.uint8)
        im[0,0]=[0,0,255]
        self.assertAlmostEqual(float(to_gray(im)[0,0]),0.299,places=4)

    def test_gaussian(self):
        k=gaussian_kernel_1d(1.6)
        self.assertAlmostEqual(float(k.sum()),1.0,places=5)
        constant=np.ones((33,37),dtype=np.float32)*0.42
        self.assertTrue(np.allclose(gaussian_blur(constant,1.6),0.42,atol=1e-6))

    def test_bilinear_resize(self):
        image=np.arange(25,dtype=np.float32).reshape(5,5)
        self.assertTrue(np.allclose(resize_bilinear(image,5,5),image))
        self.assertEqual(resize_bilinear(image,10,8).shape,(8,10))

    def test_homography_exact(self):
        p=np.array([[0.,0.],[5,0],[5,5],[0,5],[7,2],[1,3],[3,6]])
        H=np.array([[1.2,0.12,33],[-0.04,1.1,-7],[0.0003,-0.0002,1]])
        dst=transform_points(H,p)
        estimate=estimate_homography(p,dst)
        self.assertLess(np.max(np.abs(transform_points(estimate,p)-dst)),1e-6)

    def test_ransac_outliers(self):
        rng=np.random.default_rng(123)
        src=rng.uniform(0,400,size=(100,2))
        H=np.array([[0.99,0.02,48],[-0.03,1.01,16],[0.00005,0.00002,1.]])
        dst=transform_points(H,src)+rng.normal(0,0.25,size=(100,2))
        dst[:30]=rng.uniform(0,400,size=(30,2))
        result=ransac_homography(src,dst,threshold=2.5,seed=4)
        self.assertGreaterEqual(result.inliers.sum(),68)
        self.assertLess(result.mean_error,0.8)

    def test_matching(self):
        rng=np.random.default_rng(8)
        a=rng.normal(size=(30,128)).astype(np.float32)
        a/=np.linalg.norm(a,axis=1,keepdims=True)
        b=np.concatenate((a+rng.normal(0,0.003,size=a.shape),
                          rng.normal(size=(30,128))),axis=0).astype(np.float32)
        matches=match_features(a,b,ratio=0.8)
        self.assertEqual(len(matches),30)
        self.assertTrue(all(m.query_idx==m.train_idx for m in matches))

    def test_warp_identity(self):
        im=np.arange(3*15*20,dtype=np.uint8).reshape(15,20,3)
        out,mask=composite_images([im],[np.eye(3)],feather=10)
        self.assertEqual(out.shape,im.shape)
        self.assertTrue(np.all(mask))
        self.assertLess(np.abs(out.astype('float32')-im).max(),1.1)


if __name__=='__main__':
    unittest.main()
