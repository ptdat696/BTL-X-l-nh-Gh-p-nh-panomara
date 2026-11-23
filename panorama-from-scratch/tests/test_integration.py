"""Full SIFT panorama integration, uses synthetic planar scene with real overlap."""
import unittest
import numpy as np
from scripts.generate_demo_data import make_scene
from src.stitcher import stitch,StitchSettings


class TestPipeline(unittest.TestCase):
    def test_two_overlapping_crops(self):
        scene=make_scene(width=500,height=230,seed=7)
        images=[scene[:,0:330],scene[:,170:500]]
        result=stitch(images,StitchSettings(max_features=550,double_image=False,
                                            feather=25,contrast_threshold=0.02))
        self.assertEqual(len(result.pairs),1)
        self.assertGreaterEqual(int(result.pairs[0].inliers.sum()),6)
        self.assertGreater(result.panorama.shape[1],440)

    def test_four_overlapping_crops(self):
        scene=make_scene(width=660,height=240,seed=13)
        images=[scene[:,start:start+330] for start in (0,110,220,330)]
        result=stitch(images,StitchSettings(max_features=550,double_image=False,
                                            feather=25,contrast_threshold=0.02))
        self.assertEqual(len(result.pairs),3)
        self.assertTrue(all(p.inliers.sum()>=6 for p in result.pairs))
        self.assertGreater(result.panorama.shape[1],580)

    def test_three_overlapping_crops(self):
        scene=make_scene(width=600,height=240,seed=11)
        images=[scene[:,0:360],scene[:,120:480],scene[:,240:600]]
        result=stitch(images,StitchSettings(max_features=550,double_image=False,
                                            feather=25,contrast_threshold=0.02))
        self.assertEqual(len(result.pairs),2)
        self.assertTrue(all(p.inliers.sum()>=6 for p in result.pairs))
        self.assertGreater(result.panorama.shape[1],500)
        self.assertGreater(result.panorama.shape[0],200)


if __name__=='__main__':
    unittest.main()
