"""CLI: python main.py image1.jpg image2.jpg image3.jpg -o panorama.png"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import cv2  # Image load/save ONLY, no ready-made image-processing in this pipeline
import numpy as np

from src.image_utils import resize_bilinear
from src.stitcher import StitchSettings, stitch
from src.visualize import keypoint_preview, match_preview


def load_images(paths, max_width=1100):
    images=[]
    for path in paths:
        image=cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Cannot read image: {path}')
        h,w=image.shape[:2]
        if max_width and w>max_width:
            image=resize_bilinear(image,max_width,round(h*max_width/w))
        images.append(image)
    return images


def main():
    parser=argparse.ArgumentParser(description='From-scratch SIFT panorama stitcher (NumPy).')
    parser.add_argument('images',nargs='+',help='2-12 overlapping image paths, left to right')
    parser.add_argument('-o','--output',default='outputs/panorama.png')
    parser.add_argument('--max-width',type=int,default=1100,
                        help='Resize images down to this width, 0 = keep full size')
    parser.add_argument('--max-features',type=int,default=1600)
    parser.add_argument('--contrast',type=float,default=0.04)
    parser.add_argument('--ratio',type=float,default=0.78)
    parser.add_argument('--ransac-threshold',type=float,default=3.5)
    parser.add_argument('--feather',type=float,default=35.)
    parser.add_argument('--no-double',action='store_true',help='Disable SIFT initial 2x upsampling')
    parser.add_argument('--no-mutual',action='store_true',help='Disable reciprocal matching check')
    parser.add_argument('--save-debug',action='store_true',help='Export keypoint and inlier images')
    parser.add_argument('--no-crop',action='store_true')
    args=parser.parse_args()
    try:
        images=load_images(args.images,args.max_width)
        settings=StitchSettings(max_features=args.max_features,
                                contrast_threshold=args.contrast,ratio=args.ratio,
                                ransac_threshold=args.ransac_threshold,
                                double_image=not args.no_double,
                                mutual=not args.no_mutual,feather=args.feather,
                                crop=not args.no_crop)
        result=stitch(images,settings)
        output=Path(args.output)
        output.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(output),result.panorama):
            raise OSError(f'Cannot save output to {output}')
        print('Saved panorama:',output)
        for i,kps in enumerate(result.keypoints):
            print(f'  Image {i+1}: {len(kps)} SIFT features')
        for pair in result.pairs:
            print(f'  Pair {pair.source_idx+1}-{pair.dest_idx+1}: '
                  f'{len(pair.matches)} matches, {sum(pair.inliers)} RANSAC inliers, '
                  f'error={pair.reprojection_error:.2f}px')
        print('Timing:',json.dumps({k:round(v,2) for k,v in result.timings.items()}))
        if args.save_debug:
            debug=output.parent/'debug'
            debug.mkdir(parents=True,exist_ok=True)
            for idx,im in enumerate(images):
                cv2.imwrite(str(debug/f'keypoints_{idx+1}.png'),
                            keypoint_preview(im,result.keypoints[idx]))
            for pair in result.pairs:
                idx=pair.source_idx
                cv2.imwrite(str(debug/f'inliers_{idx+1}_{idx+2}.png'),
                            match_preview(images[idx],images[idx+1],
                                          result.keypoints[idx],result.keypoints[idx+1],
                                          pair.matches,pair.inliers))
            metrics={'settings':vars(settings),'shape':list(result.panorama.shape),
                     'timings_seconds':result.timings,
                     'feature_counts':[len(k) for k in result.keypoints],
                     'pairs':[{'image_a':p.source_idx+1,'image_b':p.dest_idx+1,
                               'matches':len(p.matches),'inliers':int(p.inliers.sum()),
                               'mean_reprojection_error_px':p.reprojection_error,
                               'ransac_iterations':p.trials,'homography':p.H.tolist()}
                              for p in result.pairs]}
            (debug/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
            print('Debug images and metrics:',debug)
    except (ValueError, OSError, np.linalg.LinAlgError) as exc:
        parser.exit(1,f'Error: {exc}\n')


if __name__=='__main__':
    main()
