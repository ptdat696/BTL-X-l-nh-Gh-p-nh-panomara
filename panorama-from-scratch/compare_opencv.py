"""OPTIONAL BASELINE ONLY. cv2.Stitcher_create is NEVER called in main.py/src/."""
from __future__ import annotations
import argparse
from pathlib import Path
from time import perf_counter
import cv2
from main import load_images


def main():
    parser=argparse.ArgumentParser(description='Compare custom project with OpenCV baseline')
    parser.add_argument('images',nargs='+')
    parser.add_argument('-o','--output',default='outputs/opencv_baseline.png')
    parser.add_argument('--max-width',type=int,default=1100)
    args=parser.parse_args()
    if len(args.images)<2:
        parser.error('Need >=2 images')
    images=load_images(args.images,args.max_width)
    start=perf_counter()
    stitcher=cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status,pano=stitcher.stitch(images)
    duration=perf_counter()-start
    if status!=cv2.Stitcher_OK:
        parser.exit(1,f'OpenCV baseline failed, status={status}\n')
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(out),pano)
    print(f'OpenCV baseline saved {out} ({duration:.2f}s, {pano.shape[1]}x{pano.shape[0]})')


if __name__=='__main__':
    main()
