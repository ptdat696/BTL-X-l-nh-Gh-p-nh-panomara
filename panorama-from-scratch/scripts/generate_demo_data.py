"""Generate reproducible self-created overlapping test images.

Images model crops of one planar scene; no OpenCV image-processing routines.
Use python scripts/generate_demo_data.py before first CLI demo.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import cv2  # imwrite only
import numpy as np


def make_scene(width=1000,height=420,seed=2026):
    rng=np.random.default_rng(seed)
    xx,yy=np.meshgrid(np.arange(width),np.arange(height))
    # Spatial colors, texture, independent noise and distinctive colored shapes.
    base=np.stack((55 + 0.07*xx + 16*np.sin(xx/23) + 10*np.cos(yy/15),
                   72 + 0.08*yy + 17*np.cos(xx/31) + 10*np.sin(yy/21),
                   120 + 0.06*xx + 12*np.sin((xx+yy)/35)),axis=2)
    base+=rng.normal(0,10,size=base.shape)
    scene=np.clip(base,0,255).astype(np.uint8)
    for _ in range(230):
        x,y=int(rng.integers(0,width)),int(rng.integers(0,height))
        rw,rh=int(rng.integers(4,34)),int(rng.integers(4,30))
        col=rng.integers(20,255,size=3,dtype=np.uint8)
        x1,y1=min(width,x+rw),min(height,y+rh)
        if rng.random()<0.5:
            scene[y:y1,x:x1]=col
        else:
            yy0,xx0=np.ogrid[y:y1,x:x1]
            cx,cy=(x+x1-1)/2,(y+y1-1)/2
            mask=((xx0-cx)/max((x1-x)/2,1))**2+((yy0-cy)/max((y1-y)/2,1))**2<=1
            patch=scene[y:y1,x:x1]
            patch[mask]=col
    # Non-repeating fine structures spanning image, yielding strong SIFT corners.
    for _ in range(110):
        x,y=int(rng.integers(0,width-40)),int(rng.integers(0,height-40))
        dy,dx=int(rng.integers(6,35)),int(rng.integers(6,35))
        col=rng.integers(0,250,size=3,dtype=np.uint8)
        scene[y:y+2,x:x+dx]=col
        scene[y:y+dy,x:x+2]=col
    return scene


def generate(output_dir: Path):
    source=make_scene()
    output_dir.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(output_dir/'full_scene.png'),source)
    configs={'2_images':([0,380],620),'3_images':([0,250,500],500),
             '4_images':([0,185,370,555],445)}
    for name,(offsets,width) in configs.items():
        folder=output_dir/name
        folder.mkdir(parents=True,exist_ok=True)
        for i,x in enumerate(offsets,1):
            cv2.imwrite(str(folder/f'img_{i}.png'),source[:,x:x+width])
    print('Generated:',output_dir)
    for k in configs:
        print(' ', k)


if __name__=='__main__':
    generate(Path(__file__).resolve().parents[1]/'data'/'demo')
