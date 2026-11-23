"""Debug rendering from pixel arrays (no cv2 drawing or image algorithms)."""
from __future__ import annotations
import math
import numpy as np


def _draw_dot(image,x,y,color,radius=3):
    h,w = image.shape[:2]
    cx,cy = int(round(x)),int(round(y))
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            if dx*dx+dy*dy<=radius*radius and 0<=cx+dx<w and 0<=cy+dy<h:
                image[cy+dy,cx+dx] = color


def _draw_line(image,x0,y0,x1,y1,color):
    steps = int(max(abs(x1-x0),abs(y1-y0)))+1
    for t in np.linspace(0,1,steps):
        x = int(round(x0+(x1-x0)*t))
        y = int(round(y0+(y1-y0)*t))
        if 0<=y<image.shape[0] and 0<=x<image.shape[1]:
            image[y,x] = color


def keypoint_preview(image,keypoints,limit=600):
    output = image.copy()
    for kp in keypoints[:limit]:
        _draw_dot(output,kp.x,kp.y,(0,255,255),2)
    return output


def match_preview(image1,image2,kp1,kp2,matches,inliers=None,limit=120):
    h = max(image1.shape[0],image2.shape[0])
    w1,w2 = image1.shape[1],image2.shape[1]
    out = np.zeros((h,w1+w2,3),dtype=np.uint8)
    out[:image1.shape[0],:w1] = image1
    out[:image2.shape[0],w1:] = image2
    if inliers is None:
        show = matches[:limit]
    else:
        show = [m for m,ok in zip(matches,inliers) if ok][:limit]
    rng = np.random.default_rng(10)
    for m in show:
        a,b = kp1[m.query_idx],kp2[m.train_idx]
        color = tuple(int(c) for c in rng.integers(60,255,size=3))
        _draw_line(out,a.x,a.y,b.x+w1,b.y,color)
        _draw_dot(out,a.x,a.y,color,2)
        _draw_dot(out,b.x+w1,b.y,color,2)
    return out
