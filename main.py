"""Example: python main.py img1.jpg img2.jpg img3.jpg -o panorama.png"""

import argparse
from pathlib import Path
import cv2
from src.panorama import stitch

parser = argparse.ArgumentParser(description='Panorama tự code, không dùng OpenCV Stitcher')
parser.add_argument('images', nargs='+', help='2–10 ảnh đầu vào')
parser.add_argument('-o', '--output', default='outputs/panorama.png')
args = parser.parse_args()

if not 2 <= len(args.images) <= 4:
    parser.error('Cần từ 2 đến 10 ảnh.')
images = [cv2.imread(path) for path in args.images]
result = stitch(images)
out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
if not cv2.imwrite(str(out), result.panorama):
    parser.error('Không ghi được ảnh kết quả.')
print('Đã lưu:', out)
print('Thứ tự:', ' → '.join(args.images[i] for i in result.order))
print('Inliers:', result.inlier_counts)
print('Thời gian:', round(result.seconds, 2), 'giây')
