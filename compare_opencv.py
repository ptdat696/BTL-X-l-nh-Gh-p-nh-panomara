import argparse
from pathlib import Path
import cv2
import numpy as np


def resize_image(img, max_width):
    if img.shape[1] <= max_width:
        return img
    ratio = max_width / img.shape[1]
    return cv2.resize(img, (max_width, round(img.shape[0] * ratio)),
                      interpolation=cv2.INTER_AREA)


def pair_homography(left, right, sift, matcher):
    """Tính H ánh xạ tọa độ ảnh left sang ảnh right."""
    kp1, desc1 = sift.detectAndCompute(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), None)
    kp2, desc2 = sift.detectAndCompute(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), None)
    if desc1 is None or desc2 is None or len(desc1) < 8 or len(desc2) < 8:
        raise ValueError('Không đủ đặc trưng SIFT trong một cặp ảnh.')

    knn = matcher.knnMatch(desc1, desc2, k=2)
    good = [m for m, n in knn if m.distance < 0.75 * n.distance]
    if len(good) < 10:
        # Nới ngưỡng một chút nếu dữ liệu chồng lấp tương đối ít.
        good = [m for m, n in knn if m.distance < 0.82 * n.distance]
    if len(good) < 8:
        raise ValueError(f'Chỉ có {len(good)} cặp đặc trưng phù hợp.')

    p1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    p2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    # USAC_MAGSAC nếu phiên bản OpenCV hỗ trợ, nếu không dùng RANSAC.
    method = cv2.USAC_MAGSAC if hasattr(cv2, 'USAC_MAGSAC') else cv2.RANSAC
    H, mask = cv2.findHomography(p1, p2, method, 4.0,
                                 maxIters=10000, confidence=0.999)
    if H is None or mask is None or int(mask.sum()) < 8:
        raise ValueError('Không ước lượng được Homography đáng tin cậy.')

    reproj = cv2.perspectiveTransform(p1, H).reshape(-1, 2)
    err = np.linalg.norm(reproj - p2.reshape(-1, 2), axis=1)
    median_error = float(np.median(err[mask.ravel().astype(bool)]))
    if not np.isfinite(H).all() or median_error > 6.0:
        raise ValueError(f'Homography có sai số lớn ({median_error:.1f} px).')
    return H / H[2, 2], len(good), int(mask.sum()), median_error


def solid_crop(pano, occupied):
    """Tìm hình chữ nhật lớn nhất nằm hoàn toàn trong vùng có ảnh."""
    ys, xs = np.where(occupied)
    if len(xs) == 0:
        return pano
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    mask = occupied[y0:y1, x0:x1]
    height, width = mask.shape
    levels = np.zeros(width, dtype=np.int32)
    best = (0, 0, width, height)
    max_area = 0

    for y in range(height):
        levels = np.where(mask[y], levels + 1, 0)
        stack = []  # (x bắt đầu, chiều cao)
        for x in range(width + 1):
            h = int(levels[x]) if x < width else 0
            start = x
            while stack and stack[-1][1] > h:
                old_x, old_h = stack.pop()
                area = (x - old_x) * old_h
                if area > max_area:
                    max_area = area
                    best = (old_x, y - old_h + 1, x, y + 1)
                start = old_x
            if h and (not stack or stack[-1][1] < h):
                stack.append((start, h))
    bx0, by0, bx1, by1 = best
    return pano[y0 + by0:y0 + by1, x0 + bx0:x0 + bx1]


def stitch_sift(images):
    """Ghép ảnh theo thứ tự đã cung cấp, dùng các hàm OpenCV làm đối chứng."""
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.02)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    H_pairs = []
    for i in range(len(images) - 1):
        H, count, inliers, error = pair_homography(images[i], images[i + 1], sift, matcher)
        H_pairs.append(H)
        print(f'Cặp {i + 1}-{i + 2}: {count} matches, {inliers} inliers, '
              f'sai số trung vị {error:.2f}px')
    # Sử dụng ảnh giữa làm gốc để giảm méo ở rìa panorama.
    middle = (len(images) - 1) // 2
    H_global = [None] * len(images)
    H_global[middle] = np.eye(3, dtype=np.float64)
    for i in range(middle - 1, -1, -1):
        H_global[i] = H_global[i + 1] @ H_pairs[i]
    for i in range(middle + 1, len(images)):
        H_global[i] = H_global[i - 1] @ np.linalg.inv(H_pairs[i - 1])

    corners = []
    for im, H in zip(images, H_global):
        h, w = im.shape[:2]
        p = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(p, H)
        corners.append(mapped.reshape(-1, 2))
    all_pts = np.concatenate(corners)
    if not np.isfinite(all_pts).all():
        raise ValueError('Phép chiếu không hợp lệ.')
    xmin, ymin = np.floor(all_pts.min(axis=0)).astype(int)
    xmax, ymax = np.ceil(all_pts.max(axis=0)).astype(int)
    width, height = int(xmax - xmin), int(ymax - ymin)
    if (width <= 0 or height <= 0 or
            width > 3 * sum(im.shape[1] for im in images) or
            height > 4 * max(im.shape[0] for im in images) or
            width * height > 55_000_000):
        raise ValueError('Canvas bất thường: kiểm tra các cặp Homography.')

    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float64)
    warped_images, masks, distances = [], [], []
    for im, H in zip(images, H_global):
        M = T @ H
        warped = cv2.warpPerspective(im, M, (width, height))
        mask = cv2.warpPerspective(np.full(im.shape[:2], 255, np.uint8),
                                   M, (width, height),
                                   flags=cv2.INTER_NEAREST) > 0
        warped_images.append(warped)
        masks.append(mask)
        # Chọn ảnh đóng góp chính ở mỗi pixel để tránh bóng đôi (ghosting).
        padded = cv2.copyMakeBorder(mask.astype(np.uint8), 1, 1, 1, 1,
                                    cv2.BORDER_CONSTANT, value=0)
        distances.append(cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1])

    occupied = np.any(np.stack(masks), axis=0)
    winner = np.argmax(np.stack(distances), axis=0)
    result = np.zeros((height, width, 3), dtype=np.float32)
    total_weight = np.zeros((height, width), dtype=np.float32)
    for i, (warped, mask) in enumerate(zip(warped_images, masks)):
        # Làm mềm chỉ gần ranh giới giữa các ảnh, không trộn trên toàn vùng overlap.
        weights = cv2.GaussianBlur((winner == i).astype(np.float32),
                                   (0, 0), sigmaX=12)
        weights *= mask.astype(np.float32)
        result += warped.astype(np.float32) * weights[..., None]
        total_weight += weights

    valid = total_weight > 1e-6
    result[valid] /= total_weight[valid, None]
    pano = np.clip(result, 0, 255).astype(np.uint8)
    return solid_crop(pano, valid)


def main():
    parser = argparse.ArgumentParser(description='So sánh Panorama với OpenCV')
    parser.add_argument('images', nargs='+', help='Ảnh theo thứ tự trái sang phải')
    parser.add_argument('-o', '--output', default='outputs/opencv.jpg')
    parser.add_argument('--method', choices=['auto', 'panorama', 'sift'], default='auto',
                        help='auto: Stitcher PANORAMA, nếu lỗi thì SIFT + Homography')
    parser.add_argument('--max-width', type=int, default=1400)
    args = parser.parse_args()
    if len(args.images) < 2:
        parser.error('Cần ít nhất hai ảnh.')

    images = []
    for path in args.images:
        im = cv2.imread(path)
        if im is None:
            parser.error(f'Không đọc được ảnh: {path}')
        images.append(resize_image(im, args.max_width))

    pano = None
    if args.method in ('auto', 'panorama'):
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        status, candidate = stitcher.stitch(images)
        if status == cv2.Stitcher_OK and candidate is not None:
            pano = candidate
            print('Phương pháp: OpenCV Stitcher PANORAMA')
        else:
            print(f'OpenCV Stitcher PANORAMA thất bại (mã {status}).')
            if args.method == 'panorama':
                parser.error('Không thể ghép bằng Stitcher PANORAMA.')

    if pano is None:
        print('Chuyển sang đối chứng OpenCV SIFT + BFMatcher + RANSAC + Warping...')
        try:
            pano = stitch_sift(images)
        except (ValueError, np.linalg.LinAlgError, cv2.error) as ex:
            parser.error(f'Ghép bằng OpenCV SIFT cũng thất bại: {ex}')
        print('Phương pháp: OpenCV SIFT + Homography (không phải Stitcher)')

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), pano):
        parser.error(f'Không lưu được: {out}')
    print(f'Đã lưu đối chứng: {out} ({pano.shape[1]} x {pano.shape[0]} px)')


if __name__ == '__main__':
    main()
