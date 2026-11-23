# Panorama Stitching From Scratch: SIFT + RANSAC + Warping

**Bài tập lớn Xử lý ảnh**. Cài đặt thuật toán xử lý ảnh từ đầu bằng Python + NumPy.

## 1. Cài đặt (Windows, PowerShell)

> Khuyến nghị Python 3.11 hoặc 3.12. Dùng đường dẫn ngắn (ví dụ `C:\panorama`) để tránh lỗi Windows Long Path.

```powershell
cd C:\panorama
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu PowerShell không cho kích hoạt môi trường, dùng trực tiếp `.venv\Scripts\python.exe` hoặc chạy `Set-ExecutionPolicy -Scope Process Bypass`.

**Cách nhanh trên Windows:** giải nén project vào `C:\panorama`, chạy `setup_windows.bat` một lần rồi chạy `run_demo_windows.bat`. Để mở giao diện, chạy `run_ui_windows.bat` (sẽ cài thêm Streamlit).

## 2. Chạy thử ngay với bộ ảnh mẫu

```powershell
python scripts\generate_demo_data.py
python main.py data\demo\3_images\img_1.png data\demo\3_images\img_2.png data\demo\3_images\img_3.png -o outputs\demo_panorama.png --save-debug
```

Ví dụ ghép 4 ảnh:

```powershell
python main.py data\demo\4_images\img_1.png data\demo\4_images\img_2.png data\demo\4_images\img_3.png data\demo\4_images\img_4.png -o outputs\demo_4_images.png --save-debug
```

## 3. Dùng ảnh tự chụp

```powershell
python main.py "D:\images\1.jpg" "D:\images\2.jpg" "D:\images\3.jpg" -o outputs\my_panorama.png --save-debug
```

- Ảnh nhập **theo thứ tự trái → phải**, mỗi cặp liên tiếp nên chồng lấp khoảng 30–50%.
- Tránh vật thể chuyển động và chênh lệch phối cảnh lớn; tốt nhất quay camera quanh cùng tâm khi chụp.
- Không cần GPU. SIFT tự cài đặt bằng NumPy sẽ chậm hơn thư viện C++ tối ưu.
- Chương trình tự giảm chiều rộng ảnh về tối đa 1100 px (có thể đổi `--max-width 800` để chạy nhanh).
- Nên thử với 2 ảnh trước rồi ghép 3–4 ảnh.

Tùy chỉnh:

```powershell
python main.py 1.jpg 2.jpg -o outputs\pano.png --max-width 700 --max-features 1400 --contrast 0.03 --ratio 0.82 --ransac-threshold 4 --save-debug
```

Nếu matching quá ít, thử `--ratio 0.85 --contrast 0.02`, hoặc dùng ảnh có nhiều chi tiết hơn. Nếu xử lý chậm, thử `--no-double` (đánh đổi khả năng phát hiện đặc trưng).

## 4. Giao diện web Streamlit (tùy chọn)

```powershell
pip install -r requirements-ui.txt
streamlit run app.py
```

Chọn 2–4 ảnh, bấm Ghép ảnh, xem keypoints, matching, số inliers và tải PNG.

## 5. Đánh giá đối chứng OpenCV (chỉ dùng trong tệp riêng)

```powershell
python compare_opencv.py data\demo\3_images\img_1.png data\demo\3_images\img_2.png data\demo\3_images\img_3.png -o outputs\opencv_baseline.png
```

Hàm `cv2.Stitcher_create` **chỉ** được dùng trong `compare_opencv.py`, tuyệt đối không được gọi trong pipeline tự viết.

## 6. Kiểm thử tự động

```powershell
python -m unittest discover -s tests -v
```

## 7. Cấu trúc và tự cài đặt

| Module | Thuật toán tự code |
|---|---|
| `src/image_utils.py` | Chuyển grayscale, Gaussian kernel, Gaussian blur tích chập separable, bilinear resize, gradient |
| `src/sift.py` | Gaussian pyramid, Difference of Gaussian, cực trị không gian tỷ lệ, tinh chỉnh vị trí, loại edge, gán hướng, mô tả SIFT 128 chiều |
| `src/matcher.py` | L2 nearest neighbor, Lowe Ratio Test, kiểm tra matching hai chiều |
| `src/homography.py` | Normalized DLT + SVD, biến đổi tọa độ, sai số chiếu lại |
| `src/ransac.py` | Lặp lấy mẫu RANSAC 4 điểm, tối ưu và ước lượng lại H |
| `src/warping.py` | Inverse warping, bilinear interpolation, feather blending theo biên, crop |
| `src/stitcher.py` | Pipeline multi-image với ảnh tham chiếu trung tâm |
| `src/visualize.py` | Tự vẽ keypoints và inlier matches bằng gán pixel |
| `main.py` | CLI và export `debug/metrics.json` |
| `app.py` | Giao diện Streamlit |
| `scripts/generate_demo_data.py` | Bộ dữ liệu thử nghiệm từ ảnh tự tạo |
| `compare_opencv.py` | **Duy nhất phần OpenCV baseline** |

Sử dụng hàm `cv2.imread`, `cv2.imwrite`, `cv2.imdecode`, `cv2.imencode` **chỉ cho đọc/ghi ảnh**. Các phép toán đại số như `np.linalg.solve`, `np.linalg.svd` được dùng để giải hệ phương trình DLT, không thay thế thuật toán xử lý ảnh. Không dùng `cv2.SIFT_create`, `cv2.BFMatcher`, `cv2.findHomography`, `cv2.warpPerspective` trong giải pháp chính.

## 8. Đầu ra

- `outputs/panorama.png`: ảnh panorama cuối.
- `outputs/debug/keypoints_*.png`: keypoints phát hiện.
- `outputs/debug/inliers_*_*.png`: các cặp matching giữ lại sau RANSAC.
- `outputs/debug/metrics.json`: cấu hình, thời gian, điểm đặc trưng, inliers và homography.

## 9. Giới hạn kỹ thuật

- Cần ảnh có vùng chồng lấp và đặc trưng nhận biết được; không thể ghép mọi bộ ảnh tùy ý.
- H chỉ mô tả phép biến đổi phối cảnh, nên có thể lỗi với ảnh parallax mạnh, vật ở nhiều độ sâu, chuyển động, lens distortion lớn.
- Feather blending giảm đường nối nhưng có thể ghosting ở vùng chuyển động/khác biệt phơi sáng.
- Đối với ảnh thật, nên đánh giá kết quả đầu ra, vị trí sai khớp và các trường hợp thất bại.
