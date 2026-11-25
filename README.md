# Panorama tối giản, tự code thuật toán

Project ghép **2–4 ảnh có chồng lấp** bằng SIFT + Matching + RANSAC + Homography + Warping + Blending, kèm **tự nhận diện thứ tự ảnh ngang**. Không dùng hàm OpenCV thực hiện sẵn các thuật toán này trong chương trình chính. OpenCV chỉ dùng đọc/ghi ảnh và so sánh riêng.

## Chạy bằng VS Code trên Windows, Python 3.13.5

Trong Terminal PowerShell tại thư mục project:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Giao diện mở ở `http://localhost:8501`. Chọn 2–4 ảnh, bật **Tự sắp xếp ảnh** nếu tải ảnh lộn xộn, rồi nhấn **Ghép ảnh**. Để nhanh hơn khi đã chọn ảnh theo thứ tự, tắt tùy chọn này.

## Demo 4 ảnh xáo trộn thứ tự

```powershell
.\.venv\Scripts\python.exe main.py data\4_images\img_3.png data\4_images\img_1.png data\4_images\img_4.png data\4_images\img_2.png -o outputs\demo.png
```

Đầu ra: `outputs\demo.png`.

Kiểm thử:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

So sánh với OpenCV (chỉ để đối chứng):

```powershell
.\.venv\Scripts\python.exe compare_opencv.py data\4_images\img_1.png data\4_images\img_2.png data\4_images\img_3.png data\4_images\img_4.png
```

## Code theo thành viên

| Thành viên | File | Phần tự cài đặt |
|---|---|---|
| TV1 | `src/image_utils.py`, `src/sift.py` | Grayscale, Gaussian, DoG, keypoints, orientation, SIFT descriptor 128 chiều |
| TV2 | `src/geometry.py` | Khoảng cách đặc trưng, Lowe ratio, Homography (DLT), RANSAC |
| TV3 | `src/panorama.py`, `src/warping.py`, `app.py` | Tự sắp xếp ảnh, warping, nội suy bilinear, blending, ứng dụng |

`main.py` là lệnh chạy không cần giao diện. `compare_opencv.py` **không được gọi bởi code chính**, chỉ để đối chứng theo yêu cầu đề tài.

## Những giới hạn cần biết

- Ảnh nên có vùng chồng lấp 30–50% và nhiều chi tiết; SIFT có thể thất bại với ảnh gần như trơn.
- Tự sắp xếp giả định các ảnh được chụp **quét ngang**, không nhằm xử lý panorama 360° hay mọi kiểu bố cục.
- SIFT tự code bằng Python/NumPy sẽ chậm hơn bản tối ưu trong OpenCV. Mặc định thu nhỏ đầu vào tối đa **700 px** mỗi ảnh để chạy nhanh hơn; có thể chỉnh `max_width` trong `src/panorama.py`.
- SVD (`np.linalg.svd`) chỉ được dùng để giải đại số tuyến tính, không gọi các thuật toán xử lý ảnh dựng sẵn.
