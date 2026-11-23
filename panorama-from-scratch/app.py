"""Streamlit user interface (optional): streamlit run app.py"""
from __future__ import annotations
import json
from time import perf_counter

import cv2  # image decoding/encoding only
import numpy as np
import streamlit as st

from src.image_utils import resize_bilinear
from src.stitcher import StitchSettings, stitch
from src.visualize import keypoint_preview,match_preview

st.set_page_config(page_title='Panorama from Scratch',page_icon='🖼️',layout='wide')
st.title('🖼️ Panorama Stitching: SIFT tự cài đặt')
st.write('SIFT → Matching → RANSAC + Homography → Warping → Feather Blending. '
         'OpenCV chỉ dùng để giải mã và mã hóa ảnh trong giao diện này.')

uploads=st.file_uploader('Chọn 2–4 ảnh theo thứ tự từ trái sang phải',
                         type=['jpg','jpeg','png','bmp'],accept_multiple_files=True)
c1,c2,c3=st.columns(3)
with c1:
    max_width=st.slider('Giới hạn chiều rộng ảnh (pixel)',300,1600,900,step=100)
    max_features=st.slider('Số đặc trưng SIFT tối đa',200,3000,1400,step=100)
with c2:
    contrast=st.slider('Contrast Threshold',0.005,0.12,0.04,step=0.005)
    ratio=st.slider('Lowe Ratio',0.55,0.95,0.78,step=0.01)
with c3:
    threshold=st.slider('RANSAC Threshold (pixel)',1.,12.,3.5,step=0.5)
    feather=st.slider('Feather width',0,100,35,step=5)

if uploads:
    st.write('**Thứ tự đầu vào:** '+', '.join(f.name for f in uploads))

if st.button('Ghép ảnh panorama',type='primary'):
    if not 2<=len(uploads)<=4:
        st.error('Hãy chọn từ 2 đến 4 ảnh.')
    else:
        images=[]
        for up in uploads:
            arr=np.frombuffer(up.getvalue(),dtype=np.uint8)
            im=cv2.imdecode(arr,cv2.IMREAD_COLOR)
            if im is None:
                st.error('Không đọc được '+up.name)
                st.stop()
            h,w=im.shape[:2]
            if w>max_width:
                im=resize_bilinear(im,max_width,round(h*max_width/w))
            images.append(im)
        try:
            with st.spinner('Đang xử lý các thuật toán tự cài đặt...'):
                settings=StitchSettings(max_features=max_features,
                                        contrast_threshold=contrast,ratio=ratio,
                                        ransac_threshold=threshold,feather=float(feather))
                res=stitch(images,settings)
            st.success(f'Ghép thành công {len(images)} ảnh, {res.timings["total"]:.2f}s')
            st.image(res.panorama[:,:,::-1],caption='Panorama từ code tự viết',use_container_width=True)
            ok,png=cv2.imencode('.png',res.panorama)
            if ok:
                st.download_button('Tải ảnh panorama PNG',png.tobytes(),
                                   file_name='panorama.png',mime='image/png')
            st.subheader('Kết quả định lượng')
            st.json({'Features per image':[len(k) for k in res.keypoints],
                     'Matches per pair':[len(p.matches) for p in res.pairs],
                     'RANSAC inliers':[int(p.inliers.sum()) for p in res.pairs],
                     'Reprojection error (px)':[round(p.reprojection_error,3) for p in res.pairs],
                     'Runtime (seconds)':{k:round(v,2) for k,v in res.timings.items()}})
            with st.expander('Xem keypoints và những cặp điểm inliers'):
                for i,img in enumerate(images):
                    st.image(keypoint_preview(img,res.keypoints[i])[:,:,::-1],
                             caption=f'Keypoints ảnh {i+1}',use_container_width=True)
                for p in res.pairs:
                    out=match_preview(images[p.source_idx],images[p.dest_idx],
                                      res.keypoints[p.source_idx],res.keypoints[p.dest_idx],
                                      p.matches,p.inliers)
                    st.image(out[:,:,::-1],caption=f'Inlier pair {p.source_idx+1}-{p.dest_idx+1}',
                             use_container_width=True)
        except (ValueError,np.linalg.LinAlgError) as exc:
            st.error(str(exc))
            st.info('Hãy chọn các ảnh có ít nhất 30–50% chồng lấp và đủ chi tiết, '
                    'tránh lệch góc nhìn quá lớn.')
