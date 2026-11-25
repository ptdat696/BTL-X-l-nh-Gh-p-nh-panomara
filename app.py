"""Run: python -m streamlit run app.py"""

import hashlib
import cv2
import numpy as np
import streamlit as st
from src.panorama import stitch

st.set_page_config(page_title='Panorama', layout='wide')
st.title('Ghép ảnh Panorama')

files = st.file_uploader('Chọn 2–4 ảnh', type=['png', 'jpg', 'jpeg'],
                         accept_multiple_files=True)

data = [(f.name, f.getvalue()) for f in files]
key = tuple((name, hashlib.md5(raw).hexdigest()) for name, raw in data)

if st.button('Ghép ảnh', type='primary'):
    if not 2 <= len(data) <= 4:
        st.error('Hãy chọn từ 2 đến 4 ảnh.')
    else:
        try:
            images = [cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                      for name, raw in data]
            with st.spinner('Đang xử lý...'):
                result = stitch(images)
            ok, png = cv2.imencode('.png', result.panorama)
            if not ok:
                raise ValueError('Không xuất được ảnh PNG.')
            st.session_state['result'] = (key, result, png.tobytes())
        except (ValueError, np.linalg.LinAlgError) as exc:
            st.session_state.pop('result', None)
            st.error(str(exc))

saved = st.session_state.get('result')
if saved and saved[0] == key:
    result, png = saved[1:]
    st.success(f'Hoàn thành trong {result.seconds:.1f} giây')
    st.caption('Thứ tự: ' + ' → '.join(data[i][0] for i in result.order))
    st.image(result.panorama[:, :, ::-1], use_container_width=True)
    st.download_button('Tải ảnh Panorama', png, 'panorama.png', 'image/png')
