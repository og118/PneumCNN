import numpy as np
from PIL import Image

import streamlit as st
from load_model import test

st.write("# PneumCNN")

img_buffer = st.file_uploader("Please upload a file")

if img_buffer:
    image = Image.open(img_buffer)
    img_array = np.array(image)

    st.image(img_buffer)
    res = test(img_array)

    if(res>0.1):
        st.write("## The Chest XRay is pneumonic")
    else:
        st.write("## The Chest XRay is normal")
    

# st.write(st)