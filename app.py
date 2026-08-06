import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="centered"
)

model = tf.keras.models.load_model("brain_tumor_model.keras")

labels = [
    "Glioma Tumor",
    "Meningioma Tumor",
    "No Tumor",
    "Pituitary Tumor"
]

st.markdown("""
<style>
.main{
    background-color:#0E1117;
}
.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}
div[data-testid="stFileUploader"]{
    border:2px dashed #4F8BF9;
    border-radius:15px;
    padding:15px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align:center;color:#4F8BF9;font-size:52px;'>
🧠 Brain Tumor Classification
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<h3 style='text-align:center;color:#BDBDBD;font-weight:400;'>
Deep Learning based MRI Brain Tumor Detection using Convolutional Neural Networks
</h3>
""", unsafe_allow_html=True)

st.markdown("""
<p style='text-align:center;font-size:22px;'>
Upload an MRI brain scan image to classify the tumor using a trained Deep Learning model.
</p>
""", unsafe_allow_html=True)

st.warning(
    "⚠️ This application is developed for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis."
)

uploaded_file = st.file_uploader(
    "📤 Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded MRI Image",
        width=400
    )

    img = np.array(image)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = cv2.resize(img, (150, 150))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    st.success(f"### ✅ Prediction: {labels[predicted_class]}")

    st.metric("Confidence", f"{confidence:.2f}%")

    st.subheader("📊 Prediction Probabilities")

    probabilities = {
        labels[i]: float(prediction[0][i] * 100)
        for i in range(len(labels))
    }

    st.bar_chart(probabilities)

    st.subheader("📋 Class-wise Probability")

    for i in range(len(labels)):
        st.write(
            f"**{labels[i]} : {prediction[0][i]*100:.2f}%**"
        )

st.markdown("---")

st.markdown("""
<div style="text-align:center;color:gray;font-size:16px;padding-top:10px;">

<b>Developed by Ramya Vana</b><br><br>

TensorFlow • Keras • OpenCV • NumPy • Streamlit

</div>
""", unsafe_allow_html=True)