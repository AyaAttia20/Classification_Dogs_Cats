import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("my_pet_model.keras")

model = load_model()
class_names = ["Cat", "Dog"]



st.sidebar.title("About 😺🐶")
st.sidebar.info("A simple image classifier that identifies if the image is of a **Cat** or **Dog**.")

st.title("Pet Image Classifier")
st.markdown("Upload an image and click **Classify** to see what the AI predicts.")


uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])
user_guess = st.radio("What do you think the image is?", options=["Cat", "Dog"], horizontal=True)


if st.button("🔍 Classify"):
    if uploaded_file is None:
        st.warning("Please upload an image before clicking classify.")
    else:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        # Predict
        with st.spinner("Classifying... Please wait..."):
            prediction = model.predict(img_array)

        predicted_index = 1 if prediction[0][0] > 0.5 else 0
        predicted_label = class_names[predicted_index]
        confidence = float(prediction[0][0]) if predicted_index == 1 else 1 - float(prediction[0][0])

        # Show result
        st.success(f"### 🤖 Prediction: **{predicted_label}** ({confidence:.2%} confidence)")
        st.info(f"👤 You guessed: **{user_guess}**")

        if predicted_label == user_guess:
            st.balloons()
            st.success("🎉 Great job! You guessed it right.")
        else:
            st.warning("🤔 Not quite! Try another one.")
