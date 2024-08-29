import streamlit as st
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
import os

# Define or import custom objects
def mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

custom_objects = {
    'mae': mae
}

# Define the path to your model file
model_path = 'model.h5'

@st.cache_resource(show_spinner=False)
def load_model():
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.error(f"Model file not found at: {model_path}")
        return None

model = load_model()

if model is None:
    st.stop()

st.title("Gender and Age Predictor")
st.write("Upload an image to predict the gender and age.")

# Image uploader
uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = load_img(uploaded_file, target_size=(200, 200))  # Adjust target size as per your model input
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0) / 255.0  # Normalize the image if required

    # Perform prediction
    prediction = model.predict(image)
    
    # Assuming the model has two outputs, one for gender and one for age
    gender_pred = "Male" if prediction[0][0] > 0.5 else "Female"
    age_pred = int(prediction[1][0])

    # Display results
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write(f"Predicted Gender: {gender_pred}")
    st.write(f"Predicted Age: {age_pred} years")

