import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('model.h5')

def preprocess_image(image):
    image = image.resize((128, 128))  # Resize to model's expected input size
    image = image.convert('L')  # Convert to grayscale if needed
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

def predict(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    print(f"Predictions: {predictions}")  # Debugging line
    age, gender = predictions[0]  # Assuming the model outputs age and gender in a tuple
    gender_label = "Male" if gender > 0.5 else "Female"
    return age, gender_label

# Streamlit UI
st.title("Age and Gender Prediction")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    try:
        age, gender_label = predict(image)
        st.write(f"Predicted Age: {age:.2f}")
        st.write(f"Predicted Gender: {gender_label}")
    except Exception as e:
        st.write(f"Error: {e}")
        print(f"Error: {e}")  # Debugging line
