import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
@st.cache_resource
def load_trained_model():
    return load_model("agri_waste_quality_classifier.h5")

model = load_trained_model()

# Class labels
class_labels = [
    'Banana_Stems_Contaminated', 'Banana_Stems_Dry', 'Banana_Stems_Moisturized',
    'Coconut_shells_Dry', 'Maize_Stalks_Dry', 'Rice_Straw_Dry', 'Sugarcane_bagasse_Dry'
]

# Title
st.title("🌾 Agri Waste Quality Classifier")
st.write("Upload an image of agricultural waste to classify its quality.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    img = img.resize((224, 224))  # Update if your model expects a different size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"Prediction: **{predicted_class}**")
    st.info(f"Confidence: **{confidence * 100:.2f}%**")
