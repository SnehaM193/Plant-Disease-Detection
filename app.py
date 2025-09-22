import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os

# --- Load the trained model ---
# A more robust loading mechanism to handle potential file not found errors
# and different model formats.
model = None
model_path_keras = "leaf_disease_model_stable.keras"
model_path_h5 = "leaf_disease_model_stable.h5"

try:
    if os.path.exists(model_path_keras):
        model = tf.keras.models.load_model(model_path_keras, safe_mode=False)
    elif os.path.exists(model_path_h5):
        model = tf.keras.models.load_model(model_path_h5, safe_mode=False)
    else:
        st.error("Error: Model file 'leaf_disease_model_stable.keras' or 'leaf_disease_model_stable.h5' not found.")
        st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}. Please ensure the model file is not corrupted.")
    st.stop()

# --- Class names from your dataset ---
# This list must be in the exact same order as your dataset directories.
class_names = [
    'Pepper__bell___Bacterial_spot', 
    'Pepper__bell___healthy', 
    'Potato___Early_blight', 
    'Potato___Late_blight', 
    'Potato___healthy',
    'Tomato_Bacterial_spot', 
    'Tomato_Early_blight', 
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 
    'Tomato_Mosaic_virus', 
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two-spotted_spider_mite', 
    'Tomato__Target_Spot',
    'Tomato__Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato__healthy'
]

# --- Preprocessing function to match your model's input ---
def preprocess_image(image):
    img = image.resize((128, 128))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array

# --- Streamlit UI ---
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("🌱 Plant Disease Detection")
st.write("Upload a leaf image and the model will predict the disease.")
st.markdown("---")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.markdown("---")
    
    with st.spinner('Classifying the image...'):
        # Preprocess the image and get predictions
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        predicted_class_index = np.argmax(score)
        predicted_class = class_names[predicted_class_index]


        # Refined logic to extract leaf and disease names
        parts = predicted_class.replace('_', ' ').split(' ')
        if parts[0].lower() in ['pepper', 'potato', 'tomato']:
            leaf_name = parts[0]
            if parts[1].lower() == 'bell': # Special case for Pepper bell
                leaf_name = ' '.join(parts[0:2])
                disease_name = ' '.join(parts[2:])
            else:
                disease_name = ' '.join(parts[1:])
        else:
            leaf_name = "Unknown Leaf"
            disease_name = predicted_class.replace('_', ' ')

        # Display the results
        st.success(f"**Prediction:** {leaf_name} with **{disease_name}**")
      
