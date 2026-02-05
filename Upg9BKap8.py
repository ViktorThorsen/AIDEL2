import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image, ImageOps
import requests
import numpy as np

st.set_page_config(page_title="CIFAR-100 Classifier", layout="centered")

st.title("Bildigenkänning med CNN (CIFAR-100)")
st.write("Ladda upp en bild så gissar modellen vad det är!")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('cifar100_model.keras')
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Kunde inte ladda modellen. Felmeddelande: {e}")
    st.stop()

URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
class_names = requests.get(URL).json()

file = st.file_uploader("Välj en bild (jpg, png)...", type=["jpg", "png", "jpeg"])

if file is not None:
    image = Image.open(file)
    st.image(image, caption='Uppladdad bild', use_container_width=True)
    
    size = (224, 224) 
    image_resized = ImageOps.fit(image, size, Image.LANCZOS)
    
    if image_resized.mode != "RGB":
        image_resized = image_resized.convert("RGB")
        
    img_array = np.asarray(image_resized)
    
    data = np.expand_dims(img_array, axis=0)
    data = preprocess_input(data)

    with st.spinner('Modellen analyserar bilden...'):
        prediction = model.predict(data)
    
    predicted_class_index = np.argmax(prediction[0])
    
    if predicted_class_index < len(class_names):
        predicted_class_name = class_names[predicted_class_index]
        confidence = np.max(prediction[0]) * 100

        st.divider()
        st.success(f"Gissning: **{predicted_class_name.replace('_', ' ').title()}**")
        st.write(f"Säkerhet: **{confidence:.2f}%**")
        st.progress(min(int(confidence), 100))
        
        with st.expander("Visa topp 5 gissningar"):
            top_5_indices = np.argsort(prediction[0])[-5:][::-1]
            for i in top_5_indices:
                if i < len(class_names):
                    name = class_names[i].replace('_', ' ').title()
                    score = prediction[0][i] * 100
                    st.write(f"**{name}**: {score:.2f}%")
    else:
        st.warning(f"Error")