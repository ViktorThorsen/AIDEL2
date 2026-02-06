import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(page_title="AI Bildigenkänning", layout="centered")

st.title("Bildigenkänning med förtränad AI")
st.write("Denna app använder den förtränade modellen **MobileNetV2** (tränad på ImageNet) för att identifiera objekt i dina bilder.")

@st.cache_resource
def load_model():
    model = MobileNetV2(weights='imagenet')
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Kunde inte ladda modellen: {e}")
    st.stop()

file = st.file_uploader("Ladda upp en bild...", type=["jpg", "png", "jpeg"])

if file is not None:
    image = Image.open(file)
    st.image(image, caption='Din uppladdade bild', use_container_width=True)
    
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.LANCZOS)
    
    if image_resized.mode != "RGB":
        image_resized = image_resized.convert("RGB")
        
    img_array = np.asarray(image_resized)
    data = np.expand_dims(img_array, axis=0)
    
    data = preprocess_input(data)

    with st.spinner('AI:n tänker...'):
        prediction = model.predict(data)
        top_predictions = decode_predictions(prediction, top=5)[0]

    st.divider()
    
    best_guess = top_predictions[0]
    st.success(f"Bästa gissning: **{best_guess[1].replace('_', ' ').title()}**")
    st.write(f"Säkerhetsmarginal: **{best_guess[2]*100:.2f}%**")
    st.progress(int(best_guess[2] * 100))
    
    with st.expander("Visa topp 5 gissningar"):
        for _, name, score in top_predictions:
            st.write(f"**{name.replace('_', ' ').title()}**: {score*100:.2f}%")

else:
    st.info("Ladda upp en bild för att starta analysen.")