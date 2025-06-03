import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mon_modele_arfi.h5")
    return model

model = load_model()

st.set_page_config(page_title="Détection d’anomalies ARFI", layout="centered")
st.title("🔧 Détection automatique des anomalies")
st.write("Chargez une image d’un équipement industriel pour prédiction.")

uploaded_file = st.file_uploader("Choisissez une image", type=["png", "jpg", "jpeg"])

class_names = ["Sain ✅", "Défectueux ❌", "Usé ⚠️"]

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image chargée", use_column_width=True)

    img_resized = image.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.markdown("---")
    st.subheader("Résultat de la prédiction :")
    st.success(f"**{predicted_class}**")
    st.write(f"Confiance : **{confidence * 100:.2f}%**")
