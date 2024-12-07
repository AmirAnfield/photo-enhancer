import streamlit as st
from PIL import Image
import cv2
import numpy as np

def detect_faces(image):
    """Détecte les visages dans une image."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return image

st.title("Amélioration des Photos avec IA")
st.write("Télécharge une photo pour détecter ou modifier un visage !")

uploaded_file = st.file_uploader("Choisis une photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Photo Importée", use_column_width=True)

    if st.button("Détecter les Visages"):
        st.write("Détection en cours...")
        image_array = np.array(image)
        result_image = detect_faces(image_array)
        st.image(result_image, caption="Visages Détectés", use_column_width=True)
