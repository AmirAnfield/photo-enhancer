import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("FaceSwap : Remplace un visage par un autre !")
st.write("Télécharge une photo modèle et une photo source pour échanger les visages.")

# Téléchargement des fichiers
uploaded_file1 = st.file_uploader("Photo Modèle (celle à modifier)", type=["jpg", "jpeg", "png"])
uploaded_file2 = st.file_uploader("Photo Source (ton visage)", type=["jpg", "jpeg", "png"])

def swap_faces(model_image, source_image):
    """
    Fonction pour échanger les visages entre deux images.
    Utilise OpenCV pour détecter et échanger les visages.
    """
    # Détecter les visages
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Convertir en grayscale
    gray_model = cv2.cvtColor(model_image, cv2.COLOR_BGR2GRAY)
    gray_source = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

    # Trouver les visages
    faces_model = face_cascade.detectMultiScale(gray_model, 1.1, 4)
    faces_source = face_cascade.detectMultiScale(gray_source, 1.1, 4)

    if len(faces_model) == 0 or len(faces_source) == 0:
        st.error("Aucun visage détecté dans l'une des photos.")
        return model_image

    # Utiliser le premier visage trouvé
    (x, y, w, h) = faces_model[0]
    (x2, y2, w2, h2) = faces_source[0]

    # Extraire les visages
    face_model = model_image[y:y+h, x:x+w]
    face_source = cv2.resize(source_image[y2:y2+h2, x2:x2+w2], (w, h))

    # Remplacer le visage
    model_image[y:y+h, x:x+w] = face_source

    return model_image

if uploaded_file1 and uploaded_file2:
    # Charger les images
    model_image = np.array(Image.open(uploaded_file1))
    source_image = np.array(Image.open(uploaded_file2))

    # Afficher les images importées
    st.image(model_image, caption="Photo Modèle", use_column_width=True)
    st.image(source_image, caption="Photo Source", use_column_width=True)

    # Bouton pour lancer le FaceSwap
    if st.button("Échanger les Visages"):
        st.write("Traitement en cours...")
        result_image = swap_faces(model_image, source_image)
        st.image(result_image, caption="Résultat : FaceSwap", use_column_width=True)
