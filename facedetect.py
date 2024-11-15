import cv2
import streamlit as st

def detect_faces(color_bgr, scale_factor, min_neighbors):
    # Chargement du classificateur en cascade pour la détection de visages
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialisation de la webcam
    cap = cv2.VideoCapture(0)

    # Variable pour enregistrer une image
    saved_frame = None

    while True:
        # Lecture des images de la webcam
        ret, frame = cap.read()

        if not ret:
            st.error("Impossible de lire la vidéo depuis la webcam.")
            break

        # Conversion de l'image en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Détection des visages
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

        # Dessins des rectangles autour des visages détectés
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)

        # Affichage des images avec les visages détectés
        cv2.imshow("Détection de visages", frame)

        # Sauvegarde de l'image
        saved_frame = frame.copy()

        # Sortie de la boucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libération des ressources
    cap.release()
    cv2.destroyAllWindows()

    # Retour à l'image sauvegardée
    return saved_frame

def app():
    st.title("Détection de visages avec l'algorithme Viola-Jones")
    st.markdown("""
    ### Instructions :
    1. Cliquez sur "Détecter les visages" pour activer la détection à partir de votre webcam.
    2. Ajustez les paramètres comme la **couleur des rectangles**, **scaleFactor**, et **minNeighbors** si nécessaire.
    3. Cliquez sur "Enregistrer l'image" pour sauvegarder le résultat si vous le souhaitez.
    4. Appuyez sur "q" pour quitter la détection en direct.
    """)

    # Choix de la couleur des rectangles
    color = st.color_picker("Choisissez la couleur des rectangles", "#00FF00")  # Vert par défaut
    color_bgr = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))  # Conversion en BGR

    # Ajustement scaleFactor et minNeighbors
    scale_factor = st.slider("Facteur d'échelle (scaleFactor)", 1.1, 2.0, 1.3, step=0.1)
    min_neighbors = st.slider("Nombre minimal de voisins (minNeighbors)", 1, 10, 5)

    # Détection des visages
    if st.button("Détecter les visages"):
        st.write("Appuyez sur 'q' pour arrêter la détection.")
        detected_image = detect_faces(color_bgr, scale_factor, min_neighbors)

        # Ajout de la fonctionnalité d'enregistrement
        if detected_image is not None:
            if st.button("Enregistrer l'image"):
                cv2.imwrite("image_detected.jpg", detected_image)
                st.success("Image sauvegardée sous le nom 'image_detected.jpg'")

if __name__ == "__main__":
    app()
