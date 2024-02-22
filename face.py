import subprocess

subprocess.check_call(["python", '-m', 'pip', 'install', 'opencv-python'])
import cv2
import streamlit as st

# Assurez-vous que le chemin vers le fichier XML est correct
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(scaleFactor, minNeighbors, rectangle_color):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    while True:
        # Read the frames from the webcam
        ret, frame = cap.read()
        # Check if the frame is not empty
        if frame is not None:
            # Convert the frames to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect the faces using the face cascade classifier
            faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
            # Draw rectangles around the detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), rectangle_color, 2)
            # Display the frames
            cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)
            # Save the image
            cv2.imwrite('detected_faces.png', frame)
            # Exit the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Frame is empty")
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

def app():
    st.title("Détection de visage avec l'algorithme de Viola-Jones")
    st.write("Appuyez sur le bouton ci-dessous pour commencer à détecter les visages à partir de votre webcam")
    st.markdown("Les images avec les visages détectés seront enregistrées sur votre appareil.")
    st.markdown("Vous pouvez choisir la couleur des rectangles dessinés autour des visages détectés.")
    st.markdown("Vous pouvez également ajuster les paramètres scaleFactor et minNeighbors pour la détection des visages.")
    
    # Add a color picker to choose the rectangle color
    rectangle_color = st.color_picker("Choisissez la couleur des rectangles")
    # Convert the color to BGR
    rectangle_color = tuple(int(rectangle_color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
    
    # Add sliders to adjust the scaleFactor and minNeighbors parameters
    scaleFactor = st.slider("Ajustez le paramètre scaleFactor", min_value=1.0, max_value=2.0, value=1.3, step=0.1)
    minNeighbors = st.slider("Ajustez le paramètre minNeighbors", min_value=1, max_value=10, value=5)
    
    # Add a button to start detecting faces
    if st.button("Détecter les visages"):
        # Call the detect_faces function
        detect_faces(scaleFactor, minNeighbors, rectangle_color)

if __name__ == "__main__":
    app()
