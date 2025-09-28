import streamlit as st
import cv2
from keras.models import model_from_json
import numpy as np
import tempfile

# Load the model
json_file = open("emotiondetector_new.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector_new.keras")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Helper function to preprocess the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 200.0

# Streamlit UI
st.title("Face Emotion Detection")
st.sidebar.title("Options")
option = st.sidebar.selectbox("Choose Input Source", ["Camera", "Upload Video"])

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

if option == "Camera":
    st.write("Using Camera for Emotion Detection")
    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False

    start_camera = st.button("Start Camera")
    stop_camera = st.button("Stop Camera")
    stframe = st.empty()

    if start_camera:
        st.session_state.camera_active = True

    if stop_camera:
        st.session_state.camera_active = False

    if st.session_state.camera_active:
        webcam = cv2.VideoCapture(0)
        try:
            while st.session_state.camera_active and webcam.isOpened():
                i, im = webcam.read()
                if not i:
                    st.write("Failed to access the camera.")
                    break
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(im, 1.3, 5)
                for (p, q, r, s) in faces:
                    image = gray[q:q + s, p:p + r]
                    cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
                    image = cv2.resize(image, (48, 48))
                    img = extract_features(image)
                    pred = model.predict(img)
                    prediction_label = labels[pred.argmax()]
                    cv2.putText(im, '% s' % (prediction_label), (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
                stframe.image(im, channels="BGR")
        finally:
            webcam.release()
            st.session_state.camera_active = False

elif option == "Upload Video":
    st.write("Upload a Video File for Emotion Detection")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        while video.isOpened():
            i, im = video.read()
            if not i:
                break
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(im, 1.3, 5)
            for (p, q, r, s) in faces:
                image = gray[q:q + s, p:p + r]
                cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
                image = cv2.resize(image, (48, 48))
                img = extract_features(image)
                pred = model.predict(img)
                prediction_label = labels[pred.argmax()]
                cv2.putText(im, '% s' % (prediction_label), (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
            stframe.image(im, channels="BGR")
        video.release()
