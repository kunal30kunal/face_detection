import cv2
import os
import numpy as np
import streamlit as st
from deepface import DeepFace

# ---------------------- MODEL PATHS ----------------------
MODEL_DIR = "models"
FACE_PROTO = os.path.join(MODEL_DIR, "deploy.prototxt")
FACE_MODEL = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

GENDER_PROTO = os.path.join(MODEL_DIR, "age_gender", "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(MODEL_DIR, "age_gender", "gender_net.caffemodel")

AGE_PROTO = os.path.join(MODEL_DIR, "age_gender", "age_deploy.prototxt")
AGE_MODEL = os.path.join(MODEL_DIR, "age_gender", "age_net.caffemodel")

# ---------------------- LABELS ----------------------
GENDER_LABELS = ['Male', 'Female']
AGE_LABELS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# ---------------------- LOAD MODELS ----------------------
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)

# ---------------------- DETECTION FUNCTION ----------------------
def detect_face_attributes(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), MODEL_MEAN_VALUES, swapRB=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.6:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype(int)
        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        # Gender prediction
        blob_gender = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        gender_net.setInput(blob_gender)
        gender_preds = gender_net.forward()
        gender = GENDER_LABELS[gender_preds[0].argmax()]

        # Age prediction
        blob_age = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        age_net.setInput(blob_age)
        age_preds = age_net.forward()
        age = AGE_LABELS[age_preds[0].argmax()]

        # Emotion detection using DeepFace
        try:
            temp_path = "temp_face.jpg"
            cv2.imwrite(temp_path, face)
            analysis = DeepFace.analyze(img_path=temp_path, actions=['emotion'], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion']
        except Exception:
            emotion = "Unknown"

        # Draw bounding box and label
        label = f"{gender}, {age}, {emotion}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame

# ---------------------- STREAMLIT UI ----------------------
st.set_page_config(page_title="Face Detection — Age, Gender & Emotion", layout="wide")
st.title("🧠 Real-Time Face Detection (DeepFace Enhanced)")
st.markdown("""
### NOTE:
For better result use good lighting
""")

if st.button("🎥 Start Webcam Detection"):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        result = detect_face_attributes(frame)
        stframe.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                      channels="RGB", use_container_width=True)  # ✅ Updated here

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
