import cv2
import os
import numpy as np
import streamlit as st
from deepface import DeepFace

# ==============================
# 🔧 Configuration
# ==============================
MODEL_DIR = "models"
DATASET_DIR = "my_dataset"

FACE_PROTO = os.path.join(MODEL_DIR, "deploy.prototxt")
FACE_MODEL = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
AGE_PROTO = os.path.join(MODEL_DIR, "age_gender", "age_deploy.prototxt")
AGE_MODEL = os.path.join(MODEL_DIR, "age_gender", "age_net.caffemodel")
GENDER_PROTO = os.path.join(MODEL_DIR, "age_gender", "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(MODEL_DIR, "age_gender", "gender_net.caffemodel")

AGE_LABELS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)',
              '(38-43)', '(48-53)', '(60-100)']
GENDER_LABELS = ['Male', 'Female']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# ==============================
# 🧩 Load models
# ==============================
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)


# ==============================
# 🧠 Detection (Age, Gender, Emotion)
# ==============================
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

        # Gender
        blob_gender = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        gender_net.setInput(blob_gender)
        gender_preds = gender_net.forward()
        gender = GENDER_LABELS[gender_preds[0].argmax()]

        # Age
        blob_age = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        age_net.setInput(blob_age)
        age_preds = age_net.forward()
        age = AGE_LABELS[age_preds[0].argmax()]

        # Emotion (DeepFace)
        try:
            temp_path = "temp_face.jpg"
            cv2.imwrite(temp_path, face)
            analysis = DeepFace.analyze(img_path=temp_path, actions=['emotion'], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion']
        except Exception:
            emotion = "Unknown"

        label = f"{gender}, {age}, {emotion}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


# ==============================
# 🧍‍♂️ Recognition (DeepFace)
# ==============================
def recognize_faces_deepface(frame):
    temp_path = "temp_recognition.jpg"
    cv2.imwrite(temp_path, frame)

    try:
        result = DeepFace.find(
            img_path=temp_path,
            db_path=DATASET_DIR,
            model_name="ArcFace",  # You can change to "VGG-Face" or "Facenet"
            enforce_detection=False
        )

        if len(result) > 0 and not result[0].empty:
            best_match = result[0].iloc[0]
            identity_path = best_match["identity"]
            person_name = os.path.basename(os.path.dirname(identity_path))
            cv2.putText(frame, f"{person_name}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unknown", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    except Exception as e:
        cv2.putText(frame, "Error", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        st.error(f"DeepFace error: {e}")
    return frame


# ==============================
# 📸 Capture New Faces
# ==============================
def capture_images(person_name):
    save_path = os.path.join(DATASET_DIR, person_name)
    os.makedirs(save_path, exist_ok=True)
    cap = cv2.VideoCapture(0)
    count = 0
    stframe = st.empty()
    st.info(f"📸 Capturing 20 images for {person_name}... Please face the camera.")

    while count < 20:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        img_name = os.path.join(save_path, f"{count+1}.jpg")
        cv2.imwrite(img_name, frame)
        count += 1

    cap.release()
    st.success(f"✅ Captured {count} images for {person_name}!")


# ==============================
# 🚀 Streamlit App
# ==============================
st.set_page_config(page_title="Face Detection & Recognition System", layout="wide")
st.title("🧠 Face Detection & Recognition System")

option = st.sidebar.selectbox("Choose Mode:", ["Detection", "Recognition", "Capture & Train"])

# ----- Detection -----
if option == "Detection":
    st.subheader("🔍 Face Detection (Age, Gender, Emotion)")
    if st.button("🎥 Start Webcam Detection"):
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            result = detect_face_attributes(frame)
            stframe.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        cap.release()
        cv2.destroyAllWindows()

# ----- Recognition -----
elif option == "Recognition":
    st.subheader("🧩 Face Recognition using DeepFace")
    if st.button("🎥 Start Recognition"):
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            result = recognize_faces_deepface(frame)
            stframe.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        cap.release()
        cv2.destroyAllWindows()

# ----- Capture -----
elif option == "Capture & Train":
    st.subheader("📸 Capture New Faces")
    person_name = st.text_input("Enter the person's name:")
    if person_name:
        if st.button("📷 Capture 20 Images"):
            capture_images(person_name)
