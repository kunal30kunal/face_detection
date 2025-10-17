import cv2
import os
import numpy as np
import streamlit as st
import joblib
from deepface import DeepFace

MODEL_DIR = "models"
DATASET_DIR = "my_dataset"
RECOGNIZER_PATH = os.path.join(MODEL_DIR, "face_recognizer.yml")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.pkl")

F_TXT = os.path.join(MODEL_DIR, "deploy.prototxt")
F_MOD = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
AGE_TXT = os.path.join(MODEL_DIR, "age_gender", "age_deploy.prototxt")
AGE_MOD = os.path.join(MODEL_DIR, "age_gender", "age_net.caffemodel")
GEN_TXT = os.path.join(MODEL_DIR, "age_gender", "gender_deploy.prototxt")
GEN_MODEL = os.path.join(MODEL_DIR, "age_gender", "gender_net.caffemodel")

AGE_PATTERN = ['(0-5)', '(6-8)', '(9-12)', '(15-20)', '(25-30)',
               '(30-40)', '(40-50)', '(60-100)']
GENDER_OPTION = ['Male', 'Female']
MEAN_VALUE = (78.4263377603, 87.7689143744, 114.895847746)

face_check = cv2.dnn.readNetFromCaffe(F_TXT, F_MOD)
age_check = cv2.dnn.readNetFromCaffe(AGE_TXT, AGE_MOD)
gender_check = cv2.dnn.readNetFromCaffe(GEN_TXT, GEN_MODEL)

if os.path.exists(RECOGNIZER_PATH) and os.path.exists(LABELS_PATH):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(RECOGNIZER_PATH)
    label_dict = joblib.load(LABELS_PATH)
else:
    recognizer, label_dict = None, {}


def detect_face(frame):
    h, w = frame.shape[:2]
    df = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                               MEAN_VALUE, swapRB=False)
    face_check.setInput(df)
    detection = face_check.forward()

    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence < 0.6:
            continue

        box = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype(int)

        # Skip invalid regions
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            continue

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        # Gender
        df_gender = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                          MEAN_VALUE, swapRB=False)
        gender_check.setInput(df_gender)
        gender_preds = gender_check.forward()
        gender = GENDER_OPTION[gender_preds[0].argmax()]

        # Age
        df_age = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                       MEAN_VALUE, swapRB=False)
        age_check.setInput(df_age)
        age_prediction = age_check.forward()
        age = AGE_PATTERN[age_prediction[0].argmax()]

        # Emotion
        try:
            temp_path = "temp_face.jpg"
            cv2.imwrite(temp_path, face)
            analysis = DeepFace.analyze(img_path=temp_path,
                                        actions=['emotion'],
                                        enforce_detection=False)
            emotion = analysis[0]['dominant_emotion']
        except Exception:
            emotion = "Unknown"

        info_text = f"{gender} | {age} | {emotion}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(60, 255, 60), thickness=2)
        text_position = (x1 + 5, max(y1 - 8, 15))
        cv2.putText(frame, info_text, text_position,
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.55,
                    color=(50, 255, 50), thickness=1, lineType=cv2.LINE_AA)

    return frame


def recognize_faces(frame):
    if recognizer is None or not label_dict:
        st.error("Sorry model not trained.")
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (a, b, g, h) in faces:
        roi_gray = gray[b:b+h, a:a+g]
        label_id, conf = recognizer.predict(roi_gray)
        name = next((k for k, v in label_dict.items() if v == label_id), "Unknown")

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (a, b), (a+g, b+h), color, 2)
        cv2.putText(frame, f"{name} ({conf:.1f})", (a, b - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


def capture_images(person_name):
    save_path = os.path.join(DATASET_DIR, person_name)
    os.makedirs(save_path, exist_ok=True)
    cap = cv2.VideoCapture(0)
    count = 0
    stframe = st.empty()
    st.info("Capturing images. Please face the camera...")

    while count < 20:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                      channels="RGB", use_container_width=True)
        img_name = os.path.join(save_path, f"{count+1}.jpg")
        cv2.imwrite(img_name, frame)
        count += 1

    cap.release()
    st.success(f"✅ Captured {count} images for {person_name}!")


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Face Detection & Recognition System", layout="wide")
st.markdown("""
    <style>
        body {
            background-color: #ffcccc;
            color: white;
        }
        .stApp {
            background-color: #ffcccc;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧠Face Detection & Recognition System")
option = st.sidebar.selectbox("Choose:",
                              ["Detection", "Recognition", "Capture yourself"])

# --------- DETECTION -----------
if option == "Detection":
    st.subheader("Face Detection (Age, Gender, Emotion)")
    if st.button("🎥 Webcam Detection"):
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                st.warning("⚠️ No frame detected.")
                continue
            frame = cv2.flip(frame, 1)
            result = detect_face(frame)
            if result is not None and result.size != 0:
                stframe.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                              channels="RGB", use_container_width=True)
        cap.release()
        cv2.destroyAllWindows()

# ---------  RECOGNITION -----------
elif option == "Recognition":
    st.subheader("Face Recognition (LBPH Model)")
    if st.button("Start Recognition"):
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                st.warning("⚠️ No frame detected.")
                continue
            frame = cv2.flip(frame, 1)
            result = recognize_faces(frame)
            if result is not None and result.size != 0:
                stframe.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                              channels="RGB", use_container_width=True)
        cap.release()
        cv2.destroyAllWindows()

# ---------  CAPTURE & TRAIN -----------
elif option == "Capture yourself":
    st.subheader("📸 Capture Images & Add New Person")
    person_name = st.text_input("Enter name:")

    if person_name:
        if st.button("📷 Capture 20 Images"):
            capture_images(person_name)

        if st.button("🚀 Update Model"):
            st.info("Please wait while retraining...")
            faces, labels = [], []
            label_map = {}
            current_label = 0

            for root, _, files in os.walk(DATASET_DIR):
                for file in files:
                    if file.lower().endswith((".jpg", ".png", ".jpeg")):
                        path = os.path.join(root, file)
                        label = os.path.basename(root)
                        if label not in label_map:
                            label_map[label] = current_label
                            current_label += 1
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            faces.append(img)
                            labels.append(label_map[label])

            if len(faces) > 0:
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                recognizer.train(faces, np.array(labels))
                recognizer.save(RECOGNIZER_PATH)
                joblib.dump(label_map, LABELS_PATH)
                st.success("✅ Model updated successfully!")
                st.info("Ready for recognition!")
                st.write(f"**Trained on {len(label_map)} persons.**")
            else:
                st.error("❌ No valid faces found for training.")
