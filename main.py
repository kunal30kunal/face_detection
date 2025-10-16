import cv2
import os
import numpy as np
from deepface import DeepFace

# --- Folder paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
AGE_GENDER_DIR = os.path.join(MODEL_DIR, "age_gender")

# --- Load Face Detector ---
face_proto = os.path.join(MODEL_DIR, "deploy.prototxt")
face_model = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
face_net = cv2.dnn.readNetFromCaffe(face_proto, face_model)

# --- Load Age & Gender Models ---
age_proto = os.path.join(AGE_GENDER_DIR, "age_deploy.prototxt")
age_model = os.path.join(AGE_GENDER_DIR, "age_net.caffemodel")
gender_proto = os.path.join(AGE_GENDER_DIR, "gender_deploy.prototxt")
gender_model = os.path.join(AGE_GENDER_DIR, "gender_net.caffemodel")

age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)

# --- Age and Gender categories ---
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['male', 'Female']

# --- Image source ---
image_path = os.path.join(BASE_DIR, "images", "test(1).jpg")
frame = cv2.imread(image_path)
if frame is None:
    raise FileNotFoundError(f"❌ Image not found: {image_path}")

(h, w) = frame.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))
face_net.setInput(blob)
detections = face_net.forward()

print(f"📷 Image loaded: {w}x{h}")
print(f"🧍 {detections.shape[2]} faces detected (may include low-confidence ones)")

# --- Loop through all detected faces ---
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > 0.7:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")

        # Boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        face = frame[y1:y2, x1:x2].copy()
        if face.size == 0:
            continue

        # --- Gender Prediction ---
        blob_gender = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.426, 87.769, 114.896))
        gender_net.setInput(blob_gender)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]

        # --- Age Prediction ---
        age_net.setInput(blob_gender)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]

        # --- Emotion Prediction (DeepFace) ---
        try:
            emotion_result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            emotion = emotion_result[0]['dominant_emotion']
        except Exception:
            emotion = "unknown"

        # --- Combine label ---
        label = f"{gender}, {age}, {emotion}"

        # --- Draw bounding box and label ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        print(f"🧠 Face {i+1}: {label} (confidence={confidence:.2f})")

# --- Display Output ---
cv2.imshow("Face, Age, Gender & Emotion Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()



# --- Image source ---
image_path = os.path.join(BASE_DIR, "images", "test(1).jpg")
frame = cv2.imread(image_path)
if frame is None:
    raise FileNotFoundError(f"❌ Image not found: {image_path}")

(h, w) = frame.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))
face_net.setInput(blob)
detections = face_net.forward()

print(f"📷 Image loaded: {w}x{h}")
print(f"🧍 {detections.shape[2]} faces detected (may include low-confidence ones)")

# --- Loop through all detected faces ---
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > 0.7:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")

        # Boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        face = frame[y1:y2, x1:x2].copy()
        if face.size == 0:
            continue

        # --- Gender Prediction ---
        blob_gender = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.426, 87.769, 114.896))
        gender_net.setInput(blob_gender)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]

        # --- Age Prediction ---
        age_net.setInput(blob_gender)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]

        # --- Emotion Prediction (DeepFace) ---
        try:
            emotion_result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            emotion = emotion_result[0]['dominant_emotion']
        except Exception:
            emotion = "unknown"

        # --- Combine label ---
        label = f"{gender}, {age}, {emotion}"

        # --- Draw bounding box and label ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        print(f"🧠 Face {i+1}: {label} (confidence={confidence:.2f})")

# --- Display Output ---
cv2.imshow("Face, Age, Gender & Emotion Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()


