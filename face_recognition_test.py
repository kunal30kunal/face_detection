import cv2
import os
import numpy as np
import joblib
from deepface import DeepFace

# --- Paths ---
MODEL_DIR = "models"
FACE_PROTO = os.path.join(MODEL_DIR, "deploy.prototxt")
FACE_MODEL = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
RECOGNIZER_PATH = os.path.join(MODEL_DIR, "face_recognizer.yml")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.pkl")

# --- Load models ---
print("[INFO] Loading models...")
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(RECOGNIZER_PATH)
labels = joblib.load(LABELS_PATH)
labels = {int(k): v for k, v in labels.items()}

# --- Real-time detection and recognition ---
cap = cv2.VideoCapture(0)
print("\n🎥 Press 'Q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0), swapRB=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf < 0.6:
            continue

        x1, y1, x2, y2 = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (200, 200))

        label_id, confidence = recognizer.predict(gray)
        name = labels.get(label_id, "Unknown")

        # Emotion detection via DeepFace
        try:
            cv2.imwrite("temp_face.jpg", face)
            result = DeepFace.analyze(img_path="temp_face.jpg",
                                      actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
        except Exception:
            emotion = "Unknown"

        text = f"{name} ({int(confidence)}) | {emotion}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Real-Time Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

