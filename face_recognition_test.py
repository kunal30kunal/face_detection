import cv2
import numpy as np
import os

# -------------------------------
# LOAD MODELS
# -------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("models/face_recognizer.yml")
label_map = np.load("models/label_map.npy", allow_pickle=True).item()

# -------------------------------
# TEST ON IMAGE OR WEBCAM
# -------------------------------
test_img_path = "images/test1.jpg"  # change if needed

if os.path.exists(test_img_path):
    frame = cv2.imread(test_img_path)
else:
    print("🎥 Starting webcam...")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_gray = cv2.resize(roi_gray, (200, 200))
    label_id, confidence = recognizer.predict(roi_gray)

    name = label_map.get(label_id, "Unknown")
    confidence_text = f"{100 - confidence:.2f}%"

    color = (0, 255, 0) if confidence < 70 else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, f"{name} ({confidence_text})", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

cv2.imshow("Face Recognition", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
