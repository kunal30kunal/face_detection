import cv2
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -------------------------------
# Load trained model
# -------------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("models/face_recognizer.yml")
label_map = np.load("models/label_map.npy", allow_pickle=True).item()

# -------------------------------
# Load test data
# -------------------------------
test_dir = "my_dataset"  # ✅ updated
true_labels = []
pred_labels = []

print("📊 Evaluating on test images...")

for person_name in os.listdir(test_dir):
    person_folder = os.path.join(test_dir, person_name, "test")
    if not os.path.exists(person_folder):
        continue

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, (200, 200))

        # Predict
        label_id, confidence = recognizer.predict(img)
        pred_name = label_map.get(label_id, "Unknown")

        true_labels.append(person_name)
        pred_labels.append(pred_name)

# -------------------------------
# Evaluate Results
# -------------------------------
print("\n✅ Evaluation Results:")
print("Accuracy:", accuracy_score(true_labels, pred_labels) * 100, "%")
print("\nClassification Report:\n", classification_report(true_labels, pred_labels))
print("Confusion Matrix:\n", confusion_matrix(true_labels, pred_labels))
