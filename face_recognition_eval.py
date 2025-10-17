import os
import cv2
import pickle

DATASET_DIR = "my_dataset"

print("[INFO] Loading trained model and label data...")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_recognizer.yml")

with open("label_dict.pkl", "rb") as f:
    label_dict = pickle.load(f)

print("[INFO] Starting model evaluation...")

reverse_label_dict = {v: k for k, v in label_dict.items()}

correct = 0
total = 0

for person_name in os.listdir(DATASET_DIR):
    normalized_name = person_name.lower().replace(" ", "_").replace(".", "_")

    if normalized_name not in label_dict:
        print(f"[WARN] Skipping '{person_name}' — not found in label dictionary.")
        continue

    label = label_dict[normalized_name]
    test_dir = os.path.join(DATASET_DIR, person_name, "test")

    if not os.path.exists(test_dir):
        continue

    for img_name in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pred_label, confidence = recognizer.predict(gray)
        total += 1
        if pred_label == label:
            correct += 1

if total == 0:
    print("\n⚠️ No valid test images found. Please verify dataset structure.")
else:
    accuracy = (correct / total) * 100
    print(f"\n✅ Evaluation complete! Accuracy: {accuracy:.2f}%")

