import cv2
import os
import numpy as np

# -------------------------------
# PATH SETTINGS
# -------------------------------
dataset_dir = "my_dataset"  # ✅ updated
model_save_path = "models/face_recognizer.yml"

# -------------------------------
# LOAD IMAGES & LABELS
# -------------------------------
faces = []
labels = []
label_map = {}   # e.g. {0: "Kunal", 1: "Priya", 2: "Rahul"}

print("📦 Loading dataset...")

for idx, person_name in enumerate(os.listdir(dataset_dir)):
    person_folder = os.path.join(dataset_dir, person_name)
    if not os.path.isdir(person_folder):
        continue

    label_map[idx] = person_name

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, (200, 200))
        faces.append(img)
        labels.append(idx)

faces = np.array(faces)
labels = np.array(labels)

print(f"✅ Dataset loaded: {len(faces)} images, {len(label_map)} persons")

# -------------------------------
# TRAIN LBPH FACE RECOGNIZER
# -------------------------------
print("🧠 Training face recognizer...")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)

# Save model and label map
os.makedirs("models", exist_ok=True)
recognizer.save(model_save_path)
np.save("models/label_map.npy", label_map)

print("✅ Model trained and saved at:", model_save_path)
