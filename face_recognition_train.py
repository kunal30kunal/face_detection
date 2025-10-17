import cv2
import os
import numpy as np
import pickle

DATASET_DIR = "my_dataset"
features = []
labels = []
label_dict = {}
label_id = 0

print("dataset is loading")

for person_name in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue

    normalized_name = person_name.lower().replace(" ", "_").replace(".", "_")

    if normalized_name not in label_dict:
        label_dict[normalized_name] = label_id
        label_id += 1

    for split in ["train", "test"]:
        split_dir = os.path.join(person_dir, split)
        if not os.path.exists(split_dir):
            continue

        for img_name in os.listdir(split_dir):
            img_path = os.path.join(split_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features.append(gray)
            labels.append(label_dict[normalized_name])

print(f"[INFO] Total classes: {len(label_dict)}")
print("[INFO] Sample label mapping:", label_dict)


with open("label_dict.pkl", "wb") as f:
    pickle.dump(label_dict, f)


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(features, np.array(labels))
recognizer.save("face_recognizer.yml")

print(" Model and labels saved.")


