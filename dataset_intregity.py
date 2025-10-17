import cv2
import os

DATASET_DIR = "my_dataset"

print("[INFO] Verifying dataset images...\n")

for person in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person)
    if not os.path.isdir(person_dir):
        continue

    for split in ["train", "test"]:
        split_dir = os.path.join(person_dir, split)
        if not os.path.exists(split_dir):
            continue

        for img_name in os.listdir(split_dir):
            img_path = os.path.join(split_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[❌] Cannot read: {img_path}")
            else:
                print(f"[✅] OK: {img_path}")

print("\n[INFO] Verification complete.")
