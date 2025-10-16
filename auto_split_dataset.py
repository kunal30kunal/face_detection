import os
import shutil
import random

# Path to your current dataset
dataset_dir = "my_dataset"

# Loop through each person's folder
for person_name in os.listdir(dataset_dir):
    person_path = os.path.join(dataset_dir, person_name)
    if not os.path.isdir(person_path):
        continue

    # Create train/test subfolders
    train_dir = os.path.join(person_path, "train")
    test_dir = os.path.join(person_path, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get all image files
    images = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Shuffle for randomness
    random.shuffle(images)

    # 70/30 split
    split_index = int(0.7 * len(images))
    train_images = images[:split_index]
    test_images = images[split_index:]

    # Move images to train/test folders
    for img in train_images:
        shutil.move(os.path.join(person_path, img), os.path.join(train_dir, img))
    for img in test_images:
        shutil.move(os.path.join(person_path, img), os.path.join(test_dir, img))

    print(f"✅ {person_name}: {len(train_images)} train, {len(test_images)} test images")

print("\n🎉 Dataset split completed successfully!")
