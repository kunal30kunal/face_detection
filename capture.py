import cv2
import os

# --- Configuration ---
DATASET_PATH = "my_dataset"
os.makedirs(DATASET_PATH, exist_ok=True)

# --- Input person name ---
person_name = input("Enter person's name: ").strip()
save_dir = os.path.join(DATASET_PATH, person_name)
os.makedirs(save_dir, exist_ok=True)

# --- Load Haar Cascade detector ---
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
img_count = 0
max_images = 20

print(f"\n[INFO] Capturing {max_images} images for {person_name}. Look at the camera.\n")

while True:
    success, frame = cap.read()
    if not success:
        print("❌ Could not access webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (200, 200))
        img_count += 1
        file_path = os.path.join(save_dir, f"{img_count}.jpg")
        cv2.imwrite(file_path, face)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{img_count}/{max_images}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Capturing Faces (Press Q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if img_count >= max_images:
        print("\n✅ Image capture complete.")
        break

cap.release()
cv2.destroyAllWindows()
