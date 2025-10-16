import cv2
import numpy as np
import os

# ================================
# 🧩 Step 1 — Setup paths
# ================================
FACE_PROTO = "models/deploy.prototxt"
FACE_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"

AGE_PROTO = "models/age_gender/age_deploy.prototxt"
AGE_MODEL = "models/age_gender/age_net.caffemodel"

GENDER_PROTO = "models/age_gender/gender_deploy.prototxt"
GENDER_MODEL = "models/age_gender/gender_net.caffemodel"

# ================================
# 🧠 Step 2 — Load models
# ================================
print("🔍 Checking model files...")
print("Face proto exists:", os.path.exists(FACE_PROTO))
print("Face model exists:", os.path.exists(FACE_MODEL))
print("Age proto exists:", os.path.exists(AGE_PROTO))
print("Age model exists:", os.path.exists(AGE_MODEL))
print("Gender proto exists:", os.path.exists(GENDER_PROTO))
print("Gender model exists:", os.path.exists(GENDER_MODEL))

print("\n📦 Loading models into memory...")
try:
    face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
    age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
    gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
    print("✅ Models loaded successfully!\n")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load models: {e}")

# ================================
# ⚙️ Step 3 — Define labels
# ================================
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# ================================
# 📸 Step 4 — Load input image
# ================================
img_path = "images/test(1).jpg"
print(f"📷 Loading image from: {img_path}")
image = cv2.imread(img_path)

if image is None:
    raise FileNotFoundError(f"❌ Could not find image at path: {img_path}")

(h, w) = image.shape[:2]
print(f"✅ Image loaded: {w}x{h}")

blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
                             1.0, (300, 300), (104.0, 177.0, 123.0))
face_net.setInput(blob)
detections = face_net.forward()

# ================================
# 🧩 Step 5 — Detect & Predict
# ================================
face_count = 0

for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.3:  # lower threshold to catch more faces
        face_count += 1
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        print(f"\n🧍 Face #{face_count} detected (confidence={confidence:.2f})")

        face = image[max(0, y1):min(y2, h - 1), max(0, x1):min(x2, w - 1)]

        if face.size == 0:
            print("⚠️ Skipped empty face region")
            continue

        print("📏 Face region shape:", face.shape)

        blob2 = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                      (78.4263, 87.7689, 114.8958),
                                      swapRB=False)

        # Gender prediction
        gender_net.setInput(blob2)
        gender_preds = gender_net.forward()
        print("🔹 Gender probabilities:", gender_preds)
        gender = GENDER_LIST[gender_preds[0].argmax()]

        # Age prediction
        age_net.setInput(blob2)
        age_preds = age_net.forward()
        print("🔹 Age probabilities:", age_preds)
        age = AGE_LIST[age_preds[0].argmax()]

        label = f"{gender}, {age}"
        print(f"✅ Prediction: {label}")

        # Draw box and label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# ================================
# 💾 Step 6 — Save output
# ================================
if face_count == 0:
    print("❌ No faces detected!")
else:
    print(f"\n✅ Total faces detected: {face_count}")

os.makedirs("outputs", exist_ok=True)
output_path = "outputs/age_gender_debug_output.jpg"
cv2.imwrite(output_path, image)
print(f"💾 Output saved at: {output_path}")

cv2.imshow("Age & Gender Detection - Debug", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
