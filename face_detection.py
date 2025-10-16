import cv2
import numpy as np
import os

# --- Paths ---
MODEL_PATH = "models"
IMAGE_PATH = "images/test(1).jpg"
OUTPUT_PATH = "outputs/face_detected.jpg"

# --- Load the pre-trained model ---
print("📦 Loading model...")
modelFile = os.path.join(MODEL_PATH, "res10_300x300_ssd_iter_140000.caffemodel")
configFile = os.path.join(MODEL_PATH, "deploy.prototxt")

net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
print("✅ Model loaded successfully!")

# --- Load input image ---
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise ValueError(f"❌ Could not read image at {IMAGE_PATH}")

(h, w) = image.shape[:2]

# --- Prepare input blob ---
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()

# --- Process detections ---
face_count = 0
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.6:
        face_count += 1
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{confidence*100:.1f}%"
        y = y1 - 10 if y1 - 10 > 10 else y1 + 10
        cv2.putText(image, text, (x1, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# --- Save and show result ---
os.makedirs("outputs", exist_ok=True)
cv2.imwrite(OUTPUT_PATH, image)
print(f"✅ Detected {face_count} faces. Output saved at: {OUTPUT_PATH}")

cv2.imshow("Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
