from deepface import DeepFace
import cv2
import os

# Load your image
# NOTE: Ensure the path "images/test(1).jpg" exists and contains a valid image of a face.
image_path = "images/test(1).jpg"
os.makedirs("outputs", exist_ok=True)
image = cv2.imread(image_path)

# Check if image loaded correctly
if image is None:
    print(f"Error: Could not load image from {image_path}. Please check the file path.")
else:
    try:
        # Run emotion analysis
        result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)

        # Extract dominant emotion
        # Assuming only one face is detected (result[0])
        emotion = result[0]['dominant_emotion']
        print(f"🧠 Detected Emotion: {emotion}")

        # Draw label on image
        # Note: Position (20, 40) is near the top left corner
        cv2.putText(image, f"Emotion: {emotion}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save and show output
        output_path = "outputs/emotion_output.jpg"
        cv2.imwrite(output_path, image)

        # Display the image
        cv2.imshow("Emotion Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred during DeepFace analysis or processing: {e}")



