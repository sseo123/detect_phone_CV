import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("✅ All imports successful")

# Visualization function (from docs)
MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)

def visualize(image, detection_result) -> np.ndarray:
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                         MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return image

# STEP 1: Import (already done above)

# STEP 2: Create ObjectDetector
print("Loading model...")
base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)
print("✅ Model loaded successfully")

# STEP 3: Open webcam (instead of loading image file)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Failed to open webcam")
    exit(1)

print("✅ Webcam opened. Press 'q' to quit")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture frame")
        break
    
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # STEP 3: Load image (we use webcam frame instead)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # STEP 4: Detect objects
    detection_result = detector.detect(mp_image)
    
    # STEP 5: Visualize results
    image_copy = np.copy(rgb_frame)
    annotated_image = visualize(image_copy, detection_result)
    
    # Convert back to BGR for display
    bgr_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    
    cv2.imshow("Object Detection", bgr_annotated_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Program closed")