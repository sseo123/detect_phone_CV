import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Visualization function (from docs)
MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 2
FONT_THICKNESS = 2
TEXT_COLOR = (255, 0, 0)

#function looks specifically for "cell phone" object to return
def visualize(image, detection_result) -> np.ndarray:
    for detection in detection_result.detections:
        category = detection.categories[0]
        category_name = category.category_name

        if category_name == "cell phone":
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

            probability = round(category.score, 2)
            result_text = category_name + ' (' + str(probability) + ')'
            text_location = (MARGIN + bbox.origin_x,MARGIN + ROW_SIZE + bbox.origin_y)
            cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
            
    return image

#create ObjectDetector
print("Loading model...")
base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)
print("✅ Model loaded successfully")

#open webcam 
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Failed to open webcam")
    exit(1)
print("✅ Webcam opened. Press 'q' to quit")

#for each frame run object detection, return what is found, draw boxes and lables, and show the image
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture frame")
        break
    
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Load image (we use webcam frame instead)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    #Detect objects
    detection_result = detector.detect(mp_image)
    #Visualize results
    image_copy = np.copy(rgb_frame)
    annotated_image = visualize(image_copy, detection_result)
    #Convert back to BGR for display
    bgr_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    
    cv2.imshow("Object Detection", bgr_annotated_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Program closed")