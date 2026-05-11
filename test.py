import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os

# Visualization function (from docs)
MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 2
FONT_THICKNESS = 2
TEXT_COLOR = (255, 0, 0)
phone_detected_time = None
alert_triggered = False

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
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)


    phone_found = False
    for detection in detection_result.detections:
        category = detection.categories[0]
        if category.category_name == "cell phone" and category.score > 0.5:
            phone_found = True
            break
    if phone_found:
        if phone_detected_time is None:
            phone_detected_time = time.time()
            print("phone detected")
    else:
        phone_detected_time = None
        alert_triggered = False

    current_time = time.time()
    if phone_detected_time is not None and (current_time - phone_detected_time) > 2.0:
        if not alert_triggered:
            print("lock in. put your phone down and study")
            alert_triggered = True

    image_copy = np.copy(rgb_frame)
    annotated_image = visualize(image_copy, detection_result)

    if phone_detected_time is not None:
        elapsed = current_time - phone_detected_time
        timer_text = f"phone detected: {elapsed:.1f}s"
        cv2.putText(annotated_image, timer_text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 255), 2)
    
    if alert_triggered:
        cv2.putText(annotated_image, "lock in!! Put your phone down and study", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
    bgr_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    
    cv2.imshow("Object Detection", bgr_annotated_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()