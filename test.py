import cv2
import mediapipe
import numpy as np

print("✅ OpenCV imported")
print("✅ MediaPipe imported")
print("✅ NumPy imported")
print("\nAll packages installed successfully!")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Display the frame
    cv2.imshow("Phone Detector", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()