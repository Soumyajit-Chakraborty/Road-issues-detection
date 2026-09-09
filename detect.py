import cv2
from ultralytics import YOLO

# Load your local trained weights
model = YOLO('Road Yolo.pt')  # Replace with 'best.pt' if that is your filename

# Initialize local webcam (0 is typically the default built-in camera)
cap = cv2.VideoCapture(2)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting webcam feed... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLOv8 inference on current frame (higher conf prevents false positives)
    results = model.predict(source=frame, conf=0.20, verbose=False)

    # Annotate frame with bounding boxes
    annotated_frame = results[0].plot()

    # Display live video window
    cv2.imshow("YOLOv8 Live Detection", annotated_frame)

    # Press 'q' on your keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close window
cap.release()
cv2.destroyAllWindows()
