import cv2
from ultralytics import YOLO

# Set up the external camera with OpenCV
cap = cv2.VideoCapture(0)  # 0 is typically the default camera. Change it to 1 or higher if using other cameras.

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

#define custom classes
#model.set_classes(["dice"])

while True:
    # Capture a frame from the external camera
    ret, frame = cap.read()

    # If the frame was not captured correctly, skip the loop iteration
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Run YOLO model on the captured frame and store the results
    results = model(frame)

    # Output the visual detection data, we will draw this on our camera preview window
    annotated_frame = results[0].plot()

    # Get inference time
    inference_time = results[0].speed['inference']
    fps = 1000 / inference_time  # Convert to milliseconds
    text = f'FPS: {fps:.1f}'

    # Define font and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = annotated_frame.shape[1] - text_size[0] - 10  # 10 pixels from the right
    text_y = text_size[1] + 10  # 10 pixels from the top

    # Draw the text on the annotated frame
    cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow("Camera", annotated_frame)

    # Exit the program if q is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
