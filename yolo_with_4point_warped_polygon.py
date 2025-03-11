import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

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

# Variables for polygon drawing
drawing = False
polygon_points = []

# Mouse callback function to draw polygon
def draw_polygon(event, x, y, flags, param):
    global drawing, polygon_points
    if event == cv2.EVENT_LBUTTONDOWN:
        # Only allow 4 points to be drawn
        if len(polygon_points) < 4:
            polygon_points.append((x, y))
        if len(polygon_points) == 4:
            # Automatically close the polygon after the 4th point
            cv2.polylines(frame, [np.array(polygon_points, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
        else:
            # Draw the polygon in progress
            temp_image = frame.copy()
            for i in range(1, len(polygon_points)):
                cv2.line(temp_image, polygon_points[i - 1], polygon_points[i], (0, 255, 0), 2)
            if len(polygon_points) > 0:
                cv2.line(temp_image, polygon_points[-1], (x, y), (0, 255, 0), 2)
            cv2.imshow("Camera", temp_image)
    elif event == cv2.EVENT_LBUTTONUP:
        if len(polygon_points) < 4:
            cv2.polylines(frame, [np.array(polygon_points, np.int32)], isClosed=False, color=(0, 255, 0), thickness=2)
        cv2.imshow("Camera", frame)

# Capture a single frame to draw polygon
ret, frame = cap.read()

if not ret:
    print("Error: Failed to capture image.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Show the frame to allow polygon drawing
cv2.imshow("Camera", frame)
cv2.setMouseCallback("Camera", draw_polygon)

# Wait until user finishes drawing the 4-point polygon (press "ENTER")

# Flag to check if the message has already been printed
message_printed = False
print("Draw a polygon with exactly 4 points and press 'ENTER' to proceed.")
while True:
    key = cv2.waitKey(1) & 0xFF
    
    # Check if 4 points are drawn
    if len(polygon_points) == 4:
        # Only print the message once
        if not message_printed:
            print("Polygon drawn with 4 points. Press 'ENTER' to proceed.")
            message_printed = True

    # Break if ENTER key is pressed and 4 points are drawn
    if key == 13 and len(polygon_points) == 4:
        break
    # Exit if 'q' key is pressed
    elif key == ord('q'):
        print("Exiting.")
        break

# Once the polygon is drawn, create a PolygonZone
polygon_zone = sv.PolygonZone(polygon=np.array(polygon_points))

# Define the destination points for perspective transform (4 corners of the new screen)
width, height = 640, 640  # New screen size (destination size for the mapped points)
dst_points = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype='float32')

# Compute the perspective transform matrix
matrix = cv2.getPerspectiveTransform(np.array(polygon_points, dtype='float32'), dst_points)

# Now, start video stream and apply YOLO detection
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Run YOLO model on the captured frame
    results = model(frame)

    # Output the visual detection data, we will draw this on our camera preview window
    annotated_frame = results[0].plot()

    # Draw the polygon (polygon_zone) on the frame if it is detected
    cv2.polylines(annotated_frame, [np.array(polygon_points, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

    # Perform the perspective transform to map the points to the new screen
    try:
        transformed_frame = cv2.warpPerspective(frame, matrix, (width, height))
    except cv2.error as e:
        print(f"Error in perspective transform: {e}")
        continue  # Skip this frame and move to the next one

    # Show the transformed frame as the new screen
    cv2.imshow("Transformed View", transformed_frame)

    # Get inference time and calculate FPS
    inference_time = results[0].speed['inference']
    fps = 1000 / inference_time  # Convert to milliseconds
    text = f'FPS: {fps:.1f}'

    # Define font and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = annotated_frame.shape[1] - text_size[0] - 10  # 10 pixels from the right
    text_y = text_size[1] + 10  # 10 pixels from the top

    # Draw the FPS text on the annotated frame
    cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow("Camera", annotated_frame)

    # Exit the program if q is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
