import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# Global variable to switch between webcam and image
input_source = "image"  # Change to "image" to use an image file
image_path = "grid.jpg"  # Set the path to your image file

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Variables for polygon drawing
drawing = False
polygon_points = []

# Mouse callback function to draw polygon
def draw_polygon(event, x, y, flags, param):
    global drawing, polygon_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(polygon_points) < 4:
            polygon_points.append((x, y))
        if len(polygon_points) == 4:
            cv2.polylines(frame, [np.array(polygon_points, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
        else:
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

# Initialize video capture or load image based on input_source
if input_source == "webcam":
    cap = cv2.VideoCapture(0)  # 0 is typically the default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
    ret, frame = cap.read()
else:
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Could not load image.")
        exit()
    ret = True

if not ret:
    print("Error: Failed to capture image.")
    if input_source == "webcam":
        cap.release()
    cv2.destroyAllWindows()
    exit()

cv2.imshow("Camera", frame)
cv2.setMouseCallback("Camera", draw_polygon)

print("Draw a polygon with exactly 4 points and press 'ENTER' to proceed.")
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 13 and len(polygon_points) == 4:
        break
    elif len(polygon_points) >= 4:
        print("Polygon drawn with 4 points. Press 'ENTER' to proceed.")
    elif key == ord('q'):
        print("Exiting.")
        exit()

polygon_zone = sv.PolygonZone(polygon=np.array(polygon_points))
width, height = 640, 640
dst_points = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype='float32')
matrix = cv2.getPerspectiveTransform(np.array(polygon_points, dtype='float32'), dst_points)

while True:
    if input_source == "webcam":
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
    else:
        frame = cv2.imread(image_path)
        if frame is None:
            print("Error: Could not load image.")
            break

    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.polylines(annotated_frame, [np.array(polygon_points, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
    transformed_frame = cv2.warpPerspective(frame, matrix, (width, height))
    cv2.imshow("Transformed View", transformed_frame)

    inference_time = results[0].speed['inference']
    fps = 1000 / inference_time
    text = f'FPS: {fps:.1f}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = annotated_frame.shape[1] - text_size[0] - 10
    text_y = text_size[1] + 10
    cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Camera", annotated_frame)

    if cv2.waitKey(1) == ord("q"):
        break

if input_source == "webcam":
    cap.release()
cv2.destroyAllWindows()