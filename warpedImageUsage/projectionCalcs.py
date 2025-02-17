import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import os

# Global variables
input_source = "image"  # Change to "webcam" to use the webcam
image_path = os.path.join(os.path.dirname(__file__), "Images/grid.png")  # Path to the image file
model = YOLO("yolov8n.pt")  # Load YOLOv8 model
drawing = False  # Flag for polygon drawing
polygon_points = []  # List to store polygon points

# Mouse callback function to draw polygon
def draw_polygon(event, x, y, flags, param):
    global drawing, polygon_points, frame
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

# Initialize video capture or load image
def initialize_input_source(input_source, image_path):
    if input_source == "webcam":
        cap = cv2.VideoCapture(0)  # 0 is typically the default camera
        if not cap.isOpened():
            print("Error: Could not open camera.")
            exit()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
        ret, frame = cap.read()
        return cap, frame, ret
    else:
        frame = cv2.imread(image_path)
        if frame is None:
            print("Error: Could not load image.")
            exit()
        return None, frame, True

# Draw polygon on the frame and wait for user input
def draw_polygon_interactively(frame):
    cv2.imshow("Camera", frame)
    cv2.setMouseCallback("Camera", draw_polygon)
    print("Draw a polygon with exactly 4 points and press 'ENTER' to proceed.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and len(polygon_points) == 4:  # ENTER key
            break
        elif len(polygon_points) >= 4:
            print("Polygon drawn with 4 points. Press 'ENTER' to proceed.")
        elif key == ord('q'):
            print("Exiting.")
            exit()

# Process the frame with YOLO and perspective transform
def process_frame(frame, model, polygon_points, width, height, matrix):
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.polylines(annotated_frame, [np.array(polygon_points, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
    transformed_frame = cv2.warpPerspective(frame, matrix, (width, height))
    cv2.imshow("Transformed View", transformed_frame)

    # Calculate and display FPS
    inference_time = results[0].speed['inference']
    fps = 1000 / inference_time
    text = f'FPS: {fps:.1f}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = annotated_frame.shape[1] - text_size[0] - 10
    text_y = text_size[1] + 10
    cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Camera", annotated_frame)

# Main function
def main():
    global frame, polygon_points

    # Initialize input source (webcam or image)
    cap, frame, ret = initialize_input_source(input_source, image_path)
    if not ret:
        print("Error: Failed to capture image.")
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        exit()

    # Draw polygon interactively
    draw_polygon_interactively(frame)

    # Create PolygonZone and perspective transform matrix
    polygon_zone = sv.PolygonZone(polygon=np.array(polygon_points))
    width, height = 640, 640
    dst_points = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype='float32')
    matrix = cv2.getPerspectiveTransform(np.array(polygon_points, dtype='float32'), dst_points)

    # Main loop for processing frames
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

        # Process the frame
        process_frame(frame, model, polygon_points, width, height, matrix)

        # Exit on 'q' key press
        if cv2.waitKey(1) == ord("q"):
            break

    # Release resources
    if input_source == "webcam":
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()