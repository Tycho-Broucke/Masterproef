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

# Define the destination points for perspective transform (3:2 aspect ratio)
height = 640  # Set height 
width = int(height)  # Calculate width to maintain 3:2 aspect ratio
dst_points = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype='float32')

# Compute the perspective transform matrix
matrix = cv2.getPerspectiveTransform(np.array(polygon_points, dtype='float32'), dst_points)

# Function to check if a point is inside the polygon
def is_point_inside_polygon(point, polygon):
    """
    Check if the point is inside the polygon using OpenCV's pointPolygonTest.
    
    Args:
    - point: A tuple (x, y) representing the point to check.
    - polygon: A list of points representing the polygon, e.g., [(x1, y1), (x2, y2), ...].
    
    Returns:
    - True if the point is inside the polygon.
    - False if the point is outside the polygon.
    """
    # Convert the polygon to a NumPy array
    polygon = np.array(polygon, dtype=np.int32)
    
    # Use OpenCV's pointPolygonTest to check if the point is inside the polygon
    result = cv2.pointPolygonTest(polygon, point, False)
    
    # Return True if the point is inside (result > 0), False if outside (result < 0)
    return result >= 0  # True if inside or on the polygon, False if outside

# Function to find the closest object in a list of points
def find_closest_point(points, reference_point):
    closest_point = None
    min_distance = float('inf')
    
    for point in points:
        dist = np.linalg.norm(np.array(point) - np.array(reference_point))
        if dist < min_distance:
            min_distance = dist
            closest_point = point
    return closest_point

# Start video stream and apply YOLO detection
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Initialize the transformed_frame in every iteration, even if no transformation occurs
    transformed_frame = frame.copy()

    # Run YOLO model on the captured frame
    results = model(frame)

    # Filter detections to persons, bottles, and cups (class 0, 39, 41)
    person_bottle_cup_boxes = [detection for detection in results[0].boxes if detection.cls in [0, 39, 41]]  # class 0 = "person", 39 = "bottle", 41 = "cup"

    # Output the visual detection data for persons, bottles, and cups
    annotated_frame = frame.copy()

    # Create a list to store transformed dots
    transformed_dots = []
    cups = []
    bottles = []

    for detection in person_bottle_cup_boxes:
        # Get the bounding box coordinates (x1, y1, x2, y2)
        x1, y1, x2, y2 = detection.xyxy[0].tolist()

        # Set bounding box and dot color based on the class
        if detection.cls == 41:  # Class 41 = cup
            box_color = (0, 0, 255)  # Red color for cups
            dot_color = (0, 0, 255)  # Red color for cup dots
            cups.append((int((x1 + x2) / 2), int(y2)))  # Save the center bottom of the cup
        elif detection.cls == 39:  # Class 39 = bottle
            box_color = (0, 255, 0)  # Green color for bottles
            dot_color = (0, 255, 0)  # Green color for bottle dots
            bottles.append((int((x1 + x2) / 2), int(y2)))  # Save the center bottom of the bottle

        # Draw the bounding box on the annotated frame
        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)

        # Calculate the bottom middle point of the bounding box
        bottom_middle_x = int((x1 + x2) / 2)
        bottom_middle_y = int(y2)
        dot = (bottom_middle_x, bottom_middle_y)

        # Draw the original dot on the annotated frame
        cv2.circle(annotated_frame, dot, 5, dot_color, -1)

        # Check if the dot is inside the polygon
        if is_point_inside_polygon(dot, polygon_points):
            # If the dot is inside, apply perspective transform to the coordinate
            transformed_dot = cv2.perspectiveTransform(np.array([dot], dtype=np.float32).reshape(-1, 1, 2), matrix)
            transformed_dot = transformed_dot[0][0]
            transformed_dot = (int(transformed_dot[0]), int(transformed_dot[1]))  # Convert to integer
            
            # Append the transformed dot to the list
            transformed_dots.append((transformed_dot, dot_color))  # Store color along with the transformed dot

    # Create the transformed frame only if the polygon condition is met
    transformed_frame = cv2.warpPerspective(frame, matrix, (width, height))

    # Draw each transformed dot on the transformed frame
    for transformed_dot, dot_color in transformed_dots:
        cv2.circle(transformed_frame, transformed_dot, 5, dot_color, -1)  # Use the appropriate color for each dot

    # Find the closest cup and bottle for this frame
    if cups and bottles:
        closest_cup = find_closest_point(cups, bottles[0])  # Find closest cup to the first bottle
        transformed_cup = cv2.perspectiveTransform(np.array([closest_cup], dtype=np.float32).reshape(-1, 1, 2), matrix)
        transformed_cup = transformed_cup[0][0]

        closest_bottle = bottles[0]  # Assuming the first bottle is the reference point
        transformed_bottle = cv2.perspectiveTransform(np.array([closest_bottle], dtype=np.float32).reshape(-1, 1, 2), matrix)
        transformed_bottle = transformed_bottle[0][0]

        # Draw an arrow from bottle to cup
        cv2.arrowedLine(transformed_frame, (int(transformed_bottle[0]), int(transformed_bottle[1])),
                             (int(transformed_cup[0]), int(transformed_cup[1])), (0, 165, 255), 3)  # Orange color for the arrow

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

    # Display the original annotated frame (only persons, bottles, and cups will be annotated)
    cv2.imshow("Camera", annotated_frame)

    # Display the transformed frame with transformed dots and arrow
    cv2.imshow("Transformed View", transformed_frame)

    # Exit the program if q is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
