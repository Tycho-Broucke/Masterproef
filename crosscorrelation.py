import cv2
import numpy as np

# Load the reference image and convert to grayscale
ref_image = cv2.imread('arrow.jpeg', cv2.IMREAD_GRAYSCALE)

# Check if reference image is loaded
if ref_image is None:
    print("Error: Reference image not found!")
    exit()

# Apply binary thresholding to the reference image
_, ref_thresh = cv2.threshold(ref_image, 150, 255, cv2.THRESH_BINARY)

# Open the video stream from the webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam, change it if you have multiple devices

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    # Capture a frame from the webcam
    ret, random_image = cap.read()

    # Check if the frame was captured correctly
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Convert the webcam frame to grayscale
    gray_frame = cv2.cvtColor(random_image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding to the webcam frame with a higher threshold value
    threshold_value = 128  # Increase the threshold value to better separate the arrow from the background
    _, rand_thresh = cv2.threshold(gray_frame, threshold_value, 255, cv2.THRESH_BINARY)

    # Perform Canny edge detection on both the thresholded reference image and the random image
    edges_ref = cv2.Canny(ref_thresh, 100, 200)  # Adjust the low and high thresholds for edge detection
    edges_rand = cv2.Canny(rand_thresh, 100, 200)

    # Find contours in the Canny edge-detected images
    contours_ref, _ = cv2.findContours(edges_ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_rand, _ = cv2.findContours(edges_rand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour is the arrow in both images
    contour_ref = max(contours_ref, key=cv2.contourArea)
    contour_rand = max(contours_rand, key=cv2.contourArea)

    # Step 1: Fit ellipses to the contours
    if len(contour_ref) >= 5 and len(contour_rand) >= 5:  # Minimum 5 points required to fit an ellipse
        ellipse_ref = cv2.fitEllipse(contour_ref)
        ellipse_rand = cv2.fitEllipse(contour_rand)
        
        # Extract the orientation angle of the ellipses
        angle_ref = ellipse_ref[2]  # Angle of the reference arrow
        angle_rand = ellipse_rand[2]  # Angle of the random arrow
        
        # If the angle is negative, normalize it to be positive
        angle_ref = angle_ref if angle_ref >= 0 else angle_ref + 180
        angle_rand = angle_rand if angle_rand >= 0 else angle_rand + 180
        
        # Step 2: Calculate the rotation difference
        rotation_diff = angle_rand - angle_ref
        
        # Step 3: Draw ellipses on the images for visualization
        cv2.ellipse(random_image, ellipse_rand, (0, 255, 0), 2)
        cv2.ellipse(ref_image, ellipse_ref, (0, 255, 0), 2)
        
        # Display the rotation difference on the frame
        cv2.putText(random_image, f"Rotation: {rotation_diff:.2f} degrees", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the random image with the detected ellipse and rotation
    cv2.imshow("Random Arrow - Orientation", random_image)

    # Check for 'q' key to break the loop and stop the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
