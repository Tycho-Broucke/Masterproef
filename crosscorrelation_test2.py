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

# Function to compute the orientation of a contour using image moments
def calculate_orientation(contour):
    # Compute the moments of the contour
    moments = cv2.moments(contour)
    # Calculate the orientation (angle) of the contour using the second moments
    if moments['mu02'] != 0:
        angle = 0.5 * np.arctan2(2 * moments['mu11'], moments['mu02'] - moments['mu20'])
        angle = np.degrees(angle)  # Convert the angle from radians to degrees
    else:
        angle = 0
    return angle

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

    # Apply Canny edge detection to both the reference and random images
    ref_edges = cv2.Canny(ref_thresh, 100, 200)  # Canny edge detection on the reference image
    rand_edges = cv2.Canny(rand_thresh, 100, 200)  # Canny edge detection on the random image

    # Find contours in the Canny-edged images
    contours_ref, _ = cv2.findContours(ref_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_rand, _ = cv2.findContours(rand_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour in both the reference and random images (this should be the arrow contour)
    if contours_ref and contours_rand:
        contour_ref = max(contours_ref, key=cv2.contourArea)
        contour_rand = max(contours_rand, key=cv2.contourArea)

        # Calculate the orientation of both contours
        orientation_ref = calculate_orientation(contour_ref)
        orientation_rand = calculate_orientation(contour_rand)

        # Calculate the difference in orientation between the reference and the random arrows
        orientation_diff = orientation_rand - orientation_ref
        if orientation_diff < 0:
            orientation_diff += 360  # Ensure positive angle difference

        # Print the orientation difference to the console
        print(f"Orientation Difference: {orientation_diff} degrees")

        # Draw the contours on the images
        ref_image_contours = cv2.cvtColor(ref_image, cv2.COLOR_GRAY2BGR)  # Convert reference image to BGR
        cv2.drawContours(ref_image_contours, contours_ref, -1, (0, 255, 0), 2)  # Draw contours on the reference image

        rand_image_contours = random_image.copy()  # Copy the random image to draw on it
        cv2.drawContours(rand_image_contours, contours_rand, -1, (0, 255, 0), 2)  # Draw contours on the random image

        # Overlay the orientation difference text on the random image
        cv2.putText(rand_image_contours, f"Orientation Diff: {orientation_diff:.2f} degrees", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # Display the reference and random images with contours and orientation difference
        cv2.imshow('Reference Arrow with Contours', ref_image_contours)
        cv2.imshow('Random Arrow with Contours and Orientation', rand_image_contours)
    
    # Break the loop on 'Esc' key press
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
