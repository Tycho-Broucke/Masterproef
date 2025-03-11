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

# Perform Canny edge detection on the thresholded reference image
ref_edges = cv2.Canny(ref_thresh, 50, 150)

# Open the video stream from the webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam, change it if you have multiple devices

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Capture a frame from the webcam
ret, random_image = cap.read()

# Check if the frame was captured correctly
if not ret:
    print("Error: Could not read frame from webcam.")
    exit()

# Convert the webcam frame to grayscale
gray_frame = cv2.cvtColor(random_image, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding to the webcam frame with a higher threshold value
threshold_value = 128  # Increase the threshold value to better separate the arrow from the background
_, rand_thresh = cv2.threshold(gray_frame, threshold_value, 255, cv2.THRESH_BINARY)

# Perform Canny edge detection on the thresholded webcam frame
rand_edges = cv2.Canny(rand_thresh, 50, 150)

# Display the preprocessed images (grayscale, thresholded, and edges detected)
cv2.imshow('Reference Arrow Grayscale', ref_image)
cv2.imshow('Reference Arrow Thresholded', ref_thresh)
cv2.imshow('Reference Arrow Edges', ref_edges)
cv2.imshow('Random Arrow Grayscale', gray_frame)
cv2.imshow('Random Arrow Thresholded', rand_thresh)
cv2.imshow('Random Arrow Edges', rand_edges)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Release the video capture object
cap.release()
