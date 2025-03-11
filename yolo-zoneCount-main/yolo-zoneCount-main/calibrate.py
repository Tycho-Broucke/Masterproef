import cv2
import tkinter as tk
from PIL import Image, ImageTk

CAMERAS = ['/dev/camerazonea', '/dev/camerazoneb', '/dev/camerazonec']
CAMERA_WIDTH = [4096, 1920, 4096]
CAMERA_HEIGHT = [2160, 1080, 2160]
# Global list to store points
points = []
current_camera = 0

def clear_points():
    global points
    points = []
    update_image()


def button_print():
    global points
    print(f"Result for cam {CAMERAS[current_camera]}: {points}")
    clear_points()


def button_switchCam():
    global current_camera, cap
    current_camera = (current_camera + 1) % len(CAMERAS)
    print(f"Switched to camera {CAMERAS[current_camera]}")

    cap.release()
    cap = cv2.VideoCapture(CAMERAS[current_camera])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH[current_camera])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT[current_camera])
    
    update_image()

def convert_image(image):
    # Convert the image from BGR to RGB, then to a PIL image, then to a Tkinter image
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)
    return img_tk

def update_image():
    ret, img = cap.read()
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    
    # Draw all the points
    for point in points:
        cv2.circle(img, point, 5, (0, 0, 255), -1)

    img_tk = convert_image(img)
    
    # Update the label with the new image
    label.config(image=img_tk)
    label.image = img_tk


# Function to handle quit button
def quit_callback():
    cap.release()
    cv2.destroyAllWindows()
    root.quit()


# Create a window
root = tk.Tk()
frame = tk.Frame(root)
frame.pack()

# Create control buttons
button_update = tk.Button(frame, text='Update frame', command=update_image)
button_update.pack(side='left')
button_clear = tk.Button(frame, text='Clear', command=clear_points)
button_clear.pack(side='left')
button_print = tk.Button(frame, text='Print', command=button_print)
button_print.pack(side='left')
button_switch = tk.Button(frame, text='Switch camera', command=button_switchCam)
button_switch.pack(side='left')

# Create a quit button
quit_button = tk.Button(frame, text='Quit', command=quit_callback)
quit_button.pack(side='left')

# Create a label for the image
label = tk.Label(root)
label.pack()

cap = cv2.VideoCapture(CAMERAS[current_camera])
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH[current_camera])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT[current_camera])

def draw_circle(event):
    global points, img
    x, y = event.x, event.y
    points.append((x, y))  # Add the point to the list
    update_image()


# Bind the function to the window
label.bind('<Button-1>', draw_circle)

update_image()

root.mainloop()
cap.release()
cv2.destroyAllWindows()
