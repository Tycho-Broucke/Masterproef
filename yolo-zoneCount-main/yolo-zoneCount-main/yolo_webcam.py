import torch
import cv2 as cv
import numpy as np
import os
from IPython import display
import supervision as sv
import socketio
import atexit

from datetime import datetime

sio = socketio.Client()

# log library versions
print("supervision", sv.__version__)
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

HOME = os.getcwd()
print(HOME)

display.clear_output()

# Start example
model = torch.hub.load('ultralytics/yolov5', 'yolov5m6')

colors = sv.ColorPalette.DEFAULT

ZONES = ['A', 'B', 'C']

# Define the camera settings
# cam_port: the port of the camera
# width: the width of the frame
# height: the height of the frame
# polygon: the polygon of the zone, can be set manually or using calibrate.py
camera_settings = {
    "A": {
        "cam_port": "/dev/camerazonea",
        "width": 4096,
        "height": 2160,
        "polygon": [(1000, 0), (1000, 2160), (3000, 2160), (3000, 0)],
    },
    "B": {
        "cam_port": "/dev/camerazoneb",
        "width": 1920,
        "height": 1080,
        "polygon": [(500, 0), (500, 1080), (1500, 1080), (1500, 0)],
    },
    "C": {
        "cam_port": "/dev/camerazonec",
        "width": 4096,
        "height": 2160,
        "polygon": [(1000, 0), (1000, 2160), (3000, 2160), (3000, 0)],
    }
}

for zone, settings in camera_settings.items():
    # initiate polygon zone
    polygon = np.array(settings["polygon"])
    polygon_zone = sv.PolygonZone(polygon=polygon)
    settings["polygon_zone"] = polygon_zone
    settings["polygon"] = polygon

    # set up video capture
    # settings["cam"] = cv.VideoCapture(settings["cam_port"], cv.CAP_V4L2)
    settings["cam"] = cv.VideoCapture(settings["cam_port"])
    settings["cam"].set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    settings["cam"].set(cv.CAP_PROP_BUFFERSIZE, 1)
    settings["cam"].set(cv.CAP_PROP_FRAME_WIDTH, settings["width"])
    settings["cam"].set(cv.CAP_PROP_FRAME_HEIGHT, settings["height"])
    
    # Check if the webcam is opened correctly
    # if not settings["cam"].isOpened():
    #     raise IOError(f"Cannot open webcam \"{settings['cam_port']}\"")

if any(not settings["cam"].isOpened() for _, settings in camera_settings.items()):
    not_opened_cams = [settings["cam_port"] for _, settings in camera_settings.items() if not settings["cam"].isOpened()]
    raise IOError(f"Cannot open webcam(s) {not_opened_cams}")


# fucntion to count the people in the zones, this function is called by the count_people_event function
# it will return a dictionary with the zones and the number of people in the zones
# when called, all 3 pictues are taken and then processed one by one using yolov5
def count_people(timestamp):
    zone_detections = {}
    camera_frames = {}

    # Read from all cameras
    for zone, values in camera_settings.items():
        print(f"Reading from {zone}")
        # reading the input using the camera, takes 2 frames to overwrite buffer
        result, frame = values["cam"].read()
        result, frame = values["cam"].read()
        if result:
            camera_frames[zone] = frame
        else:
            camera_frames[zone] = None
    
    print("Done reading")

    # Start processing the frames
    for zone, values in camera_settings.items():
        print(f"Processing zone {zone}")
        frame = camera_frames[zone]
        if frame is None:
            # TODO: Send error message with socket
            continue

        frame = cv.flip(frame, 1)
        cv.imwrite(f'original_{zone}.jpg', frame)
        results = model(frame, size=1280)
        detections = sv.Detections.from_yolov5(results)
        mask = values["polygon_zone"].trigger(detections=detections)
        detections = detections[(detections.class_id == 0) & (detections.confidence > 0.2) & mask]

        confidence_levels = []
        for i in range(len(detections.confidence)):
            confidence_levels.append(str(f'{detections.confidence[i]:.2f}'))

        # annotate
        box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
        frame = box_annotator.annotate(scene=frame, detections=detections, labels=confidence_levels)
        frame = sv.draw_polygon(scene=frame, polygon=values["polygon"], color=sv.Color.red(), thickness=6)

        print(f"[{timestamp}] Detections in zone {zone}: {len(detections)}")

        # Set font, scale, color, and thickness
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        font_color = (0, 0, 0)  # RGB for black
        background_color = (0, 0, 255)  # RGB for red
        thickness = 4
        # Create a label with the length of detections
        label = "Detections: {}".format(len(detections))
        # Get the width and height of the label box
        (text_width, text_height) = cv.getTextSize(label, font, fontScale=font_scale, thickness=thickness)[0]
        # Set the text start position
        text_offset_x = frame.shape[1] - text_width - 40  # adjust the -20 as per your requirement
        text_offset_y = frame.shape[0] - 40  # adjust the -20 as per your requirement
        # Make the coords of the box with a small padding of 5 pixels
        box_coords = (
            (text_offset_x, text_offset_y), (text_offset_x + text_width + 5, text_offset_y - text_height - 5))
        cv.rectangle(frame, box_coords[0], box_coords[1], background_color, cv.FILLED)
        cv.putText(frame, label, (text_offset_x, text_offset_y), font, font_scale, font_color, thickness)

        # Save the result image, for debugging
        # only the last frame per zone is saved and will be overwritten when the next question is processed
        cv.imwrite(f'result_{zone}.jpg', frame)

        cv.destroyAllWindows()

        zone_detections[zone] = len(detections)  # save the number of detections in the zone dictionary
    return zone_detections


@sio.event
# Print a message when connected to the server
def connect():
    print('Connection established')


@sio.event
# function to handle the count_people event, this is used to start counting the people in the zones
# this is called by the quiz-app when the timer of a question ends, it gives a quizId and questionId
def count_people_event(data):
    print(data)
    quiz_id = data['quizId']
    question_id = data['questionId']
    print(f'Starting count for quiz {quiz_id} question {question_id}')
    count = count_people(datetime.now())  # start counting the people in the zones
    for z in ZONES:  # make sure all zones are in the count, we will use 3
        if z not in count:  # if a zone is not in the count, set it to 0
            count[z] = 0
    print(f'Count: {count}')  # print the results
    data['results'] = [count[z] for z in ZONES]
    sio.emit('count_people_answer', data)  # send the results back to the quiz-app


@sio.event
# Print a message when disconnected from the server
def disconnect():
    print('Disconnected from server')


# release the camera when is closing
def exit_handler():
    for _, values in camera_settings.items():
        values["cam"].release()


atexit.register(exit_handler)

# Start the socket.io client
if __name__ == '__main__':
    print('Attempting to connect to the Socket.IO server')
    sio.connect('http://127.0.0.1', retry=True)
    print('Connected to the Socket.IO server')
    sio.wait()
