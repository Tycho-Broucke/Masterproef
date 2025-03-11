import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import cv2
import torch  # Assuming PyTorch-based YOLO
import numpy as np

class PersonDetector(Node):
    def __init__(self):
        super().__init__('person_detector')
        self.publisher_ = self.create_publisher(Point, 'person_coordinates', 10)
        self.timer = self.create_timer(0.1, self.detect_and_publish)  # 10 Hz

        # Load YOLO model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.cap = cv2.VideoCapture(0)  # Change to your camera source

    def detect_and_publish(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning('Failed to capture frame')
            return

        # Run YOLO detection
        results = self.model(frame)
        detections = results.xyxy[0].cpu().numpy()  # Extract detections

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if int(cls) == 0 and conf > 0.5:  # Class 0 is 'person' in YOLO
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2

                msg = Point()
                msg.x = float(x_center)
                msg.y = float(y_center)
                msg.z = 0.0  # Use if needed

                self.publisher_.publish(msg)
                self.get_logger().info(f'Published: x={msg.x}, y={msg.y}')

                break  # Only publish the first detected person

def main(args=None):
    rclpy.init(args=args)
    node = PersonDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
