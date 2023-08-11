# Import necessary libraries
from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import pyttsx3
import numpy as np
from collections import defaultdict

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Open the webcam
cap = cv2.VideoCapture(0)

# Load the YOLO model
model = YOLO("../Yolo-Weights/yolov8n.pt")

# List of class names for object detection
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Create a defaultdict to keep track of the count for each class
count = defaultdict(int)

# Camera calibration parameters
focal_length = 500  # Focal length of the camera in pixels
object_size = 0.1  # Size of the object in meters

# Ground truth bounding box coordinates
ground_truth = [(10, 10, 60, 60), (100, 100, 150, 150)]

# Timer variables
detection_interval = 20  # Detection interval in seconds
start_time = time.time()
prev_frame_time = start_time



while True:
    new_frame_time = time.time()

    # Check if it's time to perform object detection
    if new_frame_time - start_time >= detection_interval:
        start_time = new_frame_time  # Reset the start time for the next interval
        count.clear()  # Clear the count dictionary to reset the counter

        # Read a frame from the webcam
        success, img = cap.read()
        results = model(img, stream=True)
        detected_objects = []

        # Process the detection results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                class_name = classNames[cls]
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
                detected_objects.append(class_name)
                # Calculate distance
                distance = (object_size * focal_length) / h  # Distance estimation using similar triangles
                # Update the count for each detected class
                count[class_name] += 1

        # Convert the count dictionary to a formatted string for speech
        count_text = ", ".join([f"{v} {k}" for k, v in count.items()])

        engine.say(f"{count_text} detected")
        engine.say(f"The distance is: {distance:.2f} meters")  # Speak the distance
        engine.runAndWait()

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        print(fps)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('Q'):
        break

# Release the camera
cap.release()
# Destroy all open windows
cv2.destroyAllWindows()
