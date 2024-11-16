import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import serial
import time
import os
from pathlib import Path


print("Starting program and initializing components...")

py_serial = serial.Serial(port='/dev/ttyACM0', baudrate=57600)
print("Serial connection established successfully")

# YOLOv5 모델 파일 경로 설정
YOLO_MODEL_PATH = "yolov5s.pt"  # 로컬 경로 지정

# 모델 로딩 최적화
print("Loading YOLOv5 model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 로컬에 저장된 모델이 있는지 확인
if os.path.exists(YOLO_MODEL_PATH):
    print("Loading model from local file...")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH, force_reload=False)
else:
    print("Downloading model for the first time...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    # 다음 실행을 위해 모델 저장
    torch.save(model.state_dict(), YOLO_MODEL_PATH)

model = model.to(device)
model.eval()  # 평가 모드로 설정하여 최적화
print("YOLOv5 model loaded successfully")




# Define COCO instance category names for label identification
COCO_INSTANCE_CATEGORY_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# choose_labels = ['person']
choose_labels = COCO_INSTANCE_CATEGORY_NAMES
extract_idx = [i for i, label in enumerate(COCO_INSTANCE_CATEGORY_NAMES) if label in choose_labels]
print(f"Tracking objects of types: {choose_labels}")

# Set up the RealSense D455 camera
print("Initializing RealSense camera...")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
print("RealSense camera initialized successfully")

depth_scale = 0.001
print(f"Depth scale set to: {depth_scale}")

frame_count = 0
start_time = time.time()

print("Starting main detection loop...\n")
print("Press 'q' to quit the program")

while True:
    frame_count += 1
    
    # Get the latest frame from the camera
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    
    if frame_count % 30 == 0:  # Print FPS every 30 frames
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        print(f"FPS: {fps:.2f}")

    # Convert the frames to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Convert the depth image to meters
    depth_image = depth_image * depth_scale

    # Detect objects using YOLOv5
    results = model(color_image)

    # Process the results
    detections_count = 0
    for result in results.xyxy[0]:
        x1, y1, x2, y2, confidence, class_id = result

        if int(class_id) in extract_idx:
            detections_count += 1
            # Calculate the distance to the object
            object_depth = np.median(depth_image[int(y1):int(y2), int(x1):int(x2)])

            # Get the object's class name
            class_name = model.names[int(class_id)]

            # Create label with class name and distance
            label = f"{class_name}: {object_depth:.2f}m"

            # Draw a rectangle around the object
            cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (252, 119, 30), 2)

            # Draw the label
            cv2.putText(color_image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (252, 119, 30), 2)

            # Print detection information
            print(f"Detection {detections_count}: {label}, Confidence: {confidence:.2f}")
            
            if object_depth < 0.5:
                com = 'b'
                py_serial.write(com.encode('utf-8'))
                print(f"Object too close! Sending command: {com}")
            else:
                com = 'a'
                py_serial.write(com.encode('utf-8'))
                print(f"Object at safe distance. Sending command: {com}")

    if detections_count == 0 and frame_count % 30 == 0:  # Print only every 30 frames when no detections
        print("No objects detected in current frame")

    # Show the image
    cv2.imshow("Color Image", color_image)

    # Break out of the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nProgram terminated by user")
        break

# Release resources
print("\nCleaning up resources...")
cv2.destroyAllWindows()
pipeline.stop()
py_serial.close()
print("Program ended successfully")