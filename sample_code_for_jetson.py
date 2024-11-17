import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import serial

py_serial = serial.Serial(  port = '/dev/ttyACM0',  baudrate=57600)
# Load the YOLOv5 model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
# 기존 코드
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

# 새로운 코드
import os
import sys
import torch

# YOLOv5 디렉토리를 시스템 경로에 추가
YOLOV5_DIR = '/home/jetson/Downloads/detection-robot/yolov5'
sys.path.append(YOLOV5_DIR)

from models.common import DetectMultiBackend
from utils.torch_utils import select_device

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

# 학습된 모델 경로 설정
trained_model = './runs/train/exp/weights/best.pt'  # 실제 모델 경로로 수정하세요

# 모델 로드
model = DetectMultiBackend(
    weights=trained_model,
    device=device,
    dnn=False,
    data=None,
)

# 모델을 평가 모드로 설정
model.eval()

# 워밍업
model.warmup(imgsz=(1, 3, 640, 480))  # 이미지 크기를 카메라 해상도에 맞춤

# Define COCO instance category names for label identification
# COCO_INSTANCE_CATEGORY_NAMES = [
#     'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#     'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
#     'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#     'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 
#     'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#     'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#     'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
#     'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#     'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
#     'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
#     'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
#     'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
# ]
# 새로운 클래스 정의
CUSTOM_CLASSES = ['ai', 'awear', 'imr', 'gist']
choose_labels = CUSTOM_CLASSES  # 모든 클래스를 탐지
extract_idx = list(range(len(CUSTOM_CLASSES)))  # 모든 클래스의 인덱스 (0,1,2,3)


# choose_labels = ['person']
# extract_idx = [i for i, label in enumerate(COCO_INSTANCE_CATEGORY_NAMES) if label in choose_labels]

# Set up the RealSense D455 camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
# Write your yolov5 depth scale here
depth_scale = 0.001

# run state
state = 0

# Main loop

while True:

    # Get the latest frame from the camera
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    # Convert the frames to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Convert the color image to grayscale
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # Convert the depth image to meters
    depth_image = depth_image * depth_scale

    # Detect objects using YOLOv5
    results = model(color_image)

    # Process the results
    for result in results.xyxy[0]:
        x1, y1, x2, y2, confidence, class_id = result

        if int(class_id) in extract_idx:
            # 객체의 중심 x좌표 계산
            center_x = (x1 + x2) / 2
            # 이미지의 중심 x좌표 (640x480 이미지이므로 중심은 320)
            image_center = 640 / 2  # 320

            # Calculate the distance to the object
            object_depth = np.median(depth_image[int(y1):int(y2), int(x1):int(x2)])

            # 픽셀 차이를 실제 거리로 변환
            # RealSense D455의 수평 FOV는 87도
            FOV_HORIZONTAL = 87  # degrees
            pixel_to_degree = FOV_HORIZONTAL / 640  # degree/pixel

            # 중심으로부터의 픽셀 차이
            pixel_difference = center_x - image_center

            # 각도 계산 (탄젠트 사용)
            # 음수: 왼쪽, 양수: 오른쪽
            angle = pixel_difference * pixel_to_degree

            # Get the object's class name
            # class_name = model.names[int(class_id)]
            class_name = CUSTOM_CLASSES[int(class_id)]

            # Create label with class name, distance, and center position
            label = f"{class_name}: {object_depth:.2f}m, angle: {angle:.1f}"

            # Draw a rectangle around the object
            cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (252, 119, 30), 2)

            # Draw the label
            cv2.putText(color_image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (252, 119, 30), 2)

            if state == 0:
                if angle <= -15:
                    command = 'a'
                    state += 1
                    print("state is update to ", state)
                elif angle >= 15:
                    command = 'd'
                    state += 1
                    print("state is update to ", state)
                else:
                    command = 'w'
                    state += 2
                    print("state is update to ", state)

            elif state == 1:
                if angle < 30 or angle > -30:
                    command = 'w'
                    state += 1
                    print("state is update to ", state)
                else:
                    continue

            elif state == 2:
                if angle >= 30:
                    command = 'sd'
                    state += 1
                    print("state is update to ", state)
                elif angle <= -30:
                    command = 'sa'
                    state += 1
                    print("state is update to ", state)
                else:
                    continue
            
            elif state == 3:
                command = 'w'
                state += 1
                print("state is update to ", state)
            
            else:
                continue

            py_serial.write(command.encode('utf-8'))
            print(command)

    # Show the image
    cv2.imshow("Color Image", color_image)

    # Break out of the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources

cv2.destroyAllWindows()
pipeline.stop()