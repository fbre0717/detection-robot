import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import serial

py_serial = serial.Serial(  port = '/dev/ttyACM0',  baudrate=57600)
# Load the YOLOv5 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

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


choose_labels = ['person']
extract_idx = [i for i, label in enumerate(COCO_INSTANCE_CATEGORY_NAMES) if label in choose_labels]

# Set up the RealSense D455 camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
# Write your yolov5 depth scale here
depth_scale = 0.001

# run state
isrun = False
state = 0
last_command = 'n'

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
            class_name = model.names[int(class_id)]

            # Create label with class name, distance, and center position
            label = f"{class_name}: {object_depth:.2f}m, angle: {angle:.1f}"

            # Draw a rectangle around the object
            cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (252, 119, 30), 2)

            # Draw the label
            cv2.putText(color_image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (252, 119, 30), 2)

            # Print the object's class and distance
            # print(label, class_id)
            # if object_depth < 0.5:
            #     com = 'b'
            #     py_serial.write(com.encode('utf-8'))
            # else:
            #     com = 'a'
            #     py_serial.write(com.encode('utf-8'))

            # 좌/우 제어

            # new code
            # if state == 0:
            #     if angle <= -15:
            #         command = 'a'
            #     elif angle >= 15:
            #         command = 'd'
            #     else:
            #         command = 'w'
            #         state += 1
            # elif state == 1:
            #     if angle <= -30:
            #         command = 'a'
            #         state += 1
            #     elif angle >= 30:
            #         command = 'd'
            #         state += 1
            #     else:
            #         command = 'w'
            # elif state == 2:
            #     command = 'w'
            # print(command)


            if state == 0:
                if angle <= -15:
                    command = 'a'
                    state += 1
                elif angle >= 15:
                    command = 'd'
                    state += 1
                else:
                    command = 'w'
                    state += 2

            elif state == 1:
                if angle < 30 or angle > -30:
                    command = 'w'
                    state += 1
                else:
                    continue

            elif state == 2:
                if angle >= 30:
                    command = 'd'
                    state += 1
                elif angle <= -30:
                    command = 'a'
                    state += 1
                else:
                    continue
            
            elif state == 3:
                command = 'w'
                state += 1
            
            else:
                continue

            py_serial.write(command.encode('utf-8'))
            print(command)


            # if command != last_command:
            #     py_serial.write(command.encode('utf-8'))
            #     last_command = command
            #     print("Serial Serial Serial :", command)

            

            

    # Show the image
    cv2.imshow("Color Image", color_image)

    # Break out of the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources

cv2.destroyAllWindows()
pipeline.stop()