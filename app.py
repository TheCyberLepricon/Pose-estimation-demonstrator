import random
import cv2
import numpy as np
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

model_path = "models/pose_landmarker_lite.task"
video_source = 0

num_poses = 1
min_pose_detection_confidence = 0.5
min_pose_presence_confidence = 0.5
min_tracking_confidence = 0.5

r_hand_x = 0
r_hand_y = 0
l_hand_x = 0
l_hand_y = 0

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z) for landmark in pose_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


to_window = None
last_timestamp_ms = 0


def print_result(detection_result: vision.PoseLandmarkerResult, output_image: mp.Image,
                 timestamp_ms: int):
    global to_window
    global last_timestamp_ms
    global r_hand_x
    global r_hand_y
    global l_hand_x
    global l_hand_y

    if timestamp_ms < last_timestamp_ms:
        return
    last_timestamp_ms = timestamp_ms
    if detection_result.pose_landmarks:
        pose_landmarks = detection_result.pose_landmarks[0]
        # Linkerpols (index 15) en rechterpols (index 16)
        left_wrist = pose_landmarks[15]
        right_wrist = pose_landmarks[16]

        frame_height, frame_width, _ = output_image.numpy_view().shape

        # Omzetten naar pixels
        left_wrist_x_px = int(left_wrist.x * frame_width)
        left_wrist_y_px = int(left_wrist.y * frame_height)

        right_wrist_x_px = int(right_wrist.x * frame_width)
        right_wrist_y_px = int(right_wrist.y * frame_height)

        # Toevoegen aan globaal variabel
        r_hand_x = right_wrist_x_px
        r_hand_y = right_wrist_y_px
        l_hand_x = left_wrist_x_px
        l_hand_y = left_wrist_y_px

        # print("pose landmarker result: {}".format(detection_result))
        to_window = cv2.cvtColor(
            draw_landmarks_on_image(output_image.numpy_view(), detection_result), cv2.COLOR_RGB2BGR)


base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
base_options=base_options,
running_mode=vision.RunningMode.LIVE_STREAM,
num_poses=num_poses,
min_pose_detection_confidence=min_pose_detection_confidence,
min_pose_presence_confidence=min_pose_presence_confidence,
min_tracking_confidence=min_tracking_confidence,
output_segmentation_masks=False,
result_callback=print_result
)

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    # Use OpenCV’s VideoCapture to start capturing from the webcam.
    cap = cv2.VideoCapture(video_source)
    
    # Create a loop to read the latest frame from the camera using VideoCapture#read()
    fullscreen = False
    prev_time = 0
    y_1 = 0
    y_2 = 20
    x_1 = random.randint(40,600)
    x_2 = random.randint(40,600)
    score = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Image capture failed.")
            break
    
        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)
        
        # Moving objects adding to screen
        x1_1, y1_1 = x_1, y_1
        x2_1, y2_1 = x_1 + 100, y_1 + 100
        
        x1_2, y1_2 = x_2, y_2
        x2_2, y2_2 = x_2 + 100, y_2 + 100
        
        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        if to_window is not None:
            cv2.putText(to_window, f"FPS:{int(fps)}", (10,30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(to_window, f"FPS:{int(score)}", (300,30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            cv2.rectangle(to_window, (x1_1, y1_1), (x2_1, y2_1), (0, 0, 255), -1)
            cv2.rectangle(to_window, (x1_2, y1_2), (x2_2, y2_2), (255, 0, 0), -1)
            y_1 += 5
            y_2 += 5
            if y_1 > to_window.shape[0]:
                y_1 = 0
                x_1 = random.randint(40,600)
            if y_2 > to_window.shape[0]:
                y_2 = 0
                x_2 = random.randint(40,600)
            cv2.namedWindow("MediaPipe Pose Landmark", cv2.WINDOW_NORMAL)
            cv2.imshow("MediaPipe Pose Landmark", to_window)
        if r_hand_x >= x1_1  or l_hand_x >= x1_1 and r_hand_x <= x2_1 or l_hand_x <= x2_1:
            if r_hand_y >= y1_1 or l_hand_y >= y1_1 and r_hand_y <= y2_1 or r_hand_y <= y2_1:
                cv2.rectangle(to_window, (300, 100), (350, 150), (0, 255, 0), -1)
                #Dit is een test
                score += 10
        if r_hand_x >= x1_2 or l_hand_x >= x1_2 and r_hand_x <= x2_2 or l_hand_x <= x2_2:
            if r_hand_y >= y1_2 or l_hand_y >= y1_2 and r_hand_y <= y2_2 or r_hand_y <= y2_2:
                cv2.rectangle(to_window, (300, 300), (350,350), (0, 255, 0), -1)
                #Dit is een test
                score = 0
                cv2.putText(to_window, f"GAME OVER", (300,300), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)

        key = cv2.waitKey(1) & 0xFF

        # Toggle fullscreen when "F" is pressed
        if key == ord('f'):
            fullscreen = not fullscreen  # Toggle state
            if fullscreen:
                cv2.setWindowProperty("MediaPipe Pose Landmark", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty("MediaPipe Pose Landmark", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        # Exit when "Q" is pressed
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()