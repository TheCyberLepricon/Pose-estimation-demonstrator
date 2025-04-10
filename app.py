import cv2
import numpy as np
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

model_path = "models/pose_landmarker_full.task"
video_source = 0

num_poses = 1
min_pose_detection_confidence = 0.5
min_pose_presence_confidence = 0.5
min_tracking_confidence = 0.5


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
    if timestamp_ms < last_timestamp_ms:
        return
    last_timestamp_ms = timestamp_ms
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
    y = 0
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
        
        # Moving object adding to screen
        x1, y1 = 100, y
        x2, y2 = 200, y + 100
        
        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        if to_window is not None:
            cv2.putText(to_window, f"FPS:{int(fps)}", (10,30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(to_window, (x1, y1), (x2, y2), (0, 0, 0), 2)
            y += 5
            if y > to_window.shape[0]:
                y = 0
            cv2.namedWindow("MediaPipe Pose Landmark", cv2.WINDOW_NORMAL)
            cv2.imshow("MediaPipe Pose Landmark", to_window)

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