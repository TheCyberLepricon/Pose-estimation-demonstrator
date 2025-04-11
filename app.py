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
    global r_hand_x, r_hand_y, l_hand_x, l_hand_y

    if timestamp_ms < last_timestamp_ms:
        return
    last_timestamp_ms = timestamp_ms

    if detection_result.pose_landmarks:
        pose_landmarks = detection_result.pose_landmarks[0]
        left_wrist = pose_landmarks[15]
        right_wrist = pose_landmarks[16]

        frame_height, frame_width, _ = output_image.numpy_view().shape

        left_wrist_x_px = int((1 - left_wrist.x) * frame_width)
        left_wrist_y_px = int(left_wrist.y * frame_height)

        right_wrist_x_px = int((1 - right_wrist.x) * frame_width)
        right_wrist_y_px = int(right_wrist.y * frame_height)

        r_hand_x, r_hand_y = right_wrist_x_px, right_wrist_y_px
        l_hand_x, l_hand_y = left_wrist_x_px, left_wrist_y_px

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
    cap = cv2.VideoCapture(video_source)
    fullscreen = False
    prev_time = 0

    success, image = cap.read()
    if not success:
        print("Initial image capture failed.")
        exit()

    frame_height, frame_width, _ = image.shape

    score_block_x = random.randint(40, frame_width - 100)
    score_block_y = 0

    bomb_y = 20
    bomb_x = random.randint(40, frame_width - 100)

    score = 0
    cooldown_active = False
    cooldown_end_time = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Image capture failed.")
            break

        image = cv2.flip(image, 1)
        frame_height, frame_width, _ = image.shape

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)

        score_block_x1 = frame_width - score_block_x - 50
        score_block_x2 = frame_width - score_block_x
        score_block_y1 = score_block_y
        score_block_y2 = score_block_y + 50

        bomb_x1 = frame_width - bomb_x - 50
        bomb_x2 = frame_width - bomb_x
        bomb_y1 = bomb_y
        bomb_y2 = bomb_y + 50

        curr_time = time.time()
        fps = 1 / max(curr_time - prev_time, 1e-5)
        prev_time = curr_time

        if cooldown_active and curr_time >= cooldown_end_time:
            cooldown_active = False

        if to_window is not None:
            cv2.putText(to_window, f"FPS:{int(fps)}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(to_window, f"Score:{int(score)}", (300, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

            cv2.rectangle(to_window, (score_block_x1, score_block_y1), (score_block_x2, score_block_y2), (0, 255, 0), -1)
            cv2.rectangle(to_window, (bomb_x1, bomb_y1), (bomb_x2, bomb_y2), (0, 0, 255), -1)

            score_block_y += 5
            bomb_y += 5

            if score_block_y > frame_height:
                score_block_y = -80
                score_block_x = random.randint(40, frame_width - 100)
            if bomb_y > frame_height:
                bomb_y = -80
                bomb_x = random.randint(40, frame_width - 100)

            if not cooldown_active:
                if ((score_block_x1 <= r_hand_x <= score_block_x2 and score_block_y1 <= r_hand_y <= score_block_y2) or
                    (score_block_x1 <= l_hand_x <= score_block_x2 and score_block_y1 <= l_hand_y <= score_block_y2)):
                    score += 10
                    score_block_y = -80
                    score_block_x = random.randint(40, frame_width - 100)

                if ((bomb_x1 <= r_hand_x <= bomb_x2 and bomb_y1 <= r_hand_y <= bomb_y2) or
                    (bomb_x1 <= l_hand_x <= bomb_x2 and bomb_y1 <= l_hand_y <= bomb_y2)):
                    score = 0
                    cooldown_active = True
                    cooldown_end_time = curr_time + 2
                    bomb_y = -80
                    bomb_x = random.randint(40, frame_width - 100)

            if cooldown_active:
                remaining_time = int(cooldown_end_time - curr_time) + 1
                cv2.putText(to_window, f"COOLDOWN:", (250, 200), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 4)
                cv2.putText(to_window, f"{remaining_time}", (250, 250), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0))
                cv2.putText(to_window, "GAME OVER", (230, 150), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 3)

            cv2.namedWindow("MediaPipe Pose Landmark", cv2.WINDOW_KEEPRATIO)
            cv2.imshow("MediaPipe Pose Landmark", to_window)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('f'):
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty("MediaPipe Pose Landmark", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty("MediaPipe Pose Landmark", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
