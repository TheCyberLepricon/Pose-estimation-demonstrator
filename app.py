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

model_path = "models/pose_landmarker_full.task"
video_source = 0

# Detection config
num_poses = 1
min_pose_detection_confidence = 0.5
min_pose_presence_confidence = 0.5
min_tracking_confidence = 0.5

# Hand positions
r_hand_x = 0
r_hand_y = 0
l_hand_x = 0
l_hand_y = 0

# Shared state
to_window = None
last_timestamp_ms = 0
pose_detected = False
last_pose_time = time.time()

# Countdown & Game Start Flags
countdown_active = False
countdown_start_time = None
game_started = False

# Draw pose landmarks
def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z) for l in pose_landmarks])
        mp_drawing.draw_landmarks(annotated_image, proto, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                                  mp_drawing.DrawingSpec(color=(0,0,255), thickness=2))
    return annotated_image

# Pose callback
def print_result(detection_result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global to_window, last_timestamp_ms, r_hand_x, r_hand_y, l_hand_x, l_hand_y, pose_detected, last_pose_time

    if timestamp_ms < last_timestamp_ms:
        return
    last_timestamp_ms = timestamp_ms

    rgb_np = output_image.numpy_view()
    frame_height, frame_width, _ = rgb_np.shape

    if detection_result.pose_landmarks:
        pose_detected = True
        last_pose_time = time.time()
        pose_landmarks = detection_result.pose_landmarks[0]
        if len(pose_landmarks) >= 21:
            l = pose_landmarks[19]
            r = pose_landmarks[20]
            l_hand_x = int(l.x * frame_width)
            l_hand_y = int(l.y * frame_height)
            r_hand_x = int(r.x * frame_width)
            r_hand_y = int(r.y * frame_height)

    to_window = cv2.cvtColor(draw_landmarks_on_image(rgb_np, detection_result), cv2.COLOR_RGB2BGR)

# Init model
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

# Main
with vision.PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(video_source)
    fullscreen = False
    prev_time = 0
    game_active = False

    success, image = cap.read()
    if not success:
        print("Initial image capture failed.")
        exit()

    frame_height, frame_width, _ = image.shape

    # Game vars
    score_block_x = random.randint(40, frame_width - 100)
    score_block_y = 0
    bomb_y = 20
    bomb_x = random.randint(40, frame_width - 100)
    score = 0
    cooldown_active = False
    cooldown_end_time = 0
    topscore = 0
    window_name = "MediaPipe Game"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Image capture failed.")
            break

        image = cv2.flip(image, 1)
        frame_height, frame_width, _ = image.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)

        curr_time = time.time()

        # Pose timeout check
        if time.time() - last_pose_time > 3:
            pose_detected = False
            game_active = False
            game_started = False
            countdown_active = False

        if not pose_detected:
            to_window = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            cv2.putText(to_window, "Waiting for player...", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
            score_block_y = 0
            bomb_y = 20
        else:
            # Start countdown if needed
            if not game_started and not countdown_active:
                countdown_active = True
                countdown_start_time = time.time()

        if to_window is not None:

            # ⏳ Countdown logic
            if countdown_active:
                elapsed = time.time() - countdown_start_time
                if elapsed < 3:
                    count = 3 - int(elapsed)
                    cv2.putText(to_window, f"Starting in {count}", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 4)
                    cv2.imshow(window_name, to_window)
                    key = cv2.waitKey(1) & 0xFF
                    continue
                elif elapsed < 3.5:
                    cv2.putText(to_window, "GO!", (230, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6)
                    cv2.imshow(window_name, to_window)
                    key = cv2.waitKey(1) & 0xFF
                    continue
                else:
                    countdown_active = False
                    game_started = True
                    score_block_y = 0
                    bomb_y = 20
                    game_active = True

            if game_active:
                # Game logic
                speed = 1 + ((score / 20) * 0.04)
                base_speed = 6
                block_speed = base_speed ** speed

                score_block_x1 = int(frame_width - score_block_x - 75)
                score_block_x2 = int(frame_width - score_block_x)
                score_block_y1 = int(score_block_y)
                score_block_y2 = int(score_block_y + 75)

                bomb_x1 = int(frame_width - bomb_x - 75)
                bomb_x2 = int(frame_width - bomb_x)
                bomb_y1 = int(bomb_y)
                bomb_y2 = int(bomb_y + 75)

                score_block_y += block_speed
                bomb_y += block_speed

                if score_block_y > frame_height:
                    score_block_y = -80
                    score_block_x = random.randint(40, frame_width - 100)
                    score = max(score - 20, 0)
                if bomb_y > frame_height:
                    bomb_y = -80
                    bomb_x = random.randint(40, frame_width - 100)

                # Collisions
                if not cooldown_active:
                    hit_block = (score_block_x1 <= r_hand_x <= score_block_x2 and score_block_y1 <= r_hand_y <= score_block_y2) or \
                                (score_block_x1 <= l_hand_x <= score_block_x2 and score_block_y1 <= l_hand_y <= score_block_y2)
                    hit_bomb = (bomb_x1 <= r_hand_x <= bomb_x2 and bomb_y1 <= r_hand_y <= bomb_y2) or \
                               (bomb_x1 <= l_hand_x <= bomb_x2 and bomb_y1 <= l_hand_y <= bomb_y2)

                    if hit_block:
                        score += 10
                        score_block_y = -80
                        score_block_x = random.randint(40, frame_width - 100)
                    if hit_bomb:
                        achieved_score = score
                        topscore = max(topscore, score)
                        score = 0
                        cooldown_active = True
                        cooldown_end_time = curr_time + 3.5
                        bomb_y = -80
                        bomb_x = random.randint(40, frame_width - 100)

                # Draw blocks and score
                cv2.rectangle(to_window, (score_block_x1, score_block_y1), (score_block_x2, score_block_y2), (0, 255, 0), -1)
                cv2.rectangle(to_window, (bomb_x1, bomb_y1), (bomb_x2, bomb_y2), (0, 0, 255), -1)
                cv2.putText(to_window, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(to_window, f"Top: {topscore}", (frame_width - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                if cooldown_active:
                    remaining = int(cooldown_end_time - curr_time) + 1
                    cv2.putText(to_window, "GAME OVER", (180, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                    cv2.putText(to_window, f"Cooldown: {remaining}", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
                    cv2.putText(to_window, f"Score: {achieved_score}", (180, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
                    
                    if curr_time >= cooldown_end_time:
                        cooldown_active = False
                        achieved_score = 0

            cv2.imshow(window_name, to_window)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('f'):
            fullscreen = not fullscreen
            mode = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, mode)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
