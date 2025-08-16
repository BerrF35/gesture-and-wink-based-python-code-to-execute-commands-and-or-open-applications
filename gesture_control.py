import cv2
import mediapipe as mp
import os
import numpy as np
import time

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# MediaPipe Face Mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Helper functions for fingers
def finger_is_up(hand_landmarks, tip_id, pip_id, threshold=0.05):
    return (hand_landmarks.landmark[pip_id].y - hand_landmarks.landmark[tip_id].y) > threshold

def finger_is_down(hand_landmarks, tip_id, mcp_id, threshold=0.03):
    return (hand_landmarks.landmark[tip_id].y - hand_landmarks.landmark[mcp_id].y) > threshold

def spider_man_gesture(hand_landmarks, spacing_thresh=0.1, fold_thresh=0.05):
    index_up = finger_is_up(hand_landmarks, 8, 6)
    pinky_up = finger_is_up(hand_landmarks, 20, 18)
    middle_fold = finger_is_down(hand_landmarks, 12, 9, fold_thresh)
    ring_fold = finger_is_down(hand_landmarks, 16, 13, fold_thresh)
    tip_dist = abs(hand_landmarks.landmark[8].x - hand_landmarks.landmark[20].x)
    return (index_up and pinky_up and middle_fold and ring_fold and tip_dist > spacing_thresh)

def is_fist(hand_landmarks, fold_threshold=0.03):
    return all([
        finger_is_down(hand_landmarks, 4, 3, fold_threshold),
        finger_is_down(hand_landmarks, 8, 6, fold_threshold),
        finger_is_down(hand_landmarks, 12, 10, fold_threshold),
        finger_is_down(hand_landmarks, 16, 14, fold_threshold),
        finger_is_down(hand_landmarks, 20, 18, fold_threshold)
    ])

def all_fingers_up(hand_landmarks, threshold=0.05):
    return all([
        finger_is_up(hand_landmarks, 4, 3, threshold),
        finger_is_up(hand_landmarks, 8, 6, threshold),
        finger_is_up(hand_landmarks, 12, 10, threshold),
        finger_is_up(hand_landmarks, 16, 14, threshold),
        finger_is_up(hand_landmarks, 20, 18, threshold)
    ])

# Eye landmarks for wink detection (right eye)
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT = 263
RIGHT_EYE_RIGHT = 362

def eye_aspect_ratio(landmarks, top_id, bottom_id, left_id, right_id):
    top = landmarks.landmark[top_id]
    bottom = landmarks.landmark[bottom_id]
    left = landmarks.landmark[left_id]
    right = landmarks.landmark[right_id]
    vertical_dist = np.linalg.norm(np.array([top.x, top.y]) - np.array([bottom.x, bottom.y]))
    horizontal_dist = np.linalg.norm(np.array([left.x, left.y]) - np.array([right.x, right.y]))
    if horizontal_dist == 0:
        return 0
    return vertical_dist / horizontal_dist

# Gesture sequence state for screenshot (fist → open hand → fist)
gesture_state = 0
state_start_time = 0

wink_cooldown = 0
WINK_THRESHOLD = 0.2  # Adjust based on testing
WINK_COOLDOWN_TIME = 2  # seconds

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process hands
    hand_results = hands.process(img_rgb)

    # Process face
    face_results = face_mesh.process(img_rgb)

    now = time.time()

    # -- Wink Detection --
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]
        ear_right = eye_aspect_ratio(face_landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)
        if ear_right < WINK_THRESHOLD and now - wink_cooldown > WINK_COOLDOWN_TIME:
            wink_cooldown = now
            print("Wink detected — opening Snip & Sketch!")
            os.system("start ms-screenclip:")

    # -- Hand Gestures --
    if hand_results.multi_hand_landmarks:
        hand = hand_results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

        thumb_up = finger_is_up(hand, 4, 3)
        index_up = finger_is_up(hand, 8, 6)
        middle_up = finger_is_up(hand, 12, 10)
        ring_up = finger_is_up(hand, 16, 14)
        pinky_up = finger_is_up(hand, 20, 18)

        thumb_down = finger_is_down(hand, 4, 3)
        index_down = finger_is_down(hand, 8, 6)
        middle_down = finger_is_down(hand, 12, 10)
        ring_down = finger_is_down(hand, 16, 14)
        pinky_down = finger_is_down(hand, 20, 18)

        # More sensitive middle finger lock
        if middle_up and not (index_up or ring_up):
            print("Middle finger detected — locking screen!")
            os.system("rundll32.exe user32.dll,LockWorkStation")
            break

        # Open Chrome: Index finger only but stricter
        elif index_up and all([thumb_down, middle_down, ring_down, pinky_down]):
            print("Index finger detected — opening Chrome!")
            os.system("start chrome")
            break

        # Open Chrome: Spider-Man finger refined
        elif spider_man_gesture(hand):
            print("Spider-Man finger detected — opening Chrome!")
            os.system("start chrome")
            break

        # Screenshot sequence (fist → open hand → fist)
        if gesture_state == 0:
            if is_fist(hand):
                gesture_state = 1
                state_start_time = now
                print("Gesture step 1: Fist detected")
        elif gesture_state == 1:
            if all_fingers_up(hand):
                gesture_state = 2
                print("Gesture step 2: Open hand detected")
        elif gesture_state == 2:
            if is_fist(hand):
                if now - state_start_time <= 3:
                    print("Gesture complete: Opening Snip & Sketch for screenshot")
                    os.system("start ms-screenclip:")
                gesture_state = 0
        if gesture_state != 0 and now - state_start_time > 3:
            gesture_state = 0

    else:
        gesture_state = 0

    cv2.imshow("Gesture and Wink Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



