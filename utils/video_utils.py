import numpy as np                
import cv2
import mediapipe as mp
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=True)

def detect_blink(landmarks):
    left_eye_top = landmarks[159]
    left_eye_bottom = landmarks[145]
    dist = abs(left_eye_top.y-left_eye_bottom.y)
    return dist<0.015

def detect_head_movement(prev_positions, current_nose):
    prev_positions.append(current_nose)
    if len(prev_positions) > 10:
        prev_positions.popleft()
    diffs = [abs(prev_positions[i] - prev_positions[i-1]) for i in range(1, len(prev_positions))]
    return np.mean(diffs) if diffs else 0.0

def is_facing_forward(landmarks):
    left_eye = landmarks[33]  # Approx left eye
    right_eye = landmarks[263]  # Approx right eye
    dx = abs(left_eye.x - right_eye.x)
    return dx > 0.20  # Simpler proxy for forward-facing

# posture detection
def detect_posture(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose_detector.process(rgb)
    
    if result.pose_landmarks:
        left_shoulder = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        nose = result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        
        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
        head_level = nose.y < left_shoulder.y
        upright = shoulder_diff < 0.05 and head_level
        return 1 if upright else 0
    return 0.5
