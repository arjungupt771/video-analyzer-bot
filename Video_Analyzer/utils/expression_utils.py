from collections import deque
import cv2
from utils.audio_utils import extract_audio, transcribe_and_analyze_fluency, analyze_voice_confidence
from utils.video_utils import detect_blink, detect_head_movement, is_facing_forward, detect_posture
import numpy as np                  
from deepface import DeepFace
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)


def analyze_confidence(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(frame_count // 30, 1)
    
    blink_count =0
    total_blink_frames=0
    head_positions = deque()
    head_movementv_values = []
    expression_scores = []
    gaze_scores = []
    posture_scores=[]

    for i in range(0, frame_count, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            analysis = DeepFace.analyze(rgb, actions=["emotion"], enforce_detection=False)
            emotion = analysis[0]["dominant_emotion"]
        except:
            emotion = "neutral"

        # Expression score
        expr_score = {"happy": 10, "neutral": 8, "surprise": 5, "sad": 3, "angry": 1, "fear": 2, "disgust": 1}
        expression_scores.append(expr_score.get(emotion, 5))

        # Gaze estimation
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            if detect_blink(face.landmark):
                blink_count +=1
            total_blink_frames += 1
            nose_x = face.landmark[1].x
            head_movement = detect_head_movement(head_positions, nose_x)
            head_movementv_values.append(head_movement)
            if is_facing_forward(face.landmark):
                gaze_scores.append(1)  # Looking forward
            else:
                gaze_scores.append(0)  # Not looking
        
        posture_score = detect_posture(frame)
        posture_scores.append(posture_score)

    cap.release()

    if not expression_scores:
        return "neutral", 0

    avg_expression = np.mean(expression_scores)
    avg_gaze = np.mean(gaze_scores) if gaze_scores else 0.5
    avg_posture = np.mean(posture_scores)
    avg_blink_rate = blink_count / total_blink_frames if total_blink_frames>0 else 0
    blink_score = 10 - (avg_blink_rate*100)
    blink_score=max(0, min(blink_score, 10))
    
    avg_head_movement = np.mean(head_movementv_values) if head_movementv_values else 0.05
    head_score = 10-(avg_head_movement*100)
    head_score = max(0, min(head_score, 10))
    
    audio_path = extract_audio(video_path)
    transcript, fluency_score = transcribe_and_analyze_fluency(audio_path)
    voice_score = analyze_voice_confidence(audio_path)

    confidence_score = round(
        0.35*avg_expression +
        0.2*avg_gaze*10 +
        0.2*voice_score +
        0.15*fluency_score+
        0.25*avg_posture*10+
        0.1*blink_score+
        0.1*head_score,2
    )
    return emotion, confidence_score, transcript