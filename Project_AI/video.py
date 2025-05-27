import streamlit as st
import cv2
import subprocess
import librosa
import tempfile
import os
import whisper
from collections import deque
from deepface import DeepFace
import mediapipe as mp
import numpy as np
import pandas as pd
import re

# Setup MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
pose_detector = mp_pose.Pose(static_image_mode=True)

def extract_audio(video_path, audio_path = "temp_audio.wav"):
    command = f"ffmpeg -y -i {video_path} -vn -acodec pcm_s16le -ar 44100 -ac 1 {audio_path}"
    subprocess.call(command, shell=True)
    return audio_path

def transcribe_and_analyze_fluency(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    transcript = result["text"].lower()
    
    filler_words = ["um","uh","like", "you know"," i mean", "so", "actually", "basically"]
    total_words = len(transcript.split())
    filler_count = sum(len(re.findall(rf"\b{re.escape(filler)}\b", transcript)) for filler in filler_words)
    
    if total_words ==0:
        return transcript, 0.0
    filler_ratio = filler_count/total_words
    fluency_score=max(0,10-(filler_ratio*50))
    
    return transcript, round(fluency_score,2)

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
        
        
def analyze_voice_confidence(audio_path):
    y, sr = librosa.load(audio_path)
    pitch = librosa.yin(y, fmin=50, fmax=300)
    volume = np.mean(np.abs(y))
    onset_env = librosa.onset.onset_strength(y=y,sr=sr)
    speech_rate = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
    
    score =0
    if 150< speech_rate<180 : score +=3
    if volume > 0.03: score +=3
    if np.std(pitch)>10: score +=4
    
    return round(score,2)

# Calculate eye contact score (simplified)
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

# --- Streamlit UI ---
st.title("🎥 Confidence Analyzer for Interview Videos")
st.markdown("Upload interview recordings. We'll score each candidate based on facial expressions and eye contact.")

uploaded_videos = st.file_uploader("Upload videos", type=["mp4","webm"], accept_multiple_files=True)

leaderboard = []

if uploaded_videos:
    for i, video in enumerate(uploaded_videos):
        ext = os.path.splitext(video.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(video.read())
            video_path = tmp.name

        st.video(video_path)
        with st.spinner(f"Analyzing Video {i+1}..."):
            dominant_emotion, score, transcript = analyze_confidence(video_path)

        st.success(f"Video {i+1}: Emotion = **{dominant_emotion}**, Confidence Score = **{score}/10**")
        st.markdown("📄 Transcript:")
        st.text(transcript)

        leaderboard.append({
            "Video": video.name,
            "Dominant Emotion": dominant_emotion,
            "Confidence Score": score
        })

    
    st.subheader("🏆 Leaderboard: Most Confident Candidates")
    df = pd.DataFrame(leaderboard)
    df_sorted = df.sort_values(by="Confidence Score", ascending=False).reset_index(drop=True)
    df_sorted.index += 1
    st.table(df_sorted)
    st.write(transcript)
