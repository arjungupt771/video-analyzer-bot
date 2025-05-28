import streamlit as st
import tempfile
import os
from collections import deque
import pandas as pd
from utils.audio_utils import extract_audio, transcribe_and_analyze_fluency, analyze_voice_confidence
from utils.video_utils import detect_blink, detect_head_movement, is_facing_forward, detect_posture
from utils.expression_utils import analyze_confidence
from utils.scoring import evaluate_technical_answers_with_explanation

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
            qa_results, tech_score = evaluate_technical_answers_with_explanation(transcript, qa_set)

        st.success(f"Video {i+1}: Emotion = **{dominant_emotion}**, Confidence Score = **{score}/10**, Technical Score = **{tech_score}/10**")
        st.markdown("📄 Transcript:")
        st.text(transcript)

        st.markdown("🧠 **Technical Answer Evaluation:**")
        for res in qa_results:
            st.markdown(f"- **Q:** {res['Question']}")
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;💡 **Expected:** {res['Expected Answer']}")
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;🗣️ **Response:** _{res['Best Match from Response']}_")
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;📊 **Score:** {res['Score']} / 10")
        
        leaderboard.append({
            "Video": video.name,
            "Dominant Emotion": dominant_emotion,
            "Confidence Score": score,
            "Technical Score": tech_score,
            "Total Score": round((score + tech_score) / 2, 2)
        })

    st.subheader("🏆 Leaderboard: Most Confident Candidates")
    df = pd.DataFrame(leaderboard)
    df_sorted = df.sort_values(by="Confidence Score", ascending=False).reset_index(drop=True)
    df_sorted.index += 1
    st.markdown("Sorted by **Total Score** (average of Confidence and Technical)")
    st.table(df_sorted[["Video", "Dominant Emotion", "Confidence Score", "Technical Score", "Total Score"]])
