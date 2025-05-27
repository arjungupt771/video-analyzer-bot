import subprocess
import whisper
import re
import librosa
import numpy as np     


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