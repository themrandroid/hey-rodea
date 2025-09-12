import os
import json
import re
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.signal import resample

# 1. ASR for transcription with word-level timestamps
def transcribe_with_timestamps(model, audio_path, output_json):
    segments, info = model.transcribe(audio_path, word_timestamps=True)

    words = []
    for segment in segments:
        for w in segment.words:  # each word has start, end, and text
            words.append({
                "word": w.word.strip(),
                "start": round(w.start, 2),
                "end": round(w.end, 2)
            })

    # Save JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(words, f, indent=2)

    return words


# 2. Silence Detection (energy-based)
def detect_pauses(audio_path, sr=16000, frame_ms=30, silence_threshold=0.01):
    """
    Detect pauses based on short-term energy.
    Returns: list of (start, end) tuples for pauses.
    """
    audio, file_sr = sf.read(audio_path)
    if audio.ndim > 1:  # stereo → mono
        audio = np.mean(audio, axis=1)

    # Resample if needed
    if file_sr != sr:
        n_samples = int(len(audio) * sr / file_sr)
        audio = resample(audio, n_samples)

    frame_len = int(sr * frame_ms / 1000)
    energy = np.array([
        np.sqrt(np.mean(audio[i:i+frame_len]**2))
        for i in range(0, len(audio), frame_len)
    ])

    pauses = []
    in_pause = False
    pause_start = 0.0

    for i, e in enumerate(energy):
        t = i * frame_ms / 1000.0  # seconds
        if e < silence_threshold:
            if not in_pause:
                in_pause = True
                pause_start = t
        else:
            if in_pause:
                in_pause = False
                pauses.append((round(pause_start, 2), round(t, 2)))

    # If file ended in pause
    if in_pause:
        pauses.append((round(pause_start, 2), round(len(audio) / sr, 2)))

    return pauses


# 3. Pace (WPM) and Filler Count/Density
FILLER_WORDS = [
    "um", "uh", "erm", "like", "you know",
    "sort of", "kind of", "so", "hmm",
    "i mean", "actually", "basically"
]

def normalize_text(word):
    return re.sub(r"[^\w\s]", "", word.lower())

def compute_wpm_and_fillers(asr_json_path, pauses, long_pause_threshold=2.0):
    with open(asr_json_path, "r", encoding="utf-8") as f:
        words = json.load(f)

    total_words = len(words)
    if total_words == 0:
        return {
            "WPM": 0, "total_words": 0, "effective_time_sec": 0,
            "filler_count": 0, "filler_density": 0.0, "top_fillers": {}
        }

    start_time = words[0]["start"]
    end_time = words[-1]["end"]
    total_time = end_time - start_time

    long_pauses = [end - start for start, end in pauses if (end - start) > long_pause_threshold]
    effective_time = total_time - sum(long_pauses)
    if effective_time <= 0:
        effective_time = total_time

    wpm = total_words / (effective_time / 60)

    filler_count = 0
    filler_hist = {}
    for w in words:
        token = normalize_text(w["word"])
        if token in FILLER_WORDS:
            filler_count += 1
            filler_hist[token] = filler_hist.get(token, 0) + 1

    filler_density = filler_count / total_words if total_words > 0 else 0

    return {
        "WPM": round(wpm, 2),
        "total_words": total_words,
        "effective_time_sec": round(effective_time, 2),
        "total_time_sec": round(total_time, 2),
        "long_pauses_count": len(long_pauses),
        "long_pauses_total_sec": round(sum(long_pauses), 2),
        "filler_count": filler_count,
        "filler_density": round(filler_density, 4),
        "top_fillers": dict(sorted(filler_hist.items(), key=lambda x: -x[1]))
    }


# 4. Scoring Function
def compute_scores(stats, pauses, total_time_sec):
    filler_density = stats["filler_density"]
    long_pause_rate_per_min = stats["long_pauses_count"] / (total_time_sec / 60) if total_time_sec > 0 else 0

    # --- Clarity ---
    clarity = 100
    clarity -= min(30, 300 * filler_density)  # softer penalty for fillers
    clarity -= min(15, 8 * long_pause_rate_per_min)  # gentle deduction for long pauses
    clarity = max(0, min(100, round(clarity, 2)))

    # --- Confidence ---
    WPM = stats["WPM"]
    if 140 <= WPM <= 180:
        confidence = 90 + (10 - abs(WPM - 160) / 2)  # strong zone → ~90–100
    else:
        confidence = max(40, 90 - (abs(WPM - 160) / 2))  # gradual falloff, never <40
    confidence = max(0, min(100, round(confidence, 2)))

    # --- Engagement ---
    pauses_per_min = len(pauses) / (total_time_sec / 60) if total_time_sec > 0 else 0
    pause_lengths = [end - start for start, end in pauses]
    median_pause = np.median(pause_lengths) if pause_lengths else 0

    engagement = 50  # baseline
    if 6 <= pauses_per_min <= 15 and 0.3 <= median_pause <= 1.2:
        engagement += 35  # excellent pause rhythm
    elif pauses_per_min > 0:
        engagement += 20  # some effort
    else:
        engagement += 10  # almost no pauses
    engagement = max(0, min(100, round(engagement, 2)))

    return {
        "Clarity": clarity,
        "Confidence": confidence,
        "Engagement": engagement
    }


# 5. Feedback Generator
def generate_feedback(stats, scores, pauses, total_time_sec, long_thr=2.0):
    feedback = []

    # Fillers
    if stats["filler_count"] > 0:
        top_filler = max(stats["top_fillers"], key=stats["top_fillers"].get, default=None)
        if top_filler:
            feedback.append(f"You used ‘{top_filler}’ quite a few times. Try pausing briefly instead.")
        else:
            feedback.append("You used some filler words. Short pauses would sound clearer.")

    # Pace
    if stats["WPM"] > 190:
        feedback.append("You were speaking quite fast. Slow down a little so every word lands.")
    elif stats["WPM"] < 120:
        feedback.append("Your pace felt a bit slow. Try picking it up to keep energy in your voice.")
    else:
        feedback.append("Your pace felt natural, easy to follow.")

    # Pauses
    pause_lengths = [end - start for start, end in pauses]
    if any((end - start) > long_thr for start, end in pauses):
        feedback.append("There were a few long silences. Keep them shorter to stay engaging.")
    elif len(pauses) < 2:
        feedback.append("You hardly paused. A few short breaks would make your message clearer.")
    else:
        feedback.append("Good use of pauses, they gave your words space to breathe.")

    # Overall
    if scores["Clarity"] > 75:
        feedback.append("Your clarity was strong, your points came through well.")
    elif scores["Confidence"] > 75:
        feedback.append("You sounded confident and assured, nice delivery.")
    else:
        feedback.append("Solid effort! Keep practicing and your delivery will only improve.")

    return feedback

# 6. Preprocess Audio (convert to 16kHz mono WAV)
def preprocess_audio(audio_path, out_dir="processed", target_sr=16000):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    audio, file_sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if file_sr != target_sr:
        n_samples = int(len(audio) * target_sr / file_sr)
        audio = resample(audio, n_samples)

    base_name = Path(audio_path).stem
    out_path = out_dir / f"{base_name}_proc.wav"
    sf.write(out_path, audio, target_sr)

    return str(out_path)


# 7. Analyze Audio
def analyze_speech(model, audio_path, asr_out_dir="asr_json"):
    os.makedirs(asr_out_dir, exist_ok=True)

    audio_path = preprocess_audio(audio_path)

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    asr_json_path = os.path.join(asr_out_dir, f"{base_name}.json")
    if not os.path.exists(asr_json_path):
        transcribe_with_timestamps(model, audio_path, asr_json_path)

    pauses = detect_pauses(audio_path)
    stats = compute_wpm_and_fillers(asr_json_path, pauses)
    scores = compute_scores(stats, pauses, stats["total_time_sec"])
    feedback = generate_feedback(stats, scores, pauses, stats["total_time_sec"])

    return {
        "asr_json": asr_json_path,
        "pauses": pauses,
        "stats": stats,
        "scores": scores,
        "feedback": feedback
    }