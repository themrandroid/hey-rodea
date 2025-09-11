import os
import json
import collections
import contextlib
import wave
import webrtcvad
import re
import json
import numpy as np
from pydub import AudioSegment
from pathlib import Path


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


# 2. Silence Detection
# Helper: read audio (16kHz WAV mono)
def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1, "Audio must be mono"
        # sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        assert sample_rate == 16000, "Audio must be 16kHz"
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

# Frame container
class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

# Frame generator
def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n <= len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

# Speech segment collector
def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)

    triggered = False
    segments = []
    start_time = 0

    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                start_time = ring_buffer[0][0].timestamp
                ring_buffer.clear()
        else:
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                end_time = frame.timestamp + frame.duration
                segments.append((start_time, end_time))
                triggered = False
                ring_buffer.clear()

    if triggered:
        end_time = frame.timestamp + frame.duration
        segments.append((start_time, end_time))

    return segments

# High-level pause detector
def detect_pauses(audio_path, aggressiveness=2):
    pcm_data, sample_rate = read_wave(audio_path)
    vad = webrtcvad.Vad(aggressiveness)  # 0–3 (3 = most aggressive)

    frames = list(frame_generator(30, pcm_data, sample_rate))
    segments = vad_collector(sample_rate, 30, 300, vad, frames)

    # Compute pauses as gaps between speech segments
    pauses = []
    prev_end = 0.0
    for start, end in segments:
        if prev_end > 0:
            pause_dur = start - prev_end
            if pause_dur > 0.2:  # ignore very short gaps
                pauses.append((round(prev_end, 2), round(start, 2)))
        prev_end = end

    return pauses

# 3. Pace (WPM) and Filler Count/Density
FILLER_WORDS = [
    "um", "uh", "erm", "like", "you know",
    "sort of", "kind of", "so", "hmm",
    "i mean", "actually", "basically"
]

def normalize_text(word):
    """Lowercasing and stripping punctuation for matching."""
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

    # --- Effective speaking time ---
    start_time = words[0]["start"]
    end_time = words[-1]["end"]
    total_time = end_time - start_time

    long_pauses = [end - start for start, end in pauses if (end - start) > long_pause_threshold]
    effective_time = total_time - sum(long_pauses)
    if effective_time <= 0:
        effective_time = total_time  # fallback

    wpm = total_words / (effective_time / 60)

    # --- Filler detection ---
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

    # 1. Clarity
    filler_density = stats["filler_density"]
    long_pause_rate_per_min = stats["long_pauses_count"] / (total_time_sec / 60) if total_time_sec > 0 else 0

    clarity = 100
    clarity -= min(40, 400 * filler_density)
    clarity -= min(20, 10 * long_pause_rate_per_min)
    clarity = max(0, min(100, round(clarity, 2)))

    # 2. Confidence
    WPM = stats["WPM"]
    target_mid, target_span = 165, 15
    conf_score = 100 - min(50, abs(WPM - target_mid) / target_span * 100)

    # Placeholder for variability penalty
    std_local_WPM = 0  # could compute later from 30s windows
    conf_score -= min(20, 5 * std_local_WPM)
    confidence = max(0, min(100, round(conf_score, 2)))

    # 3. Engagement
    pauses_per_min = len(pauses) / (total_time_sec / 60) if total_time_sec > 0 else 0
    pause_lengths = [end - start for start, end in pauses]
    median_pause = np.median(pause_lengths) if pause_lengths else 0

    engagement = 60

    # Pause cadence bonus
    if 6 <= pauses_per_min <= 15 and 0.3 <= median_pause <= 1.2:
        engagement += 15
    elif pauses_per_min > 0:
        engagement += 10  # partial credit

    # Energy/pitch bonus (stub for MVP)
    engagement += 10  # later refine with RMS variance

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
        feedback.append(
            f"You used {stats['filler_count']} filler words"
            + (f", mostly '{top_filler}'" if top_filler else "")
            + ". Try a short silent pause when you need to think."
        )

    # Pace (WPM)
    if stats["WPM"] > 190:
        feedback.append(
            f"Your pace averaged {stats['WPM']} WPM — a bit fast. "
            "Aim for 150–180 WPM for clarity."
        )
    elif stats["WPM"] < 120:
        feedback.append(
            f"Your pace averaged {stats['WPM']} WPM — a bit slow. "
            "Aim for 150–180 WPM for better engagement."
        )

    # Pauses
    pauses_per_min = len(pauses) / (total_time_sec / 60) if total_time_sec > 0 else 0
    pause_lengths = [end - start for start, end in pauses]
    median_pause = np.median(pause_lengths) if pause_lengths else 0

    if pauses_per_min < 3:
        feedback.append(
            f"Pauses were scarce ({pauses_per_min:.1f}/min). "
            "Add brief pauses to separate ideas."
        )
    elif any((end - start) > long_thr for start, end in pauses):
        feedback.append(
            f"Several long pauses (>{long_thr}s). "
            "Consider shorter, intentional breaths between points."
        )
    else:
        feedback.append(
            f"Great balance of pauses (median {median_pause:.1f}s). "
            "Keeps the audience with you!"
        )

    # Always include a positive reinforcement
    if scores["Clarity"] > 75:
        feedback.append("Nice clarity overall — your message came through well!")
    elif scores["Confidence"] > 75:
        feedback.append("You sounded confident — good pacing and delivery.")
    else:
        feedback.append("Solid effort — keep practicing to improve further!")

    return feedback

# 6. Preprocess Audio (to 16kHz mono WAV)
def preprocess_audio(audio_path, out_dir="processed", target_sr=16000):
    """
    Convert any input audio to 16kHz mono WAV for pipeline compatibility.
    Returns the path to the processed file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # Load audio (Pydub handles MP3, WAV, etc.)
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(target_sr).set_channels(1)

    # Save processed WAV
    base_name = Path(audio_path).stem
    out_path = out_dir / f"{base_name}_proc.wav"
    audio.export(out_path, format="wav")

    return str(out_path)

# 7. Analyze Audio
def analyze_speech(model, audio_path, asr_out_dir="asr_json"):
    os.makedirs(asr_out_dir, exist_ok=True)

    # --- Preprocess first ---
    audio_path = preprocess_audio(audio_path)

    # 1. ASR
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    asr_json_path = os.path.join(asr_out_dir, f"{base_name}.json")
    if not os.path.exists(asr_json_path):
        transcribe_with_timestamps(model, audio_path, asr_json_path)
        
    # 2. Pause detection
    pauses = detect_pauses(audio_path)

    # 3. Stats
    stats = compute_wpm_and_fillers(asr_json_path, pauses)

    # 4. Scores
    scores = compute_scores(stats, pauses, stats["total_time_sec"])

    # 5. Feedback
    feedback = generate_feedback(stats, scores, pauses, stats["total_time_sec"])

    return {
        "asr_json": asr_json_path,
        "pauses": pauses,
        "stats": stats,
        "scores": scores,
        "feedback": feedback
    }