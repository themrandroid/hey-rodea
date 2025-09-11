import streamlit as st
import os
import json
import json
import html
import streamlit.components.v1 as components
from pathlib import Path
from audiorecorder import audiorecorder
from faster_whisper import WhisperModel

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: [{"file": ..., "transcript": ...}, ...]

@st.cache_resource
def load_whisper_model():
    model_size = "small"
    return WhisperModel(model_size, device="cpu", compute_type="int8")

model = load_whisper_model()

# Import your pipeline functions
from pipeline import analyze_speech, transcribe_with_timestamps

st.set_page_config(page_title="Hey Rodea", layout="wide")

# ------------------------
# Main Menu
# ------------------------
st.title("Hey Rodea")
choice = st.radio("Choose a mode:", ["Speech-to-Text", "Speech Coach"], horizontal=True)

# ------------------------  
# Recording or Upload
# ------------------------
def get_audio_input():
    """
    Provides two options: upload file or record live.
    Returns: audio_path (str) or None if no input.
    """
    st.subheader("Choose Input Method")

    tab1, tab2 = st.tabs(["Upload File", "Record Live"])

    audio_path = None

    # ---- Upload option ----
    with tab1:
        uploaded_file = st.file_uploader("Upload audio", type=["wav", "mp3"])
        if uploaded_file:
            save_dir = Path("uploads")
            save_dir.mkdir(exist_ok=True)
            audio_path = save_dir / uploaded_file.name
            with open(audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            # st.success(f"File saved: {audio_path}")

    # ---- Record option ----
    with tab2:
        recorded_file = st.audio_input("Record your voice")
        if recorded_file:
            save_dir = Path("uploads")
            save_dir.mkdir(exist_ok=True)
            audio_path = save_dir / "recording.wav"
            with open(audio_path, "wb") as f:
                f.write(recorded_file.getbuffer())
            # st.success(f"Recording saved: {audio_path}")

    return str(audio_path) if audio_path else None    

audio_path = get_audio_input()

def copy_to_clipboard(text, key):
    safe_text = html.escape(text).replace("\n", "\\n").replace("'", "\\'")
    components.html(
        f"""
        <button onclick="navigator.clipboard.writeText(`{safe_text}`)"
                style="margin:5px;padding:5px;cursor:pointer;border-radius:5px;">
            Copy
        </button>
        """,
    )
# ------------------------
# Processing
# ------------------------
if audio_path:
    os.makedirs("asr_json", exist_ok=True)
    if choice == "Speech-to-Text":
        st.subheader("Transcript")
        # Only ASR
        out_json = f"asr_json/{Path(audio_path).stem}.json"
        transcribe_with_timestamps(model, str(audio_path), out_json)

        import json
        with open(out_json, "r") as f:
            words = json.load(f)

        transcript = " ".join([w["word"] for w in words])
        # st.text_area("Transcript", transcript, height=200)
        st.markdown(f"**Transcript:**\n\n{transcript}")
        copy_to_clipboard(transcript, key="main_copy")

        transcript = " ".join([w["word"] for w in words])

        # Save to history
        st.session_state.history.append({
            "file": Path(audio_path).name,
            "transcript": transcript
        })

        with st.expander("History", expanded=False):
            if not st.session_state.history:
                st.write("No transcripts yet.")
            else:
                for i, item in enumerate(reversed(st.session_state.history), 1):
                    with st.expander(f"{i}. {item['file']}"):
                        st.markdown(f"**Transcript:**\n\n{item['transcript']}")
                        copy_to_clipboard(item['transcript'], key=f"copy_{i}")

    elif choice == "Speech Coach":
        st.subheader("Results")
        result = analyze_speech(model, str(audio_path))

        # --- Show scores in cards ---
        col1, col2, col3 = st.columns(3)

        # Helper function to render card with animated bar
        def score_card(label, score, color):
            components.html(
                f"""
        <div style="background-color:#f9f9f9;padding:20px;
                    border-radius:10px;text-align:center;
                    box-shadow:0 2px 5px rgba(0,0,0,0.1);
                    position:relative;">
            
            <div style="font-size:32px;font-weight:bold;color:{color};">
                {score}
            </div>
            <div style="font-size:16px;color:#555;margin-bottom:10px;">
                {label}
            </div>

            <div style="background:#e0e0e0;border-radius:10px;
                        overflow:hidden;height:15px;">
                <div style="
                    width:0%;
                    background:{color};
                    height:100%;
                    border-radius:10px;
                    animation: fillAnim 2s forwards;
                    --target:{score}%;
                "></div>
            </div>
        </div>

        <style>
        @keyframes fillAnim {{
            from {{ width: 0%; }}
            to   {{ width: var(--target); }}
        }}
        </style>
                """,
            )

        with col1:
            score_card("Clarity", result["scores"]["Clarity"], "#2E86C1")

        with col2:
            score_card("Confidence", result["scores"]["Confidence"], "#28B463")

        with col3:
            score_card("Engagement", result["scores"]["Engagement"], "#CA6F1E")

        # --- Feedback section ---
        st.subheader("Feedback")

        # Initialize toggle state
        if "show_feedback" not in st.session_state:
            st.session_state.show_feedback = False

        # Toggle button
        if st.button("Show/Hide Feedback"):
            st.session_state.show_feedback = not st.session_state.show_feedback

        # Conditionally display feedback
        if st.session_state.show_feedback:
            for line in result["feedback"]:
                st.write(f"- {line}")