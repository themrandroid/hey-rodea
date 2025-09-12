import streamlit as st
import os
import json
import json
import html
import streamlit.components.v1 as components
from pathlib import Path
from faster_whisper import WhisperModel

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: [{"file": ..., "transcript": ...}, ...]

if "coach_history" not in st.session_state:
    st.session_state.coach_history = []  # list of dicts: {"file": ..., "scores": ..., "feedback": ...}

@st.cache_resource
def load_whisper_model():
    model_size = "small"
    return WhisperModel(model_size, device="cpu", compute_type="int8")

model = load_whisper_model()

# Import your pipeline functions
from pipeline import analyze_speech, transcribe_with_timestamps

# Import App Design Functions
from app_design import transcript_card, donut_card, styled_feedback

st.set_page_config(page_title="Hey Rodea", layout="wide")

# ------------------------
# Main Menu
# ------------------------
st.markdown(
    """
    <h1 style="
        text-align: center;
        background: linear-gradient(90deg, #2E86C1, #28B463, #CA6F1E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: 800;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.2);
        margin-bottom: 10px;
    ">
        Hey Rodea 👋
    </h1>
    """,
    unsafe_allow_html=True,
)

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
        st.subheader("Transcript:")
        # Only ASR
        out_json = f"asr_json/{Path(audio_path).stem}.json"
        transcribe_with_timestamps(model, str(audio_path), out_json)

        import json
        with open(out_json, "r") as f:
            words = json.load(f)

        transcript = " ".join([w["word"] for w in words])
        # st.text_area("Transcript", transcript, height=200)
        transcript_card(transcript)
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
                        transcript_card(item['transcript'])
                        copy_to_clipboard(item['transcript'], key=f"copy_{i}")

    elif choice == "Speech Coach":
        st.subheader("Results:")
        result = analyze_speech(model, str(audio_path))

        if not st.session_state.coach_history or st.session_state.coach_history[-1]["file"] != Path(audio_path).name:
            st.session_state.coach_history.append({
                "file": Path(audio_path).name,
                "scores": result["scores"],
                "feedback": result["feedback"]
            })

        # --- Show scores in cards ---
        col1, col2, col3 = st.columns(3)

        with col1:
            donut_card("Clarity", result["scores"]["Clarity"], "#2E86C1", key_suffix="current")
        with col2:
            donut_card("Confidence", result["scores"]["Confidence"], "#28B463", key_suffix="current")
        with col3:
            donut_card("Engagement", result["scores"]["Engagement"], "#CA6F1E", key_suffix="current")


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
            styled_feedback(result["feedback"])

        # Save to history
        with st.expander("Coach History", expanded=False):
            if not st.session_state.coach_history:
                st.write("No coaching sessions yet.")
            else:
                for i, item in enumerate(reversed(st.session_state.coach_history), 1):
                    with st.expander(f"{i}. {item['file']}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            donut_card("Clarity", item["scores"]["Clarity"], "#2E86C1", key_suffix=f"hist_{i}")
                        with col2:
                            donut_card("Confidence", item["scores"]["Confidence"], "#28B463", key_suffix=f"hist_{i}")
                        with col3:
                            donut_card("Engagement", item["scores"]["Engagement"], "#CA6F1E", key_suffix=f"hist_{i}")
                        styled_feedback(item["feedback"])