import os
import json
import tempfile

import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Persistence (so results + chat survive a browser refresh)
# ---------------------------------------------------------------------------
STATE_FILE = os.path.join(tempfile.gettempdir(), "ai_video_assistant_state.json")


def save_state(result: dict, chat_history: list) -> None:
    """Persist everything except the live rag_chain object (not serializable)."""
    serializable = {
        "title": result.get("title"),
        "transcript": result.get("transcript"),
        "summary": result.get("summary"),
        "action_items": result.get("action_items"),
        "key_decisions": result.get("key_decisions"),
        "open_questions": result.get("open_questions"),
        "chat_history": chat_history,
    }
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable, f)
    except Exception:
        pass  # persistence is a nice-to-have, never break the app over it


def load_state():
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def clear_state() -> None:
    try:
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "restored" not in st.session_state:
    st.session_state.restored = False

# On a fresh session (e.g. after a page refresh), try to restore from disk.
if not st.session_state.restored:
    st.session_state.restored = True
    saved = load_state()
    if saved and saved.get("transcript"):
        with st.spinner("Restoring your last session…"):
            try:
                rag_chain = build_rag_chain(saved["transcript"])
                st.session_state.result = {
                    "title": saved.get("title"),
                    "transcript": saved.get("transcript"),
                    "summary": saved.get("summary"),
                    "action_items": saved.get("action_items"),
                    "key_decisions": saved.get("key_decisions"),
                    "open_questions": saved.get("open_questions"),
                    "rag_chain": rag_chain,
                }
                st.session_state.chat_history = saved.get("chat_history", [])
            except Exception:
                # If rebuilding the chat engine fails, just drop the restore attempt.
                st.session_state.result = None
                st.session_state.chat_history = []


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* ---- App background ---- */
        .stApp {
            background: linear-gradient(180deg, #0f1220 0%, #171a2b 100%);
        }

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #12152a 0%, #1a1d35 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] p {
            color: #b7bbd8 !important;
        }

        /* ---- Hero header ---- */
        .hero-title {
            font-size: 2.1rem;
            font-weight: 800;
            background: linear-gradient(90deg, #8b5cf6, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .hero-subtitle {
            color: #9ca3d4;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }

        /* ---- Cards ---- */
        .glass-card {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 1.6rem 1.8rem;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.25);
            color: #e5e7f5;
            line-height: 1.65;
        }

        /* ---- Buttons ---- */
        .stButton > button {
            border-radius: 10px;
            font-weight: 600;
            border: none;
            transition: all 0.15s ease-in-out;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #8b5cf6, #ec4899);
            color: white;
        }
        .stButton > button[kind="primary"]:hover {
            opacity: 0.9;
            transform: translateY(-1px);
        }
        .stButton > button:hover {
            transform: translateY(-1px);
        }

        /* ---- Tabs ---- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            background: rgba(255,255,255,0.03);
            padding: 6px;
            border-radius: 12px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            color: #b7bbd8;
            font-weight: 600;
            padding: 8px 16px;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, #8b5cf6, #ec4899) !important;
            color: white !important;
        }

        /* ---- Chat bubbles ---- */
        div[data-testid="stChatMessage"] {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 14px;
            padding: 0.4rem 0.6rem;
        }

        /* ---- Text input / selects ---- */
        .stTextInput input, .stSelectbox div[data-baseweb="select"] {
            border-radius: 10px !important;
        }

        h1, h2, h3, h4 { color: #f1f2fb; }
        p, li, span { color: #d7d9f0; }

        .status-pill {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            background: rgba(139, 92, 246, 0.15);
            color: #c4b5fd;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Pipeline (same logic as the CLI version, broken into steps for progress UI)
# ---------------------------------------------------------------------------
def run_pipeline(source: str, language: str) -> dict:
    with st.status("Processing your video…", expanded=True) as status:
        st.write("Reading input and preparing audio chunks…")
        chunks = process_input(source)

        st.write("Transcribing audio…")
        transcript = transcribe_all(chunks, language)

        st.write("Generating title…")
        title = generate_title(transcript)

        st.write("Creating summary…")
        summary = summarize(transcript)

        st.write("Extracting action items…")
        action_items = extract_action_items(transcript)

        st.write("Extracting key decisions…")
        decisions = extract_key_decisions(transcript)

        st.write("Extracting open questions…")
        questions = extract_questions(transcript)

        st.write("Building chat engine…")
        rag_chain = build_rag_chain(transcript)

        status.update(label="Done!", state="complete", expanded=False)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='font-size:1.6rem; font-weight:800;'>🎬 AI Video Assistant</div>",
        unsafe_allow_html=True,
    )
    st.caption("Turn any video or meeting into a summary you can chat with.")

    st.divider()

    input_mode = st.radio(
        "Video source",
        options=["YouTube URL", "Upload a file"],
        horizontal=True,
    )

    source = None

    if input_mode == "YouTube URL":
        source = st.text_input(
            "YouTube URL",
            placeholder="https://youtube.com/watch?v=...",
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload video or audio file",
            type=["mp4", "mov", "mkv", "avi", "mp3", "wav", "m4a"],
        )
        if uploaded_file is not None:
            # Persist the upload to a temp path so process_input() can read it like a normal file path
            suffix = os.path.splitext(uploaded_file.name)[1]
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file.close()
            source = tmp_file.name
            st.caption(f"✅ Ready: {uploaded_file.name}")

    language = st.selectbox("Language", options=["english", "hinglish"], index=0)

    run_clicked = st.button("Analyze Video", type="primary", use_container_width=True)

    if run_clicked:
        if not source or not str(source).strip():
            st.warning("Please enter a YouTube URL or upload a file first.")
        else:
            st.session_state.result = run_pipeline(str(source).strip(), language)
            st.session_state.chat_history = []
            save_state(st.session_state.result, st.session_state.chat_history)

    if st.session_state.result:
        st.divider()
        st.markdown("<span class='status-pill'>● Session active — auto-saved</span>", unsafe_allow_html=True)
        if st.button("Start Over", use_container_width=True):
            st.session_state.result = None
            st.session_state.chat_history = []
            clear_state()
            st.rerun()


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
result = st.session_state.result

if result is None:
    st.markdown("<div class='hero-title'>🎬 AI Video Assistant</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>Enter a YouTube URL or upload a file in the sidebar and click "
        "<b>Analyze Video</b> to get started.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="glass-card">
        You'll get a <b>title</b>, <b>summary</b>, <b>action items</b>, <b>key decisions</b>,
        <b>open questions</b>, and a <b>chat assistant</b> for the video — and it'll all still be
        here even if you refresh the page.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(f"<div class='hero-title'>{result['title']}</div>", unsafe_allow_html=True)

    tab_summary, tab_actions, tab_decisions, tab_questions, tab_transcript, tab_chat = st.tabs(
        ["📋 Summary", "✅ Action Items", "🔑 Key Decisions", "❓ Open Questions", "📝 Transcript", "💬 Chat"]
    )

    with tab_summary:
        st.markdown(f"<div class='glass-card'>{result['summary']}</div>", unsafe_allow_html=True)

    with tab_actions:
        st.markdown(f"<div class='glass-card'>{result['action_items']}</div>", unsafe_allow_html=True)

    with tab_decisions:
        st.markdown(f"<div class='glass-card'>{result['key_decisions']}</div>", unsafe_allow_html=True)

    with tab_questions:
        st.markdown(f"<div class='glass-card'>{result['open_questions']}</div>", unsafe_allow_html=True)

    with tab_transcript:
        with st.expander("Show full transcript", expanded=False):
            st.markdown(f"<div class='glass-card'>{result['transcript']}</div>", unsafe_allow_html=True)

    with tab_chat:
        st.subheader("Chat with your video")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        question = st.chat_input("Ask something about this video...")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    answer = ask_question(result["rag_chain"], question)
                st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

            # Persist after every chat turn so refresh keeps the conversation too
            save_state(result, st.session_state.chat_history)
