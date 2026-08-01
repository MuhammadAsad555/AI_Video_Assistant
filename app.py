import os

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
# Session state
# ---------------------------------------------------------------------------
# NOTE: We intentionally do NOT persist results/chat history to disk anymore.
# Everything lives only in st.session_state, so every fresh app open (new
# browser session) starts clean on the "Enter a YouTube URL..." dashboard.
# A same-tab rerun (clicking buttons, asking chat questions, etc.) still
# keeps state, because that's the same Streamlit session.
if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
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

        /* Emoji / icon glyphs need their own font stack, otherwise the
           'Inter' override above strips them down to a blank color box. */
        .emoji-icon {
            font-family: "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol",
                         "Noto Color Emoji", sans-serif !important;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* ---- Remove the white Streamlit header bar so the gradient is one
               continuous color from the very top of the page ---- */
        header[data-testid="stHeader"] {
            background: transparent;
            box-shadow: none;
        }
        div[data-testid="stDecoration"] {
            background: transparent;
        }
        div[data-testid="stToolbar"] {
            display: none;
        }

        /* ---- App background ---- */
        .stApp {
            background: linear-gradient(180deg, #0f1220 0%, #171a2b 100%);
        }
        /* Make sure the very top of the main view container also matches
           (some Streamlit versions add a solid block behind the header) */
        section[data-testid="stAppViewContainer"] {
            background: transparent;
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

        .app-icon-svg {
            vertical-align: -4px;
            margin-right: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Reusable inline SVG "clapperboard" icon (renders identically everywhere,
# unlike the 🎬 emoji which was being swallowed by the Inter font override).
CLAPPER_ICON_SVG = """
<svg class="app-icon-svg" width="30" height="30" viewBox="0 0 24 24" fill="none"
     xmlns="http://www.w3.org/2000/svg" style="display:inline-block;">
  <defs>
    <linearGradient id="clapperGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#8b5cf6"/>
      <stop offset="100%" stop-color="#ec4899"/>
    </linearGradient>
  </defs>
  <rect x="3" y="9" width="18" height="12" rx="2" fill="url(#clapperGrad)"/>
  <path d="M3 9L4.5 4.5L8 5.5L6.5 9H3Z" fill="url(#clapperGrad)"/>
  <path d="M8.5 5.7L12 6.7L10.5 9H7L8.5 5.7Z" fill="url(#clapperGrad)"/>
  <path d="M13 6.9L16.5 7.9L15 9H11.5L13 6.9Z" fill="url(#clapperGrad)"/>
  <path d="M17.5 8L21 9L21 9.3L17.5 9.3L17.5 8Z" fill="url(#clapperGrad)"/>
  <rect x="3" y="9" width="18" height="2" fill="#171a2b" opacity="0.25"/>
  <circle cx="8" cy="15.5" r="1.4" fill="#171a2b" opacity="0.3"/>
  <circle cx="16" cy="15.5" r="1.4" fill="#171a2b" opacity="0.3"/>
</svg>
"""


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
        f"<div style='font-size:1.6rem; font-weight:800; display:flex; align-items:center;'>"
        f"{CLAPPER_ICON_SVG}<span>AI Video Assistant</span></div>",
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
            import tempfile
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

    if st.session_state.result:
        st.divider()
        st.markdown("<span class='status-pill'>● Session active</span>", unsafe_allow_html=True)
        if st.button("Start Over", use_container_width=True):
            st.session_state.result = None
            st.session_state.chat_history = []
            st.rerun()


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
result = st.session_state.result

if result is None:
    st.markdown(
        f"<div class='hero-title' style='display:flex; align-items:center;'>"
        f"{CLAPPER_ICON_SVG}<span>AI Video Assistant</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='hero-subtitle'>Enter a YouTube URL or upload a file in the sidebar and click "
        "<b>Analyze Video</b> to get started.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="glass-card">
        You'll get a <b>title</b>, <b>summary</b>, <b>action items</b>, <b>key decisions</b>,
        <b>open questions</b>, and a <b>chat assistant</b> for the video.
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
