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
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


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
    st.title("🎬 AI Video Assistant")
    st.caption("Turn any video or meeting into a summary you can chat with.")

    st.divider()

    source = st.text_input(
        "YouTube URL or local file path",
        placeholder="https://youtube.com/watch?v=... or /path/to/file.mp4",
    )
    language = st.selectbox("Language", options=["english", "hinglish"], index=0)

    run_clicked = st.button("Analyze Video", type="primary", use_container_width=True)

    if run_clicked:
        if not source.strip():
            st.warning("Please enter a YouTube URL or file path first.")
        else:
            st.session_state.result = run_pipeline(source.strip(), language)
            st.session_state.chat_history = []

    if st.session_state.result:
        st.divider()
        if st.button("Start Over", use_container_width=True):
            st.session_state.result = None
            st.session_state.chat_history = []
            st.rerun()


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
result = st.session_state.result

if result is None:
    st.title("🎬 AI Video Assistant")
    st.write("Enter a YouTube URL or a local file path in the sidebar and click **Analyze Video** to get started.")
    st.info("You'll get a title, summary, action items, key decisions, open questions, and a chat assistant for the video.")
else:
    st.title(result["title"])

    tab_summary, tab_actions, tab_decisions, tab_questions, tab_transcript, tab_chat = st.tabs(
        ["📋 Summary", "✅ Action Items", "🔑 Key Decisions", "❓ Open Questions", "📝 Transcript", "💬 Chat"]
    )

    with tab_summary:
        st.subheader("Summary")
        st.write(result["summary"])

    with tab_actions:
        st.subheader("Action Items")
        st.write(result["action_items"])

    with tab_decisions:
        st.subheader("Key Decisions")
        st.write(result["key_decisions"])

    with tab_questions:
        st.subheader("Open Questions")
        st.write(result["open_questions"])

    with tab_transcript:
        st.subheader("Full Transcript")
        with st.expander("Show transcript", expanded=False):
            st.write(result["transcript"])

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