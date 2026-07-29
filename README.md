# 🎥 AI Meeting Assistant

An AI-powered Meeting Assistant built with Python that automatically transcribes, summarizes, analyzes, and allows you to chat with meeting recordings using Retrieval-Augmented Generation (RAG).

This project processes YouTube videos or local audio/video files, generates transcripts, extracts actionable insights, and enables intelligent question answering over meeting content.

---

## 🚀 Features

- 🎥 Process YouTube videos
- 📁 Upload local audio/video files
- 🎙️ Speech-to-Text using OpenAI Whisper
- 🤖 AI-powered meeting summarization
- ✅ Extract action items
- 📌 Identify key decisions
- ❓ Extract open questions
- 💬 Chat with meeting transcripts using RAG
- 🧠 ChromaDB vector database
- ⚡ Groq LLM Integration
- 🌐 Interactive Streamlit Dashboard

---

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- Groq API
- OpenAI Whisper
- ChromaDB
- HuggingFace Embeddings
- yt-dlp
- FFmpeg
- Pydub

---

## 📂 Project Structure

```
AI-Video-Assistant/
│
├── core/
│   ├── extractor.py
│   ├── rag_engine.py
│   ├── summarizer.py
│   ├── transcriber.py
│   └── vector_store.py
│
├── utils/
│   └── audio_processor.py
│
├── app.py
├── main.py
├── Requirements.txt
├── test.py
├── .env
└── README.md
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/AI_Video_Assistant.git

cd AI_Video_Assistant
```

### Create Virtual Environment

Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

### Install Dependencies

```bash
pip install -r Requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file.

```
GROQ_API_KEY=your_groq_api_key
SARVAM_API_KEY=your_sarvam_api_key
```

---

## ▶️ Run Streamlit App

```bash
streamlit run app.py
```

---

## 🖥️ Run CLI Version

```bash
python main.py
```

---

## 📌 Workflow

```
YouTube URL / Audio File
            │
            ▼
    Audio Processing
            │
            ▼
 Whisper Transcription
            │
            ▼
 Meeting Summarization
            │
            ▼
Action Items Extraction
            │
            ▼
 ChromaDB Vector Store
            │
            ▼
    RAG Question Answering
            │
            ▼
      Streamlit UI
```

---

## 📸 Screenshots

Add screenshots of the application here.

### Dashboard

```
images/dashboard.png
```

### Summary

```
images/summary.png
```

### Chat

```
images/chat.png
```

---

## 📖 Future Improvements

- Meeting History
- Dark Mode
- DOCX Export
- Email Reports
- Voice Chat
- Multi-language Support
- Meeting Analytics
- Cloud Deployment

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.

Feel free to fork the repository and submit a pull request.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Muhammad Asad**

GitHub:
https://github.com/MuhammadAsad555

---

## ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.