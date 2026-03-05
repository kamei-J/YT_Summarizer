# YT_Summarizer


A small utility that fetches YouTube video transcripts and generates concise summaries.  
Ideal for quickly understanding long videos without watching them end‑to‑end.

---

## 🚀 Features

- Download subtitles/transcripts for a given YouTube URL
- Clean and process text
- Produce a readable summary (using your choice of NLP model or service)
- Optionally save results to a file

---

## 🛠️ Installation

1. Environment SetUp
  - create a .env file in the root directory:
```bash
  GROQ_API_KEY=your_groq_api_key
```

2. Install required packages:
 ```bash
  pip install -r requirement.txt
 ```
3.  Running the Servers
 ```bash
   fastapi dev main.py
 ```
---

## 📝 Usage
```bash
  python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
```


    Supported options (example):

    --url – YouTube video link (required)
    --output – Path to save summary (defaults to stdout)
    --model – Summarization model to use (gpt, bart, etc.)
    --no-cache – Force re-download of transcript

    Example output:

    Title: The Future of AI
    Summary:
    - Introduction to current AI capabilities …
    - Discussion on ethics and regulation …
    

---
## 🧩 Project Structure

YT_Summarizer/
├── main.py             # entry point
├── app.py              # core logic (transcript fetching, summarization)
├── requirement.txt     # dependencies

├── README.md           # this file

