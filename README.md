# 🎙️ Autonomous Voice Customer Support Agent

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Serverless-000000?style=for-the-badge)](https://www.pinecone.io/)
[![Groq](https://img.shields.io/badge/Groq-Inference-F55036?style=for-the-badge)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

**An end-to-end Autonomous RAG (A-RAG) system that resolves technical support queries via voice — dynamically crawling the web, filtering noise, embedding fresh context, and synthesizing precise answers in real time.**

[Overview](#-overview) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [API Reference](#-api-reference) · [Configuration](#-configuration) · [Project Structure](#-project-structure) · [Future Scope](#-future-scope)

</div>

---

## 📌 Overview

Traditional support bots retrieve from static knowledge bases that go stale. This project takes a different approach: when a user submits a query (voice or text), the system **actively explores the web in real time**, builds a fresh vector corpus scoped to that query, and synthesizes a grounded answer — all within a single session.

### What makes it different

| Traditional RAG | This System (A-RAG) |
|---|---|
| Static knowledge base | Dynamic, query-scoped corpus built per session |
| Single retrieval round | Autonomous multi-query crawl across 10–14 targeted URLs |
| Generic embeddings | Session-isolated Pinecone namespace — no cross-user drift |
| Text input only | Voice (Whisper-large-v3) + text, both supported |
| No validation gate | Stateful clarification loop before expensive crawl begins |

---

## 🏗️ Architecture

### System Design — Level 2 DFD

<picture>
  <source srcset="level 2 dfd.drawio.svg" type="image/svg+xml">
  <img src="level 2 dfd.png" alt="Level 2 Data Flow Diagram — Autonomous Voice Customer Support Agent" width="100%">
</picture>

### High-Level Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│               (Voice Blob / Text Query via UI)               │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                     Flask API  (app.py)                      │
│   /voice → STT    /query → pipeline    /live → poll status   │
└───────┬──────────────────┬──────────────────────────────────┘
        │                  │
        ▼                  ▼
  voice_to_text.py   query_preprocessing.py
  (Whisper-large-v3) (spaCy filler removal)
                           │
                           ▼
             ┌─────────────────────────┐
             │  Validation Gate        │  question_validation.py
             │  1–2 clarification turns│  (regex + LLM Q-gen)
             └──────────┬──────────────┘
                        │  [enough_data = True]
                        ▼
             ┌─────────────────────────┐
             │  Intelligence Engine    │  the_intellegence.py
             │  Question Graph + 10–14 │  (Groq-hosted LLM)
             │  targeted search queries│
             └──────────┬──────────────┘
                        │
                        ▼
             ┌─────────────────────────┐
             │  Autonomous Crawler     │  webcrawler.py
             │  (crawl4ai + Playwright)│  Multi-engine search
             │  → webcrawler_result    │  → site-scoped URL filter
             └──────────┬──────────────┘
                        │  [subprocess watcher]
                        ▼
             ┌─────────────────────────┐
             │  Layer 1 Chunker        │  text preprocessing+chunking.py
             │  Structural clean +     │  (watch mode, offset-tracked)
             │  paragraph chunking     │
             └──────────┬──────────────┘
                        │  [subprocess watcher]
                        ▼
             ┌─────────────────────────┐
             │  Layer 2 Adaptive Filter│  2nd_layer_chunking.py
             │  NLP scoring, hard noise│  → qdrant_ready.jasonl
             │  removal, deduplication │
             └──────────┬──────────────┘
                        │  [threading watcher]
                        ▼
             ┌─────────────────────────┐
             │  RAG Pipeline           │  rag.py
             │  Pinecone upsert +      │  BAAI/bge-small-en-v1.5
             │  semantic search Top-5  │  (FastEmbed, 512-dim)
             └──────────┬──────────────┘
                        │
                        ▼
             ┌─────────────────────────┐
             │  Answer Synthesis       │  the_intellegence.py
             │  LLM over retrieved     │  (Groq-hosted LLM)
             │  chunks + user context  │
             └─────────────────────────┘
```

### Concurrency Model

`app.py` manages three long-running background processes:

```
Main Flask Process
├── CHUNKER_PROCESS      → text preprocessing+chunking.py  (subprocess, watch mode)
├── SECOND_LAYER_PROCESS → 2nd_layer_chunking.py           (subprocess, watch mode)
└── RAG_REFRESH_THREAD   → polls qdrant_ready.jasonl       (daemon thread, 2s interval)
```

This decoupled watcher pattern keeps the API responsive while ingestion proceeds in parallel.

---

## ⚡ Quick Start

### Prerequisites

- Python 3.10 or 3.12
- [Pinecone](https://www.pinecone.io/) account (free tier works)
- [Groq](https://console.groq.com/) API key (free tier available)
- Node.js (optional, for frontend tooling)

### 1. Clone the Repository

```bash
git clone https://github.com/rajarshee99/autonomous-voice-support-agent.git
cd autonomous-voice-support-agent
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm

# Install Playwright browser for crawl4ai
playwright install chromium
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=ragcustomersupport
PINECONE_NAMESPACE=voice_support_chunks
RAG_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

> ⚠️ **Never commit `.env` to version control.** The `.gitignore` should exclude it.

### 5. Create the Pinecone Index

In your Pinecone dashboard, create a serverless index named `ragcustomersupport` with:
- **Dimensions:** `512`
- **Metric:** `cosine`
- **Cloud/Region:** any (AWS us-east-1 recommended for free tier)

### 6. Run the Application

```bash
python app.py
```

Expected startup output:
```
[APP] Started chunker process (pid=XXXXX)
[APP] Started second-layer chunker process (pid=XXXXX)
 * Running on http://127.0.0.1:5000
```

Open `http://127.0.0.1:5000` in your browser.

---

## 📡 API Reference

### `POST /query`
Submit a text query. The system runs the validation gate and either asks clarifying questions or triggers the full crawl + answer pipeline.

**Request:**
```json
{ "query": "My laptop keeps getting a blue screen on startup" }
```

**Response (needs more info):**
```json
{
  "status": "needs_more_info",
  "needs_more_info": true,
  "follow_up_question": "What exact error code appears on the BSOD screen?",
  "top_level_questions": ["...", "...", "..."]
}
```

**Response (complete):**
```json
{
  "status": "complete",
  "is_complete": true,
  "llm_answer": "Based on the retrieved sources...",
  "source_websites": ["stackoverflow.com", "learn.microsoft.com"]
}
```

---

### `POST /voice`
Transcribe an audio recording to text via Groq Whisper.

**Request:** `multipart/form-data` with an `audio` file field (`.webm`, `.wav`, `.mp3`)

**Response:**
```json
{ "text": "My printer stopped working after the Windows update" }
```

---

### `GET /live`
Polling endpoint for the frontend to track crawl progress.

**Response:**
```json
{
  "is_crawling": true,
  "current_website": "stackoverflow.com",
  "websites_crawled": ["stackoverflow.com", "learn.microsoft.com"],
  "last_llm_answer": "...",
  "crawl_start_time": "2025-01-01T12:00:00"
}
```

---

### `POST /flush-crawler-results`
Reset session state and optionally clear intermediate JSONL files. Intended for browser `unload` hooks.

---

## ⚙️ Configuration

All tunable parameters are controlled via environment variables:

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Groq API key (required) |
| `PINECONE_API_KEY` | — | Pinecone API key (required) |
| `PINECONE_INDEX_NAME` | `ragcustomersupport` | Pinecone index name |
| `PINECONE_NAMESPACE` | `voice_support_chunks` | Session namespace |
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed model name |
| `RAG_TOP_K` | `5` | Number of retrieved chunks |
| `PINECONE_INDEX_DIM` | `512` | Embedding dimension |
| `MIN_CLARIFICATION_TURNS` | `1` | Minimum Q&A rounds before crawl |
| `MAX_CLARIFICATION_TURNS` | `2` | Maximum Q&A rounds before forced crawl |
| `CRAWLER_MAX_QUERIES_PER_CRAWL` | `8` | Max crawl queries per session |
| `CRAWLER_MAX_QUERIES_PER_WEBSITE` | `2` | Max queries targeting a single domain |
| `CRAWLER_PAGE_TIMEOUT_MS` | `30000` | Browser page timeout (ms) |
| `CLEAR_CRAWLER_RESULTS_ON_FLUSH` | `1` | Clear `webcrawler_result.jasonl` on flush |
| `CLEAR_ALL_JSONL_ON_FLUSH` | `0` | Clear all `.jasonl` files on flush |

---

## 📂 Project Structure

```
├── app.py                          # Flask API orchestrator, background process management
├── the_intellegence.py             # Question graph generation + answer synthesis (Groq LLM)
├── question_validation.py          # Stateful validation gate + clarifying question generation
├── query_preprocessing.py          # spaCy filler removal + text normalization
├── webcrawler.py                   # Async Playwright crawler (crawl4ai), multi-engine search
├── text preprocessing+chunking.py  # Layer 1 chunker — structural clean, paragraph split (watch mode)
├── 2nd_layer_chunking.py           # Layer 2 adaptive NLP filter — noise removal, scoring (watch mode)
├── rag.py                          # Pinecone upsert, semantic search, answer_chunk writer
├── voice_to_text.py                # Groq Whisper API wrapper
├── crawl_quality_audit.py          # Standalone audit tool — coverage and noise analysis
│
├── frontend/
│   ├── index.html                  # Single-page UI with live polling
│   └── logo.svg
│
├── level 2 dfd.drawio.svg          # System architecture diagram (SVG)
├── level 2 dfd.png                 # System architecture diagram (PNG)
│
├── FRONTEND_LIVE_OUTPUT.json       # Live state file (auto-generated, gitignore this)
├── webcrawler_result.jasonl        # Raw crawl output (auto-generated)
├── chunking.jasonl                 # Layer 1 output (auto-generated)
├── 2nd_layer_chunking.jasonl       # Layer 2 output (auto-generated)
├── qdrant_ready.jasonl             # RAG-ready chunks (auto-generated)
└── answer_chunk.jasonl             # Latest retrieval result (auto-generated)
```

> All `.jasonl` files and `FRONTEND_LIVE_OUTPUT.*` are runtime artifacts — add them to `.gitignore`.

---

## 🧠 AI Pipeline Deep Dive

### Layer 1 Scoring Formula
```
score = (overlap × 3.0)
      + min(unique_words / 15.0, 4.0)
      + min(char_length / 1000.0, 3.0)
      - (link_count × 0.2)
      - (boilerplate_hits × 1.0)
```
Where `overlap` = query term intersection with chunk tokens.

### Layer 2 Adaptive Score
```
score = 0.45 × norm(word_count)
      + 0.20 × norm(unique_ratio)
      + 0.15 × norm(avg_word_length)
      - 0.25 × norm(stopword_ratio)
      - 0.10 × norm(punctuation_ratio)
      - 0.05 × norm(numeric_ratio)
      + 0.05 × has_url
```
Hard noise is detected separately via structural heuristics (CSS patterns, nav tokens, known boilerplate URLs) before scoring.

### Query Expansion
`the_intellegence.py` scores all 45+ known technical domains against the query tokens and selects the top 10–14 for site-targeted crawling (e.g., `site:stackoverflow.com`, `site:learn.microsoft.com`, `filetype:pdf`).

---

## 🔒 Security Notes

- All API keys must be stored in `.env`, never hardcoded in source files.
- The `.env` file is excluded from version control via `.gitignore`.
- `atexit` handlers ensure browser processes and subprocesses are cleaned up on server shutdown.
- Pinecone namespaces are session-scoped to prevent cross-user context contamination.

---

## 🚀 Future Scope

This project is intentionally architected with **two modular boundaries** that make it straightforward to evolve from a prototype into a production-grade enterprise system.

---

### 🔄 Replace the Web Crawler with a Company Knowledge Base

Right now, `webcrawler.py` autonomously crawls the public web for every query. This is powerful for general-purpose support — but for enterprises, the real intelligence lives internally: in PDFs, SOPs, ticketing systems, internal wikis, product manuals, and past support transcripts.

The crawler is a **drop-in swappable component**. Because the entire pipeline downstream (chunking → filtering → embedding → Pinecone) operates on a standardized JSONL schema, replacing the crawler with a private knowledge base ingestion module requires zero changes to the rest of the system.

**What this unlocks:**

- Ingest **Confluence pages, Notion docs, SharePoint files, or Zendesk tickets** instead of crawling the open web
- Pre-build and cache a **domain-specific vector index** at deployment time — eliminating real-time crawl latency entirely
- Support **access-controlled retrieval** — different agents see different namespaces based on role or department
- Enable **multi-source hybrid retrieval** — mix internal SOPs with live web results using Reciprocal Rank Fusion (RRF)

> **Enterprise impact:** A bank's support agent that answers from internal compliance documents. A hospital's helpdesk that retrieves from clinical SOPs. A SaaS company's tier-1 support that resolves from its own knowledge base — without a single human in the loop.

---

### 🤖 Replace the Web UI with an Autonomous Voice Agent — The Telecaller of the Future

Currently, `frontend/index.html` is a browser-based interface where users type or speak into a microphone. This is just the **thin client layer** — and it is entirely replaceable.

The real vision: a **fully autonomous outbound/inbound voice agent** that eliminates the traditional telecaller role.

**What this looks like in production:**

- An AI agent **picks up inbound support calls** via Twilio or Plivo, transcribes speech in real time using Whisper, runs it through the full A-RAG pipeline, and **speaks the answer back** using a text-to-speech engine (ElevenLabs, Azure TTS, or Bark) — end to end, zero human involvement
- For **outbound scenarios**, the agent proactively calls customers, walks them through troubleshooting steps, collects confirmations, and logs the resolution — all in natural spoken conversation
- The **validation gate** (`question_validation.py`) already implements the clarification dialogue that a human telecaller would conduct — it just needs voice I/O wired around it
- With **emotion detection and escalation logic**, the agent can recognize frustration, failed resolution, or high-value customers and seamlessly hand off to a human agent at exactly the right moment

**Technology swap required:**

| Current | Production Voice Agent |
|---|---|
| `frontend/index.html` (browser mic) | Twilio Voice / Plivo SIP trunk |
| `voice_to_text.py` (Groq Whisper) | Real-time streaming STT (Deepgram / Azure) |
| Text response via `/live` polling | TTS synthesis (ElevenLabs / Azure Neural TTS) |
| Manual browser session | Persistent phone call session management |

> **Enterprise impact:** A single deployed instance of this system can handle **thousands of concurrent support calls** at a fraction of the cost of a human telecaller team — with consistent quality, zero hold times, 24/7 availability, and full call logging for compliance.

---

### 📋 Additional Roadmap Items

- [ ] **Hybrid Search** — BM25 sparse + semantic vector search with Reciprocal Rank Fusion (RRF)
- [ ] **Graph-RAG** — Neo4j relationship mapping across ingested documents for multi-hop reasoning
- [ ] **Streaming TTS** — Real-time voice response synthesis (ElevenLabs / Bark / Azure Neural TTS)
- [ ] **Local LLM fallback** — Ollama integration for air-gapped or low-latency deployments
- [ ] **Celery/Redis workers** — Production-grade task queue replacing subprocess watchers
- [ ] **Session persistence** — Redis-backed durable session storage replacing module-level globals
- [ ] **Emotion & escalation detection** — Real-time sentiment analysis to trigger human handoff

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **API** | Python 3.12, Flask |
| **LLM** | Groq-hosted LLM inference (OpenAI-compatible API) |
| **STT** | Groq Whisper-large-v3 |
| **Crawling** | crawl4ai, Playwright (Chromium), async |
| **Embeddings** | FastEmbed (`BAAI/bge-small-en-v1.5`, 512-dim) |
| **Vector DB** | Pinecone (Serverless) |
| **NLP** | spaCy (`en_core_web_sm`) |
| **Concurrency** | `threading`, `subprocess.Popen` |
| **Frontend** | Vanilla HTML/JS, polling `/live` |

---

## 🤝 Contributing

Contributions are welcome. Please follow standard fork-and-PR workflow:

```bash
# 1. Fork the repo and clone your fork
git clone https://github.com/your-username/autonomous-voice-support-agent.git

# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Make your changes and commit
git commit -m "feat: describe your change"

# 4. Push and open a Pull Request
git push origin feature/your-feature-name
```

Please ensure your branch passes basic smoke tests before opening a PR.

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built by [Rajarshee Chakraborty](https://www.linkedin.com/in/rajarshee-chakraborty99/)**
&nbsp;·&nbsp;
[GitHub](https://github.com/rajarshee99)
&nbsp;·&nbsp;
[LinkedIn](https://www.linkedin.com/in/rajarshee-chakraborty99/)

*MCA Final Year Project · AI Engineering Portfolio*

</div>
