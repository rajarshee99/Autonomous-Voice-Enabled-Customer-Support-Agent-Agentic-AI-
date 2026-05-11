# 🎙️ Autonomous Voice-Enabled Customer Support Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Framework: Flask](https://img.shields.io/badge/Framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Crawler: crawl4ai](https://img.shields.io/badge/Crawler-crawl4ai-orange.svg)](https://github.com/unclecode/crawl4ai)
[![VectorDB: Pinecone](https://img.shields.io/badge/VectorDB-Pinecone-blueviolet.svg)](https://www.pinecone.io/)

**An elite, production-grade A-RAG (Autonomous Retrieval-Augmented Generation) system engineered for complex technical troubleshooting via voice and structured reasoning.**

---

## 📑 Executive Summary

Traditional customer support bots rely on static knowledge bases that quickly become obsolete. This project implements an **Autonomous Voice Customer Support Agent**—a sophisticated AI system that doesn't just retrieve information but *actively explores* the web to resolve technical issues.

By combining **Whisper-based voice transcription**, **LLM-driven query decomposition**, and an **autonomous crawling pipeline**, the system reverse-engineers a user's problem into a structured "Question Graph." It then orchestrates a multi-layered ingestion pipeline—crawling targeted technical domains, filtering noise through adaptive scoring, and performing semantic retrieval via Pinecone—to synthesize precise, actionable resolutions.

---

## 🚀 Core Capabilities

### 1. **Autonomous Knowledge Discovery (A-RAG)**
Unlike standard RAG, this system triggers real-time web exploration. It dynamically generates search queries, targets high-authority domains (StackOverflow, Microsoft Docs, Reddit), and builds a context-specific vector corpus on the fly.

### 2. **Multi-Stage Adaptive Filtering**
Web content is inherently noisy. The architecture employs a dual-layer chunking strategy:
- **Layer 1**: Structural cleaning and paragraph-based normalization.
- **Layer 2**: Adaptive scoring using NLP metrics (stopword ratios, unique word density, and boilerplate detection) to ensure only high-signal content enters the vector database.

### 3. **Structured Reasoning & Graph Generation**
The "Intelligence" module decomposes vague user queries into a formal **Question Graph**, identifying:
- **Intents**: The core objective.
- **Entities**: Hardware/software components.
- **Constraints**: Environment-specific limitations.
- **Decomposed Questions**: Sub-problems required for a total solution.

### 4. **Session-Aware Validation Gate**
A stateful validation engine enforces a "minimum information" threshold. It engages the user in a clarifying dialogue to collect missing details (e.g., error codes, OS versions) before initiating expensive crawling operations.

---

## 🏗️ System Architecture

The system is designed with a **decoupled, process-oriented architecture** where the main API orchestrates multiple background workers.

### 🔄 High-Level Orchestration
```mermaid
graph TD
    User((User)) <-->|Voice/Web| API[Flask API Server]
    
    subgraph "Reasoning & Validation"
        API --> VAL[Question Validation Engine]
        VAL -->|Needs Detail| API
        VAL -->|Ready| Intel[Reasoning Engine: Graph Gen]
    end

    subgraph "Autonomous Ingestion Pipeline"
        Intel -->|Queries| Crawler[crawl4ai: Autonomous Crawler]
        Crawler -->|Raw JSONL| L1[Layer 1: Structural Chunker]
        L1 -->|Clean JSONL| L2[Layer 2: Adaptive Scorer]
        L2 -->|RAG Ready| RAG[Pinecone Upsert & Embed]
    end

    subgraph "Retrieval & Synthesis"
        RAG -->|Semantic Search| Sync[Answer Synthesis Engine]
        Sync -->|Final Resolution| API
    end
```

### 🧵 Background Process Management
The application manages concurrency using `subprocess` watchers and `threading` events to ensure that the RAG context remains "fresh" as the crawler progresses.
- **Chunker Watcher**: Monitors crawler output for real-time processing.
- **RAG Refresh Watcher**: Automatically triggers re-embedding when the local corpus grows.

---

## 🧠 AI Pipeline Deep Dive

### Step 1: Voice & Query Preprocessing
- **Audio**: Transcribed via `Whisper-large-v3` on Groq for sub-second latency.
- **Normalization**: `spaCy` (en_core_web_sm) is used to strip discourse markers ("um", "uh") and filler text, isolating the technical core.

### Step 2: The Validation Gate
`question_validation.py` implements a stateful gatekeeper. It analyzes query density and topic hits (Account, BSOD, Network). If the "Enough Data" threshold isn't met, it uses LLM-generated clarifying questions to bridge the gap.

### Step 3: Question Graph & Query Expansion
`the_intellegence.py` transforms the query into a multi-dimensional graph. It expands a single user question into 10–14 targeted search queries using advanced site operators (e.g., `site:learn.microsoft.com filetype:pdf`).

### Step 4: Autonomous Crawling & Noise Reduction
Using `crawl4ai`, the system bypasses search engine noise to extract markdown content.
- **Layer 2 Filtering**: Implements a custom scoring algorithm:
  $$Score = (Overlap \times 3.0) + \min(UniqueWords/15.0, 4.0) - (LinkLike \times 0.2)$$
  This ensures that "navigation chrome" and "footer junk" are discarded, while technical steps are preserved.

---

## 🗄️ Retrieval Strategy (RAG)

| Component | Strategy |
| :--- | :--- |
| **Vector DB** | Pinecone (Serverless) |
| **Embeddings** | `BAAI/bge-small-en-v1.5` (via FastEmbed) |
| **Index Dim** | 512 |
| **Retrieval** | Semantic Similarity Search ($Top\_K=5$) |
| **Namespace** | Session-isolated namespaces to prevent cross-user context drift |

---

## 📂 Project Structure

| File / Folder | Responsibility |
| :--- | :--- |
| `app.py` | Main orchestrator; manages API routes and background subprocesses. |
| `the_intellegence.py` | Reasoning module; generates graphs, expanded queries, and final synthesis. |
| `rag.py` | RAG interface; manages Pinecone connections, embeddings, and semantic search. |
| `webcrawler.py` | Autonomous explorer; uses `crawl4ai` for targeted web ingestion. |
| `question_validation.py` | Gatekeeper; enforces data sufficiency through clarifying dialogue. |
| `2nd_layer_chunking.py`| Adaptive NLP filter; scores and cleans web data for RAG readiness. |
| `voice_to_text.py` | Transcription service; interfaces with Groq/Whisper API. |
| `frontend/` | Interactive web UI with real-time progress tracking. |

---

## 🛠️ Tech Stack

- **AI/LLM**: Groq (Llama-3-70B/8B), Whisper-large-v3.
- **Backend**: Python 3.12, Flask.
- **Crawling**: `crawl4ai` (Playwright-based), asynchronous link extraction.
- **NLP/Embeddings**: `FastEmbed`, `spaCy`, `re` (Regex-based structural analysis).
- **Storage**: Pinecone (Vector), JSONL (Intermediate state).
- **Concurrency**: `threading`, `subprocess.Popen` (Watcher patterns).

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.10 or 3.12
- Pinecone Account
- Groq API Key

### 2. Environment Configuration
Create a `.env` file in the root:
```env
GROQ_API_KEY=your_groq_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=ragcustomersupport
PINECONE_NAMESPACE=voice_support_chunks
RAG_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

### 3. Dependency Installation
```bash
# Install core dependencies
pip install -r requirements.txt

# Install NLP models
python -m spacy download en_core_web_sm

# Initialize browser for crawler
playwright install chromium
```

---

## 🖥️ Running the System

Execute the main application:
```bash
python app.py
```

### Expected Startup Sequence:
1. **Flask API** initializes on `http://127.0.0.1:5000`.
2. **Chunker Watcher** starts (monitoring `webcrawler_result.jasonl`).
3. **Second-Layer Watcher** starts (monitoring `chunking.jasonl`).
4. **RAG Watcher** starts (waiting for `qdrant_ready.jasonl` signature changes).

---

## 📡 API Documentation

### `POST /query`
Main endpoint for text queries.
- **Input**: `{"query": "string"}`
- **Behavior**: Returns `needs_more_info` (with questions) or `complete` (triggering crawl).

### `POST /voice`
Endpoint for audio processing.
- **Input**: `multipart/form-data` with `audio` blob.
- **Output**: `{"text": "transcribed_string"}`

### `GET /live`
Polling endpoint for the frontend.
- **Response**: Returns current crawl status, websites found, and current LLM answer.

---

## 📐 Engineering Design Decisions

1. **Why Layered Chunking?** Web data is 80% noise (headers, footers). By using an adaptive scorer in Layer 2, we reduce vector DB costs and increase retrieval precision.
2. **Why Site Operators?** Standard search is too broad. By forcing `site:stackoverflow.com` or `site:microsoft.com`, we guarantee high-quality technical documentation.
3. **Why Watcher Subprocesses?** Decoupling ingestion from the API ensures the UI remains responsive even during heavy crawling/embedding workloads.

---

## 🛡️ Security & Scalability

- **API Security**: Keys are managed via environment variables. `atexit` ensures all browser instances and subprocesses are killed on server shutdown.
- **Scalability**: The system is modular. The crawler and chunkers can be moved to dedicated worker nodes (e.g., Celery/Redis) with minimal refactoring.
- **Error Handling**: Implements resilient JSONL parsing to handle partial writes or system interruptions during crawling.

---

## 🔮 Future Roadmap

- [ ] **Hybrid Search**: Combine BM25 keyword matching with Semantic Vector search.
- [ ] **Graph-RAG Integration**: Use Neo4j to map relationships between disparate crawled documents.
- [ ] **Local LLM Fallback**: Integrate Ollama for offline resolution of basic queries.
- [ ] **Streaming Voice Output**: Implement ElevenLabs or Bark for real-time vocal responses.

---

## 🤝 Contributing

Contributions are welcome! Please follow the standard fork-and-pull-request workflow.

1. Fork the repo.
2. Create a feature branch (`git checkout -b feature/xyz`).
3. Commit changes (`git commit -m 'Add xyz'`).
4. Push to branch (`git push origin feature/xyz`).
5. Open a Pull Request.

---

## 📄 License

This project is licensed under the **MIT License**.

---

**Developed by**: [Your Name/Portfolio]
**Purpose**: Final Year MCA Project / AI Engineering Showcase
**Contact**: [Your Email/LinkedIn]

> "Transforming technical noise into actionable intelligence."
