# LLM Orchestration — Hybrid RAG System

## Overview

This project is a modular Retrieval-Augmented Generation (RAG) system built with Python, LangChain, and LangGraph.

The application allows users to:

* Index PDF documents
* Index YouTube video transcripts
* Store embeddings in ChromaDB locally
* Perform hybrid retrieval (semantic + BM25)
* Re-rank results using CrossEncoder
* Compress retrieved context with LLM-based extraction
* Maintain conversational memory using LangGraph
* Interact with the system through a CLI interface

The architecture was designed to explore modern LLM orchestration patterns, hybrid retrieval strategies, observability, and modular AI engineering practices.

---

# Architecture

```text
                ┌────────────────────┐
                │   User Question    │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │    LangGraph App   │
                └─────────┬──────────┘
                          │
          ┌───────────────┴────────────────┐
          ▼                                ▼
 ┌──────────────────┐          ┌────────────────────┐
 │ Semantic Search  │          │    BM25 Search     │
 │   (ChromaDB)     │          │  (Keyword Search)  │
 └────────┬─────────┘          └─────────┬──────────┘
          └──────────────┬───────────────┘
                         ▼
               ┌───────────────────┐
               │ Hybrid Merge      │
               └─────────┬─────────┘
                         ▼
               ┌───────────────────┐
               │ CrossEncoder      │
               │ Re-ranking        │
               └─────────┬─────────┘
                         ▼
               ┌───────────────────┐
               │ Context Compression│
               └─────────┬─────────┘
                         ▼
               ┌───────────────────┐
               │ LLM Answer        │
               └───────────────────┘
```

---

# Features

## Retrieval-Augmented Generation (RAG)

* Hybrid retrieval pipeline
* Semantic vector search with ChromaDB
* BM25 keyword retrieval
* CrossEncoder reranking
* Context compression
* Conversational memory
* Multi-document support
* YouTube transcript ingestion

## LLM Orchestration

* LangGraph workflow orchestration
* LangChain prompt pipelines
* Stateful conversation handling
* Modular chain structure

## MLOps & Observability

The project includes an `mlops` module containing:

* Metrics
* Logging
* Tracing
* Observability utilities
* Evaluation helpers

---

# Tech Stack

## Core AI Stack

* Python 3.12+
* LangChain
* LangGraph
* ChromaDB
* Sentence Transformers
* HuggingFace Embeddings
* Groq API
* CrossEncoder reranking

## Additional Libraries

* Streamlit
* DuckDuckGo Search
* YouTube Transcript API
* yt-dlp
* rank-bm25
* pypdf
* docx2txt

---

# Project Structure

```text
llm-orchestration/
│
├── data/                     # Persistent data and vector store
├── notebooks/                # Experiments and notebooks
├── tests/                    # Tests
├── src/
│   ├── chains/               # LangChain examples and orchestration
│   ├── rag/                  # Main RAG pipeline
│   ├── mlops/                # Observability and metrics
│   └── utils/                # Utility helpers
│
├── pyproject.toml
├── poetry.lock
├── .env.example
└── README.md
```

---

# Retrieval Pipeline

## 1. Document Ingestion

Supported sources:

* PDF files
* YouTube videos

The ingestion pipeline:

1. Extracts content
2. Splits content into chunks
3. Generates embeddings
4. Stores vectors in ChromaDB

---

## 2. Hybrid Search

The retrieval process combines:

### Semantic Retrieval

Uses:

* `sentence-transformers/all-mpnet-base-v2`
* Chroma vector database
* Max Marginal Relevance (MMR)

### Keyword Retrieval

Uses:

* BM25 ranking
* Token-based retrieval

Both retrieval strategies are merged to improve recall quality.

---

## 3. Re-ranking

Results are re-ranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

This improves relevance before sending context to the LLM.

---

## 4. Context Compression

The system uses:

```python
LLMChainExtractor
```

To reduce unnecessary context and improve token efficiency.

---

## 5. Answer Generation

The current implementation uses:

```text
llama-3.1-8b-instant
```

through the Groq API.

The answer generation is orchestrated using LangGraph state machines.

---

# Installation

## Clone the Repository

```bash
git clone <repository-url>
cd llm-orchestration
```

---

## Install Dependencies

Using Poetry:

```bash
poetry install
```

Or with pip:

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file based on `.env.example`.

Example:

```env
GROQ_API_KEY=your_api_key
CHROMA_PERSIST_DIR=./data/chroma_db
```

---

# Running the Application

## Start the CLI

```bash
python -m src.rag.main
```

---

# Usage

## Index Documents

When the application starts:

```text
Digite "I" para inserir novos documento ou "C" para iniciar o chat
```

Examples:

```text
pdf /path/to/document.pdf
video https://www.youtube.com/watch?v=example
```

Then type:

```text
fim
```

---

## Start Chat

Select:

```text
C
```

Then ask questions about indexed content.

---

# Example Workflow

```text
1. Index PDFs and YouTube videos
2. Generate embeddings
3. Store chunks in ChromaDB
4. Ask questions
5. Retrieve relevant chunks
6. Re-rank results
7. Compress context
8. Generate grounded answers
```

---

# Current Capabilities

* Hybrid RAG
* Persistent vector database
* Conversational memory
* Document deduplication
* YouTube transcript ingestion
* CrossEncoder reranking
* Context compression
* Modular architecture
* Observability utilities

---

# Future Improvements

Potential next steps:

* Web UI with Streamlit
* API layer with FastAPI
* Multi-agent orchestration
* Metadata filtering UI
* Evaluation dashboards
* Redis cache layer
* GPU acceleration
* Distributed vector storage
* Airflow orchestration
* CI/CD pipelines
* Docker support
* Kubernetes deployment

---

# Author

Developed as an experimental AI orchestration and hybrid RAG project focused on modern LLM engineering patterns.

---

# License

* MIT

