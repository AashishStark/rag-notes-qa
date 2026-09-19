# rag-notes-qa

A Retrieval-Augmented Generation (RAG) system that answers questions over my own handwritten study notes (ML, DL Basics/SpringSecurity/AWS/cloud topics — EC2, EBS, storage, etc.), built as a hands-on project while transitioning toward GenAI Engineering.

## 🔴 Live

Deployed on Railway: **https://rag-notes-qa-production-65e2.up.railway.app**

- `POST /query` — ask a question, get a grounded answer with sources
- `GET /health` — service health check
- `GET /metrics` — basic request/latency/error metrics

Example:
```bash
curl -X POST https://rag-notes-qa-production-65e2.up.railway.app/query -H "Content-Type: application/json" -d "{\"question\": \"What is AWS EBS and what are its use cases?\", \"k\": 3}"
```

## What this does

Takes photographed handwritten notes, transcribes them, and turns them into a queryable Q&A system: ask a question, get an answer grounded in the actual notes — with the source chunk cited, and an explicit "I don't know" when the answer isn't in the notes rather than a hallucinated guess.

## Pipeline

```
Handwritten images (JPG/PNG)
        ↓
OCR / transcription (Gemini vision)
        ↓
Chunking (300–500 words, ~50-word overlap)
        ↓
Embeddings (sentence-transformers, all-MiniLM-L6-v2)
        ↓
Vector store (ChromaDB, persistent volume)
        ↓
Retrieval (top-k similarity search)
        ↓
Answer generation (Gemini, context-grounded)
        ↓
FastAPI (/query, /health, /metrics) → Dockerized → deployed on Railway
```

## Tech stack

- Python
- `sentence-transformers` — local embeddings, no API cost
- `chromadb` — vector database, persisted via a mounted volume
- Gemini API — transcription + answer generation
- FastAPI — REST API layer (`/query`, `/health`, `/metrics`)
- Docker — containerized deployment
- Railway — hosting

## Project structure

```
rag-notes-qa/
├── Dataset/
│   └── My Studies/          # source handwritten note photos (raw input)
├── data/
│   ├── chroma_db/           # persisted ChromaDB vector store
│   ├── transcribed_text/    # OCR output, one .txt per image
│   ├── chunks.json          # chunked text + metadata
│   ├── eval_set.json        # question/answer pairs for evaluation
│   └── eval_results.json    # generated answers vs. expected, from evaluate_rag.py
├── transcribe_notes.py      # image → transcribed text
├── chunk_notes.py           # text → overlapping chunks
├── rag_pipeline.py          # RAGPipeline: index, retrieve(), generate_answer(), CLI
├── generate_eval_set.py     # builds eval_set.json
├── evaluate_rag.py          # runs eval_set.json through the pipeline → eval_results.json
├── output_transcribe_notes.txt  # log/output from the transcription run
├── requirements.txt
├── .env                     # GEMINI_API_KEY / GOOGLE_API_KEY (not committed)
├── .gitignore
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Set your Gemini API key as an environment variable (or in `.env`):

```bash
export GOOGLE_API_KEY="your-key-here"
```

## Usage

```bash
# 1. Transcribe handwritten notes (Dataset/My Studies → data/transcribed_text/)
python transcribe_notes.py

# 2. Chunk the transcribed text → data/chunks.json
python chunk_notes.py

# 3. Build the vector index + ask questions (indexes on first run, then interactive CLI)
python rag_pipeline.py

# Or ask a single question directly:
python rag_pipeline.py --query "What is EBS used for?"

# Rebuild the index from scratch if chunks.json changed:
python rag_pipeline.py --rebuild-db
```

## Evaluation

```bash
# Generate the evaluation question set (if not already created)
python generate_eval_set.py

# Run the eval set through the pipeline → data/eval_results.json
python evaluate_rag.py
```

`evaluate_rag.py` runs a fixed set of test questions (`data/eval_set.json`) through the pipeline and saves generated answers alongside expected answers to `data/eval_results.json`, to manually check retrieval quality and catch hallucinations — including questions with no answer in the notes, to confirm the system says so rather than guessing.

## Status

✅ Deployed — Projects 3 and 4 of a self-directed GenAI Engineer transition roadmap (RAG pipeline → API → Docker → deployment → monitoring).

## Notes

- No model training/fine-tuning involved — this uses pretrained embedding and generation models as-is.
- Chunking is currently fixed-size; may revisit with semantic chunking later.
