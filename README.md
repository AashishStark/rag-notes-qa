# rag-notes-qa

A Retrieval-Augmented Generation (RAG) system that answers questions over my own handwritten study notes (AWS/cloud topics — EC2, EBS, storage, etc.), built as a hands-on project while transitioning toward GenAI Engineering.

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
Vector store (ChromaDB, local)
        ↓
Retrieval (top-k similarity search)
        ↓
Answer generation (Gemini, context-grounded)
```

## Tech stack

- Python
- `sentence-transformers` — local embeddings, no API cost
- `chromadb` — local vector database
- Gemini API — transcription + answer generation
- (Later) Docker — for deployment

## Project structure

```
rag-notes-qa/
├── data/
│   ├── raw_images/          # source handwritten note photos
│   ├── transcribed_text/    # OCR output, one .txt per image
│   ├── chunks.json          # chunked text + metadata
│   └── eval_set.json        # question/answer pairs for manual evaluation
├── src/
│   ├── preprocess.py        # image → transcribed text
│   ├── chunk.py              # text → overlapping chunks
│   ├── build_index.py        # chunks → embeddings → ChromaDB
│   ├── rag.py                 # retrieve() + generate_answer()
│   └── evaluate.py            # runs eval_set.json through the pipeline
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Set your Gemini API key as an environment variable:

```bash
export GOOGLE_API_KEY="your-key-here"
```

## Usage

```bash
# 1. Transcribe handwritten notes
python src/preprocess.py

# 2. Chunk the transcribed text
python src/chunk.py

# 3. Build the vector index
python src/build_index.py

# 4. Ask questions
python src/rag.py
```

## Evaluation

`src/evaluate.py` runs a fixed set of test questions (`data/eval_set.json`) through the pipeline and prints generated answers alongside expected answers, to manually check retrieval quality and catch hallucinations — including questions with no answer in the notes, to confirm the system says so rather than guessing.

## Status

🚧 In progress — built as Project 3 of a self-directed GenAI Engineer transition roadmap.

## Notes

- No model training/fine-tuning involved — this uses pretrained embedding and generation models as-is.
- Chunking is currently fixed-size; may revisit with semantic chunking later.
