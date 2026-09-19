"""
Step 4: Core RAG Pipeline (Embeddings + Vector DB + Retrieval + Generation)
--------------------------------------------------------------------------
1. Generates local embeddings using sentence-transformers (model: all-MiniLM-L6-v2),
   with automated local ONNX runtime fallback if system policies restrict scipy DLLs.
2. Stores & persists chunks in a local ChromaDB collection (./data/chroma_db).
3. Provides retrieve(query, k=4) for semantic search.
4. Provides generate_answer(query) using Gemini API with strict context grounding
   and explicit fallback to "I don't know based on my notes".
5. Provides an interactive CLI loop to ask questions in real-time.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (GEMINI_API_KEY / GOOGLE_API_KEY)
load_dotenv()

# Google GenAI SDK for generation
try:
    from google import genai
except ImportError:
    genai = None

# ChromaDB for vector database
try:
    import chromadb
    import chromadb.utils.embedding_functions as embedding_functions
except ImportError:
    chromadb = None
    embedding_functions = None


# Paths
DEFAULT_CHUNKS_FILE = Path("./data/chunks.json")
DEFAULT_CHROMA_DIR = Path("./data/chroma_db")
COLLECTION_NAME = "handwritten_study_notes"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Prompt template enforcing strict grounding and anti-hallucination guardrails
RAG_SYSTEM_PROMPT = """You are a helpful and precise AI study assistant answering questions based EXCLUSIVELY on a student's handwritten study notes.

Context from the study notes:
{context}

Question:
{query}

Instructions:
1. Answer the question using ONLY the facts and concepts explicitly stated in the context above.
2. If the answer cannot be found in the provided context, you MUST explicitly state: "I don't know based on my notes."
3. Do NOT extrapolate, hallucinate, assume, or incorporate external knowledge not present in the notes.
4. Keep the answer clear, technical, and faithful to the notes. Mention the relevant source note (e.g. Data 36.txt) where appropriate.
"""


def load_local_embedding_engine(model_name: str = EMBEDDING_MODEL_NAME):
    """
    Initializes a local embedding function using all-MiniLM-L6-v2.
    First attempts sentence-transformers; if system policies restrict scipy C-extensions,
    falls back seamlessly to Chroma's built-in all-MiniLM-L6-v2 ONNX engine.
    """
    # Attempt 1: sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        print(f"[RAG] Loading '{model_name}' via sentence-transformers (PyTorch)...")
        st_model = SentenceTransformer(model_name)
        return {"type": "sentence_transformers", "engine": st_model}
    except Exception as exc:
        print(f"[RAG] Note: sentence-transformers unavailable ({exc}).")

    # Attempt 2: ChromaDB native ONNX embedding function (all-MiniLM-L6-v2)
    if embedding_functions is not None:
        print(f"[RAG] Loading '{model_name}' via local ONNX runtime (Zero API cost)...")
        onnx_ef = embedding_functions.DefaultEmbeddingFunction()
        return {"type": "chroma_onnx", "engine": onnx_ef}

    raise RuntimeError("No embedding engine could be initialized. Please check chromadb / onnxruntime installation.")


class RAGPipeline:
    def __init__(
        self,
        chunks_file: Path = DEFAULT_CHUNKS_FILE,
        chroma_dir: Path = DEFAULT_CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        rebuild_db: bool = False,
        gemini_model: str = "gemini-3.5-flash-lite",
    ):
        self.chunks_file = Path(chunks_file)
        self.chroma_dir = Path(chroma_dir)
        self.collection_name = collection_name
        self.gemini_model = gemini_model

        if chromadb is None:
            raise ImportError("Missing required package 'chromadb'. Please install: pip install chromadb")

        # 1. Initialize local embedding model (no API costs)
        self.embed_info = load_local_embedding_engine(embedding_model_name)

        # 2. Initialize persistent ChromaDB vector database
        print(f"[RAG] Connecting to ChromaDB storage at '{self.chroma_dir.resolve()}'...")
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))

        # Handle collection reset if requested
        if rebuild_db:
            try:
                self.chroma_client.delete_collection(name=self.collection_name)
                print(f"[RAG] Deleted existing Chroma collection '{self.collection_name}'.")
            except Exception:
                pass

        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Handwritten study notes embeddings (all-MiniLM-L6-v2)"}
        )

        # 3. Populate collection if empty
        if self.collection.count() == 0:
            self._index_chunks()
        else:
            print(f"[RAG] Loaded existing collection with {self.collection.count()} chunks.")

        # 4. Initialize Gemini client for generation
        self.gemini_client = self._init_gemini_client()

    def _init_gemini_client(self):
        """Initializes Gemini API client with API key from environment."""
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("[Warning] No GEMINI_API_KEY or GOOGLE_API_KEY found. Retrieval will work, but generation requires an API key.")
            return None
        if genai is None:
            print("[Warning] google-genai library not installed. Generation disabled.")
            return None
        return genai.Client(api_key=api_key)

    def _encode_texts(self, texts: list[str]) -> list[list[float]]:
        """Encodes a list of text strings into embedding vectors locally."""
        engine_type = self.embed_info["type"]
        engine = self.embed_info["engine"]

        if engine_type == "sentence_transformers":
            embeddings = engine.encode(texts, show_progress_bar=True).tolist()
            return embeddings
        else:
            # chroma_onnx DefaultEmbeddingFunction
            # processes in batches if necessary
            embeddings = engine(texts)
            # Ensure float lists
            return [list(vec) for vec in embeddings]

    def _index_chunks(self):
        """Reads chunks from JSON, computes embeddings, and indexes into ChromaDB."""
        if not self.chunks_file.exists():
            raise FileNotFoundError(
                f"Chunks file '{self.chunks_file}' not found! Please run chunk_notes.py first."
            )

        print(f"[RAG] Indexing chunks from '{self.chunks_file}' into ChromaDB...")
        with open(self.chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if not chunks:
            raise ValueError(f"No chunks found in '{self.chunks_file}'.")

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            ids.append(chunk["chunk_id"])
            documents.append(chunk["text"])
            metadatas.append({
                "source_file": chunk.get("source_file", ""),
                "chunk_index": int(chunk.get("chunk_index", 0)),
                "topic_heading": chunk.get("topic_heading") or "General",
                "word_count": int(chunk.get("word_count", len(chunk["text"].split()))),
            })

        print(f"[RAG] Generating embeddings for {len(documents)} chunks (all-MiniLM-L6-v2 running locally)...")
        embeddings = self._encode_texts(documents)

        print("[RAG] Adding embeddings and metadata to ChromaDB collection...")
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        print(f"[RAG] Successfully indexed {self.collection.count()} chunks into ChromaDB!")

    def retrieve(self, query: str, k: int = 4):
        """
        Embeds a user's question and retrieves the top-k most similar chunks from ChromaDB.

        Returns a list of dicts with keys:
          - chunk_id
          - text
          - source_file
          - chunk_index
          - topic_heading
          - distance
        """
        query_embedding = self._encode_texts([query])

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, text, meta, dist in zip(ids, docs, metas, distances):
            retrieved.append({
                "chunk_id": chunk_id,
                "text": text,
                "source_file": meta.get("source_file", "Unknown"),
                "chunk_index": meta.get("chunk_index", 0),
                "topic_heading": meta.get("topic_heading", "General"),
                "distance": dist,
            })

        return retrieved

    def generate_answer(self, query: str, k: int = 4):
        """
        End-to-end RAG generation:
        1. Calls retrieve() to get relevant chunks.
        2. Builds grounded prompt with context.
        3. Calls Gemini API to generate the response.
        4. Returns the answer and sources used.
        """
        # a. Retrieve top-k chunks
        retrieved_chunks = self.retrieve(query=query, k=k)

        # Format context blocks
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            header = f"[Source #{i}: {chunk['source_file']} (Chunk {chunk['chunk_index']}, Topic: {chunk['topic_heading']})]"
            context_parts.append(f"{header}\n{chunk['text']}")

        context_text = "\n\n---\n\n".join(context_parts)

        # b. Build prompt
        prompt = RAG_SYSTEM_PROMPT.format(context=context_text, query=query)

        # c. Send prompt to Gemini API
        if not self.gemini_client:
            return {
                "query": query,
                "answer": "[Error] Gemini client not initialized. Please set GEMINI_API_KEY in your environment or .env file.",
                "sources": retrieved_chunks,
                "prompt": prompt,
            }

        try:
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
            )
            answer_text = response.text.strip() if response and response.text else "No response generated."
        except Exception as exc:
            answer_text = f"[Error generating answer: {exc}]"

        # d. Return answer with sources
        return {
            "query": query,
            "answer": answer_text,
            "sources": retrieved_chunks,
            "prompt": prompt,
        }


def run_interactive_cli(pipeline: RAGPipeline, k: int = 4):
    """Simple command-line loop to type questions and inspect answers + sources."""
    print("\n" + "=" * 70)
    print(" RAG Study Assistant — Interactive Question Loop")
    print(" (Type your question and press Enter. Type 'exit' or 'quit' to stop)")
    print("=" * 70 + "\n")

    while True:
        try:
            query = input("\nAsk a question > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Exiting RAG assistant. Goodbye!")
                break

            print("\nSearching notes and generating answer...")
            result = pipeline.generate_answer(query, k=k)

            print("\n" + "-" * 70)
            print("ANSWER:")
            print("-" * 70)
            print(result["answer"])

            print("\n" + "-" * 70)
            print("SOURCES RETRIEVED FROM NOTES:")
            print("-" * 70)
            for idx, src in enumerate(result["sources"], start=1):
                print(f"  [{idx}] {src['source_file']} (Chunk {src['chunk_index']}) | Heading: '{src['topic_heading']}' | Distance: {src['distance']:.4f}")
            print("-" * 70)

        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted. Exiting.")
            break


def main():
    parser = argparse.ArgumentParser(description="Core RAG Pipeline for Handwritten Study Notes.")
    parser.add_argument("--query", "-q", type=str, default=None, help="Single question to query and exit.")
    parser.add_argument("--k", "-k", type=int, default=4, help="Number of top chunks to retrieve (default: 4).")
    parser.add_argument("--rebuild-db", action="store_true", help="Rebuild ChromaDB collection from chunks.json.")
    parser.add_argument("--model", "-m", type=str, default="gemini-3.5-flash-lite", help="Gemini model for generation.")
    parser.add_argument("--chunks", type=str, default="./data/chunks.json", help="Path to chunks.json.")
    parser.add_argument("--chroma-dir", type=str, default="./data/chroma_db", help="Path to ChromaDB directory.")

    args = parser.parse_args()

    pipeline = RAGPipeline(
        chunks_file=Path(args.chunks),
        chroma_dir=Path(args.chroma_dir),
        rebuild_db=args.rebuild_db,
        gemini_model=args.model,
    )

    if args.query:
        # Single query mode
        print(f"\nQuestion: {args.query}")
        result = pipeline.generate_answer(args.query, k=args.k)
        print("\nAnswer:")
        print(result["answer"])
        print("\nSources Used:")
        for src in result["sources"]:
            print(f" - {src['source_file']} (Chunk {src['chunk_index']}, Topic: {src['topic_heading']})")
    else:
        # Interactive loop
        run_interactive_cli(pipeline, k=args.k)


if __name__ == "__main__":
    main()
