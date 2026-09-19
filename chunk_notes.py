"""
Step 2: Text Chunking for RAG Pipeline
--------------------------------------
Reads transcribed notes from ./data/transcribed_text/, splits them into
overlapping chunks (300-500 words with ~50 word overlap), detects topic
headings, and saves structured chunk data with metadata into ./data/chunks.json.
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path


# Regex patterns to detect section headings across various note styles
HEADING_PATTERNS = [
    re.compile(r"^#{1,4}\s+(.+)$"),                                      # Markdown: # Heading
    re.compile(r"^(?:\*\s*)?\*{1,2}([A-Za-z0-9][^\*]+?)\*{1,2}\s*$"),   # Bold: **Heading:**
    re.compile(r"^(\d+[\w\.\)]*\s+[A-Za-z0-9].*?):?$"),                 # Numbered: 6.2 EC2 ..., 8.1) INTRO
    re.compile(r"^(?:[-*]\s*)?([A-Z0-9][A-Za-z0-9\s/&_().-]{2,50})\s*:[-]?\s*$"),  # Title:
    re.compile(r"^((?:Topic\s+\d+|About\s+[A-Za-z0-9\s/&-]+).*?):?$"),  # Topic / About
]


def natural_sort_key(path: Path):
    """Sorts filenames naturally (e.g., Data 1, Data 2, ... Data 10) instead of lexicographically."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", path.name)]


def clean_heading(raw_heading: str) -> str:
    """Cleans markdown artifacts and trailing punctuation from a heading candidate."""
    cleaned = re.sub(r"[*#_`]", "", raw_heading).strip()
    return cleaned.rstrip(":").strip()


def extract_headings_with_positions(text: str):
    """
    Scans the text for topic headings and tracks the approximate word offset
    where each heading occurs.
    Returns: list of tuples -> [(word_offset, heading_text), ...]
    """
    headings = []
    lines = text.splitlines()
    running_word_count = 0

    bad_tokens = ["frac", "lambda", "sigma", "alpha", "cases", "theta", "simultaneous", "begin", "end"]

    for line in lines:
        s = line.strip()
        words_in_line = len(s.split())

        # Ignore empty lines, LaTeX blocks, and code fences
        if s and not s.startswith("$$") and not s.startswith("```"):
            for pat in HEADING_PATTERNS:
                m = pat.match(s)
                if m:
                    candidate = clean_heading(m.group(1))
                    # Filter out mathematical expressions or overly short/long strings
                    if not any(token in candidate.lower() for token in bad_tokens):
                        if 2 < len(candidate) < 60:
                            headings.append((running_word_count, candidate))
                            break

        running_word_count += words_in_line

    return headings


def determine_chunk_heading(chunk_start_word: int, chunk_end_word: int, headings: list):
    """
    Identifies the most relevant topic heading for a chunk:
    1. First heading occurring inside this chunk's word range.
    2. If none inside, the most recent preceding heading prior to this chunk.
    3. If none prior, the first heading in the file, or None if no headings detected.
    """
    if not headings:
        return None

    # Check for headings that begin inside this chunk
    for word_pos, heading in headings:
        if chunk_start_word <= word_pos < chunk_end_word:
            return heading

    # Otherwise look backwards for the most recent preceding heading
    preceding = [heading for word_pos, heading in headings if word_pos <= chunk_start_word]
    if preceding:
        return preceding[-1]

    # Fallback to the first heading in the document
    return headings[0][1]


def chunk_document(
    text: str,
    source_filename: str,
    chunk_size: int = 300,
    overlap: int = 50,
):
    """
    Splits a single document into fixed-size overlapping word chunks.
    Preserves original text layout if file fits completely within chunk_size.
    """
    words = text.split()
    if not words:
        return []

    stem = Path(source_filename).stem
    headings = extract_headings_with_positions(text)
    stride = max(1, chunk_size - overlap)
    chunks = []

    # Case A: Document fits into a single chunk
    if len(words) <= chunk_size:
        heading = determine_chunk_heading(0, len(words), headings)
        chunks.append({
            "chunk_id": f"{stem}_chunk_0",
            "text": text.strip(),
            "source_file": source_filename,
            "chunk_index": 0,
            "topic_heading": heading,
            "word_count": len(words),
        })
        return chunks

    # Case B: Document requires overlapping chunks
    start = 0
    chunk_index = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        heading = determine_chunk_heading(start, end, headings)

        chunks.append({
            "chunk_id": f"{stem}_chunk_{chunk_index}",
            "text": chunk_text,
            "source_file": source_filename,
            "chunk_index": chunk_index,
            "topic_heading": heading,
            "word_count": len(chunk_words),
        })
        chunk_index += 1

        if end >= len(words):
            break
        start += stride

    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Chunk transcribed study notes for RAG embeddings with overlap and heading metadata."
    )
    default_input = "./data/transcribed_text"
    default_output = "./data/chunks.json"

    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        default=default_input,
        help=f"Folder containing transcribed .txt files (default: '{default_input}')",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        type=str,
        default=default_output,
        help=f"Output JSON file path for chunks (default: '{default_output}')",
    )
    parser.add_argument(
        "--chunk-size",
        "-s",
        type=int,
        default=300,
        help="Target chunk size in words (default: 300 words, roughly 300-500)",
    )
    parser.add_argument(
        "--overlap",
        "-v",
        type=int,
        default=50,
        help="Overlap in words between consecutive chunks (default: 50 words)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    chunk_size = args.chunk_size
    overlap = args.overlap

    print("=" * 70)
    print(" RAG Step 2: Text Chunking for Embedding")
    print("=" * 70)
    print(f" Input Directory : {input_dir.resolve()}")
    print(f" Output File     : {output_file.resolve()}")
    print(f" Chunk Size      : {chunk_size} words")
    print(f" Overlap         : {overlap} words (stride: {chunk_size - overlap} words)")
    print("=" * 70)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    txt_files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt"]
    txt_files.sort(key=natural_sort_key)

    if not txt_files:
        print(f"Error: No .txt files found in '{input_dir}'.")
        sys.exit(1)

    all_chunks = []
    single_chunk_files = 0
    multi_chunk_files = 0

    for txt_file in txt_files:
        text = txt_file.read_text(encoding="utf-8")
        doc_chunks = chunk_document(
            text=text,
            source_filename=txt_file.name,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        if len(doc_chunks) == 1:
            single_chunk_files += 1
        elif len(doc_chunks) > 1:
            multi_chunk_files += 1

        all_chunks.extend(doc_chunks)

    # Ensure output directory exists and save JSON
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # =========================================================================
    # Summary and Sanity Check Display
    # =========================================================================
    print(f"\nProcessing Complete:")
    print(f"  Total Source Notes Processed : {len(txt_files)}")
    print(f"  Notes Fitting in 1 Chunk     : {single_chunk_files}")
    print(f"  Notes Split Across Chunks    : {multi_chunk_files}")
    print(f"  Total Chunks Created         : {len(all_chunks)}")
    print(f"  Saved Chunks To              : {output_file.resolve()}")

    # Display 2-3 sample chunks for sanity check
    print("\n" + "=" * 70)
    print(" SANITY CHECK: 3 Sample Chunks")
    print("=" * 70)

    # Pick samples: 1 single-chunk note, and 2 overlapping chunks from a multi-chunk note
    sample_indices = [0]
    # Find first multi-chunk note
    multi_sample_found = False
    for i, chunk in enumerate(all_chunks):
        if chunk["chunk_index"] == 1:
            sample_indices = [0, i - 1, i]
            multi_sample_found = True
            break

    if not multi_sample_found and len(all_chunks) >= 3:
        sample_indices = [0, 1, 2]

    for display_num, idx in enumerate(sample_indices, start=1):
        c = all_chunks[idx]
        print(f"\n--- [Example {display_num}] Chunk ID: {c['chunk_id']} ---")
        print(f"  Source File   : {c['source_file']}")
        print(f"  Chunk Index   : {c['chunk_index']}")
        print(f"  Topic Heading : {c['topic_heading']}")
        print(f"  Word Count    : {c['word_count']} words")
        print("  Text Snippet  :")
        lines = c["text"].splitlines()
        preview = lines[:4]
        for line in preview:
            print(f"    | {line}")
        if len(lines) > 4:
            print(f"    | ... ({len(lines) - 4} more line(s))")

    print("\n" + "=" * 70)
    print("Step 2 Complete! Chunks are saved and ready for Step 3/4 (Embeddings & RAG).")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
