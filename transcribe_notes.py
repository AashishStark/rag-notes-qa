"""
Step 1: Handwritten Study Notes Transcription using Gemini Vision API
---------------------------------------------------------------------
Transcribes scanned/photographed handwritten study notes into structured
plain text / Markdown files, preserving headings, bullet points, and indentation.
Marks illegible words as [unclear] for manual review.
"""

import os
import sys
import re
import time
import argparse
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

# Try importing the official modern Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
except ImportError:
    print("Error: 'google-genai' is not installed.")
    print("Please install it using: pip install google-genai pillow python-dotenv")
    sys.exit(1)

# Default prompt instructed to preserve structure and mark illegible text as [unclear]
TRANSCRIPTION_PROMPT = """You are an expert OCR and transcription assistant specializing in handwritten technical study notes (covering AWS, Spring Security, ML/DL, etc.).

Your task is to transcribe the handwritten content in this image EXACTLY as written.

Strict Instructions:
1. Exact Transcription: Transcribe the text verbatim. Do NOT summarize, explain, paraphrase, or alter any technical terms.
2. Structure Preservation: Preserve headings, sub-headings, bullet points, numbered lists, arrows/diagram labels, and indentation. Use Markdown formatting (# for headings, - or * for bullets, indented lists) to mirror the layout of the page.
3. Illegible Words: If any word, phrase, or symbol is genuinely illegible or cannot be determined with certainty, write '[unclear]' in its place. Never guess or fabricate text.
4. Output Cleanliness: Output ONLY the transcribed content. Do not include introductory notes, conversational remarks (e.g., "Here is the transcription:"), or meta commentary.
"""

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def natural_sort_key(path: Path):
    """Sorts filenames naturally (e.g. Data 1, Data 2, ... Data 10) instead of lexicographically."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", path.name)]


def get_gemini_client():
    """Loads environment variables and initializes the Gemini API client."""
    # Load .env file if available
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\n" + "=" * 70)
        print("ERROR: Gemini API Key not found!")
        print("=" * 70)
        print("Please set your Gemini API key in one of the following ways:")
        print("  1. Create a '.env' file in this folder with:")
        print("     GEMINI_API_KEY=your_actual_api_key_here")
        print("  2. Or set it in your terminal:")
        print("     PowerShell:  $env:GEMINI_API_KEY=\"your_actual_api_key_here\"")
        print("     CMD:         set GEMINI_API_KEY=your_actual_api_key_here")
        print("     Bash:        export GEMINI_API_KEY=\"your_actual_api_key_here\"")
        print("=" * 70 + "\n")
        sys.exit(1)

    return genai.Client(api_key=api_key)


def transcribe_image_with_retry(client, image_path: Path, model_name: str, max_retries: int = 3) -> str:
    """
    Sends an image and transcription prompt to Gemini API with exponential backoff for retries.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            attempt += 1
            # Open and verify the image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary (e.g., RGBA or palette images)
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")

                response = client.models.generate_content(
                    model=model_name,
                    contents=[img, TRANSCRIPTION_PROMPT],
                )

                if response and response.text:
                    return response.text.strip()
                else:
                    raise ValueError("Received an empty response from Gemini API.")

        except Exception as exc:
            err_str = str(exc)
            # Check for rate limit or transient service errors
            is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "Quota exceeded" in err_str
            if attempt < max_retries:
                wait_time = (2 ** attempt) * 2 if not is_rate_limit else 15
                print(f"\n    [Warning] Error on attempt {attempt}/{max_retries} for {image_path.name}: {exc}")
                print(f"    Waiting {wait_time}s before retrying...")
                time.sleep(wait_time)
            else:
                raise exc


def find_unclear_tags(text: str):
    """
    Finds occurrences of [unclear] in the text along with line number and snippet.
    Returns list of dicts: [{'line': line_no, 'snippet': line_content}]
    """
    results = []
    lines = text.splitlines()
    for idx, line in enumerate(lines, start=1):
        if re.search(r"\[unclear\]", line, re.IGNORECASE):
            # Clean up line snippet for compact display
            snippet = line.strip()
            if len(snippet) > 80:
                snippet = snippet[:77] + "..."
            results.append({"line": idx, "snippet": snippet})
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe handwritten study notes using Google Gemini Vision API."
    )
    # Automatically check common locations if no directory is specified
    default_input = "Dataset/My Studies" if os.path.exists("Dataset/My Studies") else "./data/raw_images"
    default_output = "./data/transcribed_text"

    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        default=default_input,
        help=f"Path to input folder containing image files (default: '{default_input}')",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=default_output,
        help=f"Path to output folder for transcribed .txt files (default: '{default_output}')",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gemini-3.5-flash-lite",
        help="Gemini vision model to use (e.g. gemini-3.7-flash, gemini-2.0-flash, gemini-1.5-flash)",
    )
    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        default=1.5,
        help="Delay in seconds between requests to respect API rate limits (default: 1.5s)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-transcribing images even if output .txt file already exists",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit number of images to process (useful for test runs)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print(" RAG Step 1: Handwritten Study Notes Transcription")
    print("=" * 70)
    print(f" Input Directory  : {input_dir.resolve()}")
    print(f" Output Directory : {output_dir.resolve()}")
    print(f" Gemini Model     : {args.model}")
    print(f" Skip Existing    : {'No (Overwrite)' if args.force else 'Yes'}")
    print("=" * 70)

    # Validate input directory
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    # Find and naturally sort all matching image files
    all_files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
    all_files.sort(key=natural_sort_key)

    if not all_files:
        print(f"No image files ({', '.join(IMAGE_EXTENSIONS)}) found in '{input_dir}'.")
        sys.exit(0)

    if args.limit:
        all_files = all_files[:args.limit]
        print(f"Processing limit set to {args.limit} images.")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Gemini client
    client = get_gemini_client()

    # Tracking metrics
    total_images = len(all_files)
    succeeded_count = 0
    failed_count = 0
    skipped_count = 0
    failed_files = []
    unclear_reports = {}  # {filename: list of {line, snippet}}

    print(f"\nFound {total_images} image(s) to process. Starting transcription...\n")

    for index, image_path in enumerate(all_files, start=1):
        output_txt_path = output_dir / f"{image_path.stem}.txt"
        prefix = f"[{index}/{total_images}] {image_path.name}"

        # Resume / Skip existing
        if output_txt_path.exists() and not args.force:
            print(f"{prefix:<35} -> Already transcribed, skipping. (--force to overwrite)")
            skipped_count += 1
            # Scan existing text for [unclear] tags so the final summary includes them
            try:
                existing_text = output_txt_path.read_text(encoding="utf-8")
                unclears = find_unclear_tags(existing_text)
                if unclears:
                    unclear_reports[image_path.name] = unclears
            except Exception:
                pass
            continue

        print(f"{prefix:<35} -> Transcribing...", end="", flush=True)

        try:
            transcription = transcribe_image_with_retry(
                client=client,
                image_path=image_path,
                model_name=args.model,
            )

            # Save transcription as .txt with UTF-8 encoding
            output_txt_path.write_text(transcription, encoding="utf-8")

            # Check for [unclear] tags
            unclears = find_unclear_tags(transcription)
            if unclears:
                unclear_reports[image_path.name] = unclears
                print(f" Done! (Found {len(unclears)} [unclear] spot(s))")
            else:
                print(" Done! (100% clear)")

            succeeded_count += 1

            # Respect rate limits between calls
            if args.delay > 0 and index < total_images:
                time.sleep(args.delay)

        except Exception as exc:
            failed_count += 1
            failed_files.append((image_path.name, str(exc)))
            print(f" FAILED! Reason: {exc}")

    # =========================================================================
    # Final Summary Report
    # =========================================================================
    print("\n" + "=" * 70)
    print(" TRANSCRIPTION SUMMARY REPORT")
    print("=" * 70)
    print(f" Total Images Evaluated : {total_images}")
    print(f" Successfully Processed : {succeeded_count}")
    print(f" Skipped (Already Done) : {skipped_count}")
    print(f" Failed Images          : {failed_count}")
    print(f" Output Location        : {output_dir.resolve()}")

    # List failed images if any
    if failed_files:
        print("\n--- Failed Images ---")
        for fname, err in failed_files:
            print(f"  * {fname}: {err}")

    # List all [unclear] occurrences for manual review
    total_unclear_spots = sum(len(spots) for spots in unclear_reports.values())
    print("\n--- Manual Review: [unclear] Occurrences ---")
    if unclear_reports:
        print(f"Total [unclear] tags across notes: {total_unclear_spots}")
        print(f"Found in {len(unclear_reports)} file(s):\n")
        for fname, spots in unclear_reports.items():
            print(f"  File: {fname} ({len(spots)} unclear spot{'s' if len(spots) > 1 else ''})")
            for spot in spots:
                print(f"    Line {spot['line']:<4}: \"{spot['snippet']}\"")
            print()
    else:
        print("No [unclear] tags found across all processed notes! All handwriting was recognized.")

    print("=" * 70)
    print("Step 1 Complete! Transcriptions are ready for Step 2 (Chunking).")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
