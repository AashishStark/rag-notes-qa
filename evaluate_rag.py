"""
Step 4 Evaluation Script: Automated RAG Benchmark Runner
--------------------------------------------------------
Loops through ./data/eval_set.json, runs generate_answer() on each question,
and displays the Question, Generated Answer, and Expected Answer Summary
side by side for easy manual comparison and hallucination checking.
Saves results to ./data/eval_results.json.
"""

import json
import time
import argparse
from pathlib import Path
from rag_pipeline import RAGPipeline

DEFAULT_EVAL_FILE = Path("./data/eval_set.json")
DEFAULT_OUTPUT_FILE = Path("./data/eval_results.json")


def run_evaluation(
    eval_file: Path = DEFAULT_EVAL_FILE,
    output_file: Path = DEFAULT_OUTPUT_FILE,
    k: int = 4,
    gemini_model: str = "gemini-3.5-flash-lite",
    delay: float = 1.0,
):
    if not eval_file.exists():
        print(f"Error: Evaluation set '{eval_file}' not found. Run generate_eval_set.py first.")
        return

    print("=" * 80)
    print(" RAG EVALUATION BENCHMARK RUNNER")
    print("=" * 80)
    print(f" Evaluation Dataset : {eval_file.resolve()}")
    print(f" Results Destination : {output_file.resolve()}")
    print(f" Top-K Retrieval     : {k}")
    print(f" Gemini Model        : {gemini_model}")
    print("=" * 80)

    with open(eval_file, "r", encoding="utf-8") as f:
        eval_items = json.load(f)

    # Initialize RAG Pipeline
    pipeline = RAGPipeline(gemini_model=gemini_model)

    results = []
    total = len(eval_items)

    print(f"\nEvaluating {total} benchmark questions...\n")

    for idx, item in enumerate(eval_items, start=1):
        q = item["question"]
        expected = item["expected_answer_summary"]
        in_corpus = item["in_corpus"]
        category = item.get("category", "General")

        print(f"[{idx}/{total}] Processing: \"{q[:65]}...\"")

        # Run RAG generation
        response = pipeline.generate_answer(query=q, k=k)
        generated_answer = response["answer"]
        sources = response["sources"]

        # Check hallucination test outcome for negative samples
        anti_hallucination_pass = None
        if not in_corpus:
            refused = "don't know" in generated_answer.lower() or "not in my notes" in generated_answer.lower() or "not mentioned" in generated_answer.lower()
            anti_hallucination_pass = refused

        result_entry = {
            "index": idx,
            "category": category,
            "question": q,
            "in_corpus": in_corpus,
            "expected_answer_summary": expected,
            "generated_answer": generated_answer,
            "anti_hallucination_check": anti_hallucination_pass,
            "sources_used": [
                {
                    "source_file": s["source_file"],
                    "chunk_index": s["chunk_index"],
                    "topic_heading": s["topic_heading"],
                    "distance": round(float(s["distance"]), 4),
                }
                for s in sources
            ],
        }
        results.append(result_entry)

        # Print detailed comparison card
        print("\n" + "=" * 80)
        print(f" QUESTION #{idx}: [{category}]")
        print("=" * 80)
        print(f"Q: {q}\n")

        print("-" * 80)
        print("EXPECTED ANSWER SUMMARY:")
        print("-" * 80)
        print(expected)

        print("\n" + "-" * 80)
        print("GENERATED RAG ANSWER:")
        print("-" * 80)
        print(generated_answer)

        print("\n" + "-" * 80)
        print("SOURCES RETRIEVED:")
        print("-" * 80)
        for s_idx, src in enumerate(sources, start=1):
            print(f"  [{s_idx}] {src['source_file']} (Chunk {src['chunk_index']}) | Heading: '{src['topic_heading']}' | Dist: {src['distance']:.4f}")

        if not in_corpus:
            status_tag = "PASS (Refused to hallucinate)" if anti_hallucination_pass else "FAIL (Hallucinated an answer)"
            print(f"\nAnti-Hallucination Guardrail Check: [{status_tag}]")

        print("=" * 80 + "\n")

        # Small delay between API calls
        if delay > 0 and idx < total:
            time.sleep(delay)

    # Save complete evaluation output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f" Evaluation Complete! {len(results)} questions evaluated.")
    print(f" Results saved to: {output_file.resolve()}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG Pipeline against benchmark questions.")
    parser.add_argument("--eval-file", "-e", type=str, default=str(DEFAULT_EVAL_FILE))
    parser.add_argument("--output-file", "-o", type=str, default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--k", "-k", type=int, default=4)
    parser.add_argument("--model", "-m", type=str, default="gemini-3.5-flash-lite")
    parser.add_argument("--delay", "-d", type=float, default=1.0)

    args = parser.parse_args()

    run_evaluation(
        eval_file=Path(args.eval_file),
        output_file=Path(args.output_file),
        k=args.k,
        gemini_model=args.model,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
