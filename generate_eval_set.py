"""
Step 3: Evaluation Set Generation for RAG Pipeline
--------------------------------------------------
Creates a curated evaluation benchmark of 10 question-answer pairs
grounded directly in the handwritten study notes (AWS, Spring Security, ML/DL).
Includes:
  - Direct factual questions
  - Multi-hop / synthesis questions combining related points
  - Negative (out-of-corpus) questions to evaluate hallucination prevention
Saves output to ./data/eval_set.json.
"""

import json
from pathlib import Path

# Curated benchmark of evaluation questions grounded in the transcribed notes
EVALUATION_QUESTIONS = [
    {
        "id": "eval_01",
        "category": "Direct Factual (AWS)",
        "question": "What is AWS EBS (Elastic Block Store) and what are its primary use cases according to the notes?",
        "expected_answer_summary": "EBS is a block storage system formatted as a series of bytes stored in disk clusters. It acts as a persistent virtual hard drive attached to an EC2 instance to store the OS, configuration, and data. Primary use cases include relational databases (RDBMS), enterprise applications, and file systems.",
        "in_corpus": True,
        "source_notes": ["Data 36.txt"]
    },
    {
        "id": "eval_02",
        "category": "Direct Factual (Deep Learning)",
        "question": "How does Dropout Regularization work during training, and how is it handled at test time?",
        "expected_answer_summary": "During training, dropout randomly drops a percentage of neurons in layers using a keep probability (e.g., keep_prob = 0.8) and scales remaining activations (dividing by keep_prob) to maintain expected matrix values. At test time, no neurons are dropped so the full network is used without modification.",
        "in_corpus": True,
        "source_notes": ["Data 2.txt"]
    },
    {
        "id": "eval_03",
        "category": "Direct Factual (Spring Security)",
        "question": "Why does Spring Security enforce CSRF protection on POST and DELETE HTTP requests, and how is the token handled?",
        "expected_answer_summary": "Without CSRF protection, state-changing requests like POST and DELETE that rely on cookies or parameters can be attacked and forged by malicious websites. Spring Security defends against this by requiring an anti-CSRF token (such as X-CSRF-TOKEN in the request header) for verification on each request.",
        "in_corpus": True,
        "source_notes": ["Data 60.txt", "Data 20.txt"]
    },
    {
        "id": "eval_04",
        "category": "Direct Factual (ML Algorithms)",
        "question": "Why is the standard squared error cost function from Linear Regression not used for Logistic Regression?",
        "expected_answer_summary": "Applying the squared error cost function to logistic regression produces a non-convex surface with many local minima, which makes gradient descent fail to find the global optimum. Logistic regression uses the logistic loss function (cross-entropy) to ensure convexity.",
        "in_corpus": True,
        "source_notes": ["Data 10.txt"]
    },
    {
        "id": "eval_05",
        "category": "Combining Points (AWS Networking & Security)",
        "question": "What are the key differences between AWS Security Groups and Network Access Control Lists (NACLs) regarding statefulness and default rules?",
        "expected_answer_summary": "Security Groups are stateful (return traffic is automatically allowed regardless of inbound rules), deny all inbound traffic by default while allowing all outbound traffic, and evaluate all rules before deciding. NACLs are stateless (return traffic must be explicitly allowed) and operate at the subnet/VPC level.",
        "in_corpus": True,
        "source_notes": ["Data 41.txt"]
    },
    {
        "id": "eval_06",
        "category": "Combining Points (AWS Storage & Lifecycle)",
        "question": "How do EBS snapshots interact with Amazon S3, and what lifecycle management practices are used for backups?",
        "expected_answer_summary": "EBS snapshots are point-in-time volume backups stored redundantly in Amazon S3 for durable, cost-effective data protection and recovery. Lifecycle management involves automating snapshot schedules (daily/weekly), applying consistent tagging policies, and setting retention rules to manage backups over time.",
        "in_corpus": True,
        "source_notes": ["Data 36.txt"]
    },
    {
        "id": "eval_07",
        "category": "Combining Points (Spring Security & Web)",
        "question": "How do CORS and CSRF differ in web application security according to the notes?",
        "expected_answer_summary": "CORS (Cross-Origin Resource Sharing) relaxes browser cross-origin restrictions to allow specific frontend origins (like port 3000) to communicate with backend APIs (like port 8080). CSRF (Cross-Site Request Forgery) is an exploit where malicious websites trigger unauthorized requests using stored cookies, prevented by issuing and checking anti-CSRF tokens.",
        "in_corpus": True,
        "source_notes": ["Data 20.txt"]
    },
    {
        "id": "eval_08",
        "category": "Combining Points (ML/DL Optimization)",
        "question": "What methods are described in the notes to address and reduce overfitting in machine learning models?",
        "expected_answer_summary": "The notes describe several techniques to reduce overfitting: 1) Regularization (L2 regularization and Dropout to spread weights), 2) Early stopping (halting training before validation error rises), 3) Data augmentation (modifying training images via rotations, cropping, flipping, and color jittering), and 4) Feature reduction.",
        "in_corpus": True,
        "source_notes": ["Data 2.txt", "Data 14.txt"]
    },
    {
        "id": "eval_09",
        "category": "Hallucination Test (Negative - AWS)",
        "question": "How does Amazon DynamoDB resolve multi-region write conflicts across Global Tables using vector clocks in the notes?",
        "expected_answer_summary": "I don't know based on my notes. The notes mention DynamoDB and DAX accelerator, but do not contain information about DynamoDB Global Tables, multi-region conflict resolution, or vector clocks.",
        "in_corpus": False,
        "source_notes": []
    },
    {
        "id": "eval_10",
        "category": "Hallucination Test (Negative - DL / LLMs)",
        "question": "What rank dimension and alpha scaling factor are recommended in the notes for fine-tuning LLMs with LoRA (Low-Rank Adaptation)?",
        "expected_answer_summary": "I don't know based on my notes. While the notes touch on deep learning, neural networks, and transformers, they do not mention LoRA (Low-Rank Adaptation), rank decomposition, or alpha scaling hyper-parameters.",
        "in_corpus": False,
        "source_notes": []
    }
]


def generate_evaluation_set(output_path: Path):
    """Saves the evaluation set as a structured JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare standard format conforming exactly to user requirements
    # fields: question, expected_answer_summary, in_corpus
    formatted_dataset = []
    for item in EVALUATION_QUESTIONS:
        formatted_dataset.append({
            "question": item["question"],
            "expected_answer_summary": item["expected_answer_summary"],
            "in_corpus": item["in_corpus"],
            "category": item["category"],
            "source_notes": item["source_notes"]
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted_dataset, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print(" RAG Step 3: Evaluation Set Generation")
    print("=" * 70)
    print(f" Output File       : {output_path.resolve()}")
    print(f" Total Questions   : {len(formatted_dataset)}")
    in_corpus_count = sum(1 for q in formatted_dataset if q["in_corpus"])
    out_corpus_count = sum(1 for q in formatted_dataset if not q["in_corpus"])
    print(f" In-Corpus Items   : {in_corpus_count} (Factual & Multi-hop synthesis)")
    print(f" Out-of-Corpus     : {out_corpus_count} (Hallucination / Negative test checks)")
    print("=" * 70)

    print("\nEvaluation Benchmark Overview:")
    for idx, item in enumerate(formatted_dataset, start=1):
        status = "[IN-CORPUS]" if item["in_corpus"] else "[OUT-OF-CORPUS (TEST NEGATIVE)]"
        print(f"\n{idx}. {status} ({item['category']})")
        print(f"   Q: {item['question']}")
        print(f"   Expected: {item['expected_answer_summary']}")
        if item["source_notes"]:
            print(f"   Source Note(s): {', '.join(item['source_notes'])}")

    print("\n" + "=" * 70)
    print("Step 3 Complete! Evaluation set is saved to ./data/eval_set.json.")
    print("Ready for Step 4 (RAG Pipeline & Automated Evaluation).")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    output_file = Path("./data/eval_set.json")
    generate_evaluation_set(output_file)
