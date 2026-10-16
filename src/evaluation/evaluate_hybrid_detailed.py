import json
from pathlib import Path
import numpy as np
from src.retrieval.hybrid_retriever import HybridRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PASSAGES_FILE = (PROJECT_ROOT / "data" / "processed" / "passages.json")
QA_FILE = (PROJECT_ROOT / "data" / "qa_dataset" / "qa_dataset.json")
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_FILE = (RESULTS_DIR / "hybrid_detailed_results.json")

TFIDF_WEIGHT = 0.7
DENSE_WEIGHT = 0.3

TOP_K = 5
CANDIDATE_K = 10

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def recall_at_k(retrieved_chunks, relevant_chunks, k):
    retrieved = retrieved_chunks[:k]
    return any(chunk in relevant_chunks for chunk in retrieved)

def reciprocal_rank(retrieved_chunks,  relevant_chunks):
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if chunk in relevant_chunks:
            return 1 / rank
    return 0.0

def calculate_metrics(question_results):
    if not question_results:
        return {
            "questions": 0,
            "recall@1": 0.0,
            "recall@3": 0.0,
            "recall@5": 0.0,
            "mrr": 0.0
        }

    r1 = np.mean([q["recall@1"] for q in question_results])
    r3 = np.mean([q["recall@3"] for q in question_results])
    r5 = np.mean([q["recall@5"] for q in question_results])
    mrr = np.mean([q["reciprocal_rank"]  for q in question_results])

    return {
        "questions": len(question_results),
        "recall@1": float(r1),
        "recall@3": float(r3),
        "recall@5": float(r5),
        "mrr": float(mrr)
    }

def evaluate_questions(retriever, questions):
    results = []
    for qa in questions:
        question = qa["question"]
        relevant_chunks = set(qa["supporting_chunks"])
        retrieved_results = retriever.search(question, top_k=TOP_K, candidate_k=CANDIDATE_K)
        retrieved_chunks = [result["chunk_id"] for result in retrieved_results]

        hit_1 = recall_at_k(retrieved_chunks, relevant_chunks, 1)
        hit_3 = recall_at_k(retrieved_chunks, relevant_chunks, 3)
        hit_5 = recall_at_k(retrieved_chunks, relevant_chunks, 5)
        rr = reciprocal_rank(retrieved_chunks, relevant_chunks)

        results.append({
            "question_id": qa["question_id"],
            "question": qa["question"],
            "language": qa["language"],
            "query_type": qa["query_type"],
            "difficulty": qa["difficulty"],
            "category": qa["category"],
            "supporting_chunks": list(relevant_chunks),
            "retrieved_chunks": retrieved_chunks,
            "recall@1": hit_1,
            "recall@3": hit_3,
            "recall@5": hit_5,
            "reciprocal_rank": rr
        })

    return results

def group_results(question_results, field):
    groups = {}
    for result in question_results:
        value = result.get(field)
        if value is None:
            value = "unknown"
        if value not in groups:
            groups[value] = []

        groups[value].append(result)

    grouped_metrics = {}

    for value, results in groups.items():
        grouped_metrics[value] = calculate_metrics(results)

    return grouped_metrics

def print_table(title, grouped_results):
    print(title)
    print(
        f"{'Group':<30}"
        f"{'Questions':>10}"
        f"{'R@1':>10}"
        f"{'R@3':>10}"
        f"{'R@5':>10}"
        f"{'MRR':>10}"
    )

    for group, metrics in grouped_results.items():
        print(
            f"{str(group):<30}"
            f"{metrics['questions']:>10}"
            f"{metrics['recall@1'] * 100:>9.2f}%"
            f"{metrics['recall@3'] * 100:>9.2f}%"
            f"{metrics['recall@5'] * 100:>9.2f}%"
            f"{metrics['mrr']:>10.4f}"
        )

def main():
    print("WEIGHTED HYBRID DETAILED EVALUATION")
    print(f"\nTF-IDF weight : {TFIDF_WEIGHT}")
    print(f"Dense weight  : {DENSE_WEIGHT}")
    print(f"Candidate K   : {CANDIDATE_K}")
    print(f"Top K         : {TOP_K}")

    passages = load_json(PASSAGES_FILE)
    qa_dataset = load_json(QA_FILE)

    answerable_questions = [q for q in qa_dataset if q["answerable"]]

    print(f"\nTotal questions       : {len(qa_dataset)}")
    print(f"Answerable questions : {len(answerable_questions)}")
    print(f"Unanswerable questions : {len(qa_dataset) - len(answerable_questions)}")

    print("\nInitializing weighted hybrid retriever...")

    retriever = HybridRetriever(passages, tfidf_weight=TFIDF_WEIGHT, dense_weight=DENSE_WEIGHT)
    
    print("\nEvaluating questions...")

    question_results = evaluate_questions(retriever, answerable_questions)

    overall = calculate_metrics(question_results)

    language_results = group_results(question_results, "language")

    query_type_results = group_results(question_results, "query_type")

    difficulty_results = group_results(question_results, "difficulty")

    category_results = group_results(question_results, "category")

    print("OVERALL RESULTS")
    print(f"\nRecall@1 : {overall['recall@1']:.4f} ({overall['recall@1'] * 100:.2f}%)")
    print(f"Recall@3 : {overall['recall@3']:.4f} ({overall['recall@3'] * 100:.2f}%)")
    print(f"Recall@5 : {overall['recall@5']:.4f} ({overall['recall@5'] * 100:.2f}%)")
    print(f"MRR      : {overall['mrr']:.4f}")
    
    print_table("\nLANGUAGE-WISE RESULTS", language_results)
    print_table("\nQUERY-TYPE RESULTS", query_type_results)
    print_table("\nDIFFICULTY-WISE RESULTS", difficulty_results)
    print_table("\nCATEGORY-WISE RESULTS", category_results)

    output = {
        "method": "Weighted Hybrid",
        "parameters": {
            "tfidf_weight": TFIDF_WEIGHT,
            "dense_weight": DENSE_WEIGHT,
            "candidate_k": CANDIDATE_K,
            "top_k": TOP_K
        },
        "dataset": {
            "total_questions": len(qa_dataset),
            "answerable_questions": len(answerable_questions),
            "unanswerable_questions": len(qa_dataset)  - len(answerable_questions)
        },
        "overall": overall,
        "by_language": language_results,
        "by_query_type": query_type_results,
        "by_difficulty": difficulty_results,
        "by_category": category_results,
        "questions": question_results
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Detailed results saved to:\n {RESULTS_FILE}")

if __name__ == "__main__":
    main()