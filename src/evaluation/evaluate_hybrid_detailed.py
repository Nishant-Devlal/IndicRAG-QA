import json
from pathlib import Path
import numpy as np
from src.retrieval.hybrid_retriever import HybridRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PASSAGES_FILE = (PROJECT_ROOT / "data" / "processed" / "passages.json")
QA_FILE = (PROJECT_ROOT / "data" / "qa_dataset" / "qa_dataset.json")
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_FILE = (RESULTS_DIR / "hybrid_detailed_results.json")

TFIDF_WEIGHT = 0.9
DENSE_WEIGHT = 0.1

TOP_K = 5
CANDIDATE_K = 10

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def recall_at_k(retrieved_chunks, relevant_chunks, k):
    retrieved = retrieved_chunks[:k]
    return any(chunk in relevant_chunks for chunk in retrieved)

def reciprocal_rank(retrieved_chunks, relevant_chunks):
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
    mrr = np.mean([q["reciprocal_rank"] for q in question_results])

    return {
        "questions": len(question_results),
        "recall@1": float(r1),
        "recall@3": float(r3),
        "recall@5": float(r5),
        "mrr": float(mrr)
    }

def evaluate_questions(retriever, questions):
    results = []
    for index, qa in enumerate(questions, start=1):
        question = qa["question"]
        print(
            f"\rEvaluating question "
            f"{index}/{len(questions)}",
            end="",
            flush=True
        )

        relevant_chunks = set(qa.get("supporting_chunks", []))
        retrieved_results = retriever.search(question, top_k=TOP_K, candidate_k=CANDIDATE_K)
        retrieved_chunks = [result["chunk_id"] for result in retrieved_results]

        hybrid_scores = [
            {
                "chunk_id": result["chunk_id"],
                "score": float(result.get("hybrid_score", result.get("rrf_score", 0.0)))
            }
            for result in retrieved_results
        ]

        if hybrid_scores:
            top_hybrid_score = (hybrid_scores[0]["score"])

        else:
            top_hybrid_score = 0.0

        if qa["answerable"]:
            hit_1 = recall_at_k(retrieved_chunks, relevant_chunks, 1)
            hit_3 = recall_at_k(retrieved_chunks, relevant_chunks, 3)
            hit_5 = recall_at_k(retrieved_chunks, relevant_chunks, 5)
            rr = reciprocal_rank(retrieved_chunks, relevant_chunks)

        else:
            hit_1 = None
            hit_3 = None
            hit_5 = None
            rr = None

        result = {
            "question_id": qa["question_id"],
            "question": qa["question"],
            "language": qa["language"],
            "query_type": qa["query_type"],
            "difficulty": qa["difficulty"],
            "category": qa["category"],
            "answerable": qa["answerable"],
            "supporting_chunks": list(relevant_chunks),
            "retrieved_chunks": retrieved_chunks,
            "hybrid_scores": hybrid_scores,
            "top_hybrid_score": top_hybrid_score,
            "recall@1": hit_1,
            "recall@3": hit_3,
            "recall@5": hit_5,
            "reciprocal_rank": rr
        }
        results.append(result)

    print("\n")

    return results

def group_results(question_results, field, answerable_only=True):
    groups = {}
    for result in question_results:
        if answerable_only and not result["answerable"]:
            continue

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
    print("\n" + title)
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

def calculate_answerability_summary(question_results):
    total = len(question_results)
    answerable = sum(1 for q in question_results if q["answerable"])
    unanswerable = total - answerable

    return {
        "total_questions": total,
        "answerable_questions": answerable,
        "unanswerable_questions": unanswerable
    }

def main():
    print("WEIGHTED HYBRID DETAILED EVALUATION")
    print(f"\nTF-IDF weight : {TFIDF_WEIGHT}")
    print(f"Dense weight  : {DENSE_WEIGHT}")
    print(f"Candidate K   : {CANDIDATE_K}")
    print(f"Top K         : {TOP_K}")

    print("\nLoading passages...")
    passages = load_json(PASSAGES_FILE)
    print(f"Loaded {len(passages)} passages.")
    
    print("\nLoading QA dataset...")
    qa_dataset = load_json(QA_FILE)
    print(f"Loaded {len(qa_dataset)} questions.")

    answerable_questions = [q for q in qa_dataset if q["answerable"]]
    unanswerable_questions = [q for q in qa_dataset if not q["answerable"]]
    print(f"\nTotal questions        : {len(qa_dataset)}")
    print(f"Answerable questions  : {len(answerable_questions)}")
    print(f"Unanswerable questions: {len(unanswerable_questions)}")
    
    print("\nInitializing weighted hybrid retriever...")
    retriever = HybridRetriever(passages, tfidf_weight=TFIDF_WEIGHT, dense_weight=DENSE_WEIGHT)

    print("\nEvaluating ALL questions...")

    question_results = evaluate_questions(retriever, qa_dataset)

    answerable_results = [q for q in question_results if q["answerable"]]

    overall = calculate_metrics(answerable_results)

    language_results = group_results(question_results, "language", answerable_only=True)
    query_type_results = group_results(question_results, "query_type", answerable_only=True)
    difficulty_results = group_results(question_results, "difficulty", answerable_only=True)
    category_results = group_results(question_results, "category", answerable_only=True)

    print("OVERALL RETRIEVAL RESULTS")
    print(f"\nAnswerable questions evaluated : {overall['questions']}")
    print(f"Recall@1 : {overall['recall@1']:.4f} ({overall['recall@1'] * 100:.2f}%)")
    print(f"Recall@3 : {overall['recall@3']:.4f} ({overall['recall@3'] * 100:.2f}%)")
    print(f"Recall@5 : {overall['recall@5']:.4f} ({overall['recall@5'] * 100:.2f}%)")
    print(f"MRR      : {overall['mrr']:.4f}")

    print_table("LANGUAGE-WISE RESULTS", language_results)
    print_table("QUERY-TYPE RESULTS", query_type_results)
    print_table("DIFFICULTY-WISE RESULTS", difficulty_results)
    print_table("CATEGORY-WISE RESULTS", category_results)

    answerability_summary = (calculate_answerability_summary(question_results))

    print("ANSWERABILITY SUMMARY")
    print(f"\nTotal        : {answerability_summary['total_questions']}")
    print(f"Answerable   : {answerability_summary['answerable_questions']}")
    print(f"Unanswerable : {answerability_summary['unanswerable_questions']}")

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
            "unanswerable_questions": len(unanswerable_questions)
        },
        "answerability_summary": answerability_summary,
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

    print("\nDetailed results saved to:")
    print(RESULTS_FILE)

if __name__ == "__main__":
    main()