import json
from pathlib import Path
import numpy as np
from src.retrieval.hybrid_retriever import HybridRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PASSAGES_FILE = PROJECT_ROOT / "data" / "processed" / "passages.json"
QA_FILE = PROJECT_ROOT / "data" / "qa_dataset" / "qa_dataset.json"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_FILE = RESULTS_DIR / "hybrid_results.json"

TFIDF_WEIGHT = 0.9
DENSE_WEIGHT = 0.1

CANDIDATE_K = 10
TOP_K = 5

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def recall_at_k(retrieved_chunks, relevant_chunks, k):
    if not relevant_chunks:
        return None

    return any(chunk in relevant_chunks for chunk in retrieved_chunks[:k])

def reciprocal_rank(retrieved_chunks, relevant_chunks):
    if not relevant_chunks:
        return None

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if chunk in relevant_chunks:
            return 1.0 / rank

    return 0.0

def calculate_retrieval_metrics(question_results):
    answerable_results = [q for q in question_results if q["answerable"] and q["supporting_chunks"]]

    if not answerable_results:
        return {
            "questions_evaluated": 0,
            "recall@1": 0.0,
            "recall@3": 0.0,
            "recall@5": 0.0,
            "mrr": 0.0
        }

    return {
        "questions_evaluated": len(answerable_results),
        "recall@1": float(np.mean([q["recall@1"] for q in answerable_results])),
        "recall@3": float(np.mean([q["recall@3"] for q in answerable_results])),
        "recall@5": float(np.mean([q["recall@5"] for q in answerable_results])),
        "mrr": float(np.mean([q["reciprocal_rank"] for q in answerable_results]))
    }

def main():
    print("WEIGHTED HYBRID RETRIEVAL EVALUATION")

    print("\nLoading passages and QA dataset...")
    passages = load_json(PASSAGES_FILE)
    qa_dataset = load_json(QA_FILE)

    total_questions = len(qa_dataset)

    answerable_count = sum(1 for q in qa_dataset if q["answerable"])
    unanswerable_count = total_questions - answerable_count

    print(f"Total questions       : {total_questions}")
    print(f"Answerable questions  : {answerable_count}")
    print(f"Unanswerable questions: {unanswerable_count}")
    print(f"Total passages        : {len(passages)}")

    print("\nRetrieval parameters:")
    print(f"TF-IDF weight : {TFIDF_WEIGHT}")
    print(f"Dense weight  : {DENSE_WEIGHT}")
    print(f"Candidate K   : {CANDIDATE_K}")
    print(f"Top K         : {TOP_K}")

    print("\nInitializing hybrid retriever...")

    retriever = HybridRetriever(passages, tfidf_weight=TFIDF_WEIGHT, dense_weight=DENSE_WEIGHT)

    print("\nEvaluating all questions...")

    question_results = []

    for index, qa in enumerate(qa_dataset, start=1):
        question = qa["question"]
        relevant_chunks = set(qa.get("supporting_chunks", []))
        results = retriever.search(question, top_k=TOP_K, candidate_k=CANDIDATE_K)
        retrieved_chunks = [result["chunk_id"] for result in results]
        hybrid_scores = [float(result.get("hybrid_score", 0.0)) for result in results]
        tfidf_scores = [float(result.get("tfidf_score", 0.0)) for result in results]
        dense_scores = [float(result.get("dense_score", 0.0)) for result in results]
        top_hybrid_score = (hybrid_scores[0] if hybrid_scores else 0.0)
        second_hybrid_score = (hybrid_scores[1] if len(hybrid_scores) > 1 else 0.0)
        score_gap = top_hybrid_score - second_hybrid_score

        if qa["answerable"] and relevant_chunks:
            hit_1 = recall_at_k(retrieved_chunks, relevant_chunks, 1)
            hit_3 = recall_at_k(retrieved_chunks, relevant_chunks, 3)
            hit_5 = recall_at_k(retrieved_chunks, relevant_chunks, 5)
            rr = reciprocal_rank(retrieved_chunks, relevant_chunks)

        else:
            hit_1 = None
            hit_3 = None
            hit_5 = None
            rr = None

        question_result = {
            "question_id": qa["question_id"],
            "question": question,
            "language": qa.get("language", "unknown"),
            "query_type": qa.get("query_type", "unknown"),
            "difficulty": qa.get("difficulty", "unknown"),
            "category": qa.get("category", "unknown"),
            "answerable": bool(qa["answerable"]),
            "supporting_chunks": list(relevant_chunks),
            "retrieved_chunks": retrieved_chunks,
            "hybrid_scores": hybrid_scores,
            "tfidf_scores": tfidf_scores,
            "dense_scores": dense_scores,
            "top_hybrid_score": top_hybrid_score,
            "second_hybrid_score": second_hybrid_score,
            "score_gap": score_gap,
            "mean_hybrid_score": (float(np.mean(hybrid_scores)) if hybrid_scores else 0.0),
            "max_tfidf_score": (max(tfidf_scores) if tfidf_scores else 0.0),
            "max_dense_score": (max(dense_scores) if dense_scores else 0.0),
            "strong_results_070": sum(score >= 0.70 for score in hybrid_scores),
            "recall@1": hit_1,
            "recall@3": hit_3,
            "recall@5": hit_5,
            "reciprocal_rank": rr
        }

        question_results.append(question_result)

        if index % 25 == 0 or index == total_questions:
            print(f"Processed {index}/{total_questions} questions...")

    overall = calculate_retrieval_metrics(question_results)

    print("RETRIEVAL RESULTS")
    print(f"Questions evaluated: {overall['questions_evaluated']}")
    print(f"Recall@1 : {overall['recall@1']:.4f} ({overall['recall@1'] * 100:.2f}%)")
    print(f"Recall@3 : {overall['recall@3']:.4f} ({overall['recall@3'] * 100:.2f}%)")
    print(f"Recall@5 : {overall['recall@5']:.4f} ({overall['recall@5'] * 100:.2f}%)")
    print(f"MRR      : {overall['mrr']:.4f}")

    saved_answerable = sum(1 for q in question_results if q["answerable"])
    saved_unanswerable = sum(1 for q in question_results if not q["answerable"])

    assert len(question_results) == total_questions
    assert saved_answerable == answerable_count
    assert saved_unanswerable == unanswerable_count

    output = {
        "method": "Weighted Hybrid",
        "parameters": {
            "tfidf_weight": TFIDF_WEIGHT,
            "dense_weight": DENSE_WEIGHT,
            "candidate_k": CANDIDATE_K,
            "top_k": TOP_K
        },
        "dataset": {
            "total_questions": total_questions,
            "answerable_questions": answerable_count,
            "unanswerable_questions": unanswerable_count
        },
        "overall": overall,
        "questions": question_results
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\nEVALUATION COMPLETED")
    print(f"Total saved questions : {len(question_results)}")
    print(f"Answerable            : {saved_answerable}")
    print(f"Unanswerable          : {saved_unanswerable}")
    print(f"\nResults saved to:\n{RESULTS_FILE}")

if __name__ == "__main__":
    main()

