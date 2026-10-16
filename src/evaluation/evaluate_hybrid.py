import json
from pathlib import Path
import numpy as np
from src.retrieval.hybrid_retriever import HybridRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PASSAGES_FILE = (PROJECT_ROOT / "data" / "processed" / "passages.json")
QA_FILE = (PROJECT_ROOT / "data" / "qa_dataset" / "qa_dataset.json")
RESULTS_DIR = (PROJECT_ROOT / "results")
RESULTS_FILE = (RESULTS_DIR / "hybrid_results.json")

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

def main():
    passages = load_json(PASSAGES_FILE)
    qa_dataset = load_json(QA_FILE)
    answerable_questions = [q for q in qa_dataset if q["answerable"]]

    TFIDF_WEIGHT = 0.7
    DENSE_WEIGHT = 0.3

    CANDIDATE_K = 10
    TOP_K = 5

    print("WEIGHTED HYBRID RETRIEVAL EVALUATION")
    print(f"\nTotal questions: {len(qa_dataset)}")
    print(f"Answerable questions: {len(answerable_questions)}")
    print(f"Unanswerable questions: {len(qa_dataset) - len(answerable_questions)}")
    print(f"\nTF-IDF weight : {TFIDF_WEIGHT}")
    print(f"Dense weight  : {DENSE_WEIGHT}")
    print(f"Candidate K   : {CANDIDATE_K}")

    print(f"Top K         : {TOP_K}")

    retriever = HybridRetriever(passages, tfidf_weight=TFIDF_WEIGHT, dense_weight=DENSE_WEIGHT)

    recall_1 = []
    recall_3 = []
    recall_5 = []
    reciprocal_ranks = []
    question_results = []

    for qa in answerable_questions:
        question = qa["question"]
        relevant_chunks = set(qa["supporting_chunks"])
        results = retriever.search(question, top_k=TOP_K, candidate_k=CANDIDATE_K)
        retrieved_chunks = [result["chunk_id"] for result in results]

        hit_1 = recall_at_k(retrieved_chunks, relevant_chunks, 1)
        hit_3 = recall_at_k(retrieved_chunks, relevant_chunks, 3)
        hit_5 = recall_at_k(retrieved_chunks, relevant_chunks, 5)
        rr = reciprocal_rank(retrieved_chunks, relevant_chunks)

        recall_1.append(hit_1)
        recall_3.append(hit_3)
        recall_5.append(hit_5)
        reciprocal_ranks.append(rr)

        question_results.append({
            "question_id": qa["question_id"],
            "question": qa["question"],
            "language": qa["language"],
            "query_type": qa["query_type"],
            "supporting_chunks": list(relevant_chunks),
            "retrieved_chunks": retrieved_chunks,
            "hybrid_scores": [result["hybrid_score"] for result in results],
            "tfidf_scores": [result["tfidf_score"] for result in results],
            "dense_scores": [result["dense_score"] for result in results],
            "recall@1": hit_1,
            "recall@3": hit_3,
            "recall@5": hit_5,
            "reciprocal_rank": rr
        })

    r1 = np.mean(recall_1)
    r3 = np.mean(recall_3)
    r5 = np.mean(recall_5)
    mrr = np.mean(reciprocal_ranks)

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
            "unanswerable_questions": len(qa_dataset) - len(answerable_questions)
        },
        "overall": {
            "recall@1": float(r1),
            "recall@3": float(r3),
            "recall@5": float(r5),
            "mrr": float(mrr)
        },
        "questions": question_results
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("OVERALL RESULTS")
    print(f"\nRecall@1 : {r1:.4f} ({r1 * 100:.2f}%)")
    print(f"Recall@3 : {r3:.4f} ({r3 * 100:.2f}%)")
    print(f"Recall@5 : {r5:.4f} ({r5 * 100:.2f}%)")
    print(f"MRR      : {mrr:.4f}")
    print(f"\nResults saved to: {RESULTS_FILE}")

if __name__ == "__main__":
    main()