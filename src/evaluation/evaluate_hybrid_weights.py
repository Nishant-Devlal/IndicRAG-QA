import json
from pathlib import Path
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval.tfidf_retriever import TFIDFRetriever
from retrieval.dense_retriever import DenseRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PASSAGES_FILE = (PROJECT_ROOT / "data" / "processed" / "passages.json")
QA_FILE = (PROJECT_ROOT / "data" / "qa_dataset" / "qa_dataset.json")

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_scores(scores):
    scores = np.array(scores, dtype=float)
    min_score = np.min(scores)
    max_score = np.max(scores)
    if max_score == min_score:
        return np.zeros_like(scores)
    return (scores - min_score) / (max_score - min_score)

def recall_at_k(retrieved, relevant, k):
    return any(chunk in relevant for chunk in retrieved[:k])

def reciprocal_rank(retrieved, relevant):
    for rank, chunk in enumerate(retrieved, start=1):
        if chunk in relevant:
            return 1 / rank
    return 0.0

def evaluate_weight(tfidf, dense, qa_dataset, alpha):
    recall_1 = []
    recall_3 = []
    recall_5 = []
    reciprocal_ranks = []
    answerable_questions = [q for q in qa_dataset if q["answerable"]]

    for qa in answerable_questions:
        query = qa["question"]
        relevant_chunks = set(qa["supporting_chunks"])
        tfidf_results = tfidf.search(query, top_k=10)
        dense_results = dense.search(query, top_k=10)

        tfidf_scores = {r["chunk_id"]: r["score"] for r in tfidf_results}
        dense_scores = {r["chunk_id"]: r["score"] for r in dense_results}

        tfidf_norm = normalize_scores(list(tfidf_scores.values()))
        dense_norm = normalize_scores(list(dense_scores.values()))
        tfidf_norm = dict(zip(tfidf_scores.keys(), tfidf_norm))
        dense_norm = dict(zip(dense_scores.keys(), dense_norm))

        candidates = set(tfidf_scores) | set(dense_scores)

        hybrid_scores = {}

        for chunk_id in candidates:
            t_score = tfidf_norm.get(chunk_id, 0.0)
            d_score = dense_norm.get(chunk_id, 0.0)
            hybrid_scores[chunk_id] = (alpha * t_score  + (1 - alpha) * d_score)

        ranked = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)

        retrieved_chunks = [chunk_id for chunk_id, score in ranked[:5]]

        hit1 = recall_at_k(retrieved_chunks, relevant_chunks, 1)
        hit3 = recall_at_k(retrieved_chunks, relevant_chunks, 3)
        hit5 = recall_at_k(retrieved_chunks, relevant_chunks, 5)
        rr = reciprocal_rank(retrieved_chunks, relevant_chunks)

        recall_1.append(hit1)
        recall_3.append(hit3)
        recall_5.append(hit5)
        reciprocal_ranks.append(rr)

    return {
        "recall@1": np.mean(recall_1),
        "recall@3": np.mean(recall_3),
        "recall@5": np.mean(recall_5),
        "mrr": np.mean(reciprocal_ranks)
    }


def main():
    passages = load_json(PASSAGES_FILE)
    qa_dataset = load_json(QA_FILE)

    print("\nLoading TF-IDF retriever...")
    tfidf = TFIDFRetriever(passages)

    print("\nLoading Dense retriever...")
    dense = DenseRetriever(passages)
    
    print("HYBRID WEIGHT EXPERIMENT")
    print(f"\nAnswerable questions: {sum(q['answerable'] for q in qa_dataset)}")

    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = []

    print("\nTF-IDF Weight | Dense Weight | R@1 | R@3 | R@5 | MRR")
    
    for alpha in weights:
        beta = 1 - alpha
        metrics = evaluate_weight(tfidf, dense, qa_dataset, alpha)
        results.append({"tfidf_weight": alpha, "dense_weight": beta, **metrics})
        print(
            f"{alpha:>12.1f} | "
            f"{beta:>11.1f} | "
            f"{metrics['recall@1'] * 100:>5.2f}% | "
            f"{metrics['recall@3'] * 100:>5.2f}% | "
            f"{metrics['recall@5'] * 100:>5.2f}% | "
            f"{metrics['mrr']:.4f}"
        )

    best_r1 = max(results, key=lambda x: x["recall@1"])
    best_mrr = max(results, key=lambda x: x["mrr"])
    best_r5 = max(results, key=lambda x: x["recall@5"])

    print("\nBEST CONFIGURATIONS")
    print(f"\nBest Recall@1: TF-IDF={best_r1['tfidf_weight']:.1f}, Dense={best_r1['dense_weight']:.1f}")
    print(f"Recall@1 = {best_r1['recall@1'] * 100:.2f}%")
    print(f"\nBest MRR: TF-IDF={best_mrr['tfidf_weight']:.1f}, Dense={best_mrr['dense_weight']:.1f}")
    print(f"MRR = {best_mrr['mrr']:.4f}")
    print(f"\nBest Recall@5: TF-IDF={best_r5['tfidf_weight']:.1f}, Dense={best_r5['dense_weight']:.1f}")
    print(f"Recall@5 = {best_r5['recall@5'] * 100:.2f}%")

    output_file = (PROJECT_ROOT / "results" / "hybrid_weight_experiment.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nExperiment saved to:\n{output_file}")

if __name__ == "__main__":
    main()