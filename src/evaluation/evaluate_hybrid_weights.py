import json
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval.tfidf_retriever import TFIDFRetriever
from retrieval.dense_retriever import DenseRetriever

PASSAGES_FILE = PROJECT_ROOT / "data" / "processed" / "passages.json"
QA_FILE = PROJECT_ROOT / "data" / "qa_dataset" / "qa_dataset.json"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def recall_at_k(retrieved, relevant, k):
    return int(any(chunk in relevant for chunk in retrieved[:k]))

def reciprocal_rank(retrieved, relevant):
    for rank, chunk in enumerate(retrieved, start=1):
        if chunk in relevant:
            return 1.0 / rank

    return 0.0

def prepare_retrieval_results(tfidf, dense, qa_dataset):
    answerable_questions = [q for q in qa_dataset if q["answerable"]]

    cached_results = []

    print("\nCaching retrieval results...")

    for index, qa in enumerate(answerable_questions, start=1):
        query = qa["question"]

        tfidf_results = tfidf.search(query, top_k=10)
        dense_results = dense.search(query, top_k=10)

        tfidf_scores = {
            result["chunk_id"]: float(result["score"])
            for result in tfidf_results
        }

        dense_scores = {
            result["chunk_id"]: float(result["score"])
            for result in dense_results
        }

        cached_results.append({
            "question": query,
            "relevant_chunks": set(qa["supporting_chunks"]),
            "tfidf_scores": tfidf_scores,
            "dense_scores": dense_scores,
        })

        if index % 25 == 0 or index == len(answerable_questions):
            print(f"Processed {index}/{len(answerable_questions)} questions")

    return cached_results


def evaluate_weight(cached_results, alpha):
    beta = 1.0 - alpha
    recall_1 = []
    recall_3 = []
    recall_5 = []
    reciprocal_ranks = []

    for item in cached_results:
        tfidf_scores = item["tfidf_scores"]
        dense_scores = item["dense_scores"]
        relevant_chunks = item["relevant_chunks"]

        candidates = set(tfidf_scores) | set(dense_scores)

        hybrid_scores = {}

        for chunk_id in candidates:
            tfidf_score = tfidf_scores.get(chunk_id, 0.0)
            dense_score = dense_scores.get(chunk_id, 0.0)
            hybrid_scores[chunk_id] = (alpha * tfidf_score + beta * dense_score)

        ranked = sorted(hybrid_scores.items(), key=lambda item: item[1], reverse=True)

        retrieved_chunks = [chunk_id for chunk_id, _ in ranked[:5]]

        recall_1.append(recall_at_k(retrieved_chunks, relevant_chunks, 1))
        recall_3.append(recall_at_k(retrieved_chunks, relevant_chunks, 3))
        recall_5.append(recall_at_k(retrieved_chunks, relevant_chunks, 5))
        reciprocal_ranks.append(reciprocal_rank(retrieved_chunks, relevant_chunks))

    return {
        "recall@1": float(np.mean(recall_1)),
        "recall@3": float(np.mean(recall_3)),
        "recall@5": float(np.mean(recall_5)),
        "mrr": float(np.mean(reciprocal_ranks)),
    }

def main():
    passages = load_json(PASSAGES_FILE)
    qa_dataset = load_json(QA_FILE)

    print("\nLoading TF-IDF retriever...")
    tfidf = TFIDFRetriever(passages)

    print("\nLoading Dense retriever...")
    dense = DenseRetriever(passages)

    answerable_count = sum(1 for q in qa_dataset if q["answerable"])

    print("\nHYBRID WEIGHT EXPERIMENT — RAW SCORES")
    print(f"Answerable questions: {answerable_count}")

    cached_results = prepare_retrieval_results(tfidf, dense, qa_dataset)

    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    results = []

    print("\nTF-IDF Weight | Dense Weight | R@1 | R@3 | R@5 | MRR")

    for alpha in weights:
        beta = 1.0 - alpha
        metrics = evaluate_weight(cached_results, alpha)

        result = {
            "tfidf_weight": alpha,
            "dense_weight": beta,
            **metrics,
        }

        results.append(result)

        print(
            f"{alpha:>12.1f} | "
            f"{beta:>11.1f} | "
            f"{metrics['recall@1'] * 100:>5.2f}% | "
            f"{metrics['recall@3'] * 100:>5.2f}% | "
            f"{metrics['recall@5'] * 100:>5.2f}% | "
            f"{metrics['mrr']:.4f}"
        )

    best_r1 = max(results, key=lambda x: x["recall@1"])
    best_r3 = max(results, key=lambda x: x["recall@3"])
    best_r5 = max(results, key=lambda x: x["recall@5"])
    best_mrr = max(results, key=lambda x: x["mrr"])

    print("\nBEST CONFIGURATIONS")

    print(
        f"\nBest Recall@1: "
        f"TF-IDF={best_r1['tfidf_weight']:.1f}, "
        f"Dense={best_r1['dense_weight']:.1f}"
    )
    print(f"Recall@1 = {best_r1['recall@1'] * 100:.2f}%")

    print(
        f"\nBest Recall@3: "
        f"TF-IDF={best_r3['tfidf_weight']:.1f}, "
        f"Dense={best_r3['dense_weight']:.1f}"
    )
    print(f"Recall@3 = {best_r3['recall@3'] * 100:.2f}%")

    print(
        f"\nBest Recall@5: "
        f"TF-IDF={best_r5['tfidf_weight']:.1f}, "
        f"Dense={best_r5['dense_weight']:.1f}"
    )
    print(f"Recall@5 = {best_r5['recall@5'] * 100:.2f}%")

    print(
        f"\nBest MRR: "
        f"TF-IDF={best_mrr['tfidf_weight']:.1f}, "
        f"Dense={best_mrr['dense_weight']:.1f}"
    )
    print(f"MRR = {best_mrr['mrr']:.4f}")

    output_file = (PROJECT_ROOT / "results" / "hybrid_weight_experiment.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nExperiment saved to:\n{output_file}")

if __name__ == "__main__":
    main()

