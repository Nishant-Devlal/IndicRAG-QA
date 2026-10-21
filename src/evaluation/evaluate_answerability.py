import json
from pathlib import Path
import numpy as np
from src.retrieval.hybrid_retriever import HybridRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PASSAGES_FILE = (PROJECT_ROOT / "data" / "processed" / "passages.json")
QA_FILE = (PROJECT_ROOT / "data" / "qa_dataset" / "qa_dataset.json")
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_FILE = (RESULTS_DIR / "answerability_threshold_results.json")

TFIDF_WEIGHT = 0.7
DENSE_WEIGHT = 0.3

CANDIDATE_K = 10
TOP_K = 5

THRESHOLD_START = 0.00
THRESHOLD_END = 0.90
THRESHOLD_STEP = 0.01

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def calculate_metrics(predictions, actuals):

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for predicted, actual in zip(predictions, actuals):
        if predicted and actual:
            tp += 1
        elif not predicted and not actual:
            tn += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1

    total = tp + tn + fp + fn

    accuracy = ((tp + tn) / total if total > 0 else 0.0)
    precision = (tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    recall = (tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)

    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main():
    print("ANSWERABILITY THRESHOLD EXPERIMENT")
    print("\nLoading passages...")
    passages = load_json(PASSAGES_FILE)
    print(f"Loaded {len(passages)} passages.")

    print("\nLoading QA dataset...")
    qa_dataset = load_json(QA_FILE)

    answerable_count = sum(1 for q in qa_dataset if q["answerable"])
    unanswerable_count = len(qa_dataset) - answerable_count

    print(f"Total questions       : {len(qa_dataset)}")
    print(f"Answerable questions : {answerable_count}")
    print(f"Unanswerable questions : {unanswerable_count}")

    print("\nInitializing weighted hybrid retriever...")
    retriever = HybridRetriever(passages, tfidf_weight=TFIDF_WEIGHT, dense_weight=DENSE_WEIGHT)

    print("\nEvaluating questions...")
    question_data = []

    for index, qa in enumerate(qa_dataset, start=1):
        question = qa["question"]
        retrieved_results = retriever.search(question, top_k=TOP_K, candidate_k=CANDIDATE_K)

        if retrieved_results:
            top_result = retrieved_results[0]
            top_hybrid_score = float(top_result.get("hybrid_score", 0.0))
            top_tfidf_score = float(top_result.get("tfidf_score", 0.0))
            top_dense_score = float(top_result.get("dense_score", 0.0))

        else:
            top_hybrid_score = 0.0
            top_tfidf_score = 0.0
            top_dense_score = 0.0

        question_data.append({
            "question_id": qa["question_id"],
            "question": question,
            "language": qa["language"],
            "query_type": qa["query_type"],
            "difficulty": qa["difficulty"],
            "category": qa["category"],
            "actual_answerable": bool(qa["answerable"]),
            "top_hybrid_score": top_hybrid_score,
            "top_tfidf_score": top_tfidf_score,
            "top_dense_score": top_dense_score,
            "retrieved_chunks": [result["chunk_id"] for result in retrieved_results]
        })

        if index % 25 == 0:
            print(f"Processed {index}/{len(qa_dataset)} questions...")

    thresholds = np.arange(THRESHOLD_START, THRESHOLD_END + THRESHOLD_STEP / 2, THRESHOLD_STEP)

    threshold_results = []

    print("\n")
    print("THRESHOLD RESULTS")
    print(
        f"{'Threshold':<12}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'TP':<8}"
        f"{'TN':<8}"
        f"{'FP':<8}"
        f"{'FN':<8}"
    )

    for threshold in thresholds:
        predictions = []
        actuals = []

        for item in question_data:
            predicted_answerable = (item["top_hybrid_score"] >= threshold)
            predictions.append(predicted_answerable)
            actuals.append(item["actual_answerable"])

        metrics = calculate_metrics(predictions, actuals)
        result = {"threshold": float(threshold), **metrics}
        threshold_results.append(result)

        print(
            f"{threshold:<12.2f}"
            f"{metrics['accuracy'] * 100:<12.2f}"
            f"{metrics['precision'] * 100:<12.2f}"
            f"{metrics['recall'] * 100:<12.2f}"
            f"{metrics['f1']:<12.4f}"
            f"{metrics['true_positive']:<8}"
            f"{metrics['true_negative']:<8}"
            f"{metrics['false_positive']:<8}"
            f"{metrics['false_negative']:<8}"
        )

    best_f1 = max(threshold_results, key=lambda x: x["f1"])
    best_accuracy = max(threshold_results, key=lambda x: x["accuracy"])
    best_precision = max(threshold_results, key=lambda x: x["precision"])

    print("\nBEST THRESHOLDS")
    print("\nBest F1 threshold:")
    print(f"Threshold : {best_f1['threshold']:.2f}")
    print(f"Accuracy  : {best_f1['accuracy'] * 100:.2f}%")
    print(f"Precision : {best_f1['precision'] * 100:.2f}%")
    print(f"Recall    : {best_f1['recall'] * 100:.2f}%")
    print(f"F1        : {best_f1['f1']:.4f}")
    print(
        f"TP={best_f1['true_positive']} "
        f"TN={best_f1['true_negative']} "
        f"FP={best_f1['false_positive']} "
        f"FN={best_f1['false_negative']}"
    )
    print("\nBest accuracy threshold:")
    print(f"Threshold : {best_accuracy['threshold']:.2f}")
    print(f"Accuracy  : {best_accuracy['accuracy'] * 100:.2f}%")
    print("\nBest precision threshold:")
    print(f"Threshold : {best_precision['threshold']:.2f}")
    print(f"Precision : {best_precision['precision'] * 100:.2f}%")

    selected_threshold = best_f1["threshold"]

    false_positives = []
    false_negatives = []

    for item in question_data:
        predicted = (item["top_hybrid_score"] >= selected_threshold)
        actual = item["actual_answerable"]
        
        if predicted and not actual:
            false_positives.append({**item, "predicted_answerable": True})

        elif not predicted and actual:
            false_negatives.append({**item, "predicted_answerable": False})

    output = {
        "method": "Hybrid Answerability Threshold",
        "retrieval_parameters": {
            "tfidf_weight": TFIDF_WEIGHT,
            "dense_weight": DENSE_WEIGHT,
            "candidate_k": CANDIDATE_K,
            "top_k": TOP_K
        },
        "dataset": {
            "total_questions": len(qa_dataset),
            "answerable_questions": answerable_count,
            "unanswerable_questions": unanswerable_count
        },
        "best_threshold": {
            "selection_metric": "f1",
            **best_f1
        },
        "best_accuracy": best_accuracy,
        "best_precision": best_precision,
        "threshold_results": threshold_results,
        "false_positives_at_best_threshold": false_positives,
        "false_negatives_at_best_threshold": false_negatives,
        "questions": question_data
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\nSAVED")
    print(f"\nDetailed results saved to:\n {RESULTS_FILE}")

if __name__ == "__main__":
    main()