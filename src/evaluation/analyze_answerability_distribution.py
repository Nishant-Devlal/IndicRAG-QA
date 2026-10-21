import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_FILE = (PROJECT_ROOT / "results" / "hybrid_detailed_results.json")

def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["questions"]

def get_score_distributions(question_results):
    answerable_scores = []
    unanswerable_scores = []

    for result in question_results:
        score = result.get("top_hybrid_score")
        answerable = result.get("answerable")

        if score is None:
            continue

        if answerable is True:
            answerable_scores.append(float(score))

        elif answerable is False:
            unanswerable_scores.append(float(score))

    return answerable_scores, unanswerable_scores

def print_statistics(name, scores):
    scores = np.array(scores)
    print(f"\n{name}")
    print(f"Questions : {len(scores)}")
    print(f"Minimum   : {scores.min():.6f}")
    print(f"Maximum   : {scores.max():.6f}")
    print(f"Mean      : {scores.mean():.6f}")
    print(f"Median    : {np.median(scores):.6f}")
    print(f"Std Dev   : {scores.std():.6f}")

    percentiles = [10, 25, 50, 75, 90, 95]
    print("\nPercentiles:")

    for p in percentiles:
        print(f"P{p:<2}       : {np.percentile(scores, p):.6f}")

def analyze_overlap(answerable_scores, unanswerable_scores):
    answerable = np.array(answerable_scores)
    unanswerable = np.array(unanswerable_scores)
    print("SCORE OVERLAP ANALYSIS")
    print(f"\nAnswerable range   : {answerable.min():.6f} - {answerable.max():.6f}")
    print(f"Unanswerable range : {unanswerable.min():.6f} - {unanswerable.max():.6f}")
    overlap_min = max(answerable.min(), unanswerable.min())
    overlap_max = min(answerable.max(), unanswerable.max())

    if overlap_min <= overlap_max:
        print(f"\nOVERLAP RANGE      : {overlap_min:.6f} - {overlap_max:.6f}")
        print("\nThe two groups overlap.")
        print("A single score threshold may therefore not perfectly separate answerable and unanswerable questions.")

    else:
        print("\nNo score overlap detected.")
        print("A score threshold may be able to separate the two groups.")

def test_candidate_thresholds(answerable_scores, unanswerable_scores):
    print("CANDIDATE THRESHOLD ANALYSIS")
    all_scores = sorted(set(answerable_scores + unanswerable_scores))

    best_threshold = None
    best_f1 = -1

    for threshold in all_scores:
        tp = sum(score >= threshold for score in answerable_scores)
        fn = sum(score < threshold for score in answerable_scores)
        fp = sum(score >= threshold for score in unanswerable_scores)
        tn = sum(score < threshold for score in unanswerable_scores)
        precision = (tp / (tp + fp) if (tp + fp) > 0 else 0)
        recall = (tp / (tp + fn) if (tp + fn) > 0 else 0)
        f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"\nBest threshold by F1 : {best_threshold:.6f}")
    print(f"Best F1             : {best_f1:.4f}")
    print("\nPerformance around selected thresholds:")

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    print(
        f"{'Threshold':<12}"
        f"{'TP':>7}"
        f"{'TN':>7}"
        f"{'FP':>7}"
        f"{'FN':>7}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
    )

    for threshold in thresholds:
        tp = sum(score >= threshold for score in answerable_scores)
        fn = sum(score < threshold for score in answerable_scores)
        fp = sum(score >= threshold for score in unanswerable_scores)
        tn = sum(score < threshold for score in unanswerable_scores)
        precision = (tp / (tp + fp) if tp + fp > 0 else 0)
        recall = (tp / (tp + fn) if tp + fn > 0 else 0)
        f1 = (2 * precision * recall / (precision + recall) if precision + recall > 0 else 0)

        print(
            f"{threshold:<12.2f}"
            f"{tp:>7}"
            f"{tn:>7}"
            f"{fp:>7}"
            f"{fn:>7}"
            f"{precision * 100:>11.2f}%"
            f"{recall * 100:>9.2f}%"
            f"{f1:>10.4f}"
        )

def plot_distribution(answerable_scores, unanswerable_scores):
    plt.figure(figsize=(10, 6))
    plt.hist(answerable_scores, bins=20, alpha=0.6, label="Answerable")
    plt.hist(unanswerable_scores, bins=20, alpha=0.6, label="Unanswerable")
    plt.xlabel("Top Hybrid Score")
    plt.ylabel("Number of Questions")
    plt.title("Top Hybrid Score Distribution: Answerable vs Unanswerable")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    output_file = (PROJECT_ROOT / "results" / "answerability_score_distribution.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"\nDistribution plot saved to:\n{output_file}")

def main():
    print("ANSWERABILITY SCORE DISTRIBUTION ANALYSIS")
    question_results = load_results(RESULTS_FILE)
    print(f"\nLoaded {len(question_results)} question results.")
    (answerable_scores, unanswerable_scores) = get_score_distributions(question_results)
    print_statistics("ANSWERABLE QUESTIONS", answerable_scores)
    print_statistics("UNANSWERABLE QUESTIONS", unanswerable_scores)
    analyze_overlap(answerable_scores, unanswerable_scores)
    test_candidate_thresholds(answerable_scores, unanswerable_scores)
    plot_distribution(answerable_scores, unanswerable_scores)

if __name__ == "__main__":
    main()