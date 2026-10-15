import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TFIDF_FILE = (PROJECT_ROOT / "results" / "tfidf_results.json")
DENSE_FILE = (PROJECT_ROOT / "results" / "dense_results.json")
HYBRID_FILE = (PROJECT_ROOT / "results" / "hybrid_results.json")

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def calculate_metrics(questions):
    if not questions:
        return {
            "count": 0,
            "recall@1": 0.0,
            "recall@3": 0.0,
            "recall@5": 0.0,
            "mrr": 0.0
        }

    return {
        "count": len(questions),
        "recall@1": np.mean([q["recall@1"] for q in questions]),
        "recall@3": np.mean([q["recall@3"] for q in questions]),
        "recall@5": np.mean([q["recall@5"] for q in questions]),
        "mrr": np.mean([q["reciprocal_rank"] for q in questions])
    }

def print_metrics(name, metrics):
    print(
        f"{name:<12}"
        f"{metrics['recall@1'] * 100:>10.2f}%"
        f"{metrics['recall@3'] * 100:>10.2f}%"
        f"{metrics['recall@5'] * 100:>10.2f}%"
        f"{metrics['mrr']:>10.4f}"
    )

def main():
    tfidf = load_json(TFIDF_FILE)
    dense = load_json(DENSE_FILE)
    hybrid = load_json(HYBRID_FILE)

    methods = {
        "TF-IDF": {
            q["question_id"]: q
            for q in tfidf["questions"]
        },

        "Dense": {
            q["question_id"]: q
            for q in dense["questions"]
        },

        "Hybrid": {
            q["question_id"]: q
            for q in hybrid["questions"]
        }
    }

    print("OVERALL RETRIEVAL COMPARISON")
    print(
        f"\n{'Method':<12}"
        f"{'R@1':>10}"
        f"{'R@3':>10}"
        f"{'R@5':>10}"
        f"{'MRR':>10}"
    )

    for method, questions in methods.items():
        metrics = calculate_metrics(list(questions.values()))
        print_metrics(method, metrics)

    query_types = sorted(
        set(q["query_type"] for q in methods["TF-IDF"].values())
    )

    print("\nRESULTS BY QUERY TYPE")
    for query_type in query_types:
        print(f"\n{'QUERY TYPE: ' + query_type.upper()}")
        print(
            f"{'Method':<12}"
            f"{'Questions':>10}"
            f"{'R@1':>10}"
            f"{'R@3':>10}"
            f"{'R@5':>10}"
            f"{'MRR':>10}"
        )

        for method, questions in methods.items():
            filtered_questions = [q for q in questions.values() if q["query_type"] == query_type]
            metrics = calculate_metrics(filtered_questions)
            print(
                f"{method:<12}"
                f"{metrics['count']:>10}"
                f"{metrics['recall@1'] * 100:>10.2f}%"
                f"{metrics['recall@3'] * 100:>10.2f}%"
                f"{metrics['recall@5'] * 100:>10.2f}%"
                f"{metrics['mrr']:>10.4f}"
            )

    languages = sorted(set(q["language"] for q in methods["TF-IDF"].values()))

    print("\nRESULTS BY LANGUAGE")
    for language in languages:
        print(f"\nLANGUAGE: {language.upper()}")
        print(
            f"{'Method':<12}"
            f"{'Questions':>10}"
            f"{'R@1':>10}"
            f"{'R@3':>10}"
            f"{'R@5':>10}"
            f"{'MRR':>10}"
        )

        for method, questions in methods.items():
            filtered_questions = [q for q in questions.values() if q["language"] == language]
            metrics = calculate_metrics(filtered_questions)
            print(
                f"{method:<12}"
                f"{metrics['count']:>10}"
                f"{metrics['recall@1'] * 100:>10.2f}%"
                f"{metrics['recall@3'] * 100:>10.2f}%"
                f"{metrics['recall@5'] * 100:>10.2f}%"
                f"{metrics['mrr']:>10.4f}"
            )

if __name__ == "__main__":
    main()