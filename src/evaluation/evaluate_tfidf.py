import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PASSAGES_FILE = PROJECT_ROOT / "data" / "processed" / "passages.json"
QA_FILE = PROJECT_ROOT /"data" / "qa_dataset" / "qa_dataset.json"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "tfidf_results.json"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class TFIDFRetriever:
    def __init__(self, passages):
        self.passages = passages
        self.texts = [passage["text"] for passage in passages]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.document_matrix = self.vectorizer.fit_transform(self.texts)

    def search(self, query, top_k=5):
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.document_matrix).flatten()
        ranked_indices = np.argsort(scores)[::-1]
        results = []

        for index in ranked_indices[:top_k]:
            results.append({
                "chunk_id": self.passages[index]["chunk_id"],
                "score": float(scores[index])
            })

        return results

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
    retriever = TFIDFRetriever(passages)

    print("TF-IDF RETRIEVAL EVALUATION")

    answerable_questions = [q for q in qa_dataset if q["answerable"]]

    print(f"\nTotal questions: {len(qa_dataset)}")
    print(f"Answerable questions: {len(answerable_questions)}")
    print(f"Unanswerable questions: {len(qa_dataset) - len(answerable_questions)}")

    recall_1 = []
    recall_3 = []
    recall_5 = []
    reciprocal_ranks = []
    question_results = []

    for qa in answerable_questions:
        question = qa["question"]
        relevant_chunks = set(qa["supporting_chunks"])
        results = retriever.search(question, top_k=5)
        retrieved_chunks = [result["chunk_id"] for result in results]

        recall_1.append(recall_at_k(retrieved_chunks, relevant_chunks, 1))
        recall_3.append(recall_at_k(retrieved_chunks, relevant_chunks, 3))
        recall_5.append(recall_at_k(retrieved_chunks, relevant_chunks, 5))
        reciprocal_ranks.append(reciprocal_rank(retrieved_chunks, relevant_chunks))
        rr = reciprocal_rank(retrieved_chunks, relevant_chunks)

        question_results.append({
            "question_id": qa["question_id"],
            "question": qa["question"],
            "query_type": qa["query_type"],
            "language": qa["language"],
            "supporting_chunks": list(relevant_chunks),
            "retrieved_chunks": retrieved_chunks,
            "scores": [result["score"] for result in results],
            "recall@1": recall_at_k(retrieved_chunks, relevant_chunks, 1),
            "recall@3": recall_at_k(retrieved_chunks, relevant_chunks, 3),
            "recall@5": recall_at_k(retrieved_chunks, relevant_chunks, 5),
            "reciprocal_rank": rr
        })

    r1 = np.mean(recall_1)
    r3 = np.mean(recall_3)
    r5 = np.mean(recall_5)
    mrr = np.mean(reciprocal_ranks)
    
    output = {
        "method": "TF-IDF",
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
    
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {RESULTS_FILE}")

    print("OVERALL RESULTS")
    print(f"\nRecall@1 : {r1:.4f} ({r1 * 100:.2f}%)")
    print(f"Recall@3 : {r3:.4f} ({r3 * 100:.2f}%)")
    print(f"Recall@5 : {r5:.4f} ({r5 * 100:.2f}%)")
    print(f"MRR      : {mrr:.4f}")

if __name__ == "__main__":
    main()