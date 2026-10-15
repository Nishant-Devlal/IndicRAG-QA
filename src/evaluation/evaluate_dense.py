import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PASSAGES_FILE = (PROJECT_ROOT / "data" / "processed" / "passages.json")
QA_FILE = (PROJECT_ROOT / "data" / "qa_dataset" / "qa_dataset.json")
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_FILE = RESULTS_DIR / "dense_results.json"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class DenseRetriever:
    def __init__(self, passages):
        self.passages = passages
        print("\nLoading multilingual model...")
        # self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        MODEL_NAME = ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        self.model = SentenceTransformer(MODEL_NAME)
        self.texts = [passage["text"] for passage in passages]

        print("Creating passage embeddings...")
        embeddings = self.model.encode(
            self.texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        self.embeddings = embeddings.astype("float32")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)

        print(f"Indexed {len(passages)} passages.")

    def search(self, query, top_k=5):
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        query_embedding = query_embedding.astype("float32")
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, index in zip(scores[0], indices[0]):
            results.append(
                {
                    "chunk_id":self.passages[index]["chunk_id"],
                    "score":float(score)
                }
            )
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
    retriever = DenseRetriever(passages)
    answerable_questions = [q for q in qa_dataset if q["answerable"]]

    print("DENSE RETRIEVAL EVALUATION")
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
            "scores": [result["score"] for result in results],
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
        "method": "Dense",
        "model": (
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        ),
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

    print(f"\nResults saved to: {RESULTS_FILE}")

    print("OVERALL RESULTS")
    print(f"\nRecall@1 : {r1:.4f} ({r1 * 100:.2f}%)")
    print(f"Recall@3 : {r3:.4f} ({r3 * 100:.2f}%)")
    print(f"Recall@5 : {r5:.4f} ({r5 * 100:.2f}%)")
    print(f"MRR      : {mrr:.4f}")

if __name__ == "__main__":
    main()