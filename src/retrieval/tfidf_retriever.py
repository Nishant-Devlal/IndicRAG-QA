import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PASSAGES_FILE = PROJECT_ROOT / "data" / "processed" / "passages.json"

def load_passages():
    with open(PASSAGES_FILE, "r", encoding="utf-8") as f:
        passages = json.load(f)
    return passages

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
        print(f"Indexed {len(self.passages)} passages.")

    def search(self, query, top_k=5):
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.document_matrix).flatten()
        ranked_indices = np.argsort(scores)[::-1]

        results = []

        for index in ranked_indices[:top_k]:
            passage = self.passages[index]
            results.append({
                "chunk_id": passage["chunk_id"],
                "document_id": passage["document_id"],
                "language": passage["language"],
                "page": passage["page"],
                "score": float(scores[index]),
                "text": passage["text"]
            })

        return results

def main():
    passages = load_passages()
    retriever = TFIDFRetriever(passages)

    print("TF-IDF RETRIEVER TEST")
    query = input("\nEnter your question: ")
    results = retriever.search(query, top_k=5)

    print("\nTop 5 results:\n")
    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Language: {result['language']}")
        print(f"Page: {result['page']}")
        print(f"Score: {result['score']:.4f}")
        print(f"\nText:\n{result['text']}")

if __name__ == "__main__":
    main()