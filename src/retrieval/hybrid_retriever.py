import json
from pathlib import Path
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PASSAGES_FILE = (PROJECT_ROOT / "data" / "processed" / "passages.json")

def load_passages():
    with open(PASSAGES_FILE, "r", encoding="utf-8") as file:
        return json.load(file)

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
        print(f"TF-IDF indexed {len(self.passages)} passages.")

    def search(self, query, top_k=10):
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.document_matrix).flatten()
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for index in ranked_indices:
            passage = self.passages[index]
            results.append({
                "chunk_id": passage["chunk_id"],
                "score": float(scores[index])
            })
        return results

class DenseRetriever:
    def __init__(self, passages):
        self.passages = passages
        print("\nLoading multilingual embedding model...")
        self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
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
        print(f"Dense retriever indexed {len(self.passages)} passages.")

    def search(self, query, top_k=10):
        query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        query_embedding = query_embedding.astype("float32")
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.passages)))
        results = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            passage = self.passages[index]
            results.append({
                "chunk_id": passage["chunk_id"],
                "score": float(score)
            })
        return results

class HybridRetriever:
    def __init__(self, passages, tfidf_weight=0.7, dense_weight=0.3):
        self.passages = passages
        self.tfidf_weight = tfidf_weight
        self.dense_weight = dense_weight

        if tfidf_weight < 0 or dense_weight < 0:
            raise ValueError("Weights cannot be negative.")

        if not np.isclose(tfidf_weight + dense_weight, 1.0):
            raise ValueError("TF-IDF and dense weights must sum to 1.0.")

        if not passages:
            raise ValueError("The passages list is empty.")

        print("\nInitializing TF-IDF retriever...")
        self.tfidf = TFIDFRetriever(passages)

        print("\nInitializing dense retriever...")
        self.dense = DenseRetriever(passages)

        self.passage_lookup = {
            passage["chunk_id"]: passage
            for passage in passages
        }

    def search(self, query, top_k=5, candidate_k=10):
        if not query or not query.strip():
            return []

        if top_k < 1 or candidate_k < 1:
            raise ValueError("top_k and candidate_k must be positive integers.")

        tfidf_results = self.tfidf.search(query, top_k=min(candidate_k, len(self.passages)))
        dense_results = self.dense.search(query, top_k=min(candidate_k, len(self.passages)))
        
        tfidf_scores = {result["chunk_id"]: result["score"] for result in tfidf_results}
        dense_scores = {result["chunk_id"]: result["score"]  for result in dense_results}
        
        all_chunks = set(tfidf_scores) | set(dense_scores)

        hybrid_scores = {}

        for chunk_id in all_chunks:
            tfidf_score = tfidf_scores.get(chunk_id, 0.0)
            dense_score = dense_scores.get(chunk_id, 0.0)
            hybrid_score = (self.tfidf_weight * tfidf_score + self.dense_weight * dense_score)
            hybrid_scores[chunk_id] = hybrid_score

        ranked_chunks = sorted(
            hybrid_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        results = []

        for chunk_id, hybrid_score in ranked_chunks[:top_k]:
            passage = self.passage_lookup[chunk_id]
            results.append({
                "chunk_id": chunk_id,
                "document_id": passage["document_id"],
                "language": passage["language"],
                "page": passage["page"],
                "tfidf_score": float(tfidf_scores.get(chunk_id, 0.0)),
                "dense_score": float(dense_scores.get(chunk_id, 0.0)),
                "hybrid_score": float(hybrid_score),
                "text": passage["text"]
            })

        return results


def main():
    passages = load_passages()

    TFIDF_WEIGHT = 0.9
    DENSE_WEIGHT = 0.1

    retriever = HybridRetriever(passages, tfidf_weight=TFIDF_WEIGHT, dense_weight=DENSE_WEIGHT)

    print("\nWEIGHTED HYBRID RETRIEVER")
    print(f"TF-IDF weight: {TFIDF_WEIGHT}")
    print(f"Dense weight:  {DENSE_WEIGHT}")

    query = input("\nEnter your question: ").strip()
    results = retriever.search(query, top_k=5, candidate_k=10)

    print("\nTop hybrid retrieval results:\n")

    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        print(f"Rank: {rank}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Document: {result['document_id']}")
        print(f"Language: {result['language']}")
        print(f"Page: {result['page']}")
        print(f"TF-IDF score: {result['tfidf_score']:.4f}")
        print(f"Dense score:  {result['dense_score']:.4f}")
        print(f"Hybrid score: {result['hybrid_score']:.4f}")
        print(f"\nText:\n{result['text']}\n")

if __name__ == "__main__":
    main()

