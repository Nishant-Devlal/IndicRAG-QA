import json
from pathlib import Path
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PASSAGES_FILE = (PROJECT_ROOT / "data" / "processed" / "passages.json")

def load_passages():
    with open(PASSAGES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

class TFIDFRetriever:
    def __init__(self, passages):
        self.passages = passages
        self.texts = [passage["text"] for passage in passages]
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
        self.document_matrix = (self.vectorizer.fit_transform( self.texts))

    def search(self, query, top_k=10):
        query_vector = (self.vectorizer.transform([query]))
        scores = cosine_similarity(query_vector, self.document_matrix).flatten()
        ranked_indices = np.argsort(scores)[::-1]

        results = []

        for index in ranked_indices[:top_k]:
            results.append({
                "chunk_id": self.passages[index]["chunk_id"],
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
        dimension = (self.embeddings.shape[1])
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)
        
        print(f"Indexed {len(passages)} passages.")

    def search(self, query, top_k=10):
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        query_embedding = (query_embedding.astype("float32"))
        scores, indices = (self.index.search(query_embedding, top_k))

        results = []

        for score, index in zip(scores[0], indices[0]):
            results.append({
                "chunk_id": self.passages[index]["chunk_id"],
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
            raise ValueError("TF-IDF and Dense weights must sum to 1.0.")

        print("\nInitializing TF-IDF retriever...")
        self.tfidf = TFIDFRetriever(passages)

        print("\nInitializing Dense retriever...")
        self.dense = DenseRetriever(passages)

    @staticmethod
    def normalize_scores(scores):
        if not scores:
            return {}

        values = np.array(list(scores.values()), dtype=float)

        min_score = values.min()
        max_score = values.max()

        if np.isclose(max_score, min_score):
            return {
                chunk_id: 0.0
                for chunk_id in scores
            }

        normalized = {}

        for chunk_id, score in scores.items():
            normalized[chunk_id] = ((score - min_score) / (max_score - min_score))

        return normalized

    def search(self, query, top_k=5, candidate_k=10):
        tfidf_results = (self.tfidf.search(query, top_k=candidate_k))
        dense_results = (self.dense.search(query, top_k=candidate_k))
        
        tfidf_scores = {result["chunk_id"]:result["score"] for result in tfidf_results}
        dense_scores = {result["chunk_id"]:result["score"]  for result in dense_results}

        tfidf_normalized = (self.normalize_scores(tfidf_scores))
        dense_normalized = (self.normalize_scores(dense_scores))

        all_chunks = (set(tfidf_scores.keys()) | set(dense_scores.keys()))

        hybrid_scores = {}

        for chunk_id in all_chunks:
            tfidf_score = (tfidf_normalized.get(chunk_id, 0.0))
            dense_score = (dense_normalized.get(chunk_id, 0.0))
            hybrid_score = (self.tfidf_weight * tfidf_score + self.dense_weight * dense_score)
            hybrid_scores[chunk_id] = hybrid_score

        ranked_chunks = sorted(
            hybrid_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        passage_lookup = {passage["chunk_id"]:passage for passage in self.passages}

        results = []

        for chunk_id, hybrid_score in (ranked_chunks[:top_k]):
            passage = (passage_lookup[chunk_id])
            results.append({
                "chunk_id": chunk_id,
                "document_id": passage["document_id"],
                "language": passage["language"],
                "page": passage["page"],
                "tfidf_score": float(tfidf_scores.get(chunk_id, 0.0)),
                "dense_score": float(dense_scores.get(chunk_id, 0.0)),
                "tfidf_normalized": float(tfidf_normalized.get(chunk_id, 0.0)),
                "dense_normalized": float(dense_normalized.get(chunk_id, 0.0)),
                "hybrid_score": float(hybrid_score),
                "text": passage["text"]
            })

        return results

def main():
    passages = load_passages()

    TFIDF_WEIGHT = 0.7
    DENSE_WEIGHT = 0.3

    retriever = HybridRetriever(passages, tfidf_weight=TFIDF_WEIGHT, dense_weight=DENSE_WEIGHT)

    print("WEIGHTED HYBRID RETRIEVER")
    print(f"\nTF-IDF weight : {TFIDF_WEIGHT}")
    print(f"Dense weight  : {DENSE_WEIGHT}")
    query = input("\nEnter your question: ")
    results = retriever.search(query, top_k=5, candidate_k=10)
    print("\nTop 5 hybrid results:\n")
    for rank, result in enumerate(results, start=1):
        print(f"Rank: {rank}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Language: {result['language']}")
        print(f"TF-IDF score: {result['tfidf_score']:.4f}")
        print(f"Dense score: {result['dense_score']:.4f}")
        print(f"Normalized TF-IDF: {result['tfidf_normalized']:.4f}")
        print(f"Normalized Dense: {result['dense_normalized']:.4f}")
        print(f"Hybrid score: {result['hybrid_score']:.4f}")
        print(f"\nText:\n {result['text']}")

if __name__ == "__main__":
    main()