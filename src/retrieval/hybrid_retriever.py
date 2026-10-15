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
        self.document_matrix = (self.vectorizer.fit_transform(self.texts))

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
        print(
            "\nLoading multilingual "
            "embedding model..."
        )

        self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.texts = [passage["text"] for passage in passages]

        print("Creating embeddings...")

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

    def search(self, query, top_k=10):
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        query_embedding = query_embedding.astype("float32")
        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for score, index in zip(scores[0], indices[0]):
            results.append({
                "chunk_id": self.passages[index]["chunk_id"],
                "score": float(score)
            })

        return results

class HybridRetriever:
    def __init__(self, passages,  rrf_k=60):
        self.passages = passages
        self.rrf_k = rrf_k
        
        print("\nInitializing TF-IDF retriever...")
        self.tfidf = TFIDFRetriever(passages)
        
        print("\nInitializing dense retriever...")
        self.dense = DenseRetriever(passages)

    def search(self, query, top_k=5, candidate_k=10):
        tfidf_results = self.tfidf.search(query, top_k=candidate_k)
        dense_results = self.dense.search(query, top_k=candidate_k)

        rrf_scores = {}

        for rank, result in enumerate(tfidf_results, start=1):
            chunk_id = result["chunk_id"]
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0.0
            rrf_scores[chunk_id] += (1.0 / (self.rrf_k + rank))

        for rank, result in enumerate(dense_results, start=1):
            chunk_id = result["chunk_id"]
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0.0
            rrf_scores[chunk_id] += (1.0 / (self.rrf_k + rank))

        ranked_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        passage_lookup = {
            passage["chunk_id"]: passage
            for passage in self.passages
        }

        results = []

        for chunk_id, rrf_score in (ranked_chunks[:top_k]):
            passage = passage_lookup[chunk_id]
            results.append({
                "chunk_id": chunk_id,
                "document_id": passage["document_id"],
                "language": passage["language"],
                "page": passage["page"],
                "rrf_score": float(rrf_score),
                "text": passage["text"]
            })

        return results

def main():
    passages = load_passages()
    retriever = HybridRetriever(passages)

    print("HYBRID RRF RETRIEVER")
    query = input("\nEnter your question: ")
    results = retriever.search(query, top_k=5, candidate_k=10)
    
    print("\nTop 5 hybrid results:\n")
    for rank, result in enumerate(results, start=1):
        print(f"Rank: {rank}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Language: {result['language']}")
        print(f"RRF Score: {result['rrf_score']:.6f}")
        print(f"\nText:\n {result['text']}")

if __name__ == "__main__":
    main()