import json
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PASSAGES_FILE = (PROJECT_ROOT / "data" / "processed" / "passages.json")

class DenseRetriever:
    def __init__(self, passages):
        self.passages = passages
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
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)

        print(f"Indexed {len(self.passages)} passages.")

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
            passage = self.passages[index]
            results.append({
                "chunk_id": passage["chunk_id"],
                "document_id": passage["document_id"],
                "language": passage["language"],
                "page": passage["page"],
                "score": float(score),
                "text": passage["text"]
            })

        return results

def load_passages():
    with open(PASSAGES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    passages = load_passages()
    retriever = DenseRetriever(passages)
    
    print("DENSE RETRIEVER TEST")
    query = input("\nEnter your question: ")
    results = retriever.search(query, top_k=5)
    print("\nTop 5 results:\n")
    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Language: {result['language']}")
        print(f"Score: {result['score']:.4f}")
        print(f"\nText:\n{result['text']}")

if __name__ == "__main__":
    main()