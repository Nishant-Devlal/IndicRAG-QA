import json
from pathlib import Path
from src.retrieval.hybrid_retriever import HybridRetriever
from src.answerability.answerability_checker import AnswerabilityChecker

PROJECT_ROOT = Path(__file__).resolve().parent
PASSAGES_FILE = (PROJECT_ROOT / "data" / "processed" / "passages.json")

def main():
    with open(PASSAGES_FILE, "r", encoding="utf-8") as file:
        passages = json.load(file)

    if not passages:
        print("No passages found in passages.json.")
        return

    print(f"Loaded {len(passages)} passages.")

    retriever = HybridRetriever(passages, tfidf_weight=0.9, dense_weight=0.1,)
    checker = AnswerabilityChecker(
        min_relevance_score=0.0,
        min_score_margin=0.0,
        min_hybrid_score=None,
        max_passages=5,
        device="cpu",
    )

    question = input("\nEnter your question: ").strip()

    if not question:
        print("Please enter a valid question.")
        return

    retrieved_results = retriever.search(question, top_k=5, candidate_k=10,)

    if not retrieved_results:
        print("\nNo passages were retrieved.")
        return

    print("\nINITIAL HYBRID RETRIEVAL")
    for rank, passage in enumerate(retrieved_results, start=1):
        print(f"Rank: {rank}")
        print(f"Chunk ID: {passage['chunk_id']}")
        print(f"Hybrid score: {passage['hybrid_score']:.4f}")
        print(f"Text: {passage['text'][:500]}")

    result = checker.check(question=question, retrieved_results=retrieved_results,)

    print("\nANSWERABILITY VERIFICATION")
    print(f"Question: {question}")
    print(f"Accepted by gate: {result['answerable']}")
    print(f"Reason: {result['reason']}")

    if result["top_relevance_score"] is not None:
        print("Top relevance score: {result['top_relevance_score']:.4f}")

    print("\nPassages ranked by cross-encoder relevance:")
    for rank, passage in enumerate(result["verified_passages"], start=1,):
        print(f"Rank: {rank}")
        print(f"Chunk ID: {passage.get('chunk_id')}")
        print(f"Relevance score: {passage['relevance_score']:.4f}")
        print(f"Hybrid score: {passage['hybrid_score']:.4f}")
        print(f"Text: {passage.get('text', '')[:500]}")

if __name__ == "__main__":
    main()
