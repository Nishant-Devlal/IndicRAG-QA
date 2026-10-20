import json
from pathlib import Path
from src.generation.rag_pipeline import RAGPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]

QA_FILE = (PROJECT_ROOT / "data" / "qa_dataset" / "qa_dataset.json")
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_FILE = (RESULTS_DIR / "rag_evaluation_results.json")

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    print("INDICRAG-QA END-TO-END EVALUATION")
    qa_dataset = load_json(QA_FILE)
    
    print(f"\nTotal questions: {len(qa_dataset)}")
    answerable_questions = [q for q in qa_dataset if q["answerable"]]
    unanswerable_questions = [q for q in qa_dataset if not q["answerable"]]
    print(f"Answerable questions: {len(answerable_questions)}")
    print(f"Unanswerable questions: {len(unanswerable_questions)}")

    print("\nInitializing RAG pipeline...")
    pipeline = RAGPipeline(
        tfidf_weight=0.7,
        dense_weight=0.3,
        candidate_k=10,
        top_k=5
    )

    print("\nPipeline ready.")
    print("\nEvaluating questions...\n")

    results = []

    for index, qa in enumerate(qa_dataset, start=1):
        print(f"[{index}/{len(qa_dataset)}] {qa['question_id']}")
        try:
            result = pipeline.answer(qa["question"])
            retrieved_results = result.get("retrieved_chunks", [])
            retrieved_chunks = [r["chunk_id"] for r in retrieved_results]
            hybrid_scores = [
                {
                    "chunk_id": r["chunk_id"],
                    "score": r.get("hybrid_score", 0.0)
                }
                for r in retrieved_results
            ]

            top_hybrid_score = (hybrid_scores[0]["score"] if hybrid_scores else 0.0)

            evaluation_result = {
                "question_id": qa["question_id"],
                "question": qa["question"],
                "language": qa["language"],
                "query_type": qa["query_type"],
                "difficulty": qa["difficulty"],
                "category": qa["category"],
                "answerable": qa["answerable"],
                "supporting_chunks": qa.get("supporting_chunks", []),
                "retrieved_chunks": retrieved_chunks,
                "hybrid_scores": hybrid_scores,
                "top_hybrid_score": top_hybrid_score,
                "generated_answer": result.get("answer", ""),
                "sources": result.get("sources", [])
            }

            results.append(evaluation_result)

        except Exception as e:
            print(f"ERROR processing {qa['question_id']}: {e}")
            results.append({
                "question_id": qa["question_id"],
                "question": qa["question"],
                "language": qa["language"],
                "query_type": qa["query_type"],
                "difficulty": qa["difficulty"],
                "category": qa["category"],
                "answerable": qa["answerable"],
                "supporting_chunks": qa.get("supporting_chunks", []),
                "retrieved_chunks": [],
                "hybrid_scores": [],
                "top_hybrid_score": 0.0,
                "generated_answer": "",
                "sources": [],
                "error": str(e)
            })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "method":
            "End-to-End RAG",

        "parameters": {
            "tfidf_weight": 0.7,
            "dense_weight": 0.3,
            "candidate_k": 10,
            "top_k": 5,
            "generation_model": "Qwen/Qwen2.5-1.5B-Instruct"
        },

        "dataset": {
            "total_questions": len(qa_dataset),
            "answerable_questions": len(answerable_questions),
            "unanswerable_questions": len(unanswerable_questions)
        },

        "questions":
            results
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("EVALUATION COMPLETE")
    print(f"\nResults saved to:\n {RESULTS_FILE}")

if __name__ == "__main__":
    main()