import json
from pathlib import Path
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.context_builder import ContextBuilder
from src.generation.prompt_builder import PromptBuilder
from src.generation.answer_generator import AnswerGenerator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PASSAGES_FILE = (PROJECT_ROOT / "data" / "processed" / "passages.json")

class RAGPipeline:
    def __init__(self, tfidf_weight=0.7, dense_weight=0.3, candidate_k=10, top_k=5):
        self.tfidf_weight = tfidf_weight
        self.dense_weight = dense_weight
        self.candidate_k = candidate_k
        self.top_k = top_k

        with open(PASSAGES_FILE, "r", encoding="utf-8") as f:
            self.passages = json.load(f)

        print(f"\nLoaded {len(self.passages)} passages.")
        print("\nInitializing Hybrid Retriever...")
        self.retriever = HybridRetriever(self.passages, tfidf_weight=tfidf_weight, dense_weight=dense_weight)

        print("\nInitializing Context Builder...")
        self.context_builder = ContextBuilder(max_context_chars=12000)

        print("\nInitializing Prompt Builder...")
        self.prompt_builder = PromptBuilder()

        print("\nInitializing Answer Generator...")
        self.answer_generator = AnswerGenerator()

        print("\nRAG Pipeline initialized successfully.")

    def detect_language(self, question):
        devanagari_chars = 0
        latin_chars = 0
        
        for char in question:
            if "\u0900" <= char <= "\u097F":
                devanagari_chars += 1

            elif char.isalpha():
                latin_chars += 1

        if devanagari_chars > 0 and latin_chars > 0:
            return "hinglish"

        elif devanagari_chars > 0:
            return "hindi"

        return "english"

    def answer(self, question):
        if not question or not question.strip():
            return {
                "answer": "Please enter a question.",
                "sources": [],
                "retrieved_chunks": []
            }

        question = question.strip()

        language = self.detect_language(question)
        retrieved_results = self.retriever.search(question, top_k=self.top_k, candidate_k=self.candidate_k)
        context, sources = (self.context_builder.build_context_with_sources(retrieved_results))
        prompt = self.prompt_builder.build_prompt(question=question, context=context, language=language)
        answer = self.answer_generator.generate(prompt)

        return {
            "question": question,
            "language": language,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved_results
        }

def main():
    print("IndicRAG-QA")
    print("Evidence-Grounded Multilingual Question Answering")
    pipeline = RAGPipeline(tfidf_weight=0.7, dense_weight=0.3, candidate_k=10, top_k=5)

    while True:
        question = input("\nEnter your question (type 'exit' to quit): ")
        if question.lower().strip() == "exit":
            print("\nExiting...")
            break

        result = pipeline.answer(question)
        print("ANSWER")
        print(result["answer"])

        print("SOURCES")
        for source in result["sources"]:
            print(f"\nChunk ID : {source['chunk_id']}")
            print(f"Document : {source['document_id']}")
            print(f"Language : {source['language']}")
            print(f"Page     : {source['page']}")

if __name__ == "__main__":
    main()