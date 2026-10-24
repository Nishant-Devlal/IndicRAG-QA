import json
from pathlib import Path
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.context_builder import ContextBuilder
from src.generation.prompt_builder import PromptBuilder
from src.generation.answer_generator import AnswerGenerator
from src.answerability import AnswerabilityChecker

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PASSAGES_FILE = PROJECT_ROOT / "data" / "processed" / "passages.json"

class RAGPipeline:
    def __init__(self, tfidf_weight=0.9, dense_weight=0.1, candidate_k=10, top_k=5):
        self.tfidf_weight = tfidf_weight
        self.dense_weight = dense_weight
        self.candidate_k = candidate_k
        self.top_k = top_k
        
        with open(PASSAGES_FILE, "r", encoding="utf-8") as f:
            self.passages = json.load(f)

        print(f"\nLoaded {len(self.passages)} passages.")
        
        print("\nInitializing Hybrid Retriever...")
        self.retriever = HybridRetriever(self.passages, tfidf_weight=tfidf_weight, dense_weight=dense_weight)

        print("\nInitializing Answerability Checker...")
        self.answerability_checker = AnswerabilityChecker(
            min_relevance_score=0.0,
            min_score_margin=0.0,
            min_hybrid_score=None,
            max_passages=5,
            device="cpu"
        )

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

    def _abstention_response(self, question, language, retrieved_results, verification):
        abstention_messages = {
            "english": (
                "I couldn't find sufficiently relevant supporting "
                "information in the retrieved documents to answer "
                "this question reliably."
            ),
            "hindi": (
                "प्राप्त दस्तावेज़ों में इस प्रश्न का विश्वसनीय उत्तर "
                "देने के लिए पर्याप्त प्रासंगिक जानकारी नहीं मिली।"
            ),
            "hinglish": (
                "Retrieved documents mein is question ka reliable "
                "answer dene ke liye sufficient relevant information "
                "nahi mili."
            )
        }

        return {
            "question": question,
            "language": language,
            "answer": abstention_messages.get(
                language,
                abstention_messages["english"]
            ),
            "sources": [],
            "retrieved_chunks": retrieved_results,
            "answerable": False,
            "answerability_check": verification
        }

    def answer(self, question):
        if not question or not question.strip():
            return {
                "question": question,
                "language": None,
                "answer": "Please enter a question.",
                "sources": [],
                "retrieved_chunks": [],
                "answerable": False,
                "answerability_check": None
            }

        question = question.strip()
        language = self.detect_language(question)
        retrieved_results = self.retriever.search(question, top_k=self.top_k, candidate_k=self.candidate_k)

        if not retrieved_results:
            verification = {
                "answerable": False,
                "reason": "No passages were retrieved.",
                "verified_passages": []
            }

            return self._abstention_response(question, language, retrieved_results, verification)

        print("\nChecking answerability...")
        verification = self.answerability_checker.check(question, retrieved_results)

        print("Answerability check:", verification.get("answerable", False))
        print("Reason:", verification.get("reason", "No reason provided."))

        if not verification.get("answerable", False):
            return self._abstention_response(question, language, retrieved_results, verification)

        verified_passages = verification.get("verified_passages", [])

        if not verified_passages:
            verification["answerable"] = False
            verification["reason"] = (
                "The checker accepted the question but returned "
                "no verified passages."
            )
            return self._abstention_response(question, language, retrieved_results, verification)

        context, sources = (self.context_builder.build_context_with_sources(verified_passages))

        # Safety check: never generate with empty context
        if not context or not context.strip():
            verification["answerable"] = False
            verification["reason"] = ("No usable context was produced from verified passages.")
            return self._abstention_response(question, language, retrieved_results, verification)

        # Build prompt
        prompt = self.prompt_builder.build_prompt(question=question, context=context, language=language)

        print("\nGenerating answer...")
        answer = self.answer_generator.generate(prompt)
        return {
            "question": question,
            "language": language,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved_results,
            "answerable": True,
            "answerability_check": verification
        }

def main():
    print("IndicRAG-QA")
    print("Evidence-Grounded Multilingual Question Answering")
    pipeline = RAGPipeline(tfidf_weight=0.9, dense_weight=0.1, candidate_k=10, top_k=5)

    while True:
        question = input("\nEnter your question (type 'exit' to quit): ")
        if question.lower().strip() == "exit":
            print("\nExiting...")
            break

        result = pipeline.answer(question)

        print("QUESTION")
        print(result["question"])

        print("\nLANGUAGE")
        print(result["language"])

        print("\nANSWER")
        print(result["answer"])

        print("\nANSWERABILITY CHECK")
        print(result["answerable"])

        verification = result.get("answerability_check")

        if verification:
            print("Reason:", verification.get("reason", "Not specified"))

        print("\nSOURCES")

        if result["sources"]:
            for source in result["sources"]:
                print(f"\nChunk ID : {source.get('chunk_id', 'N/A')}")
                print(f"Document : {source.get('document_id', 'N/A')}")
                print(f"Language : {source.get('language', 'N/A')}")
                print(f"Page     : {source.get('page', 'N/A')}")
        else:
            print("No verified sources available.")

if __name__ == "__main__":
    main()
    