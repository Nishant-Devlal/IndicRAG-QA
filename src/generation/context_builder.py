from typing import List, Dict

class ContextBuilder:
    def __init__(self, max_context_chars=12000):
        self.max_context_chars = max_context_chars

    def build_context(self, retrieved_results: List[Dict]) -> str:
        if not retrieved_results:
            return ""

        context_parts = []
        current_length = 0

        for rank, result in enumerate(retrieved_results, start=1):
            chunk_id = result.get("chunk_id", "UNKNOWN")
            document_id = result.get("document_id", "UNKNOWN")
            language = result.get("language", "UNKNOWN")
            page = result.get("page", "UNKNOWN")
            text = result.get( "text", "").strip()

            if not text:
                continue

            passage = (
                f"[Passage {rank}]\n"
                f"Chunk ID: {chunk_id}\n"
                f"Document: {document_id}\n"
                f"Language: {language}\n"
                f"Page: {page}\n"
                f"Content:\n{text}"
            )

            passage_length = len(passage)

            if (current_length + passage_length > self.max_context_chars):
                break

            context_parts.append(passage)

            current_length += (passage_length + 2)

        return "\n\n".join(context_parts)

    def build_context_with_sources(self, retrieved_results: List[Dict]):
        context = self.build_context(retrieved_results)
        sources = []
        for result in retrieved_results:
            sources.append({
                "chunk_id": result.get("chunk_id"),
                "document_id": result.get("document_id"),
                "language": result.get("language"),
                "page": result.get("page")
            })

        return context, sources

if __name__ == "__main__":
    sample_results = [
        {
            "chunk_id": "SAMAGRA_SHIKSHA_ENGLISH_C002",
            "document_id": "SAMAGRA_SHIKSHA_ENGLISH",
            "language": "english",
            "page": 2,
            "score": 0.82,
            "text":
                "Samagra Shiksha supports new "
                "pedagogical and curricular structure "
                "of school education (5+3+3+4)."
        },
        {
            "chunk_id": "SAMAGRA_SHIKSHA_ENGLISH_C003",
            "document_id": "SAMAGRA_SHIKSHA_ENGLISH",
            "language": "english",
            "page": 3,
            "score": 0.76,
            "text":
                "Middle Stage (3): experiential learning "
                "in sciences, mathematics, arts, social "
                "sciences and humanities."
        }
    ]

    builder = ContextBuilder()

    context, sources = (builder.build_context_with_sources(sample_results))

    print("GENERATED CONTEXT")
    print(context)

    print("SOURCES")
    for source in sources:
        print(source)