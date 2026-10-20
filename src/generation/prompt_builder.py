class PromptBuilder:
    def build_prompt(self, question: str, context: str, language: str) -> str:
        if not context.strip():
            return ("No relevant information was retrieved from the provided documents.")

        prompt = f"""
You are an evidence-grounded question answering assistant.
Answer the USER QUESTION using ONLY the RETRIEVED CONTEXT.
Rules:
1. Give a complete, direct answer.
2. Do not answer with only a keyword or phrase.
3. Identify the exact information requested by the question.
4. Preserve numbers, dates, names, percentages and technical terms.
5. If the question asks "what", provide the specific item requested.
6. If the question asks "how", explain the relevant process or structure.
7. Prefer the most relevant passage when multiple passages are provided.
8. Ignore passages that are unrelated to the question.
9. Do not use outside knowledge.
10. Do not invent information.
11. If the answer is not present in the context, say: "I couldn't find that information in the provided documents."

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

ANSWER:
"""
        return prompt.strip()

if __name__ == "__main__":
    builder = PromptBuilder()
    question = ("What pedagogical and curricular structure of school education does Samagra Shiksha support under NEP 2020?")
    context = """
[Passage 1]
Chunk ID: SAMAGRA_SHIKSHA_ENGLISH_C002
Document: SAMAGRA_SHIKSHA_ENGLISH
Language: english
Page: 2
Content:
Samagra Shiksha supports new pedagogical and curricular
structure of school education (5+3+3+4).
"""
    prompt = builder.build_prompt(question=question, context=context, language="English")
    print("\nGENERATED PROMPT")
    print(prompt)