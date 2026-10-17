class PromptBuilder:
    def build_prompt(self, question: str, context: str, language: str) -> str:
        if not context.strip():
            return ("No relevant information was retrieved from the provided documents.")

        prompt = f"""
You are an evidence-grounded question answering assistant.

Your task is to answer the user's question using ONLY the
information provided in the context below.

IMPORTANT RULES:
1. Use only the provided context.
2. Do not use outside knowledge.
3. Do not invent or assume information.
4. If the answer cannot be found in the context, say:
   "I couldn't find that information in the provided documents."
5. Give a complete but concise answer.
6. Answer every part of the user's question.
7. Do not give only a phrase, keyword, or partial answer.
8. Preserve important numbers, dates, names, percentages,
   eligibility conditions, and other factual details..
9. Answer in the same language as the user's question.
10. For Hinglish questions, answer naturally in Hinglish.
11. Do not mention information that is not supported by the context.

USER QUESTION:
{question}

LANGUAGE:
{language}

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