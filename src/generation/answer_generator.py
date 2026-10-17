from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class AnswerGenerator:
    def __init__(self, model_name="google/flan-t5-base", max_input_length=512, max_output_length=256):
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        print("\nLoading answer generation model...")
        print(f"Model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        print(f"Device: {self.device}")
        print("Answer generation model loaded.")

    def generate(self, prompt):
        if not prompt or not prompt.strip():
            return ("I couldn't find that information in the provided documents.")

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_input_length)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_output_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer.strip()

        if not answer:
            return ("I couldn't find that information in the provided documents.")

        return answer


if __name__ == "__main__":
    generator = AnswerGenerator()
    test_prompt = """
You are an evidence-grounded question answering assistant.

Answer the user's question using ONLY the provided context.

Do not use outside knowledge.
Do not invent information.

If the answer cannot be found in the context, say:
"I couldn't find that information in the provided documents."

Answer in the same language as the question.

Context:
Samagra Shiksha supports the new pedagogical and
curricular structure of school education (5+3+3+4).

Question:
What pedagogical and curricular structure does
Samagra Shiksha support?

Answer:
"""
    answer = generator.generate(test_prompt)
    print("\nGENERATED ANSWER")
    print(answer)