from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class AnswerGenerator:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", max_input_length=2048, max_output_length=128):
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        print("\nLoading answer generation model...")
        print(f"Model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        #self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        self.model.to(self.device)
        self.model.eval()

        print(f"Device: {self.device}")
        print("Answer generation model loaded.")

    def generate(self, prompt):
        if not prompt or not prompt.strip():
            return ("I couldn't find that information in the provided documents.")

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt")
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.max_output_length,
                do_sample=False,
                temperature=None,
                top_p=None,
                repetition_penalty=1.1
            )

        generated_tokens = outputs[0][inputs.shape[-1]:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        if not answer:
            return ("I couldn't find that information in the provided documents.")

        return answer

if __name__ == "__main__":
    generator = AnswerGenerator()
    test_prompt = """
You are an evidence-grounded question answering assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Give a complete and direct answer.
2. Answer in 1-3 sentences.
3. Do not answer with only a keyword, number, or phrase.
4. Use the exact information from the context.
5. Preserve numbers, dates, names, percentages and technical terms.
6. Do not use outside knowledge.
7. Do not invent information.
8. If the answer cannot be found in the context, say:
"I couldn't find that information in the provided documents."

Context:

[Passage 1]

Samagra Shiksha supports new pedagogical and curricular
structure of school education (5+3+3+4).

Question:

What pedagogical and curricular structure of school
education does Samagra Shiksha support under NEP 2020?

Answer:
"""
    answer = generator.generate(test_prompt)
    print("\nGENERATED ANSWER")
    print(answer) 
