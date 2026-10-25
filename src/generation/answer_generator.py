import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

# model_name="Qwen/Qwen2.5-1.5B-Instruct"
class AnswerGenerator:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", max_input_length=2048, max_output_length=128,):
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.device = "cpu"

        print("\nLoading answer generation model...")
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")

        try:
            logger.info("Loading tokenizer: %s", self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            logger.info("Tokenizer loaded successfully.")
            logger.info("Loading causal language model.")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )

            logger.info("Model weights loaded successfully.")

            self.model.to(self.device)
            self.model.eval()

            print("Answer generation model loaded successfully.")

        except Exception:
            logger.exception("Failed to initialize the answer generation model.")
            raise

    def generate(self, prompt):
        if not prompt or not prompt.strip():
            return (
                "I couldn't find that information "
                "in the provided documents."
            )

        try:
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]

            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_length,
            )

            inputs = inputs.to(self.device)

            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=inputs,
                    max_new_tokens=self.max_output_length,
                    do_sample=False,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated_tokens = outputs[0][inputs.shape[-1]:]

            answer = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip()

            if not answer:
                return (
                    "I couldn't find that information "
                    "in the provided documents."
                )

            return answer

        except Exception:
            logger.exception("Answer generation failed.")
            raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
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

