from typing import Any, Dict, List, Optional
from sentence_transformers import CrossEncoder

class AnswerabilityChecker:
    MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        min_relevance_score: float = 0.0,
        min_score_margin: float = 0.0,
        min_hybrid_score: Optional[float] = None,
        max_passages: int = 5,
        max_length: int = 256,
        device: str = "cpu",
    ):
        if max_passages < 1:
            raise ValueError("max_passages must be at least 1.")

        if max_length < 1:
            raise ValueError("max_length must be at least 1.")

        self.min_relevance_score = min_relevance_score
        self.min_score_margin = min_score_margin
        self.min_hybrid_score = min_hybrid_score
        self.max_passages = max_passages
        self.max_length = max_length
        self.device = device

        print("\nLoading answerability verification model...")
        print(f"Model: {model_name}")
        print(f"Device: {device}")

        self.model = CrossEncoder(model_name, device=device, max_length=max_length,)

        print("Answerability verification model loaded.")

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0,) -> float:
        try:
            number = float(value)

            if number != number:
                return default

            if number == float("inf") or number == float("-inf"):
                return default

            return number

        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_passage_text(passage: Dict[str, Any],) -> str:
        text = passage.get("text", "")
        if not isinstance(text, str):
            return ""
        return text.strip()

    def _score_passages(self, question: str, passages: List[Dict[str, Any]],) -> List[Dict[str, Any]]:
        pairs = []
        valid_passages = []

        for passage in passages:
            passage_text = self._normalize_passage_text(passage)
            if not passage_text:
                continue
            pairs.append((question, passage_text))
            valid_passages.append(passage)

        if not pairs:
            return []

        scores = self.model.predict(pairs, batch_size=8, show_progress_bar=False,)

        scored_results = []

        for passage, score in zip(valid_passages, scores):
            result = dict(passage)
            result["relevance_score"] = self._safe_float(score)
            result["hybrid_score"] = self._safe_float(passage.get("hybrid_score", 0.0))
            scored_results.append(result)

        scored_results.sort(key=lambda item: item["relevance_score"], reverse=True,)

        return scored_results

    def check(self, question: str, retrieved_results: List[Dict[str, Any]],) -> Dict[str, Any]:
        if not isinstance(question, str) or not question.strip():
            return {
                "answerable": False,
                "reason": "The question is empty or invalid.",
                "top_relevance_score": None,
                "second_relevance_score": None,
                "relevance_margin": None,
                "top_hybrid_score": None,
                "verified_passages": [],
                "model_name": self.MODEL_NAME,
                "verification_type": "cross_encoder_relevance",
            }

        if not retrieved_results:
            return {
                "answerable": False,
                "reason": "No passages were retrieved.",
                "top_relevance_score": None,
                "second_relevance_score": None,
                "relevance_margin": None,
                "top_hybrid_score": None,
                "verified_passages": [],
                "model_name": self.MODEL_NAME,
                "verification_type": "cross_encoder_relevance",
            }

        passages = retrieved_results[:self.max_passages]
        scored_results = self._score_passages(question.strip(), passages,)

        if not scored_results:
            return {
                "answerable": False,
                "reason": "Retrieved passages contain no usable text.",
                "top_relevance_score": None,
                "second_relevance_score": None,
                "relevance_margin": None,
                "top_hybrid_score": None,
                "verified_passages": [],
                "model_name": self.MODEL_NAME,
                "verification_type": "cross_encoder_relevance",
            }

        top_result = scored_results[0]

        top_score = top_result["relevance_score"]

        second_score = (
            scored_results[1]["relevance_score"]
            if len(scored_results) > 1
            else None
        )

        score_margin = (
            top_score - second_score
            if second_score is not None
            else None
        )

        top_hybrid_score = top_result["hybrid_score"]

        reasons = []

        if top_score < self.min_relevance_score:
            reasons.append("The top passage did not meet the minimum cross-encoder relevance threshold.")

        if (self.min_hybrid_score is not None and top_hybrid_score < self.min_hybrid_score):
            reasons.append("The top passage did not meet the minimum hybrid retrieval score.")

        if (self.min_score_margin > 0.0 and score_margin is not None and score_margin < self.min_score_margin):
            reasons.append("The difference between the top two relevance scores was below the configured margin.")

        accepted = len(reasons) == 0

        if accepted:
            reason = ("Retrieved evidence passed the configured relevance checks."
                      "Factual support has not been independently verified.")
        else:
            reason = " ".join(reasons)

        return {
            "answerable": accepted,
            "reason": reason,
            "top_relevance_score": float(top_score),
            "second_relevance_score": (
                float(second_score)
                if second_score is not None
                else None
            ),
            "relevance_margin": (
                float(score_margin)
                if score_margin is not None
                else None
            ),
            "top_hybrid_score": float(top_hybrid_score),
            "verified_passages": scored_results,
            "model_name": self.MODEL_NAME,
            "verification_type": "cross_encoder_relevance",
        }
