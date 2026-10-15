import json
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parents[2]
QA_FILE = BASE_DIR / "data" / "qa_dataset" / "qa_dataset.json"
PASSAGES_FILE = BASE_DIR / "data" / "processed" / "passages.json"

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    except FileNotFoundError:
        print(f"\nERROR: File not found:")
        print(path)
        raise SystemExit(1)

    except json.JSONDecodeError as e:
        print(f"\nERROR: Invalid JSON in:")
        print(path)
        print(f"Line: {e.lineno}, Column: {e.colno}")
        print(e.msg)
        raise SystemExit(1)

def main():
    print("DATASET VALIDATION")

    qa_data = load_json(QA_FILE)
    passages_data = load_json(PASSAGES_FILE)

    print(f"\nQA questions loaded     : {len(qa_data)}")
    print(f"Passages loaded         : {len(passages_data)}")

    errors = []
    warnings = []

    required_fields = [
        "question_id",
        "question",
        "language",
        "query_type",
        "category",
        "difficulty",
        "answerable",
        "expected_answer",
        "supporting_chunks"
    ]

    for i, q in enumerate(qa_data):
        for field in required_fields:
            if field not in q:
                errors.append(f"Question index {i}: missing field '{field}'")

    question_ids = [q.get("question_id") for q in qa_data]

    duplicate_ids = [
        qid for qid, count in Counter(question_ids).items()
        if count > 1
    ]

    if duplicate_ids:
        for qid in duplicate_ids:
            errors.append(f"Duplicate question_id: {qid}")

    valid_languages = {
        "English",
        "Hindi",
        "Hinglish"
    }

    valid_query_types = {
        "monolingual",
        "cross-lingual",
        "code-mixed"
    }

    valid_difficulties = {
        "easy",
        "medium",
        "hard"
    }

    for q in qa_data:
        qid = q.get("question_id", "UNKNOWN")

        if q.get("language") not in valid_languages:
            errors.append(f"{qid}: invalid language '{q.get('language')}'")

        if q.get("query_type") not in valid_query_types:
            errors.append(f"{qid}: invalid query_type '{q.get('query_type')}'")

        if q.get("difficulty") not in valid_difficulties:
            errors.append(f"{qid}: invalid difficulty '{q.get('difficulty')}'")

    for q in qa_data:
        qid = q.get("question_id", "UNKNOWN")
        answerable = q.get("answerable")
        expected_answer = q.get("expected_answer")
        supporting_chunks = q.get("supporting_chunks")

        if answerable is True:
            if expected_answer is None or not str(expected_answer).strip():
                errors.append(f"{qid}: answerable=True but expected_answer is empty")

            if not supporting_chunks:
                errors.append(f"{qid}: answerable=True but supporting_chunks is empty")

        elif answerable is False:
            if expected_answer is not None:
                errors.append(f"{qid}: answerable=False but expected_answer is not null")

            if supporting_chunks:
                errors.append(f"{qid}: answerable=False but supporting_chunks is not empty")

        else:
            errors.append(f"{qid}: answerable must be true or false")

    passage_ids = set()
    for passage in passages_data:
        chunk_id = passage.get("chunk_id")
        if chunk_id is None:
            errors.append("A passage is missing 'chunk_id'")
        else:
            if chunk_id in passage_ids:
                errors.append(f"Duplicate passage chunk_id: {chunk_id}")
            passage_ids.add(chunk_id)

    for q in qa_data:
        qid = q.get("question_id", "UNKNOWN")
        for chunk_id in q.get("supporting_chunks", []):
            if chunk_id not in passage_ids:
                errors.append(f"{qid}: supporting chunk does not exist: {chunk_id}")

    answerable_count = sum(
        1 for q in qa_data
        if q.get("answerable") is True
    )

    unanswerable_count = sum(
        1 for q in qa_data
        if q.get("answerable") is False
    )

    language_counts = Counter(
        q.get("language")
        for q in qa_data
    )

    query_type_counts = Counter(
        q.get("query_type")
        for q in qa_data
    )

    difficulty_counts = Counter(
        q.get("difficulty")
        for q in qa_data
    )

    category_counts = Counter(
        q.get("category")
        for q in qa_data
    )

    print("\nDATASET STATISTICS")
    print(f"\nTotal questions : {len(qa_data)}")
    print(f"Answerable     : {answerable_count}")
    print(f"Unanswerable   : {unanswerable_count}")

    print("\nLanguages:")
    for key, value in language_counts.items():
        print(f"  {key:<15} {value}")

    print("\nQuery Types:")
    for key, value in query_type_counts.items():
        print(f"  {key:<15} {value}")

    print("\nDifficulty:")
    for key, value in difficulty_counts.items():
        print(f"  {key:<15} {value}")

    print("\nCategories:")
    for key, value in category_counts.items():
        print(f"  {str(key):<25} {value}")

    if errors:
        print(f"VALIDATION FAILED")
        print(f"{len(errors)} error(s) found.")
        print("\nERRORS:")
        for error in errors:
            print(f"❌ {error}")

    else:
        print("\nVALIDATION PASSED")
        print("No critical errors found.")

    if warnings:
        print("\nWARNINGS:")
        for warning in warnings:
            print(f"⚠️ {warning}")

if __name__ == "__main__":
    main()