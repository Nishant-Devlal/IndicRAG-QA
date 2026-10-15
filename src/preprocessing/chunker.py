import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "documents_raw.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 400
CHUNK_OVERLAP = 60

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def create_chunks(words, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        start += chunk_size - overlap
    return chunks

def process_document(document):
    passages = []
    document_id = document["document_id"]
    language = document["language"]
    category = document["category"]
    scheme = document["scheme"]
    chunk_number = 1

    for page in document["pages"]:
        page_number = page["page_number"]
        text = clean_text(page["text"])
        if not text:
            continue
        words = text.split()
        chunks = create_chunks(words)

        for chunk in chunks:
            passage = {
                "chunk_id": f"{document_id}_C{chunk_number:03d}",
                "document_id": document_id,
                "scheme": scheme,
                "category": category,
                "language": language,
                "page": page_number,
                "text": chunk
            }
            passages.append(passage)
            chunk_number += 1

    return passages


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)
    print(f"\nDocuments loaded: {len(documents)}")
    all_passages = []
    
    for document in documents:
        print(f"\nProcessing: {document['filename']}")
        passages = process_document(document)
        all_passages.extend(passages)
        print(f"Generated passages: {len(passages)}")

    output_file = OUTPUT_DIR / "passages.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_passages, f,
            ensure_ascii=False,
            indent=2
        )

    print(f"\nTotal passages: {len(all_passages)}")
    print(f"\nSaved to:")
    print(output_file)

if __name__ == "__main__":
    main()