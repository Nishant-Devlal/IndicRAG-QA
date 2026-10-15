import fitz
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents" / "education_schemes"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    pages = []
    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text")
        pages.append({"page_number": page_number, "text": text})
    document.close()
    return pages

def detect_language(filename):
    filename = filename.lower()
    if "hindi" in filename:
        return "hindi"
    if "english" in filename:
        return "english"
    return "unknown"

def process_pdf(pdf_path):
    pages = extract_text_from_pdf(pdf_path)
    language = detect_language(pdf_path.name)
    relative_path = pdf_path.relative_to(DOCUMENTS_DIR)
    parts = relative_path.parts

    category = parts[0] if len(parts) >= 2 else "unknown"
    scheme = parts[-2] if len(parts) >= 2 else "unknown"

    document_id = pdf_path.stem.upper()

    return {
        "document_id": document_id,
        "filename": pdf_path.name,
        "language": language,
        "category": category,
        "scheme": scheme,
        "relative_path": str(relative_path),
        "pages": pages
    }

def main():
    all_documents = []
    pdf_files = list(DOCUMENTS_DIR.rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files.")

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path}")
        document = process_pdf(pdf_path)
        all_documents.append(document)

    output_file = OUTPUT_DIR / "documents_raw.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_documents, f, ensure_ascii=False, indent=2)

    print()
    print(f"Saved extracted data to:")
    print(output_file)

if __name__ == "__main__":
    main()