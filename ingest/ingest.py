import os
from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader
import chromadb
from fastembed import TextEmbedding

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_PATH = os.getenv("CHROMA_PATH", str(PROJECT_ROOT / "chroma_db"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "docs")


def extract_pages(pdf_path: Path) -> List[Dict]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def main():
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {DATA_DIR}. Add a PDF and re-run.")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    ids, docs, metadatas = [], [], []

    for pdf_path in pdfs:
        pages = extract_pages(pdf_path)
        for p in pages:
            for ci, chunk in enumerate(chunk_text(p["text"])):
                doc_id = f"{pdf_path.stem}_p{p['page']}_c{ci}"
                ids.append(doc_id)
                docs.append(chunk)
                metadatas.append({"filename": pdf_path.name, "page": p["page"]})

    vectors = list(embedder.embed(docs))
    collection.add(ids=ids, documents=docs, embeddings=vectors, metadatas=metadatas)

    print(f"✅ Ingested {len(docs)} chunks into Chroma at {CHROMA_PATH}, collection={COLLECTION_NAME}")


if __name__ == "__main__":
    main()
