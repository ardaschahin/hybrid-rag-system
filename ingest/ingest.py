import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from pypdf import PdfReader
import chromadb
from fastembed import TextEmbedding

import fitz  # PyMuPDF

from vlm_provider import caption_image  # local import


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_PATH = os.getenv("CHROMA_PATH", str(PROJECT_ROOT / "chroma_db"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "docs")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

ENABLE_VLM_CAPTIONS = os.getenv("ENABLE_VLM_CAPTIONS", "0").strip() == "1"
CAPTION_MAX_PAGES = int(os.getenv("CAPTION_MAX_PAGES", "0"))      # 0 = unlimited
CAPTION_MIN_IMAGE_COUNT = int(os.getenv("CAPTION_MIN_IMAGE_COUNT", "1"))
CAPTION_MIN_DRAWING_COUNT = int(os.getenv("CAPTION_MIN_DRAWING_COUNT", "1"))
CAPTION_DPI = int(os.getenv("CAPTION_DPI", "150"))

# Candidate detection toggles
CAPTION_USE_DRAWINGS = os.getenv("CAPTION_USE_DRAWINGS", "1").strip() == "1"
CAPTION_USE_TEXT_CUES = os.getenv("CAPTION_USE_TEXT_CUES", "1").strip() == "1"

# Optional: reset collection (recommended for clean demos)
RESET_COLLECTION = os.getenv("RESET_COLLECTION", "0").strip() == "1"

# Text cues: generic + bilingual-ish (not model-dependent, only our detector)
DEFAULT_CUE_REGEX = r"(diagram|figure|table|chart|flowchart|illustration|example below|in the diagram|see (the )?diagram|şekil|tablo|aşağıdaki|örnek|bkz)"
CAPTION_TEXT_CUE_REGEX = os.getenv("CAPTION_TEXT_CUE_REGEX", DEFAULT_CUE_REGEX)
_CUE_RE = re.compile(CAPTION_TEXT_CUE_REGEX, re.IGNORECASE)

# Optional: infer a rough "section" label from page text (helps class questions)
CLASS_IN_TEXT_RE = re.compile(r"\bClass\s+([A-Z])\b", re.IGNORECASE)


def extract_pages_text(pdf_path: Path) -> List[Dict[str, Any]]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = " ".join(text.split())
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


def page_image_count(page: fitz.Page) -> int:
    return len(page.get_images(full=True))


def page_drawing_count(page: fitz.Page) -> int:
    # Vektör çizimler için. Bazı PDF’lerde şekil tamamen "drawings" olabilir.
    try:
        dr = page.get_drawings()
        return len(dr) if dr else 0
    except Exception:
        return 0


def infer_section_from_text(page_text: str) -> Optional[str]:
    # Generic heuristic: if page mentions "Class X", tag as "Class X"
    if not page_text:
        return None
    head = page_text[:250]
    m = CLASS_IN_TEXT_RE.search(head)
    if m:
        return f"Class {m.group(1).upper()}"
    return None


def is_visual_candidate(page: fitz.Page, page_text: str) -> Tuple[bool, Dict[str, Any]]:
    img_n = page_image_count(page)
    drw_n = page_drawing_count(page) if CAPTION_USE_DRAWINGS else 0
    cue_hit = bool(_CUE_RE.search(page_text or "")) if CAPTION_USE_TEXT_CUES else False

    ok = (img_n >= CAPTION_MIN_IMAGE_COUNT) or (drw_n >= CAPTION_MIN_DRAWING_COUNT) or cue_hit
    meta = {"images": img_n, "drawings": drw_n, "cue_hit": cue_hit}
    return ok, meta


def render_page_png(page: fitz.Page, dpi: int = 150) -> bytes:
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


def _get_or_reset_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    if RESET_COLLECTION:
        try:
            client.delete_collection(name=COLLECTION_NAME)
            print(f"[CHROMA] Deleted collection={COLLECTION_NAME} (RESET_COLLECTION=1)", flush=True)
        except Exception:
            pass
    return client.get_or_create_collection(name=COLLECTION_NAME)


def main():
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {DATA_DIR}. Add a PDF and re-run.")

    print(f"[INGEST] CHROMA_PATH={CHROMA_PATH} COLLECTION={COLLECTION_NAME}")
    print(f"[INGEST] EMBED_MODEL={EMBED_MODEL}")
    print(f"[INGEST] RESET_COLLECTION={RESET_COLLECTION}")
    print(f"[INGEST] ENABLE_VLM_CAPTIONS={ENABLE_VLM_CAPTIONS}")
    if ENABLE_VLM_CAPTIONS:
        print(f"[INGEST] CAPTION_DPI={CAPTION_DPI}")
        print(f"[INGEST] CAPTION_MIN_IMAGE_COUNT={CAPTION_MIN_IMAGE_COUNT} CAPTION_MIN_DRAWING_COUNT={CAPTION_MIN_DRAWING_COUNT}")
        print(f"[INGEST] CAPTION_USE_DRAWINGS={CAPTION_USE_DRAWINGS} CAPTION_USE_TEXT_CUES={CAPTION_USE_TEXT_CUES}")
        print(f"[INGEST] CAPTION_TEXT_CUE_REGEX={CAPTION_TEXT_CUE_REGEX}")
        print(f"[INGEST] CAPTION_MAX_PAGES={CAPTION_MAX_PAGES}")
        print(f"[INGEST] VLM_MODEL={os.getenv('OLLAMA_VLM_MODEL','llava:13b')} VLM_MIN_CONFIDENCE={os.getenv('VLM_MIN_CONFIDENCE','0.60')}")
    print("", flush=True)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = _get_or_reset_collection(client)
    embedder = TextEmbedding(model_name=EMBED_MODEL)

    ids: List[str] = []
    docs: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for pdf_path in pdfs:
        print(f"[PDF] Processing: {pdf_path.name}", flush=True)

        pages = extract_pages_text(pdf_path)
        page_text_map = {p["page"]: (p.get("text") or "") for p in pages}

        # ---- TEXT CHUNKS ----
        text_added = 0
        for p in pages:
            page_no = p["page"]
            text = (p["text"] or "").strip()
            if not text:
                continue

            section = infer_section_from_text(text)

            for ci, chunk in enumerate(chunk_text(text)):
                doc_id = f"{pdf_path.stem}_p{page_no}_text_c{ci}"
                ids.append(doc_id)
                docs.append(chunk)
                md = {
                    "filename": pdf_path.name,
                    "page": page_no,
                    "kind": "text",
                }
                if section:
                    md["section"] = section
                metadatas.append(md)
                text_added += 1

        print(f"[PDF] Text chunks added: {text_added}", flush=True)

        # ---- CAPTIONS ----
        cap_added = 0
        if ENABLE_VLM_CAPTIONS:
            doc = fitz.open(str(pdf_path))
            total_pages = doc.page_count
            caption_pages_done = 0

            for page_index in range(total_pages):
                page_no = page_index + 1
                page = doc.load_page(page_index)
                page_text = page_text_map.get(page_no, "") or ""

                ok, cand_meta = is_visual_candidate(page, page_text)
                if not ok:
                    continue

                print(
                    f"[CAPTION] {pdf_path.name} page={page_no}/{total_pages} "
                    f"images={cand_meta['images']} drawings={cand_meta['drawings']} cue={cand_meta['cue_hit']} render...",
                    flush=True,
                )

                try:
                    png = render_page_png(page, dpi=CAPTION_DPI)
                    cap, dbg = caption_image(
                        png,
                        pdf_name=pdf_path.name,
                        page_no=page_no,
                        page_text_hint=page_text,
                    )
                except Exception as e:
                    print(f"[VLM] ERROR {pdf_path.name} page={page_no}: {e}", flush=True)
                    continue

                print(
                    f"[VLM] {pdf_path.name} page={page_no} "
                    f"conf={dbg.get('confidence',0):.2f} reason={dbg.get('reason')} latency={dbg.get('latency_s')}s",
                    flush=True,
                )

                if cap:
                    section = infer_section_from_text(page_text)
                    cap_id = f"{pdf_path.stem}_p{page_no}_caption_c0"
                    ids.append(cap_id)
                    docs.append("FIGURE/TABLE CAPTION:\n" + cap)

                    md = {
                        "filename": pdf_path.name,
                        "page": page_no,
                        "kind": "caption",
                        "vlm_model": dbg.get("vlm_model"),
                        "vlm_confidence": dbg.get("confidence"),
                        "cand_images": cand_meta["images"],
                        "cand_drawings": cand_meta["drawings"],
                        "cand_cue_hit": cand_meta["cue_hit"],
                    }
                    if section:
                        md["section"] = section
                    metadatas.append(md)

                    cap_added += 1
                    caption_pages_done += 1

                if CAPTION_MAX_PAGES and caption_pages_done >= CAPTION_MAX_PAGES:
                    print(f"[CAPTION] Reached CAPTION_MAX_PAGES={CAPTION_MAX_PAGES}.", flush=True)
                    break

            doc.close()

        print(f"[PDF] Caption chunks added: {cap_added}\n", flush=True)

    if not docs:
        raise SystemExit("No content extracted.")

    print(f"[EMBED] Embedding {len(docs)} chunks...", flush=True)
    vectors = list(embedder.embed(docs))

    print(f"[CHROMA] Writing {len(docs)} chunks to collection={COLLECTION_NAME} ...", flush=True)

    # If collection not reset, avoid spam by filtering ids already present
    if not RESET_COLLECTION:
        try:
            existing = set()
            # light check: only fetch ids we are about to insert
            # chroma get can accept ids list; if some are missing it returns those found
            got = collection.get(ids=ids, include=[])
            for _id in (got.get("ids") or []):
                if isinstance(_id, str):
                    existing.add(_id)

            if existing:
                keep_ids, keep_docs, keep_vecs, keep_metas = [], [], [], []
                for i, _id in enumerate(ids):
                    if _id in existing:
                        continue
                    keep_ids.append(_id)
                    keep_docs.append(docs[i])
                    keep_vecs.append(vectors[i])
                    keep_metas.append(metadatas[i])

                ids, docs, vectors, metadatas = keep_ids, keep_docs, keep_vecs, keep_metas
                print(f"[CHROMA] Skipping {len(existing)} already-existing ids.", flush=True)
        except Exception:
            pass

    if not docs:
        print("[CHROMA] Nothing new to add (all ids existed).", flush=True)
        return

    collection.add(ids=ids, documents=docs, embeddings=vectors, metadatas=metadatas)

    text_count = sum(1 for m in metadatas if m.get("kind") == "text")
    cap_count = sum(1 for m in metadatas if m.get("kind") == "caption")

    print(f"\n✅ Ingested {len(docs)} chunks into Chroma at {CHROMA_PATH}, collection={COLLECTION_NAME}")
    print(f"   - text chunks: {text_count}")
    print(f"   - caption chunks: {cap_count}")
    print(f"   - ENABLE_VLM_CAPTIONS={ENABLE_VLM_CAPTIONS}")
    print(f"   - RESET_COLLECTION={RESET_COLLECTION}")


if __name__ == "__main__":
    main()
