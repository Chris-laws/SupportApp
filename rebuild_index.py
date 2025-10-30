from __future__ import annotations

import argparse
import json
import unicodedata
from hashlib import sha1
from pathlib import Path
from typing import Dict, List

from app.chatbot.modules.chunker import load_and_chunk_pdf
from app.chatbot.modules.embeddings import get_embeddings, save_embeddings_to_faiss


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return " ".join(text.split())


def build_records(data_dir: Path, chunk_size: int, overlap: int) -> List[dict]:
    records: List[dict] = []
    chunk_idx = 0
    for pdf in sorted(data_dir.glob("*.pdf")):
        chunks = load_and_chunk_pdf(str(pdf), chunk_size_tokens=chunk_size, overlap_tokens=overlap)
        print(f"{pdf.name}: {len(chunks)} chunks")
        page_chunk_counts: Dict[tuple[str, int], int] = {}
        for chunk in chunks:
            doc = Path(chunk.get("source") or pdf).name
            page = int(chunk.get("page") or 0)
            page_key = (doc, page)
            order = page_chunk_counts.get(page_key, 0)
            page_chunk_counts[page_key] = order + 1

            text = chunk.get("content") or ""
            span_digest = sha1(normalize(text)[:600].encode("utf-8", "ignore")).hexdigest()

            chunk.update(
                {
                    "chunk_index": chunk_idx,
                    "doc": doc,
                    "stable_id": f"{doc}:{page}:{order}",
                    "span_hash": span_digest,
                    "chunk_idx_in_page": order,
                }
            )
            records.append(chunk)
            chunk_idx += 1
    print(f"Total chunks: {len(records)}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild FAISS index with custom chunk size")
    parser.add_argument("--data-dir", type=Path, default=Path("app/chatbot/data"))
    parser.add_argument("--output", type=Path, default=Path("app/chatbot/data/faiss_index/index"))
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--overlap", type=int, default=60)
    args = parser.parse_args()

    pdf_dir = args.data_dir
    if not pdf_dir.exists():
        raise FileNotFoundError(pdf_dir)

    records = build_records(pdf_dir, args.chunk_size, args.overlap)
    contents = [record["content"] for record in records]
    embeddings = get_embeddings(contents)

    save_embeddings_to_faiss(embeddings, records, str(args.output))
    meta_path = args.output.parent / "index_meta.json"
    meta_path.write_text(json.dumps({
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "chunks": len(records),
    }, indent=2), encoding="utf-8")
    print(f"Saved index to {args.output} (metadata -> {meta_path})")


if __name__ == "__main__":
    main()
