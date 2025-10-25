from __future__ import annotations

import os
from typing import List

import numpy as np

from modules.chunker import load_and_chunk_pdf
from modules.embeddings import get_embeddings, load_faiss_index, save_embeddings_to_faiss
from modules.llm import query_ollama
from modules.reranker import CrossEncoderReranker
from modules.retriever import (
    HybridRetriever,
    generate_multi_queries,
    merge_ranked_results,
    select_context_window,
)

BASE_DIR = os.path.dirname(__file__)
INDEX_BASE_PATH = os.path.join(BASE_DIR, "data", "faiss_index", "index")
PROMPT_PATH = os.path.join(BASE_DIR, "prompts", "system_prompts.txt")

RERANKER_WEIGHT = 0.6
RERANKER_CANDIDATES = 50
RETRIEVAL_TOP_K = 20
MULTI_QUERY_VARIANTS = 4

try:
    RERANKER = CrossEncoderReranker()
except Exception as exc:  # noqa: BLE001
    print(f"Warnung: Reranker konnte nicht geladen werden: {exc}")
    RERANKER = None


def _load_system_prompt() -> str:
    try:
        with open(PROMPT_PATH, "r", encoding="utf-8") as handle:
            return handle.read()
    except FileNotFoundError:
        return (
            "Du bist ein Assistenzsystem fuer den bankinternen IT-Support. Nutze NUR den folgenden Kontext, "
            "um die Frage zu beantworten. Wenn die Information nicht enthalten ist, sage, dass sie im Material "
            "nicht belegt ist. Jeder Absatz endet mit (Dokument, Seite X).\n\n"
            "KONTEXT:\n{{context}}\n\nFRAGE: {{question}}"
        )


def build_faiss_index() -> None:
    pdf_dir = os.path.join(BASE_DIR, "data")
    if not os.path.exists(pdf_dir):
        print("PDF-Verzeichnis existiert nicht:", pdf_dir)
        return

    pdf_files = [name for name in os.listdir(pdf_dir) if name.lower().endswith(".pdf")]
    print("Gefundene PDFs:", pdf_files)
    if not pdf_files:
        print("Keine PDFs zum Verarbeiten gefunden!")
        return

    chunk_records: List[dict] = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"Verarbeite PDF: {pdf_path}")
        chunks = load_and_chunk_pdf(pdf_path)

        for chunk in chunks:
            record = dict(chunk)
            record["source"] = pdf_path
            chunk_records.append(record)

    if not chunk_records:
        print("Keine Chunks erstellt.")
        return

    for idx, record in enumerate(chunk_records):
        record["chunk_index"] = idx

    texts = [record.get("content", "") for record in chunk_records]
    embeddings = get_embeddings(texts)
    save_embeddings_to_faiss(embeddings, chunk_records, INDEX_BASE_PATH)
    print("FAISS-Index gebaut und gespeichert.")


def ask_question(question: str) -> str:
    index, chunk_records, embeddings = load_faiss_index(INDEX_BASE_PATH)
    if not chunk_records:
        raise RuntimeError("FAISS-Index konnte nicht geladen werden oder ist leer.")

    retriever = HybridRetriever(chunk_records, embeddings=embeddings, faiss_index=index)

    query_variants = generate_multi_queries(question, total_variants=MULTI_QUERY_VARIANTS)
    print("Suchanfragen: " + " | ".join(query_variants))

    query_embeddings = get_embeddings(query_variants)
    result_batches = []
    for variant, embedding in zip(query_variants, query_embeddings):
        kwargs = {
            "question": variant,
            "query_embedding": np.array(embedding, dtype=np.float32),
            "top_k": RETRIEVAL_TOP_K,
        }
        if RERANKER is not None:
            kwargs.update(
                {
                    "reranker": RERANKER,
                    "reranker_k": RERANKER_CANDIDATES,
                    "reranker_weight": RERANKER_WEIGHT,
                }
            )
        result_batches.append(retriever.retrieve(**kwargs))

    ranked_chunks = merge_ranked_results(result_batches, RETRIEVAL_TOP_K)
    if not ranked_chunks:
        raise RuntimeError("Keine relevanten Chunks gefunden.")

    context_chunks = select_context_window(ranked_chunks, max_chunks=6, min_chunks=4, max_chars=1800)
    context = "\n\n".join(chunk.get("content", "") for chunk in context_chunks if chunk.get("content"))

    system_prompt = _load_system_prompt()
    final_prompt = system_prompt.replace("{{context}}", context).replace("{{question}}", question)

    answer = query_ollama(final_prompt)

    print("\nTop-Quellen:")
    for chunk in context_chunks:
        source_path = str(chunk.get("source") or "")
        doc_name = os.path.basename(source_path) if source_path else "unbekannt"
        page = chunk.get("page")
        page_display = (page + 1) if isinstance(page, int) else page
        matched = ", ".join(chunk.get("matched_terms", [])) or "-"
        print(
            f"- {doc_name} | Seite {page_display} | Score {chunk.get('score', 0.0):.3f} | Matches: {matched}"
        )

    print("\nAntwort:\n", answer)
    return answer


if __name__ == "__main__":
    if not os.path.exists(INDEX_BASE_PATH + ".index"):
        build_faiss_index()
    else:
        print("Index existiert bereits - wird nicht neu gebaut.")
