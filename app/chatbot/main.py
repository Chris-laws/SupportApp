from __future__ import annotations

import os
from typing import List


from modules.chunker import load_and_chunk_pdf
from modules.embeddings import get_embeddings, load_faiss_index, save_embeddings_to_faiss
from modules.llm import query_ollama
from modules.reranker import CrossEncoderReranker
from modules.retriever import HybridRetriever, rewrite_query_with_llama3, select_context_window

BASE_DIR = os.path.dirname(__file__)
INDEX_BASE_PATH = os.path.join(BASE_DIR, "data", "faiss_index", "index")
PROMPT_PATH = os.path.join(BASE_DIR, "prompts", "system_prompts.txt")

RERANKER_WEIGHT = 0.65
RERANKER_CANDIDATES = 40

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
        return "Nutze den folgenden Kontext, um die Frage zu beantworten:\n\n{{context}}\n\nFrage: {{question}}"


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
            record.setdefault("source", pdf_path)
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

    optimized_query = rewrite_query_with_llama3(question)
    print(f"Optimierte Suchanfrage: {optimized_query}")

    query_embedding = get_embeddings([optimized_query])[0]
    top_k = 15
    rerank_kwargs = {}
    if RERANKER is not None:
        rerank_kwargs.update(
            {
                "reranker": RERANKER,
                "reranker_k": max(top_k * 2, RERANKER_CANDIDATES),
                "reranker_weight": RERANKER_WEIGHT,
            }
        )
    ranked_chunks = retriever.retrieve(question, query_embedding, top_k=top_k, **rerank_kwargs)
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
