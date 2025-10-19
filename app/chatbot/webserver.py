from __future__ import annotations

import os
import re
import uuid
from typing import Dict, List, Optional, Sequence, Set, Tuple

from fastapi import Body, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from modules.embeddings import get_embeddings, load_faiss_index
from modules.llm import query_ollama
from modules.retriever import HybridRetriever, rewrite_query_with_llama3, select_context_window
from modules.reranker import CrossEncoderReranker

BASE_DIR = os.path.dirname(__file__)
INDEX_BASE_PATH = os.path.join(BASE_DIR, "data", "faiss_index", "index")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

app = FastAPI()
jobs: Dict[str, Dict[str, object]] = {}

print("Lade FAISS-Index...")
index, chunk_records, embeddings = load_faiss_index(INDEX_BASE_PATH)

if not chunk_records:
    raise RuntimeError("Der FAISS-Index ist leer oder fehlerhaft. Bitte neu erstellen.")

retriever = HybridRetriever(chunk_records, embeddings=embeddings, faiss_index=index)
try:
    reranker = CrossEncoderReranker()
except Exception as exc:  # noqa: BLE001
    print(f"Warnung: Reranker konnte nicht geladen werden: {exc}")
    reranker = None

RERANKER_WEIGHT = 0.65
RERANKER_CANDIDATES = 40


PROMPT_TEMPLATE = "Nutze den folgenden Kontext, um die Frage zu beantworten:\n\n{context}\n\nFrage: {question}"


TOKEN_PATTERN = re.compile(r"[a-z0-9]{3,}", re.IGNORECASE)
_TEXT_NORMALIZE = str.maketrans({
    '\u00e4': 'ae',
    '\u00f6': 'oe',
    '\u00fc': 'ue',
    '\u00df': 'ss',
    '\u00c4': 'ae',
    '\u00d6': 'oe',
    '\u00dc': 'ue',
    '\u00e1': 'a',
    '\u00e0': 'a',
    '\u00e2': 'a',
    '\u00e9': 'e',
    '\u00e8': 'e',
    '\u00ea': 'e',
    '\u00f3': 'o',
    '\u00f2': 'o',
    '\u00f4': 'o',
    '\u00fa': 'u',
    '\u00f9': 'u',
    '\u00fb': 'u',
})



_COMMON_STOPWORDS = {
    "ich",
    "wie",
    "ein",
    "eine",
    "einen",
    "einem",
    "einer",
    "der",
    "die",
    "das",
    "den",
    "und",
    "oder",
    "zu",
    "in",
    "auf",
    "mit",
    "fÃ¼r",
    "von",
    "was",
    "ist",
    "sind",
    "soll",
    "sollte",
    "man",
    "wir",
    "sie",
    "ihr",
}

SECOND_SOURCE_MIN_COVERAGE = 0.675


def build_prompt(question: str, context: str) -> str:
    return PROMPT_TEMPLATE.format(context=context, question=question)


def _sanitize_chunks(chunks: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    numeric_fields = {"score", "bm25_score", "vector_score", "l2_distance", "keyword_coverage", "section_score"}
    sanitized: List[Dict[str, object]] = []
    for chunk in chunks:
        cleaned = dict(chunk)
        for field in numeric_fields:
            if field in cleaned and cleaned[field] is not None:
                try:
                    cleaned[field] = float(cleaned[field])
                except (TypeError, ValueError):
                    pass
        if "matched_terms" in cleaned and cleaned["matched_terms"] is not None:
            cleaned["matched_terms"] = [str(term) for term in cleaned["matched_terms"]]
        sanitized.append(cleaned)
    return sanitized


def _tokenize_text(text: str) -> Set[str]:
    if not text:
        return set()
    normalized = text.lower().translate(_TEXT_NORMALIZE)
    tokens = {token for token in TOKEN_PATTERN.findall(normalized) if len(token) > 2}
    tokens.difference_update(_COMMON_STOPWORDS)
    return tokens

def _build_source_entry(chunk: Dict[str, object], max_snippet_len: int = 200) -> Dict[str, object]:
    source_path = str(chunk.get("source") or "")
    doc_name = os.path.basename(source_path) if source_path else "Unbekannt"
    page = chunk.get("page")
    page_display = (page + 1) if isinstance(page, int) else page
    section = chunk.get("normalized_section_heading") or chunk.get("section_heading")
    content = str(chunk.get("content") or "").strip()
    snippet = re.sub(r"\s+", " ", content) if content else ""
    if max_snippet_len and len(snippet) > max_snippet_len:
        snippet = snippet[:max_snippet_len].rstrip() + "..."
    entry: Dict[str, object] = {
        "source": doc_name,
        "page": page_display,
        "section_heading": section,
        "snippet": snippet,
        "score": float(chunk.get("score", 0.0) or 0.0),
        "matched_terms": [str(term) for term in (chunk.get("matched_terms") or [])],
        "path": None,
    }
    if source_path:
        try:
            entry["path"] = os.path.relpath(source_path, BASE_DIR)
        except ValueError:
            entry["path"] = source_path
    return entry


def _select_secondary_source(primary: Dict[str, object], context_chunks: Sequence[Dict[str, object]]) -> Optional[Dict[str, object]]:
    primary_key = (primary.get("source"), primary.get("page"))
    for chunk in context_chunks:
        if chunk is primary:
            continue
        if (chunk.get("source"), chunk.get("page")) == primary_key:
            continue
        coverage = float(chunk.get("keyword_coverage") or 0.0)
        if coverage >= SECOND_SOURCE_MIN_COVERAGE:
            return chunk
    return None


def _select_source_chunk(answer: str | None, chunks: Sequence[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not chunks:
        return None
    if not answer:
        return chunks[0]
    answer_tokens = _tokenize_text(answer)
    if not answer_tokens:
        return chunks[0]
    best_chunk = chunks[0]
    best_overlap = -1
    best_combined = -1.0
    for chunk in chunks:
        chunk_tokens = _tokenize_text(str(chunk.get("content") or ""))
        overlap = len(answer_tokens & chunk_tokens)
        chunk_score = float(chunk.get("score", 0.0) or 0.0)
        combined = overlap + 0.01 * chunk_score
        if overlap > best_overlap or (overlap == best_overlap and combined > best_combined):
            best_overlap = overlap
            best_combined = combined
            best_chunk = chunk
    return best_chunk

def _sources_from_answer(answer: str | None, context_chunks: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    selected = _select_source_chunk(answer, context_chunks)
    if not selected:
        return []
    sources = [_build_source_entry(selected)]
    secondary = _select_secondary_source(selected, context_chunks)
    if secondary is not None:
        sources.append(_build_source_entry(secondary))
    return sources


def summarize_sources(chunks: Sequence[Dict[str, object]], max_snippet_len: int = 200) -> List[Dict[str, object]]:
    if not chunks:
        return []

    primary = chunks[0]
    entries = [_build_source_entry(primary, max_snippet_len=max_snippet_len)]
    secondary = _select_secondary_source(primary, chunks)
    if secondary is not None:
        entries.append(_build_source_entry(secondary, max_snippet_len=max_snippet_len))
    return entries

def _filter_context_chunks(
    chunks: Sequence[Dict[str, object]],
    *,
    max_context_chunks: int = 4,
    max_documents: Optional[int] = None,
    min_cross_doc_coverage: float = 0.3,
) -> List[Dict[str, object]]:
    if not chunks:
        return []

    def _coverage(chunk: Dict[str, object]) -> float:
        return float(chunk.get("keyword_coverage") or 0.0)

    def _score(chunk: Dict[str, object]) -> float:
        return float(chunk.get("score") or 0.0)

    def _signal_count(chunk: Dict[str, object]) -> int:
        matches = chunk.get("matched_terms") or []
        return sum(1 for term in matches if len(term) >= 4 and term not in _COMMON_STOPWORDS)

    def _rank_sources() -> List[Tuple[object, Dict[str, float]]]:
        source_stats: Dict[object, Dict[str, float]] = {}
        for chunk in chunks:
            source = chunk.get("source")
            stats = source_stats.setdefault(source, {"signal": 0.0, "coverage": 0.0, "score": 0.0})
            stats["signal"] = max(stats["signal"], float(_signal_count(chunk)))
            stats["coverage"] = max(stats["coverage"], _coverage(chunk))
            stats["score"] = max(stats["score"], _score(chunk))
        return sorted(
            source_stats.items(),
            key=lambda item: (
                item[1]["signal"] > 0,
                item[1]["signal"],
                item[1]["coverage"],
                item[1]["score"],
            ),
            reverse=True,
        )

    ranked_sources = _rank_sources()
    if ranked_sources:
        top_coverage = ranked_sources[0][1]["coverage"]
        cross_doc_threshold = max(0.15, min(min_cross_doc_coverage, top_coverage * 0.6 if top_coverage else min_cross_doc_coverage))
    else:
        cross_doc_threshold = min_cross_doc_coverage

    if max_documents is None:
        inferred = 0
        for idx, (_source, stats) in enumerate(ranked_sources):
            if idx == 0:
                inferred = 1
                continue
            if stats["signal"] <= 0:
                break
            if stats["coverage"] < cross_doc_threshold:
                break
            if inferred >= max_context_chunks:
                break
            inferred += 1
        max_documents = max(1, inferred)
    else:
        cross_doc_threshold = min_cross_doc_coverage
        max_documents = max(1, min(max_context_chunks, max_documents))

    primary = max(chunks, key=lambda chunk: (_signal_count(chunk), _coverage(chunk), _score(chunk)))
    if _signal_count(primary) == 0 and len(chunks) > 1:
        primary = chunks[0]

    ordered: List[Dict[str, object]] = [primary]
    ordered.extend(chunk for chunk in chunks if chunk is not primary and _signal_count(chunk) > 0)
    ordered.extend(chunk for chunk in chunks if chunk is not primary and _signal_count(chunk) == 0)

    selected: List[Dict[str, object]] = []
    seen: Set[Tuple[object, object]] = set()

    primary_key = (primary.get("source"), primary.get("page"))
    selected.append(primary)
    seen.add(primary_key)
    primary_source = primary.get("source")
    allowed_sources: Set[object] = {primary_source}

    for chunk in ordered[1:]:
        chunk_key = (chunk.get("source"), chunk.get("page"))
        if chunk_key in seen:
            continue

        source = chunk.get("source")
        coverage = _coverage(chunk)
        signals = _signal_count(chunk)

        if source != primary_source:
            if len(allowed_sources) >= max_documents:
                continue
            if signals == 0:
                continue
            if coverage < cross_doc_threshold:
                continue
            allowed_sources.add(source)

        selected.append(chunk)
        seen.add(chunk_key)

        if len(selected) >= max_context_chunks:
            break

    if len(selected) < max_context_chunks:
        for chunk in ordered[1:]:
            if len(selected) >= max_context_chunks:
                break
            chunk_key = (chunk.get("source"), chunk.get("page"))
            if chunk_key in seen:
                continue
            if chunk.get("source") != primary_source:
                continue
            selected.append(chunk)
            seen.add(chunk_key)

    if not selected:
        return list(chunks[:max_context_chunks])
    return selected

def prepare_context(question: str, top_k: int = 15):
    optimized_query = rewrite_query_with_llama3(question)
    query_embedding = get_embeddings([optimized_query])[0]
    retrieve_kwargs: Dict[str, object] = {}
    if reranker is not None:
        retrieve_kwargs.update(
            {
                "reranker": reranker,
                "reranker_k": max(top_k * 2, RERANKER_CANDIDATES),
                "reranker_weight": RERANKER_WEIGHT,
            }
        )
    ranked_chunks = retriever.retrieve(
        question,
        query_embedding,
        top_k=top_k,
        **retrieve_kwargs,
    )

    context_chunks_raw = select_context_window(ranked_chunks, max_chunks=6, min_chunks=4, max_chars=1800)

    sanitized_ranked = _sanitize_chunks(ranked_chunks)
    sanitized_context = _sanitize_chunks(context_chunks_raw)
    context_chunks = _filter_context_chunks(sanitized_context)

    context_text = "\n\n".join(chunk.get("content", "") for chunk in context_chunks if chunk.get("content"))
    sources = summarize_sources(context_chunks)
    return context_text, sources, sanitized_ranked, context_chunks, optimized_query


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


async def process_question(job_id: str, question: str):
    try:
        context, _, ranked_chunks, context_chunks, optimized_query = prepare_context(question)
        final_prompt = build_prompt(question, context)
        answer = query_ollama(final_prompt)
        sources = _sources_from_answer(answer, context_chunks)
        jobs[job_id] = {
            "status": "completed",
            "answer": answer,
            "sources": sources,
            "context": context,
            "chunks": context_chunks,
            "ranked_chunks": ranked_chunks,
            "optimized_query": optimized_query,
        }
    except Exception as exc:  # noqa: BLE001
        jobs[job_id] = {
            "status": "error",
            "answer": None,
            "sources": [],
            "context": None,
            "chunks": [],
            "ranked_chunks": [],
            "error": str(exc),
        }


@app.post("/ask")
async def ask_question_json(payload: Dict[str, object] = Body(...)):
    question = str(payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Frage fehlt")

    try:
        context, _, ranked_chunks, context_chunks, optimized_query = prepare_context(question)
        final_prompt = build_prompt(question, context)
        answer = query_ollama(final_prompt)
        sources = _sources_from_answer(answer, context_chunks)
        return JSONResponse(
            {
                "answer": answer,
                "sources": sources,
                "context": context,
                "chunks": context_chunks,
                "ranked_chunks": ranked_chunks,
                "optimized_query": optimized_query,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"answer": f"Fehler: {exc}"}, status_code=500)


@app.post("/ask-html", response_class=HTMLResponse)
async def ask_question_html(request: Request, question: str = Form(...)):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "processing",
        "answer": None,
        "sources": [],
        "context": None,
        "chunks": [],
        "ranked_chunks": [],
    }

    await process_question(job_id, question)

    job = jobs[job_id]
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "question": question,
            "answer": job.get("answer"),
            "sources": job.get("sources", []),
        },
    )


@app.get("/result/{job_id}", response_class=HTMLResponse)
async def get_result(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job-ID nicht gefunden")

    if job["status"] == "processing":
        return """
        <html>
            <head>
                <title>Antwort wird verarbeitet</title>
                <meta http-equiv="refresh" content="2" />
            </head>
            <body>
                <h1>Antwort wird noch erstellt...</h1>
                <p>Bitte warten...</p>
            </body>
        </html>
        """
    elif job["status"] == "completed":
        answer_html = (job.get("answer", "") or "").replace("\n", "<br>")
        sources_rows = "".join(
            f"<tr><td>{src.get('source', '-')}</td>"
            f"<td>{src.get('page', '-')}</td>"
            f"<td>{src.get('section_heading') or '-'}</td>"
            f"<td>{', '.join(src.get('matched_terms', [])) or '-'}"  # type: ignore[str-bytes-safe]
            f"</td><td>{src.get('score', 0):.3f}</td>"
            f"<td>{src.get('snippet', '-')}</td></tr>"
            for src in job.get("sources", [])
        )
        if sources_rows:
            sources_table = f"""
                <table border=\"1\" cellpadding=\"6\" cellspacing=\"0\">
                    <thead>
                        <tr>
                            <th>Dokument</th>
                            <th>Seite</th>
                            <th>Abschnitt</th>
                            <th>Matches</th>
                            <th>Score</th>
                            <th>Ausschnitt</th>
                        </tr>
                    </thead>
                    <tbody>
                        {sources_rows}
                    </tbody>
                </table>
            """
        else:
            sources_table = "<p>Keine Quellen gefunden.</p>"

        return f"""
        <html>
            <head>
                <title>Antwort</title>
            </head>
            <body>
                <h1>Antwort</h1>
                <p><strong>Antwort:</strong> {answer_html}</p>
                <h2>Quellen</h2>
                {sources_table}
                <a href=\"/\">Neue Frage stellen</a>
            </body>
        </html>
        """
    else:
        error_html = job.get("error", "Unbekannter Fehler")
        return f"""
        <html>
            <head>
                <title>Fehler</title>
            </head>
            <body>
                <h1>Fehler bei der Bearbeitung</h1>
                <p>{error_html}</p>
                <a href=\"/\">Neue Frage stellen</a>
            </body>
        </html>
        """


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)







