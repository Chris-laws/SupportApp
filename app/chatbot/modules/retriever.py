from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple

import numpy as np
import requests
import yaml

if TYPE_CHECKING:
    from .reranker import CrossEncoderReranker

from .embeddings import get_embeddings

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]{2,}")
_SECTION_PATTERN = re.compile(r"\b\d+(?:\.\d+)+\b")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_DEFAULT_SYNONYMS: Mapping[str, Set[str]] = {
    "kennwort": {"passwort", "password", "pwd"},
    "passwort": {"kennwort", "password", "pwd"},
    "password": {"kennwort", "passwort"},
    "aendern": {"wechseln", "reset", "zuruecksetzen"},
    "zuruecksetzen": {"reset", "zurucksetzen"},
    "reset": {"zuruecksetzen", "zurucksetzen"},
    "wechseln": {"aendern"},
    "anmelden": {"login", "einloggen"},
    "login": {"anmelden", "einloggen"},
    "rufnummer": {"telefonnummer", "nummer", "callerid", "anruferid"},
    "rufnummernuebermittlung": {"nummernuebermittlung", "callerid", "anruferid"},
    "unterdruecken": {"verbergen", "nicht senden", "clir"},
    "clir": {"rufnummernunterdrueckung", "nummernunterdrueckung", "rufnummernuebermittlung deaktivieren"},
}

_DEFAULT_PHRASES: Mapping[str, Set[str]] = {
    "kennwort aendern": {"passwort aendern", "kennwort reset", "passwort reset", "kennwort zuruecksetzen"},
    "passwort zuruecksetzen": {"kennwort zuruecksetzen", "kennwort reset", "password reset", "passwort reset"},
    "passwort aendern": {"kennwort aendern", "passwort reset"},
    "rufnummer unterdruecken": {"rufnummernunterdrueckung", "clir aktivieren", "rufnummernuebermittlung deaktivieren"},
    "rufnummernunterdrueckung": {"clir", "rufnummer unterdruecken", "nummernuebermittlung deaktivieren"},
}


def _load_synonym_tables() -> Tuple[Mapping[str, Set[str]], Mapping[str, Set[str]]]:
    synonyms: Dict[str, Set[str]] = {key: set(values) for key, values in _DEFAULT_SYNONYMS.items()}
    phrases: Dict[str, Set[str]] = {key: set(values) for key, values in _DEFAULT_PHRASES.items()}

    config_path = Path(__file__).resolve().parents[1] / "Config" / "synonyms.yml"
    if not config_path.exists():
        return synonyms, phrases

    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Konnte Synonym-Konfiguration nicht laden (%s): %s", config_path, exc)
        return synonyms, phrases

    for key, values in (data.get("synonyms") or {}).items():
        normalized_key = str(key).lower()
        synonyms[normalized_key] = {str(value).lower() for value in values}
    for key, values in (data.get("phrases") or {}).items():
        normalized_key = str(key).lower()
        phrases[normalized_key] = {str(value).lower() for value in values}
    return synonyms, phrases


_DOMAIN_SYNONYMS, _DOMAIN_PHRASES = _load_synonym_tables()

_CHAR_MAP = str.maketrans({
    "\u00e4": "ae",
    "\u00f6": "oe",
    "\u00fc": "ue",
    "\u00c4": "ae",
    "\u00d6": "oe",
    "\u00dc": "ue",
    "\u00df": "ss",
})


@dataclass
class KeywordSet:
    bm25_terms: Set[str]
    expanded_terms: Dict[str, Set[str]]
    section_numbers: Set[str]
    raw_question: str

    @property
    def canonical_term_count(self) -> int:
        return len(self.expanded_terms)


class KeywordGenerator:
    def __init__(self) -> None:
        self._token_pattern = _TOKEN_PATTERN

    def extract(self, question: str) -> KeywordSet:
        normalized_question = question.lower().translate(_CHAR_MAP)
        tokens = self._tokenize(normalized_question)
        base_terms = set(tokens)

        expanded_terms: Dict[str, Set[str]] = {}
        for term in base_terms:
            variants = {term}
            variants.update(_DOMAIN_SYNONYMS.get(term, set()))
            expanded_terms[term] = {self._normalize_phrase(v) for v in variants}

        phrases = self._extract_phrases(tokens)
        for phrase in phrases:
            variants = {phrase}
            variants.update(_DOMAIN_PHRASES.get(phrase, set()))
            expanded_terms[phrase] = {self._normalize_phrase(v) for v in variants}

        bm25_terms = set()
        for variants in expanded_terms.values():
            for variant in variants:
                bm25_terms.update(self._tokenize(variant))

        section_numbers = set(_SECTION_PATTERN.findall(question))

        return KeywordSet(
            bm25_terms=bm25_terms or base_terms,
            expanded_terms=expanded_terms,
            section_numbers=section_numbers,
            raw_question=question,
        )

    def _tokenize(self, text: str) -> List[str]:
        return [match.group() for match in self._token_pattern.finditer(text)]

    def _extract_phrases(self, tokens: Sequence[str]) -> Set[str]:
        phrases: Set[str] = set()
        for window in (2, 3):
            for idx in range(len(tokens) - window + 1):
                phrases.add(" ".join(tokens[idx : idx + window]))
        return phrases

    def _normalize_phrase(self, phrase: str) -> str:
        return " ".join(self._tokenize(phrase.lower().translate(_CHAR_MAP)))


class BM25FieldIndex:
    def __init__(self, chunks: Sequence[Mapping[str, object]], field_weights: Mapping[str, float] | None = None) -> None:
        self.field_weights = dict(field_weights or {"content": 1.0, "heading": 1.8, "title": 1.2})
        self.doc_term_freqs: List[Dict[str, float]] = []
        self.doc_lengths: List[float] = []
        self.doc_freqs: MutableMapping[str, int] = Counter()
        self.avg_doc_len: float = 0.0
        self.num_docs = 0
        self.k1 = 1.5
        self.b = 0.75

        self._build(chunks)

    def _build(self, chunks: Sequence[Mapping[str, object]]) -> None:
        total_length = 0.0
        for chunk in chunks:
            term_freqs: Dict[str, float] = {}
            doc_length = 0.0

            for field, weight in self.field_weights.items():
                text_value = self._get_field_text(chunk, field)
                if not text_value:
                    continue
                tokens = self._tokenize(text_value)
                if not tokens:
                    continue
                counts = Counter(tokens)
                doc_length += weight * sum(counts.values())
                for term, freq in counts.items():
                    term_freqs[term] = term_freqs.get(term, 0.0) + weight * freq

            if not term_freqs:
                self.doc_term_freqs.append({})
                self.doc_lengths.append(1.0)
                continue

            for term in term_freqs.keys():
                self.doc_freqs[term] += 1

            self.doc_term_freqs.append(term_freqs)
            self.doc_lengths.append(doc_length if doc_length > 0 else 1.0)
            total_length += doc_length if doc_length > 0 else 1.0

        self.num_docs = len(self.doc_term_freqs)
        self.avg_doc_len = (total_length / self.num_docs) if self.num_docs else 0.0

    def search(self, query_terms: Iterable[str], top_k: int = 50) -> List[Tuple[int, float]]:
        scores: Dict[int, float] = {}
        for term in query_terms:
            doc_freq = self.doc_freqs.get(term)
            if not doc_freq:
                continue
            idf = self._idf(doc_freq)
            for doc_id, term_freqs in enumerate(self.doc_term_freqs):
                term_frequency = term_freqs.get(term)
                if not term_frequency:
                    continue
                score = self._score_term(term_frequency, doc_id, idf)
                scores[doc_id] = scores.get(doc_id, 0.0) + score

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    def _idf(self, doc_freq: int) -> float:
        numerator = (self.num_docs - doc_freq + 0.5)
        denominator = doc_freq + 0.5
        return math.log((numerator / denominator) + 1)

    def _score_term(self, term_frequency: float, doc_id: int, idf: float) -> float:
        doc_length = self.doc_lengths[doc_id]
        if self.avg_doc_len == 0:
            norm = term_frequency
        else:
            norm = term_frequency + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_len)
        return idf * (term_frequency * (self.k1 + 1)) / (norm if norm else 1)

    def _tokenize(self, text: str) -> List[str]:
        normalized = text.lower().translate(_CHAR_MAP)
        return [match.group() for match in _TOKEN_PATTERN.finditer(normalized)]

    def _get_field_text(self, chunk: Mapping[str, object], field: str) -> str:
        if field == "content":
            return str(chunk.get("content") or "")
        if field == "heading":
            return str(chunk.get("normalized_section_heading") or chunk.get("section_heading") or "")
        if field == "title":
            return str(chunk.get("document_title") or "")
        return ""


def rewrite_query_with_llama3(question: str) -> str:
    prompt = (
        "Du agierst als Recherche-Assistent fuer den bankinternen IT-Support. "
        "Formuliere die folgende Frage als praezise Suchanfrage mit Kernbegriffen, Synonymen "
        "und Produktbezeichnungen (Deutsch plus sinnvolle englische Varianten). "
        "Entferne Fueller, behalte Parameter wie Versionen oder Systeme und gib nur eine einzige Zeile aus.\n\n"
        "- Verwende gaengige Synonyme wie Kennwort/Passwort/password, Softphone/agree21Voice, Outlook/E-Mail usw.\n"
        "- Nutze Stichwoerter getrennt durch Kommas oder Leerzeichen; keine Saetze, keine erlaeuternden Worte.\n"
        "- Gib ausschliesslich die Suchanfrage als eine Zeile ohne Erklaerung aus.\n\n"
        f"Originalfrage: {question}\n"
        "Suchanfrage:"
    )

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.2,
                "repeat_penalty": 1.1,
            },
        },
    )
    if response.status_code != 200:
        raise Exception(f"Fehler beim Zugriff auf Ollama: {response.text}")

    data = response.json()
    raw = (data.get("response") or "").strip()
    if not raw:
        return question

    if "Suchanfrage:" in raw:
        candidate = raw.split("Suchanfrage:", 1)[-1]
    else:
        candidate = raw

    if "```" in candidate:
        parts = candidate.split("```")
        blocks = [block.strip() for block in parts[1::2] if block.strip()]
        if blocks:
            candidate = blocks[0]

    lines = [line.strip("` ") for line in candidate.splitlines() if line.strip()]
    if lines:
        candidate = lines[-1]

    cleaned = " ".join(candidate.split())
    return cleaned.strip("` ")


class HybridRetriever:
    def __init__(
        self,
        chunk_records: Sequence[Mapping[str, object]],
        embeddings: np.ndarray | None = None,
        faiss_index: object | None = None,
    ) -> None:
        self.chunk_records = list(chunk_records)
        self.embeddings = embeddings
        self.faiss_index = faiss_index
        self.keyword_generator = KeywordGenerator()
        self.bm25_index = BM25FieldIndex(self.chunk_records)
        self.max_page = self._compute_max_page()

    def _compute_max_page(self) -> int:
        pages = [chunk.get("page") for chunk in self.chunk_records]
        numeric_pages = [p for p in pages if isinstance(p, int)]
        return max(numeric_pages) if numeric_pages else 0

    def retrieve(
        self,
        question: str,
        query_embedding: np.ndarray,
        top_k: int = 15,
        bm25_k: int = 50,
        reranker: "CrossEncoderReranker | None" = None,
        reranker_k: int | None = None,
        reranker_weight: float = 0.6,
    ) -> List[Dict[str, object]]:
        keywords = self.keyword_generator.extract(question)
        bm25_hits = self.bm25_index.search(keywords.bm25_terms, top_k=bm25_k)

        dense_hits: List[Tuple[int, float]] = []
        dense_global_k = max(bm25_k, 80)
        if self.faiss_index is not None and self.chunk_records:
            query = np.array([query_embedding], dtype=np.float32)
            distances, indices = self.faiss_index.search(query, dense_global_k)
            dense_hits = [
                (int(idx), float(score))
                for idx, score in zip(indices[0], distances[0])
                if idx >= 0
            ]
        elif not bm25_hits and self.chunk_records:
            # Fallback: dense Hits erzeugen, wenn kein BM25 und kein Index vorhanden
            texts = [str(chunk.get("content") or "") for chunk in self.chunk_records]
            dense_embeddings = get_embeddings(texts)
            sims = dense_embeddings @ query_embedding
            dense_hits = sorted([(idx, float(score)) for idx, score in enumerate(sims)], key=lambda item: item[1], reverse=True)[:dense_global_k]

        # Kandidatenpool vereinigen
        cand_scores: Dict[int, float] = {}
        for idx, score in bm25_hits:
            cand_scores[idx] = max(cand_scores.get(idx, score), score)
        for idx, score in dense_hits:
            cand_scores[idx] = max(cand_scores.get(idx, score), score)

        candidate_indices = list(cand_scores.keys())
        if not candidate_indices:
            candidate_indices = list(range(len(self.chunk_records)))
            bm25_hits = [(idx, 0.0) for idx in candidate_indices]
        else:
            max_pool = max(bm25_k, dense_global_k)
            candidate_indices = [
                idx for idx, _ in sorted(cand_scores.items(), key=lambda item: item[1], reverse=True)[:max_pool]
            ]
            bm25_hits = [(idx, cand_scores.get(idx, 0.0)) for idx in candidate_indices]

        vector_scores, l2_distances = self._vector_scores(query_embedding, candidate_indices)

        max_bm25 = max((score for _, score in bm25_hits), default=0.0)
        max_vec = max(vector_scores.values(), default=0.0)
        max_l2 = max(l2_distances.values(), default=0.0)

        results: List[Dict[str, object]] = []
        for idx, bm25_score in bm25_hits:
            chunk = dict(self.chunk_records[idx])
            vec_score = vector_scores.get(idx, 0.0)
            l2_distance = l2_distances.get(idx, 0.0)
            match_info = self._match_keywords(chunk, keywords)

            score = self._aggregate_score(
                bm25_score=bm25_score,
                vector_score=vec_score,
                l2_distance=l2_distance,
                keyword_coverage=match_info["coverage"],
                section_score=match_info["section_score"],
                page=chunk.get("page"),
                max_bm25=max_bm25,
                max_vec=max_vec,
                max_l2=max_l2,
            )

            chunk.update(
                {
                    "bm25_score": bm25_score,
                    "vector_score": vec_score,
                    "l2_distance": l2_distance,
                    "matched_terms": match_info["matches"],
                    "keyword_coverage": match_info["coverage"],
                    "section_hit": match_info["section_score"] > 0,
                    "section_score": match_info["section_score"],
                    "score": score,
                    "chunk_index": idx,
                }
            )
            results.append(chunk)

        results.sort(key=lambda item: item["score"], reverse=True)
        if reranker is not None and results:
            candidate_count = reranker_k if reranker_k is not None else min(len(results), max(top_k * 2, 30))
            results = self._apply_reranking(
                question,
                results,
                reranker=reranker,
                candidate_count=candidate_count,
                weight=reranker_weight,
            )
        else:
            for chunk in results:
                base = float(chunk.get("score", 0.0))
                chunk.setdefault("retriever_score", base)
                chunk.setdefault("combined_score", base)
        return results[:top_k]

    def _apply_reranking(
        self,
        question: str,
        results: Sequence[Mapping[str, object]],
        *,
        reranker: "CrossEncoderReranker",
        candidate_count: int,
        weight: float,
    ) -> List[Dict[str, object]]:
        if not results:
            return []
        candidate_count = min(len(results), max(1, candidate_count))
        weight = min(max(weight, 0.0), 1.0)
        head: List[Dict[str, object]] = []
        for chunk in results[:candidate_count]:
            enriched = dict(chunk)
            base_score = float(chunk.get("score", 0.0))
            enriched.setdefault("retriever_score", base_score)
            enriched.setdefault("combined_score", base_score)
            head.append(enriched)
        reranked = reranker.rerank(question, head, top_k=candidate_count)
        if not reranked:
            reranked = head
        base_max = max(
            (float(chunk.get("retriever_score", chunk.get("score", 0.0))) for chunk in reranked),
            default=0.0,
        )
        rerank_max = max((float(chunk.get("rerank_score", 0.0)) for chunk in reranked), default=0.0)
        for chunk in reranked:
            base_score = float(chunk.get("retriever_score", chunk.get("score", 0.0)))
            rerank_score = float(chunk.get("rerank_score", 0.0))
            base_norm = base_score / base_max if base_max else 0.0
            rerank_norm = rerank_score / rerank_max if rerank_max else 0.0
            combined = (1.0 - weight) * base_norm + weight * rerank_norm
            chunk["combined_score"] = combined
            chunk["score"] = combined
            chunk.setdefault("retriever_score", base_score)
        tail: List[Dict[str, object]] = []
        for chunk in results[candidate_count:]:
            rest_chunk = dict(chunk)
            rest_base = float(rest_chunk.get("score", 0.0))
            rest_chunk.setdefault("retriever_score", rest_base)
            rest_chunk.setdefault("combined_score", rest_base)
            tail.append(rest_chunk)
        merged = reranked + tail
        merged.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return merged

    def _vector_scores(
        self,
        query_embedding: np.ndarray,
        candidate_indices: Sequence[int],
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        if self.embeddings is not None and len(self.embeddings) >= max(candidate_indices, default=-1) + 1:
            candidate_matrix = self.embeddings[candidate_indices]
        else:
            texts = [str(self.chunk_records[idx].get("content") or "") for idx in candidate_indices]
            candidate_matrix = get_embeddings(texts)

        vector_scores: Dict[int, float] = {}
        l2_distances: Dict[int, float] = {}
        for idx, embedding in zip(candidate_indices, candidate_matrix):
            vector_scores[idx] = float(np.dot(query_embedding, embedding))
            l2_distances[idx] = float(np.linalg.norm(query_embedding - embedding))
        return vector_scores, l2_distances

    def _match_keywords(self, chunk: Mapping[str, object], keywords: KeywordSet) -> Dict[str, object]:
        content = str(chunk.get("content") or "").lower()
        content_squeezed = str(chunk.get("content_squeezed") or "")
        heading = str(chunk.get("normalized_section_heading") or chunk.get("section_heading") or "").lower()
        heading_squeezed = str(chunk.get("normalized_section_heading_squeezed") or "")
        title = str(chunk.get("document_title") or "").lower()

        matches: Set[str] = set()
        canonical_hits: Set[str] = set()
        for canonical, variants in keywords.expanded_terms.items():
            for variant in variants:
                if not variant:
                    continue
                variant_text = variant.lower()
                if self._contains_variant(variant_text, content, content_squeezed, heading, heading_squeezed, title):
                    matches.add(variant_text)
                    canonical_hits.add(canonical)
                    break

        section_score = self._section_similarity(heading, keywords.section_numbers)
        coverage = len(canonical_hits) / keywords.canonical_term_count if keywords.canonical_term_count else 0.0
        return {
            "matches": sorted(matches),
            "coverage": coverage,
            "section_score": section_score,
        }

    def _contains_variant(
        self,
        variant: str,
        content: str,
        content_squeezed: str,
        heading: str,
        heading_squeezed: str,
        title: str,
    ) -> bool:
        squeezed_variant = variant.replace(" ", "")
        if squeezed_variant and (squeezed_variant in content_squeezed or squeezed_variant in heading_squeezed):
            return True
        if variant in content or variant in heading or variant in title:
            return True
        return False

    def _section_similarity(self, heading: str, query_sections: Set[str]) -> float:
        if not heading or not query_sections:
            return 0.0
        heading_sections = set(_SECTION_PATTERN.findall(heading))
        if not heading_sections:
            return 0.0
        for query_section in query_sections:
            if query_section in heading_sections:
                return 1.0
            for section in heading_sections:
                if section.startswith(query_section) or query_section.startswith(section):
                    return 0.7
        return 0.0

    def _aggregate_score(
        self,
        bm25_score: float,
        vector_score: float,
        l2_distance: float,
        keyword_coverage: float,
        section_score: float,
        page: object,
        max_bm25: float,
        max_vec: float,
        max_l2: float,
    ) -> float:
        bm25_norm = bm25_score / max_bm25 if max_bm25 else 0.0
        vector_norm = (vector_score + 1) / 2 if max_vec else (vector_score + 1) / 2
        l2_norm = 1 - (l2_distance / max_l2) if max_l2 else 0.0
        page_norm = 0.0
        if isinstance(page, int) and self.max_page > 0:
            page_norm = 1 - (page / (self.max_page + 1))

        return (
            0.4 * bm25_norm
            + 0.35 * vector_norm
            + 0.15 * keyword_coverage
            + 0.05 * section_score
            + 0.03 * l2_norm
            + 0.02 * page_norm
        )


def _trim_chunk_content(
    chunk: Mapping[str, Any], max_sentences: int = 8
) -> str:
    content = str(chunk.get("content") or "").strip()
    if not content:
        return content

    normalized = (
        content.replace("•", ". ")
        .replace("▪", ". ")
        .replace("◦", ". ")
        .replace("●", ". ")
    )
    sentences = [sentence.strip() for sentence in _SENTENCE_SPLIT_RE.split(normalized) if sentence.strip()]
    if not sentences:
        return content

    matched_terms = [str(term).lower() for term in (chunk.get("matched_terms") or []) if term]
    if not matched_terms:
        return " ".join(sentences[:max_sentences]).strip()

    target_index = 0
    for idx, sentence in enumerate(sentences):
        lower_sentence = sentence.lower()
        if any(term in lower_sentence for term in matched_terms):
            target_index = idx
            break
    start = max(0, target_index - 1)
    excerpt = sentences[start : start + max_sentences]
    return " ".join(excerpt).strip()


def _keyword_present(chunk: Mapping[str, Any], expected_tokens: Set[str]) -> bool:
    if not expected_tokens:
        return False
    content = str(chunk.get("content") or "").lower()
    if not content:
        return False
    return any(token in content for token in expected_tokens)


def select_context_window(
    ranked_chunks: Sequence[Mapping[str, Any]],
    *,
    max_chunks: int = 6,
    min_chunks: int = 4,
    max_chars: int = 1800,
    expected_tokens: Set[str] | None = None,
) -> List[Mapping[str, Any]]:
    keyword_hits = [chunk for chunk in ranked_chunks if chunk.get("matched_terms")]
    others = [chunk for chunk in ranked_chunks if not chunk.get("matched_terms")]

    ordered: List[Mapping[str, Any]] = []
    seen: Set[Tuple[Any, Any]] = set()
    trimmed_cache: Dict[int, str] = {}
    for group in (keyword_hits, others):
        for chunk in group:
            key = (chunk.get("source"), chunk.get("page"))
            if key in seen:
                continue
            seen.add(key)
            ordered.append(chunk)

    if expected_tokens:
        expected_tokens_lower = {token.lower() for token in expected_tokens if token}
        prioritized = [chunk for chunk in ordered if _keyword_present(chunk, expected_tokens_lower)]
        remainder = [chunk for chunk in ordered if chunk not in prioritized]
        ordered = prioritized + remainder

    def _try_add(chunk: Mapping[str, Any], *, force: bool = False) -> bool:
        nonlocal char_count
        cache_key = id(chunk)
        if cache_key not in trimmed_cache:
            trimmed_cache[cache_key] = _trim_chunk_content(chunk)
        content = trimmed_cache[cache_key] or str(chunk.get("content") or "")
        if not content:
            return False
        if len(selected) >= max_chunks:
            return False
        proposed = char_count + len(content)
        if not force and selected and proposed > max_chars and len(selected) >= min_chunks:
            return False
        selected.append(chunk)
        char_count = proposed
        doc_selected.add(chunk.get("source"))
        return True

    selected: List[Mapping[str, Any]] = []
    char_count = 0
    doc_selected: Set[Any] = set()

    for chunk in ordered:
        source = chunk.get("source")
        if not chunk.get("matched_terms") or source in doc_selected:
            continue
        _try_add(chunk, force=True)
        if len(selected) >= max_chunks:
            break

    if len(selected) < min_chunks:
        for chunk in ordered:
            if chunk in selected:
                continue
            if _try_add(chunk):
                if len(selected) >= max_chunks:
                    break

    if not selected:
        selected = list(ordered[: max_chunks])

    trimmed: List[Mapping[str, Any]] = []
    for chunk in selected:
        cache_key = id(chunk)
        trimmed_text = trimmed_cache.get(cache_key)
        if trimmed_text is None:
            trimmed_text = _trim_chunk_content(chunk)
        trimmed_chunk = dict(chunk)
        trimmed_chunk["content_full"] = chunk.get("content")
        trimmed_chunk["content"] = trimmed_text
        trimmed.append(trimmed_chunk)
    return trimmed


__all__ = [
    "KeywordGenerator",
    "HybridRetriever",
    "rewrite_query_with_llama3",
    "select_context_window",
]
