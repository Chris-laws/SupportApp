from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
import time
import unicodedata
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha1
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import numpy as np
import yaml

from app.chatbot.modules.embeddings import get_embeddings, load_faiss_index
from app.chatbot.modules.llm import query_ollama
from app.chatbot.modules.retriever import (
    HybridRetriever,
    generate_multi_queries,
    merge_ranked_results,
    select_context_window,
)

try:
    from app.chatbot.modules.reranker import CrossEncoderReranker
except Exception:  # noqa: BLE001
    CrossEncoderReranker = None  # type: ignore[assignment]

Pair = Tuple[str, int]
logger = logging.getLogger(__name__)


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


class KeywordNormalizer:
    def __init__(self, synonym_table: dict) -> None:
        self.variant_map: dict[str, str] = {}
        self.display_map: dict[str, str] = {}
        for base, variants in (synonym_table or {}).items():
            base_text = str(base)
            base_norm = normalize_text(base_text)
            if not base_norm:
                continue
            self.variant_map[base_norm] = base_norm
            self.display_map.setdefault(base_norm, base_text)
            for variant in variants or []:
                v_text = str(variant)
                v_norm = normalize_text(v_text)
                if v_norm:
                    self.variant_map[v_norm] = base_norm
                    self.display_map.setdefault(base_norm, base_text)

    def canonical_keyword(self, keyword: str) -> str:
        norm_kw = normalize_text(keyword)
        canonical = self.variant_map.get(norm_kw, norm_kw)
        if canonical not in self.variant_map:
            self.variant_map[canonical] = canonical
        if canonical not in self.display_map:
            self.display_map[canonical] = str(keyword)
        return canonical

    def canonical_tokens(self, text: str) -> set[str]:
        norm_text = normalize_text(text)
        tokens = set(norm_text.split())
        result: set[str] = set()
        for tok in tokens:
            result.add(self.variant_map.get(tok, tok))

        padded = f" {norm_text} "
        for variant, canonical in self.variant_map.items():
            if " " not in variant:
                continue
            needle = f" {variant} "
            if needle in padded:
                result.add(canonical)
        return result

    def display(self, canonical: str) -> str:
        return self.display_map.get(canonical, canonical)


def load_ground_truth(path: Path) -> List[dict]:
    entries: List[dict] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    entries.append(json.loads(line))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict) and "samples" in data:
            entries = data["samples"]
    return entries


def recall_at_k(gt_pairs: Sequence[Pair], ranked_pairs: Sequence[Pair], k: int) -> float:
    if not gt_pairs:
        return 0.0
    gt = set(gt_pairs)
    topk = set(ranked_pairs[:k])
    return len(gt & topk) / len(gt)


def mrr(gt_pairs: Sequence[Pair], ranked_pairs: Sequence[Pair]) -> float:
    gt = set(gt_pairs)
    for idx, pair in enumerate(ranked_pairs, start=1):
        if pair in gt:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(gt_pairs: Sequence[Pair], ranked_pairs: Sequence[Pair], k: int) -> float:
    if not gt_pairs:
        return 0.0
    gt = set(gt_pairs)
    dcg = 0.0
    for idx, pair in enumerate(ranked_pairs[:k], start=1):
        rel = 1.0 if pair in gt else 0.0
        if idx == 1:
            dcg += rel
        else:
            dcg += rel / math.log2(idx + 1)
    ideal_len = min(len(gt), k)
    idcg = 0.0
    for idx in range(1, ideal_len + 1):
        if idx == 1:
            idcg += 1.0
        else:
            idcg += 1.0 / math.log2(idx + 1)
    return dcg / idcg if idcg > 0 else 0.0


def rank_first_rel(gt_pairs: Sequence[Pair], ranked_pairs: Sequence[Pair]) -> int | None:
    gt = set(gt_pairs)
    for idx, pair in enumerate(ranked_pairs, start=1):
        if pair in gt:
            return idx
    return None


def factual_check(answer: str, contexts: Iterable[str], threshold: float = 0.7) -> bool:
    context_tokens = set()
    for ctx in contexts:
        context_tokens.update(normalize_text(ctx).split())
    answer_tokens = [tok for tok in normalize_text(answer).split() if len(tok) > 2]
    if not answer_tokens:
        return False
    covered = sum(1 for tok in answer_tokens if tok in context_tokens)
    return covered / len(answer_tokens) >= threshold


def tokens_from_text(text: str) -> int:
    return len(normalize_text(text).split())


def dedupe_pairs(pairs: Iterable[Pair]) -> List[Pair]:
    seen: set[Pair] = set()
    result: List[Pair] = []
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            result.append(pair)
    return result


def stable_id(doc: str, page: int, chunk_order: int) -> str:
    return f"{doc}:{page}:{chunk_order}"


def build_chunk_metadata() -> dict[int, dict]:
    index, records, embeddings = load_faiss_index("app/chatbot/data/faiss_index/index")
    doc_page_orders: dict[Tuple[str, int], int] = defaultdict(int)
    mapping: dict[int, dict] = {}
    for rec in records:
        chunk_idx = int(rec.get("chunk_index", -1))
        if chunk_idx < 0:
            continue
        doc = Path(rec.get("source") or "").name
        page = rec.get("page")
        if page is None:
            continue
        order = doc_page_orders[(doc, int(page))]
        doc_page_orders[(doc, int(page))] += 1
        text = (rec.get("content") or "")
        digest = sha1(normalize_text(text)[:600].encode("utf-8", "ignore")).hexdigest()
        mapping[chunk_idx] = {
            "doc": doc,
            "page": int(page),
            "order": order,
            "span_hash": digest,
        }
    return mapping


def load_models(args_models: Sequence[str]) -> List[str]:
    if not args_models:
        return ["mistral:instruct"]
    return list(args_models)


def load_synonyms_and_mandatory(syn_path: Path, mand_path: Path) -> tuple[dict, dict]:
    synonyms_data = load_yaml(syn_path).get("synonyms", {})
    mandatory_data = load_yaml(mand_path)
    return synonyms_data, mandatory_data or {}


def canonical_keyword_set(normalizer: KeywordNormalizer, keywords: Sequence[str]) -> set[str]:
    result: set[str] = set()
    for kw in keywords:
        canonical = normalizer.canonical_keyword(kw)
        if canonical:
            result.add(canonical)
    return result


_CITATION_RE = re.compile(r"\([^)]*?\bS\.\s*\d+\)", re.IGNORECASE)


def analyze_answer_format(
    answer: str,
    expected_set: set[str],
    normalizer: KeywordNormalizer,
) -> tuple[bool, bool, bool, int]:
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    bullet_lines = [line for line in lines if line.startswith("- ")]
    bullet_count = len(bullet_lines)
    if not bullet_lines:
        return False, False, "nicht im kontext" in answer.lower(), 0

    keyword_ok = True
    citation_ok = True
    for line in bullet_lines:
        body = line[2:].strip()
        prefix = body.split(":", 1)[0].strip()
        canonical = normalizer.canonical_keyword(prefix) if prefix else ""
        if expected_set:
            if canonical not in expected_set:
                keyword_ok = False
        else:
            if not canonical:
                keyword_ok = False

        if not _CITATION_RE.search(line):
            citation_ok = False

    contains_nic = "nicht im kontext" in answer.lower()
    return keyword_ok, citation_ok, contains_nic, bullet_count


def kendall_tau(pre_ranks: Sequence[int]) -> float:
    n = len(pre_ranks)
    if n < 2:
        return 0.0
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if pre_ranks[i] == pre_ranks[j]:
                continue
            if pre_ranks[i] < pre_ranks[j]:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 0.0
    return (concordant - discordant) / total


def evaluate_keywords(
    qid: str,
    expected_keywords: Sequence[str],
    answer: str,
    normalizer: KeywordNormalizer,
    mandatory_map: dict,
) -> tuple[float, list[str]]:
    expected_set = canonical_keyword_set(normalizer, expected_keywords)
    mandatory = canonical_keyword_set(normalizer, mandatory_map.get(qid, []))
    answer_tokens = normalizer.canonical_tokens(answer)
    if mandatory and not mandatory.issubset(answer_tokens):
        missing = [normalizer.display(item) for item in sorted(mandatory - answer_tokens)]
        return 0.0, missing
    if not expected_set:
        return 0.0, []
    present_expected = answer_tokens & expected_set
    true_pos = len(present_expected)
    total_expected = len(expected_set)
    recall = true_pos / total_expected if total_expected else 0.0
    precision = 1.0 if true_pos else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    missing = [normalizer.display(item) for item in sorted(expected_set - present_expected)]
    return float(f1), missing


def ensure_output(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_chunk_lookup(records: Sequence[Mapping[str, object]]) -> dict[int, Pair]:
    lookup: dict[int, Pair] = {}
    for record in records or []:
        chunk_idx = record.get("chunk_index")
        source = record.get("source") or ""
        page = record.get("page")
        if chunk_idx is None or page is None or not source:
            continue
        doc = Path(str(source)).name
        if not doc:
            continue
        try:
            idx_int = int(chunk_idx)
            page_int = int(page)
        except (TypeError, ValueError):
            continue
        lookup[idx_int] = (doc, page_int)
    return lookup


def parse_ground_truth_entry(entry: dict, chunk_lookup: dict[int, Pair] | None = None) -> tuple[str, str, List[Pair], List[str]]:
    qid = entry.get("id") or entry.get("question_id")
    query = entry.get("query") or entry.get("question")
    relevant_pairs: List[Pair] = []
    relevant = entry.get("relevant") or entry.get("ground_truth", {}).get("relevant", [])
    for item in relevant:
        doc = Path(item.get("doc") or item.get("document") or "").name
        page = item.get("page")
        if doc and page is not None:
            relevant_pairs.append((doc, int(page)))
    chunk_indices = (
        entry.get("relevant_chunk_indices")
        or entry.get("ground_truth", {}).get("relevant_chunk_indices")
        or []
    )
    missing_chunks: List[int] = []
    for chunk_idx in chunk_indices:
        try:
            idx_int = int(chunk_idx)
        except (TypeError, ValueError):
            continue
        if chunk_lookup and idx_int in chunk_lookup:
            relevant_pairs.append(chunk_lookup[idx_int])
        else:
            missing_chunks.append(idx_int)
    if missing_chunks and qid:
        logger.warning("Ground Truth: unbekannte Chunk-Indizes %s fuer %s", missing_chunks, qid)
    expected_keywords = entry.get("expected_keywords") or entry.get("ground_truth", {}).get("expected_keywords", [])
    return qid, query, relevant_pairs, expected_keywords


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation mit doc/page Ground-Truth")
    parser.add_argument("--ground-truth", type=Path, default=Path("app/chatbot/evaluation/questions_curated20.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("app/chatbot/evaluation/results_pipeline.csv"))
    parser.add_argument("--models", nargs="*", default=["mistral:instruct", "gemma:2b", "phi3:latest"])
    parser.add_argument("--k", type=int, default=20, help="Anzahl der Chunks fuer Retrieval-Metriken (k-retrieve)")
    parser.add_argument("--k-context", type=int, default=8, help="Anzahl der Chunks, die in den Prompt aufgenommen werden")
    parser.add_argument("--multi-queries", type=int, default=2, help="Anzahl der Query-Varianten fuer Multi-Query-Retrieval")
    parser.add_argument("--candidate-pool", type=int, default=40, help="Anzahl der finalen Retriever-Kandidaten (top-k)")
    parser.add_argument("--retrieval-mode", choices=["hybrid", "bm25", "dense"], default="hybrid")
    parser.add_argument("--max-context-chars", type=int, default=2200)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--reranker-k", type=int, default=50, help="Wie viele Kandidaten an den Cross-Encoder geben")
    parser.add_argument("--disable-query-rewrite", action="store_true")
    parser.add_argument("--disable-reranker", action="store_true", help="Cross-Encoder fuer schnellere Laeufe deaktivieren")
    parser.add_argument("--synonyms", type=Path, default=Path("app/chatbot/Config/synonyms.yml"))
    parser.add_argument("--mandatory-keywords", type=Path, default=Path("app/chatbot/Config/mandatory_keywords.yml"))
    parser.add_argument("--log-missing-keywords", action="store_true")
    parser.add_argument("--log-retrieval-details", action="store_true", help="Log Anteil BM25/Dense je Anfrage")
    parser.add_argument("--max-per-doc", type=int, default=2, help="Maximale Anzahl Chunks pro Dokument im finalen Ranking")
    parser.add_argument("--max-workers", type=int, default=max(1, min(4, os.cpu_count() or 2)), help="Parallelisierte Fragenverarbeitung")
    parser.add_argument("--llm-workers", type=int, default=1, help="Parallele Ollama-Anfragen (1 = sequenziell)")
    parser.add_argument("--answer-max-tokens", type=int, default=160, help="Maximale Token fuer LLM-Antworten")
    parser.add_argument("--max-questions", type=int, default=None, help="Optional: Nur die ersten N Fragen verarbeiten")
    args = parser.parse_args()

    if args.disable_query_rewrite:
        os.environ["SUPPORTAPP_DISABLE_QUERY_REWRITE"] = "1"

    if args.log_retrieval_details:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    ground_truth_entries = load_ground_truth(args.ground_truth)
    if args.max_questions is not None:
        ground_truth_entries = ground_truth_entries[: max(0, args.max_questions)]
    if not ground_truth_entries:
        print(f"Keine Ground-Truth-Daten in {args.ground_truth}", file=sys.stderr)
        sys.exit(1)

    synonyms_data, mandatory_map = load_synonyms_and_mandatory(args.synonyms, args.mandatory_keywords)
    keyword_normalizer = KeywordNormalizer(synonyms_data)

    models = load_models(args.models)

    index, records, embeddings = load_faiss_index("app/chatbot/data/faiss_index/index")
    retriever = HybridRetriever(records, embeddings=embeddings, faiss_index=index)
    chunk_lookup = build_chunk_lookup(records)
    reranker = None
    if not args.disable_reranker and CrossEncoderReranker is not None:
        try:
            reranker = CrossEncoderReranker()
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: Reranker konnte nicht geladen werden ({exc}), fahre ohne.")
            reranker = None

    use_reranker = reranker is not None and not args.disable_reranker
    if args.max_workers > 1 and use_reranker:
        print("INFO: Cross-Encoder wird fuer parallele Ausfuehrung deaktiviert.", file=sys.stderr)
        use_reranker = False
    reranker_in_use = reranker if use_reranker else None

    ensure_output(args.output)
    fieldnames = [
        "question_id",
        "query",
        "model",
        "retrieval_mode",
        "recall@k",
        "ndcg@k",
        "mrr",
        "rank_first_rel",
        "keyword_f1",
        "missing_keywords",
        "time_s",
        "tokens_in",
        "tokens_out",
        "factual_correct",
        "top_k",
        "bm25_in_topk",
        "dense_in_topk",
        "overlap_in_topk",
        "rerank_delta_top1",
        "avg_score_delta_topk",
        "kendall_tau_topk",
        "format_bullet_keyword",
        "format_citation_per_bullet",
        "contains_nicht_im_kontext",
        "bullet_count",
        "error_bucket",
    ]

    llm_options: Dict[str, object] = {
        "temperature": 0.05,
        "top_p": 0.8,
    }
    if args.answer_max_tokens:
        llm_options["num_predict"] = args.answer_max_tokens

    llm_semaphore = threading.BoundedSemaphore(max(1, args.llm_workers))
    pool_size = max(args.candidate_pool, args.k)
    bm25_pool = max(200, pool_size * 3)
    reranker_candidates = args.reranker_k if args.reranker_k is not None else max(pool_size * 2, 150)
    model_order = {model: idx for idx, model in enumerate(models)}
    question_order = {
        (entry.get("id") or entry.get("question_id") or str(idx)): idx for idx, entry in enumerate(ground_truth_entries)
    }

    def process_entry(entry: dict) -> List[dict]:
        qid, query, relevant_pairs, expected_keywords = parse_ground_truth_entry(entry, chunk_lookup)
        rows: List[dict] = []
        if not query:
            return rows
        if not relevant_pairs:
            print(f"WARN: Keine relevanten Dokumente fuer {qid}")

        query_variants = generate_multi_queries(query, total_variants=args.multi_queries)
        query_embeddings = get_embeddings(query_variants)
        result_batches = []
        for variant, embedding in zip(query_variants, query_embeddings):
            kwargs = {
                "question": variant,
                "query_embedding": np.array(embedding, dtype=np.float32),
                "top_k": pool_size,
                "bm25_k": bm25_pool,
                "reranker_weight": 0.7,
                "mode": args.retrieval_mode,
                "question_id": qid,
                "log_details": args.log_retrieval_details,
            }
            if reranker_in_use is not None:
                kwargs["reranker"] = reranker_in_use
                kwargs["reranker_k"] = reranker_candidates
            result_batches.append(retriever.retrieve(**kwargs))

        ranked_chunks = merge_ranked_results(result_batches, pool_size, max_per_doc=args.max_per_doc)

        ranked_pairs: List[Pair] = []
        contexts: List[str] = []
        for chunk in ranked_chunks:
            doc = Path(chunk.get("source") or "").name
            page = chunk.get("page")
            if doc and page is not None:
                ranked_pairs.append((doc, int(page)))
            contexts.append(str(chunk.get("content") or ""))

        ranked_pairs = dedupe_pairs(ranked_pairs)
        expected_canonical = canonical_keyword_set(keyword_normalizer, expected_keywords)

        top_chunks = ranked_chunks[: args.k]
        bm25_in_topk = sum(1 for chunk in top_chunks if chunk.get("from_bm25"))
        dense_in_topk = sum(1 for chunk in top_chunks if chunk.get("from_dense"))
        overlap_in_topk = sum(1 for chunk in top_chunks if chunk.get("from_bm25") and chunk.get("from_dense"))
        if top_chunks:
            top1 = top_chunks[0]
            retriever_score_top1 = float(top1.get("retriever_score", top1.get("score", 0.0)))
            rerank_delta_top1 = float(top1.get("score", retriever_score_top1)) - retriever_score_top1
            deltas = [
                float(chunk.get("score", chunk.get("retriever_score", 0.0))) - float(chunk.get("retriever_score", 0.0))
                for chunk in top_chunks
            ]
            avg_score_delta_topk = float(np.mean(deltas))
            pre_ranks = [int(chunk.get("pre_rank", idx + 1)) for idx, chunk in enumerate(top_chunks)]
            kendall_tau_topk = kendall_tau(pre_ranks)
        else:
            rerank_delta_top1 = 0.0
            avg_score_delta_topk = 0.0
            kendall_tau_topk = 0.0

        recall = recall_at_k(relevant_pairs, ranked_pairs, args.k)
        ndcg = ndcg_at_k(relevant_pairs, ranked_pairs, args.k)
        mrr_value = mrr(relevant_pairs, ranked_pairs[: args.k])
        first_rank = rank_first_rel(relevant_pairs, ranked_pairs)

        context_candidates = ranked_chunks[: args.k]
        window = select_context_window(
            context_candidates,
            max_chunks=args.k_context,
            min_chunks=min(args.k_context, 3),
            max_chars=args.max_context_chars,
            expected_tokens=expected_canonical,
        )
        context_text = "\n\n".join(str(chunk.get("content") or "") for chunk in window if chunk.get("content"))
        keyword_clause = ""
        if expected_keywords:
            keyword_list = ", ".join(expected_keywords)
            keyword_clause = (
                "Pflicht-Schluesselwoerter (exakt in dieser Schreibweise verwenden): "
                f"{keyword_list}.\n"
                "Formatiere die Antwort als Liste mit kurzen Saetzen. Jede Zeile beginnt mit '- ' und enthaelt GENAU EIN "
                "Pflicht-Schluesselwort, gefolgt von einem belegten Kurzsatz und der Quelle im Format '(Dokument, S.X)'.\n"
                "Setze keine Synonyme ein. Wenn eine Information fehlt, verwende statt eines Satzes den Eintrag "
                "'- <Schluesselwort>: Nicht im Kontext'.\n"
                "Beende die Antwort mit der Zeile 'Fehlend: ...' (verwende '-' falls nichts fehlt). Fuehre danach die Zeile "
                "'Quellen: ...' mit allen verwendeten Quellen auf.\n"
            )

        prompt = (
            "Beantworte die Frage ausschliesslich anhand der bereitgestellten Dokumentenabschnitte.\n"
            "Wenn sich eine geforderte Information nicht sicher aus dem Kontext belegen laesst, schreibe 'Nicht im Kontext'.\n"
            f"{keyword_clause}\n"
            f"FRAGE:\n{query}\n\nKONTEXT:\n{context_text}\n\nANTWORT:"
        )
        tokens_in = tokens_from_text(prompt)
        factual_context = contexts[: args.k] or contexts or [""]

        for model_id in models:
            start = time.time()
            try:
                with llm_semaphore:
                    answer = query_ollama(
                        prompt,
                        model=model_id,
                        language="Deutsch",
                        options=dict(llm_options),
                        timeout=args.timeout,
                        stream=True,
                    )
            except Exception as exc:  # noqa: BLE001
                answer = f"Antwort fehlgeschlagen: {exc}"
            duration = time.time() - start

            f1_score, missing_keywords = evaluate_keywords(
                qid,
                expected_keywords,
                answer,
                keyword_normalizer,
                mandatory_map,
            )
            factual = factual_check(answer, factual_context)
            tokens_out = tokens_from_text(answer)

            format_keyword_ok, format_citation_ok, contains_nic, bullet_count = analyze_answer_format(
                answer,
                expected_canonical,
                keyword_normalizer,
            )

            if first_rank is None or first_rank == "":
                error_bucket = "B"
            elif not factual:
                error_bucket = "A"
            elif not (format_keyword_ok and format_citation_ok):
                error_bucket = "C"
            else:
                error_bucket = "OK"

            rows.append(
                {
                    "question_id": qid,
                    "query": query,
                    "model": model_id,
                    "retrieval_mode": args.retrieval_mode,
                    "recall@k": round(recall, 4),
                    "ndcg@k": round(ndcg, 4),
                    "mrr": round(mrr_value, 4),
                    "rank_first_rel": first_rank if first_rank is not None else "",
                    "keyword_f1": round(f1_score, 4),
                    "missing_keywords": ", ".join(missing_keywords)
                    if args.log_missing_keywords and missing_keywords
                    else "",
                    "time_s": round(duration, 4),
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "factual_correct": int(factual),
                    "top_k": args.k,
                    "bm25_in_topk": bm25_in_topk,
                    "dense_in_topk": dense_in_topk,
                    "overlap_in_topk": overlap_in_topk,
                    "rerank_delta_top1": round(rerank_delta_top1, 6),
                    "avg_score_delta_topk": round(avg_score_delta_topk, 6),
                    "kendall_tau_topk": round(kendall_tau_topk, 6),
                    "format_bullet_keyword": int(format_keyword_ok),
                    "format_citation_per_bullet": int(format_citation_ok),
                    "contains_nicht_im_kontext": int(contains_nic),
                    "bullet_count": bullet_count,
                    "error_bucket": error_bucket,
                }
            )
        return rows

    worker_count = max(1, min(args.max_workers, len(ground_truth_entries)))
    all_rows: List[dict] = []
    if worker_count == 1:
        for entry in ground_truth_entries:
            all_rows.extend(process_entry(entry))
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {executor.submit(process_entry, entry): entry for entry in ground_truth_entries}
            for future in as_completed(future_map):
                entry = future_map[future]
                qid = entry.get("id") or entry.get("question_id") or "<unknown>"
                try:
                    all_rows.extend(future.result())
                except Exception as exc:  # noqa: BLE001
                    print(f"FEHLER bei {qid}: {exc}", file=sys.stderr)

    def sort_key(row: dict) -> tuple[int, int]:
        return (
            question_order.get(row.get("question_id"), 0),
            model_order.get(row.get("model"), 0),
        )

    all_rows.sort(key=sort_key)

    with args.output.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Ergebnisse gespeichert in {args.output}")


if __name__ == "__main__":
    main()
