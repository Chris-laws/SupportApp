from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from app.chatbot.modules.llm import query_ollama


def tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    token: List[str] = []
    for ch in text.lower():
        if ch.isalnum():
            token.append(ch)
        else:
            if token:
                tokens.append("".join(token))
                token = []
    if token:
        tokens.append("".join(token))
    return tokens


def load_dataset(path: Path) -> Tuple[dict, List[dict]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    documents = raw.get("documents") or {}
    samples = raw.get("samples") or []
    if not documents or not samples:
        raise ValueError("Dataset muss 'documents' und 'samples' enthalten.")
    return documents, samples


def prepare_index(documents: dict) -> Tuple[dict, dict, dict]:
    doc_tokens: dict[str, Counter[str]] = {}
    df: defaultdict[str, int] = defaultdict(int)
    for doc_id, text in documents.items():
        tokens = tokenize(text)
        counter = Counter(tokens)
        doc_tokens[doc_id] = counter
        for token in counter:
            df[token] += 1

    n_docs = len(documents)
    idf: dict[str, float] = {}
    for token, freq in df.items():
        idf[token] = math.log((n_docs + 1) / (freq + 1)) + 1.0

    doc_vectors: dict[str, dict[str, float]] = {}
    doc_norms: dict[str, float] = {}
    for doc_id, counter in doc_tokens.items():
        vec: dict[str, float] = {}
        norm_sq = 0.0
        for token, tf in counter.items():
            weight = tf * idf[token]
            vec[token] = weight
            norm_sq += weight * weight
        doc_vectors[doc_id] = vec
        doc_norms[doc_id] = math.sqrt(norm_sq) if norm_sq else 1.0

    return doc_vectors, doc_norms, idf


def vectorize_query(query: str, idf: dict[str, float]) -> Tuple[dict[str, float], float]:
    counter = Counter(tokenize(query))
    vec: dict[str, float] = {}
    norm_sq = 0.0
    for token, tf in counter.items():
        weight = tf * idf.get(token, 0.0)
        if weight:
            vec[token] = weight
            norm_sq += weight * weight
    return vec, math.sqrt(norm_sq) if norm_sq else 1.0


def retrieve_top_k(
    query: str,
    documents: dict[str, str],
    doc_vectors: dict[str, dict[str, float]],
    doc_norms: dict[str, float],
    idf: dict[str, float],
    k: int,
) -> List[Tuple[str, float]]:
    q_vec, q_norm = vectorize_query(query, idf)
    scores: List[Tuple[str, float]] = []
    for doc_id, doc_vec in doc_vectors.items():
        dot = 0.0
        for token, weight in q_vec.items():
            dot += weight * doc_vec.get(token, 0.0)
        if dot > 0:
            score = dot / (q_norm * doc_norms[doc_id] or 1.0)
            scores.append((doc_id, score))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:k]


def recall_at_k(relevant: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    rel_set = set(relevant)
    retrieved_set = set(retrieved[:k])
    return len(rel_set & retrieved_set) / len(rel_set)


def dcg_at_k(relevant: Sequence[str], ranked: Sequence[str], k: int) -> float:
    rel_set = set(relevant)
    dcg = 0.0
    for idx, doc_id in enumerate(ranked[:k], start=1):
        gain = 1.0 if doc_id in rel_set else 0.0
        if gain:
            dcg += gain / math.log2(idx + 1)
    return dcg


def ndcg_at_k(relevant: Sequence[str], ranked: Sequence[str], k: int) -> float:
    ideal = dcg_at_k(relevant, relevant, k)
    if ideal == 0.0:
        return 0.0
    actual = dcg_at_k(relevant, ranked, k)
    return actual / ideal


def mrr_at_k(relevant: Sequence[str], ranked: Sequence[str], k: int) -> float:
    rel_set = set(relevant)
    for idx, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in rel_set:
            return 1.0 / idx
    return 0.0


def keyword_f1(reference_keywords: Sequence[str], answer: str) -> float:
    if not reference_keywords:
        return 0.0
    answer_tokens = set(tokenize(answer))
    ref_tokens = {token for keyword in reference_keywords for token in tokenize(keyword)}
    true_pos = len(answer_tokens & ref_tokens)
    if true_pos == 0:
        return 0.0
    precision = true_pos / len(answer_tokens)
    recall = true_pos / len(ref_tokens)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def factual_check(answer: str, contexts: Iterable[str], threshold: float = 0.7) -> bool:
    context_tokens = Counter()
    for ctx in contexts:
        context_tokens.update(tokenize(ctx))
    answer_tokens = [tok for tok in tokenize(answer) if len(tok) > 2]
    if not answer_tokens:
        return False
    covered = sum(1 for tok in answer_tokens if context_tokens[tok] > 0)
    ratio = covered / len(answer_tokens)
    return ratio >= threshold


def ensure_output(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Einfache Retrieval & LLM Evaluation")
    parser.add_argument("--dataset", type=Path, required=True, help="Pfad zur JSON-Datei mit documents & samples")
    parser.add_argument("--output", type=Path, required=True, help="Pfad zur Ergebnis-CSV")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mistral:7b-instruct", "gemma:2b", "phi3:mini"],
        help="Liste der Ollama-Modelle",
    )
    parser.add_argument("--k", type=int, default=5, help="Top-k fuer Retrieval und Metriken")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout pro LLM-Anfrage (Sekunden)")
    args = parser.parse_args()

    documents, samples = load_dataset(args.dataset)
    doc_vectors, doc_norms, idf = prepare_index(documents)

    ensure_output(args.output)
    fieldnames = [
        "query",
        "model",
        "recall@k",
        "ndcg@k",
        "mrr",
        "keyword_f1",
        "time_s",
        "factual_correct",
    ]

    with args.output.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for sample in samples:
            query = sample["query"]
            relevant_docs = sample.get("relevant_docs", [])
            keywords = sample.get("reference_keywords", [])

            retrieved_pairs = retrieve_top_k(
                query,
                documents,
                doc_vectors,
                doc_norms,
                idf,
                args.k,
            )
            retrieved_ids = [doc_id for doc_id, _ in retrieved_pairs]
            retrieved_texts = [documents[doc_id] for doc_id in retrieved_ids]

            recall = recall_at_k(relevant_docs, retrieved_ids, args.k)
            ndcg = ndcg_at_k(relevant_docs, retrieved_ids, args.k)
            mrr = mrr_at_k(relevant_docs, retrieved_ids, args.k)

            context_text = "\n\n".join(retrieved_texts)
            prompt_base = (
                "Beantworte die folgende Frage ausschliesslich mit den angegebenen Dokumentauszuegen.\n\n"
                f"FRAGE:\n{query}\n\n"
                f"DOKUMENTAUSZUEGE:\n{context_text}\n\nANTWORT:"
            )

            for model_id in args.models:
                start = time.time()
                try:
                    answer = query_ollama(
                        prompt_base,
                        model=model_id,
                        language="Deutsch",
                        options={"timeout": args.timeout},
                    )
                except Exception as exc:  # noqa: BLE001
                    answer = f"Antwort fehlgeschlagen: {exc}"
                duration = time.time() - start

                f1 = keyword_f1(keywords, answer)
                factual = factual_check(answer, retrieved_texts)

                writer.writerow(
                    {
                        "query": query,
                        "model": model_id,
                        "recall@k": round(recall, 4),
                        "ndcg@k": round(ndcg, 4),
                        "mrr": round(mrr, 4),
                        "keyword_f1": round(f1, 4),
                        "time_s": round(duration, 4),
                        "factual_correct": int(factual),
                    }
                )


if __name__ == "__main__":
    main()
