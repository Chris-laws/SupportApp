from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Mapping, Sequence

import numpy as np


def recall_at_k(ranked_items: Sequence[int], relevant_items: Iterable[int], k: int) -> float:
    relevant_set = set(relevant_items)
    if not relevant_set:
        return 0.0
    hits = sum(1 for item in ranked_items[:k] if item in relevant_set)
    return hits / len(relevant_set)


def precision_at_k(ranked_items: Sequence[int], relevant_items: Iterable[int], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant_set = set(relevant_items)
    hits = sum(1 for item in ranked_items[:k] if item in relevant_set)
    return hits / k


def average_precision(ranked_items: Sequence[int], relevant_items: Iterable[int], k: int | None = None) -> float:
    relevant_set = set(relevant_items)
    if not relevant_set:
        return 0.0
    if k is None:
        k = len(ranked_items)
    hits = 0
    score = 0.0
    for idx, item in enumerate(ranked_items[:k], start=1):
        if item in relevant_set:
            hits += 1
            score += hits / idx
    return score / len(relevant_set)


def ndcg_at_k(ranked_items: Sequence[int], relevance: Mapping[int, float] | Iterable[int], k: int) -> float:
    if isinstance(relevance, Mapping):
        rel_lookup = relevance
    else:
        rel_lookup = {item: 1.0 for item in relevance}
    if not rel_lookup:
        return 0.0

    dcg = 0.0
    for idx, item in enumerate(ranked_items[:k], start=1):
        gain = rel_lookup.get(item)
        if gain is None:
            continue
        dcg += (2**gain - 1) / math.log2(idx + 1)

    ideal_gains = sorted(rel_lookup.values(), reverse=True)
    idcg = 0.0
    for idx, gain in enumerate(ideal_gains[:k], start=1):
        idcg += (2**gain - 1) / math.log2(idx + 1)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def _tokenize(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]


def _extract_ngrams(tokens: Sequence[str], n: int) -> Counter[str]:
    return Counter(" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(candidate: str, references: Sequence[str], max_n: int = 4) -> float:
    if not candidate or not references:
        return 0.0
    candidate_tokens = _tokenize(candidate)
    reference_tokens = [_tokenize(ref) for ref in references]

    precisions: list[float] = []
    for n in range(1, max_n + 1):
        cand_ngrams = _extract_ngrams(candidate_tokens, n)
        if not cand_ngrams:
            precisions.append(0.0)
            continue
        max_refs = Counter()
        for ref_tokens in reference_tokens:
            max_refs |= _extract_ngrams(ref_tokens, n)
        clipped = sum(min(count, max_refs[ng]) for ng, count in cand_ngrams.items())
        precisions.append(clipped / sum(cand_ngrams.values()))

    if any(p == 0 for p in precisions):
        return 0.0

    log_prec = sum(math.log(p) for p in precisions) / max_n

    ref_lengths = [len(tokens) for tokens in reference_tokens]
    cand_len = len(candidate_tokens)
    closest_ref = min(ref_lengths, key=lambda r: abs(r - cand_len))
    if cand_len > closest_ref:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1 - closest_ref / max(cand_len, 1))

    return brevity_penalty * math.exp(log_prec)


def rouge_l(candidate: str, reference: str) -> float:
    if not candidate or not reference:
        return 0.0
    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)
    m, n = len(ref_tokens), len(cand_tokens)
    if m == 0 or n == 0:
        return 0.0

    lcs = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == cand_tokens[j - 1]:
                lcs[i, j] = lcs[i - 1, j - 1] + 1
            else:
                lcs[i, j] = max(lcs[i - 1, j], lcs[i, j - 1])

    lcs_len = lcs[m, n]
    precision = lcs_len / n
    recall = lcs_len / m
    if precision + recall == 0:
        return 0.0
    beta = precision / (recall + 1e-12)
    return (1 + beta**2) * precision * recall / (recall + beta**2 * precision + 1e-12)

