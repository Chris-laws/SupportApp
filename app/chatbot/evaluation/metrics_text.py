from __future__ import annotations

import re
from typing import List

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from rouge_score import rouge_scorer

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize_de(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def bleu_precise(pred: str, refs: List[str]) -> float:
    pred_tok = tokenize_de(pred or "")
    refs_tok = [tokenize_de(r) for r in refs or []]
    if not pred_tok or not refs_tok:
        return 0.0
    smoother = SmoothingFunction().method4
    return corpus_bleu([refs_tok], [pred_tok], smoothing_function=smoother)


def rouge_all(pred: str, refs: List[str]) -> dict:
    if not refs:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeLsum": 0.0}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
    best = {"rouge1": 0.0, "rouge2": 0.0, "rougeLsum": 0.0}
    for reference in refs:
        scores = scorer.score(reference or "", pred or "")
        for key in best:
            best[key] = max(best[key], scores[key].fmeasure)
    return best
