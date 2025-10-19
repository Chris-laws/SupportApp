from __future__ import annotations

from typing import Any, List, Mapping, Sequence

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        question: str,
        chunks: Sequence[Mapping[str, Any]],
        *,
        top_k: int,
    ) -> List[Mapping[str, Any]]:
        if not chunks:
            return []

        pairs = [(question, str(chunk.get("content") or "")) for chunk in chunks]
        scores = self.model.predict(pairs)

        scored: List[tuple[float, Mapping[str, Any]]] = [
            (float(score), chunk) for score, chunk in zip(scores, chunks)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

        reranked: List[Mapping[str, Any]] = []
        for score, chunk in scored[:top_k]:
            enriched = dict(chunk)
            enriched["rerank_score"] = score
            reranked.append(enriched)
        return reranked


__all__ = [
    "CrossEncoderReranker",
]
