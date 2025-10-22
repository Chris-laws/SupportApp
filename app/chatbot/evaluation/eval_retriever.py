from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import List

from app.chatbot.evaluation.dataset import load_questions
from app.chatbot.evaluation.metrics import ndcg_at_k, recall_at_k
from app.chatbot.modules.embeddings import get_embeddings, load_faiss_index
from app.chatbot.modules.retriever import HybridRetriever, rewrite_query_with_llama3
from app.chatbot.modules.reranker import CrossEncoderReranker


@dataclass
class EvalConfig:
    index_base: Path
    dataset_path: Path
    top_k: int = 10
    bm25_k: int = 80
    use_rewrite: bool = True
    use_reranker: bool = True


def run_eval(config: EvalConfig) -> None:
    index, chunk_records, embeddings = load_faiss_index(str(config.index_base))
    if not chunk_records:
        raise RuntimeError("Kein Index geladen. Bitte zuerst den Index aufbauen.")

    retriever = HybridRetriever(chunk_records, embeddings=embeddings, faiss_index=index)
    reranker = None
    if config.use_reranker:
        try:
            reranker = CrossEncoderReranker()
        except Exception as exc:  # noqa: BLE001
            print(f"Warnung: Reranker nicht verfuegbar: {exc}")

    questions = load_questions(config.dataset_path)
    recalls: List[float] = []
    ndcgs: List[float] = []

    for sample in questions:
        question = sample["question"]
        ground = sample["ground_truth"]
        relevant_indices = ground.get("relevant_chunk_indices") or []

        rewritten = rewrite_query_with_llama3(question) if config.use_rewrite else question
        query_embedding = get_embeddings([rewritten])[0]

        ranked = retriever.retrieve(
            question,
            query_embedding,
            top_k=config.top_k,
            bm25_k=config.bm25_k,
            reranker=reranker,
            reranker_weight=0.6 if reranker else 0.0,
            reranker_k=max(config.top_k * 2, 100) if reranker else None,
        )

        candidate_indices = [int(chunk.get("chunk_index", -1)) for chunk in ranked]
        recalls.append(recall_at_k(candidate_indices, relevant_indices, config.top_k))
        ndcgs.append(ndcg_at_k(candidate_indices, relevant_indices, config.top_k))

    print("----- Retriever Evaluation -----")
    print(f"Fragen: {len(questions)}")
    print(f"Top-k: {config.top_k} | BM25-k: {config.bm25_k}")
    print(f"Query-Rewrite: {'an' if config.use_rewrite else 'aus'}")
    print(f"Re-Ranker: {'an' if config.use_reranker else 'aus'}")
    print(f"Recall@{config.top_k}: {mean(recalls):.3f}")
    print(f"nDCG@{config.top_k}: {mean(ndcgs):.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluiere den Retriever mit Recall/nDCG auf dem Gold-Dataset.")
    parser.add_argument("--index-base", required=True, help="Pfad zur Basis des Index (ohne Endung)")
    parser.add_argument("--dataset", required=True, help="JSON-Datei mit Fragen und Ground-Truth")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bm25-k", type=int, default=80)
    parser.add_argument("--no-rewrite", action="store_true", help="Query-Rewrite deaktivieren")
    parser.add_argument("--no-reranker", action="store_true", help="Cross-Encoder Reranker deaktivieren")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EvalConfig(
        index_base=Path(args.index_base),
        dataset_path=Path(args.dataset),
        top_k=args.top_k,
        bm25_k=args.bm25_k,
        use_rewrite=not args.no_rewrite,
        use_reranker=not args.no_reranker,
    )
    run_eval(config)


if __name__ == "__main__":
    main()

