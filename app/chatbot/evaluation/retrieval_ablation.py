from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from app.chatbot.evaluation.dataset import load_questions, save_results
from app.chatbot.evaluation.metrics import ndcg_at_k, recall_at_k
from app.chatbot.modules.embeddings import get_embeddings, load_faiss_index
from app.chatbot.modules.retriever import HybridRetriever, rewrite_query_with_llama3
from app.chatbot.modules.reranker import CrossEncoderReranker


@dataclass
class AblationSetting:
    name: str
    top_k: int = 5
    enable_rewrite: bool = True
    enable_reranker: bool = True


ABLATIONS: Sequence[AblationSetting] = (
    AblationSetting("baseline", top_k=5, enable_rewrite=True, enable_reranker=True),
    AblationSetting("no_reranker", top_k=5, enable_rewrite=True, enable_reranker=False),
    AblationSetting("no_rewrite", top_k=5, enable_rewrite=False, enable_reranker=True),
    AblationSetting("k3", top_k=3, enable_rewrite=True, enable_reranker=True),
    AblationSetting("k8", top_k=8, enable_rewrite=True, enable_reranker=True),
)


@dataclass
class EvalRow:
    question_id: str
    question: str
    setting: str
    recall_at_k: float
    ndcg_at_k: float
    latency_seconds: float | None
    retrieved_indices: List[int] = field(default_factory=list)
    retrieved_sources: List[str] = field(default_factory=list)


def run(
    *,
    index_base: Path,
    dataset_path: Path,
    output_path: Path,
) -> List[EvalRow]:
    dataset = load_questions(dataset_path)
    index, chunk_records, embeddings = load_faiss_index(str(index_base))
    if not chunk_records:
        raise RuntimeError("FAISS index leer. Bitte zuerst `python app/chatbot/main.py` ausfuehren.")

    retriever = HybridRetriever(chunk_records, embeddings=embeddings, faiss_index=index)
    try:
        reranker = CrossEncoderReranker()
    except Exception as exc:  # noqa: BLE001
        print("WARN: Reranker konnte nicht geladen werden:", exc)
        reranker = None

    rewrite_cache: Dict[str, str] = {}
    rows: List[EvalRow] = []

    for setting in ABLATIONS:
        print(f">> Setting {setting.name}")
        for sample in dataset:
            question = sample["question"]
            ground = sample["ground_truth"]
            relevant_indices: List[int] = list(ground.get("relevant_chunk_indices") or [])
            if not relevant_indices:
                relevant_indices = []

            query_text = question
            if setting.enable_rewrite:
                if question not in rewrite_cache:
                    rewrite_cache[question] = rewrite_query_with_llama3(question)
                query_text = rewrite_cache[question]

            embedding = get_embeddings([query_text])[0]

            retrieve_kwargs: Dict[str, object] = {}
            if setting.enable_reranker and reranker is not None:
                retrieve_kwargs.update(
                    {
                        "reranker": reranker,
                        "reranker_k": max(setting.top_k * 2, 30),
                        "reranker_weight": 0.65,
                    }
                )

            start = time.perf_counter()
            ranked = retriever.retrieve(
                question,
                embedding,
                top_k=setting.top_k,
                **retrieve_kwargs,
            )
            latency = time.perf_counter() - start

            ranked_indices = [int(chunk.get("chunk_index", -1)) for chunk in ranked]
            ranked_sources = [str(chunk.get("source")) for chunk in ranked]

            rows.append(
                EvalRow(
                    question_id=sample["question_id"],
                    question=question,
                    setting=setting.name,
                    recall_at_k=recall_at_k(ranked_indices, relevant_indices, setting.top_k),
                    ndcg_at_k=ndcg_at_k(ranked_indices, relevant_indices, setting.top_k),
                    latency_seconds=latency,
                    retrieved_indices=ranked_indices,
                    retrieved_sources=ranked_sources,
                )
            )
    save_results(output_path, (asdict(row) for row in rows))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG Retrieval Ablationen (Kapitel 6, F1)")
    parser.add_argument("--index-base", required=True, help="Pfad zur Basis des FAISS Index (ohne Endung)")
    parser.add_argument("--dataset", required=True, help="Pfad zur Fragenliste (JSON)")
    parser.add_argument("--output", required=True, help="Pfad zur Ergebnisdatei (JSON)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        index_base=Path(args.index_base),
        dataset_path=Path(args.dataset),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()

