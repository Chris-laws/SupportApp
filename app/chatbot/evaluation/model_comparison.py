from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:
    from pynvml import NVMLError, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit
except Exception:  # noqa: BLE001
    nvmlInit = None  # type: ignore[assignment]

from app.chatbot.evaluation.dataset import load_questions, save_results
from app.chatbot.evaluation.metrics import corpus_bleu, ndcg_at_k, recall_at_k, rouge_l
from app.chatbot.modules.embeddings import get_embeddings, load_faiss_index
from app.chatbot.modules.llm import query_ollama
from app.chatbot.modules.retriever import HybridRetriever, rewrite_query_with_llama3, select_context_window
from app.chatbot.modules.reranker import CrossEncoderReranker


@dataclass
class GeneratorSetting:
    name: str
    model: str
    temperature: float = 0.2
    top_p: float = 0.9


DEFAULT_GENERATORS: Sequence[GeneratorSetting] = (
    GeneratorSetting(name="llama3", model="llama3"),
    GeneratorSetting(name="mistral", model="mistral"),
    GeneratorSetting(name="phi3_mini", model="phi3:mini"),
)


@dataclass
class EvalRow:
    question_id: str
    question: str
    generator: str
    recall_at_k: float
    ndcg_at_k: float
    bleu: float
    rouge_l: float
    latency_seconds: float
    response: str
    tokens_per_second: float | None
    vram_delta_gb: float | None


class ModelComparator:
    def __init__(
        self,
        index_base: Path,
        dataset_path: Path,
        generators: Sequence[GeneratorSetting],
        top_k: int = 5,
        device_index: int = 0,
    ) -> None:
        self.generators = generators
        self.top_k = top_k
        self.dataset = load_questions(dataset_path)

        index, chunk_records, embeddings = load_faiss_index(str(index_base))
        if not chunk_records:
            raise RuntimeError("FAISS index leer. Bitte `python app/chatbot/main.py` ausfuehren.")
        self.retriever = HybridRetriever(chunk_records, embeddings=embeddings, faiss_index=index)

        try:
            self.reranker = CrossEncoderReranker()
        except Exception as exc:  # noqa: BLE001
            print("WARN: Reranker konnte nicht geladen werden:", exc)
            self.reranker = None

        self._rewrite_cache: Dict[str, str] = {}

        if nvmlInit is not None:
            try:
                nvmlInit()
                self._nvml_handle = nvmlDeviceGetHandleByIndex(device_index)
            except Exception as exc:  # noqa: BLE001
                print("WARN: NVML nicht verfuegbar:", exc)
                self._nvml_handle = None
        else:
            self._nvml_handle = None

    def _rewrite(self, question: str) -> str:
        if question not in self._rewrite_cache:
            self._rewrite_cache[question] = rewrite_query_with_llama3(question)
        return self._rewrite_cache[question]

    def _gpu_usage(self) -> float | None:
        if not self._nvml_handle:
            return None
        try:
            info = nvmlDeviceGetMemoryInfo(self._nvml_handle)
            return info.used / (1024**3)
        except Exception:  # noqa: BLE001
            return None

    def _retrieve(self, question: str, optimized_query: str):
        embedding = get_embeddings([optimized_query])[0]
        retrieve_kwargs: Dict[str, object] = {}
        if self.reranker is not None:
            retrieve_kwargs.update(
                {
                    "reranker": self.reranker,
                    "reranker_k": max(self.top_k * 2, 30),
                    "reranker_weight": 0.65,
                }
            )
        ranked = self.retriever.retrieve(question, embedding, top_k=self.top_k, **retrieve_kwargs)
        return ranked

    def evaluate(self) -> List[EvalRow]:
        rows: List[EvalRow] = []

        for generator in self.generators:
            print(f">> Generator {generator.name}")
            for sample in self.dataset:
                question = sample["question"]
                ground = sample["ground_truth"]
                relevant_indices: List[int] = list(ground.get("relevant_chunk_indices") or [])

                optimized = self._rewrite(question)
                ranked = self._retrieve(question, optimized)
                ranked_indices = [int(chunk.get("chunk_index", -1)) for chunk in ranked]

                recall = recall_at_k(ranked_indices, relevant_indices, self.top_k)
                ndcg = ndcg_at_k(ranked_indices, relevant_indices, self.top_k)

                context_chunks = select_context_window(ranked, max_chunks=6, min_chunks=4, max_chars=1800)
                context_text = "\n\n".join(str(chunk.get("content") or "") for chunk in context_chunks)

                prompt = (
                    "Nutze den folgenden Kontext, um die Frage zu beantworten:\n\n"
                    f"{context_text}\n\nFrage: {question}"
                )

                start = time.perf_counter()
                gpu_before = self._gpu_usage()
                response = query_ollama(
                    prompt,
                    model=generator.model,
                    language="Deutsch",
                )
                latency = time.perf_counter() - start
                gpu_after = self._gpu_usage()
                vram_delta = None
                if gpu_before is not None and gpu_after is not None:
                    vram_delta = max(0.0, gpu_after - gpu_before)

                token_estimate = len(response.split())
                tokens_per_second = token_estimate / latency if latency > 0 else None

                references = ground.get("reference_answers") or []
                bleu = corpus_bleu(response, references) if references else 0.0
                rouge = max((rouge_l(response, ref) for ref in references), default=0.0)

                rows.append(
                    EvalRow(
                        question_id=sample["question_id"],
                        question=question,
                        generator=generator.name,
                        recall_at_k=recall,
                        ndcg_at_k=ndcg,
                        bleu=bleu,
                        rouge_l=rouge,
                        latency_seconds=latency,
                        response=response,
                        tokens_per_second= tokens_per_second,
                        vram_delta_gb=vram_delta,
                    )
                )
        return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modellvergleich fuer Kapitel 6 (F2)")
    parser.add_argument("--index-base", required=True, help="Pfad zur Basis des FAISS Index (ohne Endung)")
    parser.add_argument("--dataset", required=True, help="Pfad zur Fragenliste (JSON)")
    parser.add_argument("--output", required=True, help="Pfad zur Ergebnisdatei (JSON)")
    parser.add_argument("--models", nargs="*", help="Ollama Modellnamen", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generators = (
        tuple(GeneratorSetting(name=model.replace(":", "_"), model=model) for model in args.models)
        if args.models
        else DEFAULT_GENERATORS
    )

    comparator = ModelComparator(
        index_base=Path(args.index_base),
        dataset_path=Path(args.dataset),
        generators=generators,
        top_k=args.top_k,
        device_index=args.device_index,
    )
    rows = comparator.evaluate()

    save_results(args.output, (asdict(row) for row in rows))


if __name__ == "__main__":
    main()

