from __future__ import annotations

import json
import itertools
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from app.chatbot.evaluation.metrics_text import bleu_precise, rouge_all
from app.chatbot.modules.embeddings import get_embeddings, load_faiss_index
from app.chatbot.modules.llm import query_ollama
from app.chatbot.modules.retriever import HybridRetriever, rewrite_query_with_llama3, select_context_window
from app.chatbot.modules.reranker import CrossEncoderReranker

INDEX_BASE = Path("app/chatbot/data/faiss_index/index")
GOLD_PATH = Path("data/eval/gold.json")
MODELS_YML = Path("data/eval/models.yml")
OUT_DIR = Path("data/eval")
OUT_CSV = OUT_DIR / "eval_f2_llms.csv"


def keyword_f1(answer: str, keywords: list[str]) -> float:
    if not answer or not keywords:
        return 0.0
    predicted = set(answer.lower().split())
    gold = {kw.lower() for kw in keywords}
    inter = len(predicted & gold)
    if inter == 0:
        return 0.0
    precision = inter / len(predicted)
    recall = inter / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def build_shared_context(retriever: HybridRetriever, reranker: CrossEncoderReranker, question: str, top_k: int = 15) -> str:
    optimized = rewrite_query_with_llama3(question)
    embedding = get_embeddings([optimized])[0]
    ranked = retriever.retrieve(
        question,
        embedding,
        top_k=top_k,
        bm25_k=max(80, top_k * 5),
        reranker=reranker,
        reranker_k=max(2 * top_k, 100),
        reranker_weight=0.6,
    )
    window = select_context_window(ranked, max_chunks=6, min_chunks=4, max_chars=1800)
    return "

".join(str(chunk.get("content") or "") for chunk in window if chunk.get("content"))


def ensure_output_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_output_dir()
    model_cfg = yaml.safe_load(MODELS_YML.read_text(encoding="utf-8"))
    llm_variants = list(model_cfg.get("llm", {}).items())
    decoding_presets = model_cfg.get("decoding", {})

    index, records, embeddings = load_faiss_index(str(INDEX_BASE))
    retriever = HybridRetriever(records, embeddings=embeddings, faiss_index=index)
    reranker = CrossEncoderReranker()

    gold_samples = json.loads(GOLD_PATH.read_text(encoding="utf-8"))

    rows: list[dict] = []
    for item in gold_samples:
        query = item["query"]
        keywords = item.get("expected_answer_keywords", [])
        references = item.get("references", [])
        context = build_shared_context(retriever, reranker, query)

        for size_tag, llm_model in llm_variants:
            for dec_tag, dec_opts in decoding_presets.items():
                prompt = (
                    "Nutze ausschließlich den folgenden CONTEXT zur Beantwortung."
                    "

CONTEXT:
" + context + "

FRAGE:
" + query +
                    "

ANTWORT mit Quellenangabe (Dokument, S.X):"
                )
                options = {
                    "temperature": dec_opts.get("temperature", 0.0),
                    "top_p": dec_opts.get("top_p", 1.0),
                    "max_tokens": dec_opts.get("max_tokens", 512),
                }

                start = time.time()
                answer = query_ollama(prompt, model=llm_model, language="Deutsch", options=options)
                end = time.time()

                kw_f1 = keyword_f1(answer, keywords)
                bleu = bleu_precise(answer, references)
                rouge = rouge_all(answer, references)

                rows.append({
                    "query": query,
                    "llm_size": size_tag,
                    "llm_model": llm_model,
                    "decoding": dec_tag,
                    "answer_f1_kw": round(kw_f1, 4),
                    "bleu": round(bleu, 4),
                    "rouge1": round(rouge["rouge1"], 4),
                    "rouge2": round(rouge["rouge2"], 4),
                    "rougeLsum": round(rouge["rougeLsum"], 4),
                    "generation_time_s": round(end - start, 4),
                    "answer_length": len(answer or ""),
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    summary = df.groupby(["llm_size", "decoding"])[["answer_f1_kw", "bleu", "rougeLsum", "generation_time_s", "answer_length"]].mean()
    print(summary)

    # Boxplots nach Größe und Decoding
    for metric in ["answer_f1_kw", "bleu", "rougeLsum"]:
        plt.figure()
        df.boxplot(column=[metric], by="llm_size")
        plt.title(f"{metric} nach LLM-Größe")
        plt.suptitle("")
        plt.savefig(OUT_DIR / f"f2_box_{metric}_by_size.png")
        plt.close()

        plt.figure()
        df.boxplot(column=[metric], by="decoding")
        plt.title(f"{metric} nach Decoding-Profil")
        plt.suptitle("")
        plt.savefig(OUT_DIR / f"f2_box_{metric}_by_decoding.png")
        plt.close()

    # Histogramme der ROUGE-Lsum pro Größe
    for tag in df["llm_size"].unique():
        plt.figure()
        df[df["llm_size"] == tag]["rougeLsum"].plot(kind="hist", bins=10, title=f"ROUGE-Lsum Histogram {tag}")
        plt.xlabel("ROUGE-Lsum")
        plt.savefig(OUT_DIR / f"f2_hist_rougeL_{tag}.png")
        plt.close()

    # Scatter BLEU vs ROUGE-Lsum je Größe
    plt.figure()
    marker_cycle = itertools.cycle(["o", "s", "^", "D", "x", "+"])
    for tag in sorted(df["llm_size"].unique()):
        subset = df[df["llm_size"] == tag]
        plt.scatter(subset["bleu"], subset["rougeLsum"], label=tag, marker=next(marker_cycle))
    plt.xlabel("BLEU")
    plt.ylabel("ROUGE-Lsum")
    plt.title("BLEU vs ROUGE-Lsum (LLM-Größe)")
    plt.legend()
    plt.savefig(OUT_DIR / "f2_scatter_bleu_rougeL.png")
    plt.close()


if __name__ == "__main__":
    main()
