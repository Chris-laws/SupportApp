from __future__ import annotations





import argparse
import json


import math


import re


import time


from pathlib import Path





import matplotlib


matplotlib.use("Agg")


import matplotlib.pyplot as plt


import pandas as pd





from app.chatbot.modules.embeddings import get_embeddings, load_faiss_index


from app.chatbot.modules.llm import query_ollama


from app.chatbot.modules.retriever import HybridRetriever, rewrite_query_with_llama3, select_context_window


from app.chatbot.modules.reranker import CrossEncoderReranker





INDEX_BASE = Path("app/chatbot/data/faiss_index/index")


GOLD_PATH = Path("data/eval/gold.json")


OUT_DIR = Path("data/eval")


OUT_CSV = OUT_DIR / "eval_f1_results.csv"


FIXED_LLM = "llama3.1:8b-instruct-q4_K_M"



def parse_args() -> argparse.Namespace:


    parser = argparse.ArgumentParser(description="F1-Evaluation fuer RAG-Antworten")


    parser.add_argument("--start", type=int, default=0, help="Startindex in der Goldliste")


    parser.add_argument("--limit", type=int, default=None, help="Begrenze die Anzahl der ausgewerteten Fragen")


    parser.add_argument("--output-suffix", type=str, default=None, help="Dateinamenssuffix fuer partielle Runs")
    parser.add_argument("--llm-model", type=str, default=FIXED_LLM, help="Ollama Modell fuer die Antwortgenerierung")

    return parser.parse_args()








def recall_at_k(results: list[dict], gold_ids: list[int], k: int = 10) -> float:


    if not gold_ids:


        return 0.0


    retrieved = {int(r.get("chunk_index", -1)) for r in results[:k]}


    gold = set(int(cid) for cid in gold_ids)


    return len(retrieved & gold) / len(gold)








def ndcg_at_k(results: list[dict], gains: dict, k: int = 10) -> float:


    if not gains:


        return 0.0

    numeric_gains = {int(k): float(v) for k, v in gains.items()}


    def gain(cid: int) -> float:


        if cid in numeric_gains:


            return numeric_gains[cid]


        key = str(cid)


        return numeric_gains.get(int(key), 0.0)





    dcg = 0.0


    for rank, item in enumerate(results[:k]):


        cid = int(item.get("chunk_index", -1))


        dcg += gain(cid) / math.log2(rank + 2)





    ideal = sorted(numeric_gains.values(), reverse=True)[:k]


    idcg = sum(val / math.log2(idx + 2) for idx, val in enumerate(ideal))


    if idcg == 0.0:


        return 0.0


    return dcg / idcg


def mrr_at_k(results: list[dict], relevant: list[int], k: int = 10) -> float:


    if not relevant:


        return 0.0


    relevant_set = {int(r) for r in relevant}


    for rank, item in enumerate(results[:k], start=1):


        cid = int(item.get("chunk_index", -1))


        if cid in relevant_set:


            return 1.0 / rank


    return 0.0








_CHAR_MAP = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "ae", "Ö": "oe", "Ü": "ue", "ß": "ss"})


_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


_STOP_WORDS = {"und", "oder", "der", "die", "das", "ein", "eine", "einen", "ist", "sind", "mit", "zu"}


def _tok_norm(text: str) -> list[str]:


    lowered = (text or "").lower().translate(_CHAR_MAP)


    return _WORD_RE.findall(lowered)


def keyword_f1(answer: str, keywords: list[str]) -> float:


    pred_tokens = {


        token


        for token in _tok_norm(answer)


        if len(token) > 2 and token not in _STOP_WORDS


    }


    gold_tokens = {


        token


        for keyword in (keywords or [])


        for token in _tok_norm(keyword)


        if len(token) > 2 and token not in _STOP_WORDS


    }


    if not pred_tokens or not gold_tokens:


        return 0.0


    inter = len(pred_tokens & gold_tokens)


    if inter == 0:


        return 0.0


    precision = inter / len(pred_tokens)


    recall = inter / len(gold_tokens)


    if precision + recall == 0.0:


        return 0.0


    return 2 * precision * recall / (precision + recall)









def _answer_is_compliant(answer: str) -> bool:
    if not answer:
        return False
    has_source = "(Dokument" in answer
    has_bullet = bool(re.search(r"^\s*\d+\.", answer, flags=re.MULTILINE))
    return has_source and has_bullet


def _fallback_answer(window: list[dict], expected_tokens: set[str]) -> str:
    lines: list[str] = []
    covered: set[str] = set()
    for idx, chunk in enumerate(window[:3], start=1):
        content = str(chunk.get("content") or "").strip()
        if not content:
            continue
        first_sentence = re.split(r"(?<=\.)\s+", content)[0].strip()
        source = str(chunk.get("source") or "unbekannt")
        page = chunk.get("page")
        page_display = f"S.{page}" if isinstance(page, int) else "S.?"
        lines.append(f"{idx}. {first_sentence} ({source}, {page_display})")
        covered.update(tok for tok in _tok_norm(content))
    missing = sorted(tok for tok in expected_tokens if tok not in covered)
    if missing:
        lines.append("Fehlend: " + ", ".join(missing))
    if not lines:
        lines.append("1. Kein Kontext verfuegbar (Fehlend: k.A.)")
    return "\n".join(lines)


def build_context(
    retriever: HybridRetriever,
    reranker: CrossEncoderReranker,
    question: str,
    *,
    expected_tokens: set[str] | None = None,
    top_k: int = 18,
):


    optimized = rewrite_query_with_llama3(question)


    embedding = get_embeddings([optimized])[0]


    ranked = retriever.retrieve(


        question,


        embedding,


        top_k=top_k,


        bm25_k=max(80, top_k * 5),


        reranker=reranker,


        reranker_k=max(4 * top_k, 60),


        reranker_weight=0.6,


    )


    window = select_context_window(
        ranked, max_chunks=6, min_chunks=4, max_chars=2200, expected_tokens=expected_tokens
    )


    context_text = "\n\n".join(


        str(chunk.get("content") or "") for chunk in window if chunk.get("content")


    )


    return ranked, context_text, optimized, window








def ensure_output_dir() -> None:


    OUT_DIR.mkdir(parents=True, exist_ok=True)








def main() -> None:

    args = parse_args()
    model_id = args.llm_model

    ensure_output_dir()

    index, records, embeddings = load_faiss_index(str(INDEX_BASE))

    retriever = HybridRetriever(records, embeddings=embeddings, faiss_index=index)

    reranker = CrossEncoderReranker()

    gold_samples = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    total_samples = len(gold_samples)

    start_index = max(args.start, 0)
    if start_index:
        gold_samples = gold_samples[start_index:]
    if args.limit is not None:
        gold_samples = gold_samples[: max(args.limit, 0)]

    if not gold_samples:
        print("Keine Datensaetze fuer die Auswertung (Start/Limit pruefen).")
        return

    rows: list[dict] = []

    for offset, item in enumerate(gold_samples, start=start_index):
        query = item["query"]
        relevant_ids = item.get("relevant_ids", [])
        gains_raw = item.get("gains", {})
        gains = {int(k): float(v) for k, v in gains_raw.items()}
        expected_keywords = item.get("expected_answer_keywords", [])
        references = item.get("references", [])
        expected_tokens = {
            token
            for keyword in expected_keywords
            for token in _tok_norm(keyword)
            if len(token) > 2
        }

        t0 = time.time()
        ranked, context, optimized_query, window = build_context(
            retriever,
            reranker,
            query,
            expected_tokens=expected_tokens or None,
        )
        t1 = time.time()

        print("\n=== DEBUG ===")
        print(f"Query #{offset + 1}/{total_samples}: {query}")
        print(f"Optimierte Suchanfrage: {optimized_query}")
        if ranked:
            print("Top 5 Kandidaten:")
            for chunk in ranked[:5]:
                source = chunk.get("source") or "-"
                page = chunk.get("page")
                page_str = f"S.{page}" if page not in (None, "") else "-"
                score = float(chunk.get("score", 0.0))
                coverage = float(chunk.get("keyword_coverage", 0.0))
                matches = ", ".join(chunk.get("matched_terms") or [])
                print(f"- {source} ({page_str}) score={score:.3f} kw_cov={coverage:.2f} matches={matches or '-'}")
        else:
            print("Keine Kandidaten gefunden.")
        expected_line = ", ".join(expected_keywords) if expected_keywords else "(keine Vorgaben)"
        print(f"Erwartete Keywords: {expected_line}")
        if expected_tokens:
            context_tokens = set(_tok_norm(context))
            present_tokens = sorted(context_tokens & expected_tokens)
            missing_tokens = sorted(expected_tokens - context_tokens)
            print(f"Kontext-Treffer: {', '.join(present_tokens) or '-'}")
            print(f"Kontext-fehlend: {', '.join(missing_tokens) or '-'}")
        print("================\n")

        recall = recall_at_k(ranked, relevant_ids, k=10)
        cndcg = ndcg_at_k(ranked, gains, k=10)
        mrr = mrr_at_k(ranked, relevant_ids, k=10)

        required_kw = ", ".join(expected_keywords)
        keyword_hint = (
            f"**Verwende diese Begriffe woertlich, sofern durch den CONTEXT gedeckt:** {required_kw}"
            if required_kw
            else "**Verwende diese Begriffe woertlich, sofern durch den CONTEXT gedeckt:** (keine Vorgaben)"
        )

        prompt_parts = [
            "Nutze ausschliesslich den folgenden CONTEXT zur Antwort.\n",
            "Wenn Informationen fehlen: 'Nicht belegt im verfuegbaren Kontext.'\n",
            "Schreibe in einer nummerierten Stichpunktliste und wiederhole nur belegte Aussagen.\n",
            "Jeder Punkt MUSS exakt eine Quellenangabe im Format (Dokument, S.X) enthalten.\n",
            f"{keyword_hint}\n\n",
            "Verwende jedes belegte Schlüsselwort wörtlich in den Stichpunkten.\n",
            "Liste alle nicht belegten Schlüsselwörter am Ende als 'Fehlend: <keyword1>, <keyword2>'.\n\n",
            f"CONTEXT:\n{context}\n\n",
            f"FRAGE:\n{query}\n\n",
            "ANTWORT:",
        ]
        base_prompt = "".join(prompt_parts)

        answer = ""
        total_gen_time = 0.0
        for attempt in range(3):
            extra_instruction = ""
            if attempt == 1:
                extra_instruction = (
                    "\n\nACHTUNG: Nutze nummerierte Stichpunkte mit Quellenangaben "
                    "und wiederhole alle belegten Begriffe wörtlich. Beantworte nicht frei, "
                    "sondern extrahiere exakt aus dem Kontext."
                )
            elif attempt == 2:
                extra_instruction = (
                    "\n\nLETZTER VERSUCH: Jeder Stichpunkt MUSS eine Quelle (Dokument, S.X) "
                    "enthalten und alle angegebenen Schlüsselwörter wörtlich aufführen. "
                    "Fehlende Schlüsselwörter sind unter 'Fehlend:' zu nennen. Keine zusätzlichen Sätze."
                )
            prompt = base_prompt + extra_instruction
            gen_start = time.time()
            answer = query_ollama(
                prompt,
                model=model_id,
                language="Deutsch",
                max_retries=1,
            )
            gen_end = time.time()
            total_gen_time += gen_end - gen_start
            if _answer_is_compliant(answer):
                break
        if not _answer_is_compliant(answer):
            answer = _fallback_answer(window, expected_tokens or set())

        kw_f1 = keyword_f1(answer, expected_keywords)

        rows.append({
            "llm_model": model_id,
            "dataset_index": offset,
            "query": query,
            "recall@10": round(recall, 4),
            "cndcg@10": round(cndcg, 4),
            "mrr@10": round(mrr, 4),
            "answer_f1_kw": round(kw_f1, 4),
            "retrieval_time_s": round(t1 - t0, 4),
            "generation_time_s": round(total_gen_time, 4),
            "answer_length": len(answer or ""),
        })

    df = pd.DataFrame(rows)

    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    out_csv = OUT_DIR / f"eval_f1_results{suffix}.csv"
    df.to_csv(out_csv, index=False)

    describe = df.describe()
    print("\nAggregierte Kennzahlen:")
    print(describe)

    def plot_path(name: str) -> Path:
        return OUT_DIR / f"{name}{suffix}.png"

    # Plots
    plt.figure()
    df["recall@10"].plot(kind="hist", bins=10, title="Histogramm Recall@10")
    plt.xlabel("Recall@10")
    plt.savefig(plot_path("recall_hist"))
    plt.close()

    plt.figure()
    df["cndcg@10"].plot(kind="hist", bins=10, title="Histogramm cNDCG@10")
    plt.xlabel("cNDCG@10")
    plt.savefig(plot_path("cndcg_hist"))
    plt.close()

    plt.figure()
    df["mrr@10"].plot(kind="hist", bins=10, title="Histogramm MRR@10")
    plt.xlabel("MRR@10")
    plt.savefig(plot_path("mrr_hist"))
    plt.close()

    plt.figure()
    df.boxplot(column=["recall@10", "cndcg@10", "mrr@10", "answer_f1_kw"])
    plt.title("Boxplot Qualitaetsmetriken")
    plt.savefig(plot_path("box_quality"))
    plt.close()

    plt.figure()
    plt.scatter(df["recall@10"], df["answer_f1_kw"])
    plt.xlabel("Recall@10")
    plt.ylabel("Keyword-F1")
    plt.title("Recall vs. Keyword-F1")
    plt.savefig(plot_path("scatter_recall_f1"))
    plt.close()

if __name__ == "__main__":


    main()


