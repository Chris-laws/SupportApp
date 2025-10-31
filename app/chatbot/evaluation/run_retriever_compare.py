from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
EVAL_DIR = BASE_DIR / "app" / "chatbot" / "evaluation"
DATA_DIR = BASE_DIR / "app" / "chatbot" / "data" / "faiss_index"
RESULTS_DIR = EVAL_DIR / "retriever_k10"
DATASET_PATH = EVAL_DIR / "questions_curated20.jsonl"

EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "intfloat/e5-base-v2",
]

CHUNK_STRATEGIES = [
    "fixed_500",
    "semantic_overlap20",
]

RETRIEVER_SETUPS = [
    ("bm25", True),
    ("dense", True),
    ("hybrid", False),
    ("hybrid", True),
]

TOP_K = 10
SAMPLE_SIZE = 100
SEED = 13


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


@dataclass
class RunConfig:
    embedding: str
    chunk: str
    retriever_mode: str
    enable_reranker: bool

    def label(self) -> str:
        rerank_suffix = "reranker" if self.enable_reranker else "noranker"
        return f"{slugify(self.embedding)}__{slugify(self.chunk)}__{self.retriever_mode}__{rerank_suffix}"

    def index_base(self) -> Path:
        return DATA_DIR / f"index_{slugify(self.embedding)}_{slugify(self.chunk)}"

    def output_path(self) -> Path:
        name = f"results_k{TOP_K}_{self.label()}.csv"
        return RESULTS_DIR / name


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_index(config: RunConfig, force: bool = False) -> None:
    index_base = config.index_base()
    marker = Path(str(index_base) + ".index")
    if marker.exists() and not force:
        return
    env = os.environ.copy()
    env["SUPPORTAPP_EMBEDDING_MODEL"] = config.embedding
    env["SUPPORTAPP_CHUNK_STRATEGY"] = config.chunk
    env["SUPPORTAPP_INDEX_BASE"] = str(index_base)
    ensure_directory(index_base.parent)
    main_script = BASE_DIR / "app" / "chatbot" / "main.py"
    cmd = [sys.executable, str(main_script)]
    subprocess.run(cmd, env=env, check=True)


def run_pipeline(config: RunConfig, force: bool = False) -> None:
    output_path = config.output_path()
    if output_path.exists() and not force:
        return
    env = os.environ.copy()
    env["SUPPORTAPP_EMBEDDING_MODEL"] = config.embedding
    env["SUPPORTAPP_CHUNK_STRATEGY"] = config.chunk
    env["SUPPORTAPP_INDEX_BASE"] = str(config.index_base())
    env["SUPPORTAPP_RUN_LABEL"] = config.label()
    ensure_directory(output_path.parent)
    cmd = [
        sys.executable,
        "-m",
        "app.chatbot.evaluation.pipeline_eval",
        "--ground-truth",
        str(DATASET_PATH),
        "--output",
        str(output_path),
        "--models",
        "llama3:8b",
        "--k",
        str(TOP_K),
        "--k-context",
        "8",
        "--candidate-pool",
        "80",
        "--retrieval-mode",
        config.retriever_mode,
        "--index-base",
        str(config.index_base()),
        "--sample-size",
        str(SAMPLE_SIZE),
        "--random-seed",
        str(SEED),
        "--run-label",
        config.label(),
        "--timeout",
        "90",
    ]
    if config.enable_reranker:
        cmd.append("--enable-reranker")
    subprocess.run(cmd, env=env, check=True)


def paired_t_test(x: Iterable[float], y: Iterable[float]) -> Tuple[float, float]:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    mask = ~np.isnan(x_arr) & ~np.isnan(y_arr)
    diffs = x_arr[mask] - y_arr[mask]
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan")
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    if std_diff == 0.0:
        if mean_diff == 0.0:
            return 0.0, 1.0
        t_stat = math.copysign(math.inf, mean_diff)
        return t_stat, 0.0
    t_stat = mean_diff / (std_diff / math.sqrt(n))
    try:
        from scipy import stats  # type: ignore

        p_value = float(stats.t.sf(abs(t_stat), df=n - 1) * 2)
    except Exception:
        p_value = float("nan")
    return t_stat, p_value


def collect_metrics(df: pd.DataFrame) -> Dict[str, float]:
    metrics = {
        "recall_mean": df["recall@k"].mean(),
        "recall_std": df["recall@k"].std(ddof=1),
        "ndcg_mean": df["ndcg@k"].mean(),
        "ndcg_std": df["ndcg@k"].std(ddof=1),
        "mrr_mean": df["mrr"].mean(),
        "mrr_std": df["mrr"].std(ddof=1),
        "keyword_f1_mean": df["keyword_f1"].mean(),
        "keyword_f1_std": df["keyword_f1"].std(ddof=1),
        "factual_mean": df["factual_correct"].mean(),
        "factual_std": df["factual_correct"].std(ddof=1),
        "time_mean": df["time_s"].mean(),
        "time_std": df["time_s"].std(ddof=1),
        "tokens_in_mean": df["tokens_in"].mean(),
        "tokens_out_mean": df["tokens_out"].mean(),
        "bm25_in_topk_mean": df["bm25_in_topk"].mean(),
        "dense_in_topk_mean": df["dense_in_topk"].mean(),
    }
    return metrics


def aggregate_results(result_paths: List[Path]) -> pd.DataFrame:
    combos: Dict[Tuple[str, str, str, int], pd.DataFrame] = {}
    meta_records: List[Dict[str, object]] = []
    for path in result_paths:
        df = pd.read_csv(path)
        if df.empty:
            continue
        embedding = df["embedding_model"].iloc[0]
        chunk_strategy = df["chunk_strategy"].iloc[0]
        retriever_mode = df["retrieval_mode"].iloc[0]
        reranker_flag = int(df["reranker_enabled"].iloc[0])
        key = (embedding, chunk_strategy, retriever_mode, reranker_flag)
        combos[key] = df
    summary_rows: List[Dict[str, object]] = []
    for key, df in combos.items():
        embedding, chunk_strategy, retriever_mode, reranker_flag = key
        metrics = collect_metrics(df)
        row: Dict[str, object] = {
            "embedding_model": embedding,
            "chunk_strategy": chunk_strategy,
            "retrieval_mode": retriever_mode,
            "reranker_enabled": reranker_flag,
            **metrics,
        }
        baseline_key = (embedding, chunk_strategy, "hybrid", 1)
        if key == baseline_key:
            row.update(
                {
                    "delta_keyword_f1": 0.0,
                    "delta_factual": 0.0,
                    "delta_recall": 0.0,
                    "delta_time": 0.0,
                    "t_keyword_f1": 0.0,
                    "p_keyword_f1": 1.0,
                    "t_factual": 0.0,
                    "p_factual": 1.0,
                }
            )
        elif baseline_key in combos:
            baseline_df = combos[baseline_key]
            merged = df.merge(
                baseline_df[["sample_index", "keyword_f1", "factual_correct"]],
                on="sample_index",
                suffixes=("_cur", "_base"),
            )
            row["delta_keyword_f1"] = (
                metrics["keyword_f1_mean"] - baseline_df["keyword_f1"].mean()
            )
            row["delta_factual"] = metrics["factual_mean"] - baseline_df["factual_correct"].mean()
            row["delta_recall"] = metrics["recall_mean"] - baseline_df["recall@k"].mean()
            row["delta_time"] = metrics["time_mean"] - baseline_df["time_s"].mean()
            t_f1, p_f1 = paired_t_test(merged["keyword_f1_cur"], merged["keyword_f1_base"])
            t_fact, p_fact = paired_t_test(
                merged["factual_correct_cur"], merged["factual_correct_base"]
            )
            row["t_keyword_f1"] = t_f1
            row["p_keyword_f1"] = p_f1
            row["t_factual"] = t_fact
            row["p_factual"] = p_fact
        else:
            row.update(
                {
                    "delta_keyword_f1": float("nan"),
                    "delta_factual": float("nan"),
                    "delta_recall": float("nan"),
                    "delta_time": float("nan"),
                    "t_keyword_f1": float("nan"),
                    "p_keyword_f1": float("nan"),
                    "t_factual": float("nan"),
                    "p_factual": float("nan"),
                }
            )
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.sort_values(
        ["embedding_model", "chunk_strategy", "retrieval_mode", "reranker_enabled"],
        inplace=True,
    )
    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Vergleich k=10 Retriever-Konfigurationen")
    parser.add_argument("--force", action="store_true", help="Vorhandene Artefakte ignorieren und neu rechnen")
    parser.add_argument(
        "--skip-evals",
        action="store_true",
        help="Bewertungen ueberspringen und nur bereits erzeugte CSVs aggregieren",
    )
    args = parser.parse_args()

    ensure_directory(RESULTS_DIR)

    configs = [
        RunConfig(embedding=embedding, chunk=chunk, retriever_mode=retriever, enable_reranker=rerank)
        for embedding in EMBEDDING_MODELS
        for chunk in CHUNK_STRATEGIES
        for retriever, rerank in RETRIEVER_SETUPS
    ]

    if not args.skip_evals:
        for config in configs:
            build_index(config, force=args.force)
            run_pipeline(config, force=args.force)

    result_paths = sorted(RESULTS_DIR.glob("results_k*.csv"))
    if not result_paths:
        print("Keine Ergebnisdateien gefunden.", file=sys.stderr)
        sys.exit(1)

    summary_df = aggregate_results(result_paths)
    summary_path = RESULTS_DIR / "summary_retriever_k10.csv"
    summary_df.to_csv(summary_path, index=False)

    json_path = RESULTS_DIR / "summary_retriever_k10.json"
    summary_df.to_json(json_path, orient="records", indent=2)

    print(f"Aggregierte Ergebnisse gespeichert in {summary_path}")


if __name__ == "__main__":
    main()
