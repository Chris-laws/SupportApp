from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # ensure numeric types for new columns
    numeric_cols = [
        "recall@k",
        "ndcg@k",
        "mrr",
        "keyword_f1",
        "time_s",
        "tokens_in",
        "tokens_out",
        "factual_correct",
        "bm25_in_topk",
        "dense_in_topk",
        "overlap_in_topk",
        "rerank_delta_top1",
        "avg_score_delta_topk",
        "kendall_tau_topk",
        "format_bullet_keyword",
        "format_citation_per_bullet",
        "contains_nicht_im_kontext",
        "bullet_count",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def basic_summary(df: pd.DataFrame) -> dict:
    summary = {
        "recall_mean": df["recall@k"].mean(),
        "ndcg_mean": df["ndcg@k"].mean(),
        "mrr_mean": df["mrr"].mean(),
        "keyword_f1_mean": df["keyword_f1"].mean(),
        "factual_rate": df["factual_correct"].mean(),
        "time_mean": df["time_s"].mean(),
        "tokens_in_mean": df["tokens_in"].mean(),
        "tokens_out_mean": df["tokens_out"].mean(),
        "format_bullet_keyword_pct": df["format_bullet_keyword"].mean(),
        "format_citation_per_bullet_pct": df["format_citation_per_bullet"].mean(),
        "nicht_im_kontext_pct": df["contains_nicht_im_kontext"].mean(),
        "avg_bullet_count": df["bullet_count"].mean(),
        "bm25_in_topk_mean": df["bm25_in_topk"].mean(),
        "dense_in_topk_mean": df["dense_in_topk"].mean(),
        "overlap_in_topk_mean": df["overlap_in_topk"].mean(),
        "avg_rerank_delta_top1": df["rerank_delta_top1"].mean(),
        "avg_score_delta_topk": df["avg_score_delta_topk"].mean(),
        "kendall_tau_topk_mean": df["kendall_tau_topk"].mean(),
    }
    for bucket in ["A", "B", "C", "OK"]:
        col = f"error_{bucket.lower()}_pct"
        summary[col] = (df["error_bucket"] == bucket).mean()
    return summary


def aggregate_runs(run_map: dict[str, Path], output: Path) -> None:
    rows: list[dict] = []
    for label, path in run_map.items():
        df = load_csv(path)
        if df.empty:
            continue
        base = basic_summary(df)
        base.update(
            {
                "label": label,
                "model": df["model"].iloc[0],
                "retrieval_mode": df["retrieval_mode"].iloc[0],
                "top_k": int(df["top_k"].iloc[0]),
                "row_count": len(df),
            }
        )
        rows.append(base)
    if not rows:
        return
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output, index=False, float_format="%.6f")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return float(pd.Series(x).corr(pd.Series(y), method="pearson"))


def correlation_summary(df: pd.DataFrame) -> dict:
    recall = df["recall@k"].to_numpy()
    ndcg = df["ndcg@k"].to_numpy()
    mrr = df["mrr"].to_numpy()
    f1 = df["keyword_f1"].to_numpy()
    factual = df["factual_correct"].to_numpy()
    return {
        "spearman_recall_f1": spearman(recall, f1),
        "spearman_ndcg_f1": spearman(ndcg, f1),
        "spearman_mrr_f1": spearman(mrr, f1),
        "spearman_recall_factual": spearman(recall, factual),
        "spearman_ndcg_factual": spearman(ndcg, factual),
        "spearman_mrr_factual": spearman(mrr, factual),
        "pearson_recall_f1": pearson(recall, f1),
        "pearson_ndcg_f1": pearson(ndcg, f1),
        "pearson_mrr_f1": pearson(mrr, f1),
        "pearson_recall_factual": pearson(recall, factual),
        "pearson_ndcg_factual": pearson(ndcg, factual),
        "pearson_mrr_factual": pearson(mrr, factual),
    }


def export_correlations(run_map: dict[str, Path], output: Path) -> None:
    rows: list[dict] = []
    for label, path in run_map.items():
        df = load_csv(path)
        if df.empty:
            continue
        stats = correlation_summary(df)
        stats.update(
            {
                "label": label,
                "model": df["model"].iloc[0],
                "retrieval_mode": df["retrieval_mode"].iloc[0],
                "top_k": int(df["top_k"].iloc[0]),
            }
        )
        rows.append(stats)
    if not rows:
        return
    pd.DataFrame(rows).to_csv(output, index=False, float_format="%.6f")


def stability_stats(paths: Iterable[Path], metric: str) -> tuple[float, float]:
    values = []
    for path in paths:
        df = load_csv(path)
        if df.empty:
            continue
        if metric == "keyword_f1":
            values.append(df["keyword_f1"].mean())
        elif metric == "factual":
            values.append(df["factual_correct"].mean())
        elif metric == "time":
            values.append(df["time_s"].mean())
    if not values:
        return float("nan"), float("nan")
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def write_stability_report(
    runs: dict[str, list[Path]],
    output: Path,
) -> None:
    records: list[dict] = []
    for label, paths in runs.items():
        mean_f1, std_f1 = stability_stats(paths, "keyword_f1")
        mean_fact, std_fact = stability_stats(paths, "factual")
        mean_time, std_time = stability_stats(paths, "time")
        records.append(
            {
                "label": label,
                "runs": len(paths),
                "keyword_f1_mean": mean_f1,
                "keyword_f1_std": std_f1,
                "factual_mean": mean_fact,
                "factual_std": std_fact,
                "time_mean": mean_time,
                "time_std": std_time,
            }
        )
    if not records:
        return
    pd.DataFrame(records).to_csv(output, index=False, float_format="%.6f")


def export_retrieval_mix(source: Path, output: Path) -> None:
    df = load_csv(source)
    if df.empty:
        return
    cols = [
        "question_id",
        "model",
        "retrieval_mode",
        "top_k",
        "bm25_in_topk",
        "dense_in_topk",
        "overlap_in_topk",
        "rank_first_rel",
        "rerank_delta_top1",
        "avg_score_delta_topk",
        "kendall_tau_topk",
        "keyword_f1",
        "factual_correct",
        "error_bucket",
    ]
    df[cols].to_csv(output, index=False, float_format="%.6f")


if __name__ == "__main__":
    base = Path("app/chatbot/evaluation")
    run_map = {
        "hybrid_k12_ctx8_llama3": base / "results_hybrid_k12_ctx8_llama3_8b.csv",
        "hybrid_k12_ctx8_phi3": base / "results_hybrid_k12_ctx8_phi3.csv",
        "hybrid_k12_ctx8_gemma": base / "results_hybrid_k12_ctx8_gemma.csv",
    }
    aggregate_runs(run_map, base / "summary_runs.csv")
    export_correlations(run_map, base / "summary_run_correlations.csv")

    stability_runs = {
        "hybrid_k12_ctx8_llama3": [
            base / "results_hybrid_k12_ctx8_llama3_8b.csv",
            base / "results_hybrid_k12_ctx8_llama3_8b_run2.csv",
            base / "results_hybrid_k12_ctx8_llama3_8b_run3.csv",
        ],
    }
    write_stability_report(stability_runs, base / "summary_stability.csv")

    export_retrieval_mix(base / "results_hybrid_k12_ctx8_llama3_8b.csv", base / "retrieval_mix_hybrid_k12_ctx8_llama3.csv")
