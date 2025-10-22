from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Liste Evaluationsfaelle mit hohem Recall aber niedrigem Keyword-F1."
    )
    parser.add_argument(
        "--results",
        default="data/eval/eval_f1_results.csv",
        help="Pfad zur CSV mit Eval-Ergebnissen.",
    )
    parser.add_argument(
        "--recall-threshold",
        type=float,
        default=0.8,
        help="Mindestrecal laut dem ein Fall als relevant gilt.",
    )
    parser.add_argument(
        "--f1-threshold",
        type=float,
        default=0.2,
        help="Obergrenze fuer Keyword-F1 damit der Fall gelistet wird.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximale Anzahl an Faellen, 0 fuer alle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.results)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    filtered = df[
        (df["recall@10"] >= args.recall_threshold) & (df["answer_f1_kw"] <= args.f1_threshold)
    ].copy()
    filtered = filtered.sort_values(by="answer_f1_kw")

    if args.limit > 0:
        filtered = filtered.head(args.limit)

    if filtered.empty:
        print("Keine Faelle unter den angegebenen Schwellenwerten gefunden.")
        return

    cols = [
        "dataset_index",
        "query",
        "recall@10",
        "cndcg@10",
        "mrr@10",
        "answer_f1_kw",
        "retrieval_time_s",
        "generation_time_s",
    ]
    print(filtered[cols].to_string(index=False))


if __name__ == "__main__":
    main()
