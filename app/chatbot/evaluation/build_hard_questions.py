from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence


def load_jsonl(path: Path) -> List[dict]:
    entries: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def write_jsonl(path: Path, entries: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False))
            handle.write("\n")


def extract_expected(entry: dict) -> List[str]:
    if entry.get("expected_keywords"):
        return list(entry["expected_keywords"])
    ground_truth = entry.get("ground_truth") or {}
    if ground_truth.get("expected_keywords"):
        return list(ground_truth["expected_keywords"])
    return []


def extract_relevant(entry: dict) -> List[dict]:
    ground_truth = entry.get("ground_truth") or {}
    if ground_truth.get("relevant"):
        return list(ground_truth["relevant"])
    return []


def set_expected(entry: dict, keywords: Sequence[str]) -> None:
    keywords = sorted(dict.fromkeys(str(kw) for kw in keywords if kw))
    if not keywords:
        return
    entry["expected_keywords"] = keywords
    ground_truth = dict(entry.get("ground_truth") or {})
    ground_truth["expected_keywords"] = keywords
    entry["ground_truth"] = ground_truth


def update_relevant(entry: dict, relevant_items: Sequence[dict]) -> None:
    if not relevant_items:
        return
    ground_truth = dict(entry.get("ground_truth") or {})
    seen = set()
    deduped: List[dict] = []
    for item in relevant_items:
        doc = item.get("doc") or item.get("document")
        page = item.get("page")
        key = (doc, page)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    if deduped:
        ground_truth["relevant"] = deduped
        entry["ground_truth"] = ground_truth


def make_detail_entry(entry: dict) -> dict:
    detail = copy.deepcopy(entry)
    original_id = entry.get("id") or entry.get("question_id") or "item"
    detail["id"] = f"{original_id}_detail"
    detail["query"] = (
        f"{entry.get('query') or entry.get('question')} "
        "Bitte nenne mindestens zwei konkrete Details und zitiere jede Quelle eindeutig."
    )
    set_expected(detail, extract_expected(entry))
    update_relevant(detail, extract_relevant(entry))
    return detail


def make_combination_entry(entry_a: dict, entry_b: dict) -> dict:
    combined = {
        "id": f"{(entry_a.get('id') or 'A')}__{(entry_b.get('id') or 'B')}_combo",
        "query": (
            f"{entry_a.get('query') or entry_a.get('question')} UND "
            f"{entry_b.get('query') or entry_b.get('question')}."
            " Beantworte beide Teile getrennt, mit klaren Quellenangaben pro Stichpunkt."
        ),
    }
    expected = extract_expected(entry_a) + extract_expected(entry_b)
    if expected:
        set_expected(combined, expected)
    relevant = extract_relevant(entry_a) + extract_relevant(entry_b)
    if relevant:
        update_relevant(combined, relevant)
    return combined


def build_hard_dataset(entries: List[dict], target_size: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    hard_entries: List[dict] = []

    # Detail-oriented variants
    for entry in entries:
        hard_entries.append(make_detail_entry(entry))

    # Combination questions (pairwise)
    for idx in range(0, len(entries) - 1, 2):
        hard_entries.append(make_combination_entry(entries[idx], entries[idx + 1]))

    rng.shuffle(hard_entries)
    if len(hard_entries) > target_size:
        hard_entries = hard_entries[:target_size]
    return hard_entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Erzeuge anspruchsvollere Testfragen aus vorhandenen Datensaetzen.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("app/chatbot/evaluation/questions_curated100.jsonl"),
        help="Quelle im JSONL-Format",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("app/chatbot/evaluation/questions_curated100_hard.jsonl"),
        help="Zieldatei für die erzeugten Fragen",
    )
    parser.add_argument("--size", type=int, default=100, help="Zielanzahl der Fragen")
    parser.add_argument("--seed", type=int, default=42, help="Zufalls-Seed für Mischung")
    args = parser.parse_args()

    entries = load_jsonl(args.input)
    if not entries:
        raise SystemExit(f"Keine Daten in {args.input} gefunden.")

    hard_entries = build_hard_dataset(entries, target_size=args.size, seed=args.seed)
    write_jsonl(args.output, hard_entries)
    print(f"{len(hard_entries)} anspruchsvollere Fragen gespeichert in {args.output}")


if __name__ == "__main__":
    main()
