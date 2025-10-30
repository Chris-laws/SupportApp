import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

from app.chatbot.modules.embeddings import load_faiss_index


def ascii_clean(text: str, collapse_spaces: bool = True) -> str:
    replacements = {
        "\u00df": "ss",
        "\u2013": "-",
        "\u2014": "-",
        "\u201c": "\"",
        "\u201d": "\"",
        "\u201e": "\"",
    }
    text = str(text)
    for src, tgt in replacements.items():
        text = text.replace(src, tgt)
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    if collapse_spaces:
        ascii_text = " ".join(ascii_text.split())
    return ascii_text


def ascii_keyword(text: str) -> str:
    return ascii_clean(text).lower()


def dedupe(seq: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def main() -> None:
    gold_path = Path("data/eval/questions_gold.json")
    curated_path = Path("app/chatbot/evaluation/questions_curated20.jsonl")
    extra_path = Path("data/eval/questions.json")

    gold_entries = json.loads(gold_path.read_text(encoding="utf-8"))
    gold_by_id: Dict[str, dict] = {entry["question_id"]: entry for entry in gold_entries}
    question_to_id = {entry["question"]: entry["question_id"] for entry in gold_entries}
    max_id = max(int(entry["question_id"][1:]) for entry in gold_entries if entry["question_id"].startswith("G"))

    extras = json.loads(extra_path.read_text(encoding="utf-8"))
    _, records, _ = load_faiss_index("app/chatbot/data/faiss_index/index")
    page_chunk_map: Dict[Tuple[str, int], List[Tuple[int, int]]] = {}
    for rec in records:
        idx = rec.get("chunk_index")
        if idx is None:
            continue
        doc_name = rec.get("doc") or Path(rec.get("source") or "").name
        page = int(rec.get("page", 0))
        doc_ascii = ascii_clean(doc_name)
        order = int(rec.get("chunk_idx_in_page", 0))
        page_chunk_map.setdefault((doc_ascii, page), []).append((order, int(idx)))

    for key, items in page_chunk_map.items():
        items.sort()

    curated_entries: Dict[str, dict] = {}
    if curated_path.exists():
        with curated_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    payload = json.loads(line)
                    curated_entries[payload["id"]] = payload

    for extra in extras:
        ascii_question = ascii_clean(extra.get("question", ""))
        if not ascii_question:
            continue
        existing_id = question_to_id.get(ascii_question)
        if existing_id:
            qid = existing_id
        else:
            max_id += 1
            qid = f"G{max_id:03d}"

        doc_refs = []
        seen_doc_pages = set()
        for src in extra.get("ground_truth", {}).get("relevant_sources", []):
            if "#page=" not in src:
                continue
            doc_name, page_str = src.split("#page=", 1)
            doc_name = ascii_clean(Path(doc_name).name)
            try:
                page = max(int(page_str) - 1, 0)
            except ValueError:
                continue
            key = (doc_name, page)
            if key in seen_doc_pages:
                continue
            seen_doc_pages.add(key)
            doc_refs.append({"doc": doc_name, "page": page})

        if not doc_refs:
            original_indices = extra.get("ground_truth", {}).get("relevant_chunk_indices", [])
            for idx in original_indices:
                meta_key = next((k for k in page_chunk_map if any(item[1] == int(idx) for item in page_chunk_map[k])), None)
                if meta_key:
                    doc_refs.append({"doc": meta_key[0], "page": meta_key[1]})

        chunk_indices: List[int] = []
        for ref in doc_refs:
            key = (ref["doc"], ref["page"])
            for _, idx in page_chunk_map.get(key, [])[:2]:
                if idx not in chunk_indices:
                    chunk_indices.append(idx)
        if not chunk_indices:
            chunk_indices = [int(idx) for idx in extra.get("ground_truth", {}).get("relevant_chunk_indices", [])]

        reference_answers = [ascii_clean(ans) for ans in extra.get("ground_truth", {}).get("reference_answers", [])]
        reference_answers = [ans for ans in reference_answers if ans]
        expected_keywords = [ascii_keyword(kw) for kw in extra.get("ground_truth", {}).get("expected_keywords", [])]
        expected_keywords = [kw for kw in expected_keywords if kw]
        expected_keywords = dedupe(expected_keywords)

        entry = {
            "question_id": qid,
            "question": ascii_question,
            "ground_truth": {
                "relevant_chunk_indices": chunk_indices,
                "reference_answers": reference_answers,
                "expected_keywords": expected_keywords,
            },
        }
        gold_by_id[qid] = entry
        question_to_id[ascii_question] = qid

        curated_entries[qid] = {
            "id": qid,
            "query": ascii_question,
            "relevant": doc_refs,
            "expected_keywords": expected_keywords,
        }

    updated_gold = sorted(gold_by_id.values(), key=lambda item: item["question_id"])
    gold_path.write_text(json.dumps(updated_gold, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with curated_path.open("w", encoding="utf-8") as fh:
        for qid in sorted(curated_entries):
            fh.write(json.dumps(curated_entries[qid], ensure_ascii=False))
            fh.write("\n")

    print(f"Dataset now contains {len(updated_gold)} questions up to {updated_gold[-1]['question_id']}.")


if __name__ == "__main__":
    main()
