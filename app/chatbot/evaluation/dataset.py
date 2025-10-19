from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List


def load_questions(path: str | Path) -> List[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_results(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(rows), handle, ensure_ascii=False, indent=2)

