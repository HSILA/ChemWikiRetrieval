#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
HOT = ROOT / "hotpotqa"
OUT = HOT / "mteb_clean"
DECISIONS = HOT / "topic_filter_gemma4.jsonl"
MATCHES = HOT / "matches.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def latest_clean_ids() -> set[str]:
    latest: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(DECISIONS):
        latest[str(row["id"])] = row
    return {qid for qid, row in latest.items() if row.get("chemistry") is True}


def main() -> int:
    clean_ids = latest_clean_ids()
    matches = {str(row["id"]): row for row in load_jsonl(MATCHES) if str(row["id"]) in clean_ids}

    corpus_ds = cast(Any, load_dataset("mteb/hotpotqa", "corpus"))
    corpus_by_id: dict[str, dict[str, Any]] = {str(row["_id"]): dict(row) for split in corpus_ds.values() for row in split}

    doc_ids = sorted({str(doc_id) for row in matches.values() for doc_id in row.get("relevant_doc_ids", [])})
    corpus_rows = [
        {"_id": doc_id, "title": corpus_by_id[doc_id].get("title", ""), "text": corpus_by_id[doc_id].get("text", "")}
        for doc_id in doc_ids
        if doc_id in corpus_by_id
    ]

    queries_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    qrels_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    for qid in sorted(matches):
        row = matches[qid]
        split = str(row.get("split") or "")
        if split not in queries_by_split:
            continue
        queries_by_split[split].append({"_id": qid, "text": row.get("question", "")})
        for doc_id in row.get("relevant_doc_ids", []):
            qrels_by_split[split].append({"query-id": qid, "corpus-id": str(doc_id), "score": "1"})

    write_jsonl(OUT / "corpus.jsonl", corpus_rows)
    for split, rows in queries_by_split.items():
        write_jsonl(OUT / "queries" / f"{split}.jsonl", rows)
    for split, rows in qrels_by_split.items():
        write_jsonl(OUT / "qrels" / f"{split}.jsonl", rows)

    summary = {
        "source_dataset": "mteb/hotpotqa",
        "chemistry_true_queries": sum(len(v) for v in queries_by_split.values()),
        "corpus_docs": len(corpus_rows),
        "splits": {split: {"queries": len(queries_by_split[split]), "qrels": len(qrels_by_split[split])} for split in queries_by_split},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
