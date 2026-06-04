#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
HOT = ROOT / "hotpotqa"
CHEM_ARTICLES = ROOT / "graph/data/chemistry_bfs_depth5/pruned_hybrid/final_chemistry_articles.jsonl"
SPLIT_OUTPUTS = {
    "train": HOT / "mteb_train_matches.json",
    "dev": HOT / "mteb_dev_matches.json",
    "test": HOT / "mteb_test_matches.json",
}
MATCHES = HOT / "matches.jsonl"
SUMMARY = HOT / "mteb_filter_summary.json"


def normalize_title(title: str) -> str:
    return str(title).replace("_", " ").strip()


def load_chemistry_titles(path: Path) -> set[str]:
    titles: set[str] = set()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            title = obj.get("article_display_title") or obj.get("article_title") or ""
            title = normalize_title(title)
            if title:
                titles.add(title)
    return titles


def load_mteb() -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, list[str]]]]:
    queries_ds = cast(Any, load_dataset("mteb/hotpotqa", "queries", trust_remote_code=True))
    queries: dict[str, Any] = {str(row["_id"]): dict(row) for split in queries_ds.values() for row in split}

    corpus_ds = cast(Any, load_dataset("mteb/hotpotqa", "corpus", trust_remote_code=True))
    corpus: dict[str, Any] = {str(row["_id"]): dict(row) for split in corpus_ds.values() for row in split}

    qrels_ds = cast(Any, load_dataset("mteb/hotpotqa", "default", trust_remote_code=True))
    qrels_by_split: dict[str, dict[str, list[str]]] = {}
    for split_name, split_rows in qrels_ds.items():
        grouped: dict[str, list[str]] = defaultdict(list)
        for row in split_rows:
            qid = row.get("query-id") or row.get("query_id")
            doc_id = row.get("corpus-id") or row.get("corpus_id")
            if qid and doc_id:
                grouped[str(qid)].append(str(doc_id))
        qrels_by_split[str(split_name)] = dict(grouped)
    return queries, corpus, qrels_by_split


def split_matches(split_name: str, qrels: dict[str, list[str]], queries: dict[str, Any], corpus: dict[str, Any], chem_titles: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for qid, doc_ids in qrels.items():
        matching_titles: list[str] = []
        for doc_id in doc_ids:
            title = corpus.get(doc_id, {}).get("title", "")
            if normalize_title(title) in chem_titles and title not in matching_titles:
                matching_titles.append(title)
        if matching_titles:
            rows.append({
                "id": qid,
                "question": queries.get(qid, {}).get("text", ""),
                "split": split_name,
                "matching_corpus_titles": matching_titles,
                "relevant_doc_ids": doc_ids,
            })
    rows.sort(key=lambda r: str(r["id"]))
    return rows


def main() -> int:
    HOT.mkdir(parents=True, exist_ok=True)
    chem_titles = load_chemistry_titles(CHEM_ARTICLES)
    queries, corpus, qrels_by_split = load_mteb()

    summary: dict[str, Any] = {
        "chemistry_article_count": len(chem_titles),
        "source_dataset": "mteb/hotpotqa",
        "filter_signal": "qrel-linked corpus document titles only",
        "splits": {},
    }
    all_rows: list[dict[str, Any]] = []
    for split_name, qrels in qrels_by_split.items():
        rows = split_matches(split_name, qrels, queries, corpus, chem_titles)
        SPLIT_OUTPUTS[split_name].write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        all_rows.extend(rows)
        summary["splits"][split_name] = {
            "total_queries": len(qrels),
            "matches": len(rows),
            "fraction": len(rows) / len(qrels) if qrels else 0.0,
        }

    all_rows.sort(key=lambda r: str(r["id"]))
    with MATCHES.open("w", encoding="utf-8") as out:
        for row in all_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary["unique_matches"] = len({row["id"] for row in all_rows})
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
