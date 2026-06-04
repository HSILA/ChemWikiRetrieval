#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "hotpotqa" / "tmp_mteb_broad_chemistry"
CHEM_ARTICLES = ROOT / "graph" / "data" / "chemistry_bfs_depth5" / "pruned_hybrid" / "final_chemistry_articles.jsonl"
ORIG_TRAIN = ROOT / "hotpot_train_v1.1.json"
ORIG_DEV_FULLWIKI = ROOT / "hotpot_dev_fullwiki_v1.json"
ORIG_DEV_DISTRACTOR = ROOT / "hotpot_dev_distractor_v1.json"
MATCHES = ROOT / "hotpotqa" / "matches.jsonl"
DECISIONS = ROOT / "hotpotqa" / "topic_filter_gemma4.jsonl"


def normalize_title(title: Any) -> str:
    return str(title or "").replace("_", " ").strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
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


def load_original_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(row.get("_id")) for row in data if row.get("_id")}


def load_chemistry_titles() -> set[str]:
    titles: set[str] = set()
    with CHEM_ARTICLES.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            title = normalize_title(row.get("article_display_title") or row.get("article_title"))
            if title:
                titles.add(title)
    return titles


def latest_gemma_true_ids() -> set[str]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(DECISIONS):
        if row.get("id"):
            latest[str(row["id"])] = row
    return {qid for qid, row in latest.items() if row.get("chemistry") is True and not row.get("error")}


def match_file_ids() -> set[str]:
    return {str(row["id"]) for row in read_jsonl(MATCHES) if row.get("id")}


def verify_mteb_against_original(queries_ds: Any, qrels_ds: Any) -> dict[str, Any]:
    original_train_ids = load_original_ids(ORIG_TRAIN)
    original_dev_fullwiki_ids = load_original_ids(ORIG_DEV_FULLWIKI)
    original_dev_distractor_ids = load_original_ids(ORIG_DEV_DISTRACTOR)

    mteb_query_ids_by_split: dict[str, set[str]] = {}
    mteb_total_query_ids: set[str] = set()
    for split, rows in queries_ds.items():
        ids: set[str] = set()
        for raw_row in rows:
            row = cast(dict[str, Any], raw_row)
            ids.add(str(row["_id"]))
        mteb_query_ids_by_split[str(split)] = ids
        mteb_total_query_ids.update(ids)

    qrel_qids_by_split: dict[str, set[str]] = defaultdict(set)
    qrel_rows_by_split: dict[str, int] = defaultdict(int)
    for split, rows in qrels_ds.items():
        for raw_row in rows:
            row = cast(dict[str, Any], raw_row)
            qid = row.get("query-id") or row.get("query_id")
            if qid:
                qrel_qids_by_split[str(split)].add(str(qid))
                qrel_rows_by_split[str(split)] += 1

    train_plus_dev = qrel_qids_by_split.get("train", set()) | qrel_qids_by_split.get("dev", set())
    test_ids = qrel_qids_by_split.get("test", set())

    checks = {
        "mteb_total_equals_original_train_plus_one_dev": mteb_total_query_ids == (original_train_ids | original_dev_fullwiki_ids),
        "mteb_qrel_train_plus_dev_equals_original_train": train_plus_dev == original_train_ids,
        "mteb_qrel_test_equals_original_dev_fullwiki": test_ids == original_dev_fullwiki_ids,
        "mteb_qrel_test_equals_original_dev_distractor": test_ids == original_dev_distractor_ids,
        "qrel_query_ids_subset_of_query_ids": all(qids <= mteb_total_query_ids for qids in qrel_qids_by_split.values()),
    }

    return {
        "original_counts": {
            "train": len(original_train_ids),
            "dev_fullwiki": len(original_dev_fullwiki_ids),
            "dev_distractor": len(original_dev_distractor_ids),
            "train_plus_one_dev_unique": len(original_train_ids | original_dev_fullwiki_ids),
        },
        "mteb_query_counts": {split: len(ids) for split, ids in sorted(mteb_query_ids_by_split.items())},
        "mteb_total_queries": len(mteb_total_query_ids),
        "mteb_qrel_counts": dict(sorted(qrel_rows_by_split.items())),
        "mteb_qrel_query_counts": {split: len(ids) for split, ids in sorted(qrel_qids_by_split.items())},
        "checks": checks,
        "verified_roughly_identical": all(checks.values()),
    }


def build_variant(
    name: str,
    selected_qids: set[str],
    queries_by_id: dict[str, dict[str, Any]],
    qrels_by_split: dict[str, list[dict[str, str]]],
    chemistry_titles: set[str],
) -> dict[str, Any]:
    variant_out = OUT / name
    selected_qrels_by_split: dict[str, list[dict[str, str]]] = {"train": [], "dev": [], "test": []}
    selected_queries_by_split: dict[str, list[dict[str, str]]] = {"train": [], "dev": [], "test": []}
    selected_positive_doc_ids: set[str] = set()

    for split, qrels in qrels_by_split.items():
        qids_in_split = sorted({row["query-id"] for row in qrels if row["query-id"] in selected_qids})
        for qid in qids_in_split:
            row = queries_by_id[qid]
            selected_queries_by_split[split].append({"_id": qid, "text": row["text"]})
        for row in qrels:
            if row["query-id"] in selected_qids:
                selected_qrels_by_split[split].append(row)
                selected_positive_doc_ids.add(row["corpus-id"])

    corpus_rows: list[dict[str, str]] = []
    chemistry_title_doc_ids: set[str] = set()
    positive_docs_added_by_force: set[str] = set()
    corpus_ids: set[str] = set()

    corpus_ds = load_dataset("mteb/hotpotqa", "corpus")
    for rows in corpus_ds.values():
        for raw_row in rows:
            row = cast(dict[str, Any], raw_row)
            doc_id = str(row["_id"])
            title = normalize_title(row.get("title"))
            is_chem_doc = title in chemistry_titles
            is_positive_doc = doc_id in selected_positive_doc_ids
            if is_chem_doc or is_positive_doc:
                corpus_rows.append({"_id": doc_id, "title": str(row.get("title") or ""), "text": str(row.get("text") or "")})
                corpus_ids.add(doc_id)
                if is_chem_doc:
                    chemistry_title_doc_ids.add(doc_id)
                if is_positive_doc and not is_chem_doc:
                    positive_docs_added_by_force.add(doc_id)

    corpus_rows.sort(key=lambda row: row["_id"])
    write_jsonl(variant_out / "corpus" / "corpus.jsonl", corpus_rows)
    for split in ("train", "dev", "test"):
        selected_queries_by_split[split].sort(key=lambda row: row["_id"])
        selected_qrels_by_split[split].sort(key=lambda row: (row["query-id"], row["corpus-id"]))
        write_jsonl(variant_out / "queries" / f"{split}.jsonl", selected_queries_by_split[split])
        write_jsonl(variant_out / "qrels" / f"{split}.jsonl", selected_qrels_by_split[split])

    all_qrels = [row for rows in selected_qrels_by_split.values() for row in rows]
    qrel_qids = {row["query-id"] for row in all_qrels}
    qrel_corpus_ids = {row["corpus-id"] for row in all_qrels}
    qrels_per_query = Counter(row["query-id"] for row in all_qrels)
    missing_positive_docs = sorted(qrel_corpus_ids - corpus_ids)
    unreferenced_corpus_docs = corpus_ids - qrel_corpus_ids

    summary = {
        "variant": name,
        "selection": "Gemma-cleaned qrel-title chemistry queries" if name == "gemma_true" else "qrel-title chemistry queries before Gemma cleanup",
        "queries": {split: len(rows) for split, rows in selected_queries_by_split.items()},
        "queries_total": sum(len(rows) for rows in selected_queries_by_split.values()),
        "qrels": {split: len(rows) for split, rows in selected_qrels_by_split.items()},
        "qrels_total": len(all_qrels),
        "qrel_query_ids": len(qrel_qids),
        "qrel_unique_corpus_ids": len(qrel_corpus_ids),
        "qrels_per_query_distribution": dict(sorted(Counter(qrels_per_query.values()).items())),
        "corpus_docs": len(corpus_rows),
        "corpus_docs_from_chemistry_title_filter": len(chemistry_title_doc_ids),
        "positive_docs_added_by_force": len(positive_docs_added_by_force),
        "unreferenced_corpus_docs": len(unreferenced_corpus_docs),
        "missing_positive_docs": len(missing_positive_docs),
        "missing_positive_doc_ids_sample": missing_positive_docs[:20],
        "output_dir": str(variant_out),
    }
    (variant_out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    chemistry_titles = load_chemistry_titles()

    queries_ds = load_dataset("mteb/hotpotqa", "queries")
    qrels_ds = load_dataset("mteb/hotpotqa", "default")
    verification = verify_mteb_against_original(queries_ds, qrels_ds)

    summary: dict[str, Any] = {
        "source_dataset": "mteb/hotpotqa",
        "chemistry_article_count": len(chemistry_titles),
        "verification": verification,
    }

    if not verification["verified_roughly_identical"]:
        summary["status"] = "stopped_before_build_because_mteb_did_not_verify"
        (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2

    queries_by_id: dict[str, dict[str, Any]] = {}
    for split, rows in queries_ds.items():
        for raw_row in rows:
            row = cast(dict[str, Any], raw_row)
            queries_by_id[str(row["_id"])] = {"_id": str(row["_id"]), "text": str(row["text"]), "split": str(split)}

    qrels_by_split: dict[str, list[dict[str, str]]] = {"train": [], "dev": [], "test": []}
    for split, rows in qrels_ds.items():
        split_name = str(split)
        for raw_row in rows:
            row = cast(dict[str, Any], raw_row)
            qid = str(row.get("query-id") or row.get("query_id"))
            doc_id = str(row.get("corpus-id") or row.get("corpus_id"))
            qrels_by_split[split_name].append({"query-id": qid, "corpus-id": doc_id, "score": "1"})

    title_match_ids = match_file_ids()
    gemma_true_ids = latest_gemma_true_ids()
    if not title_match_ids:
        raise RuntimeError(f"No title match IDs found at {MATCHES}")
    if not gemma_true_ids:
        raise RuntimeError(f"No Gemma true IDs found at {DECISIONS}")

    variants = {
        "title_match": build_variant("title_match", title_match_ids, queries_by_id, qrels_by_split, chemistry_titles),
        "gemma_true": build_variant("gemma_true", gemma_true_ids, queries_by_id, qrels_by_split, chemistry_titles),
    }
    summary["status"] = "built"
    summary["variants"] = variants
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
