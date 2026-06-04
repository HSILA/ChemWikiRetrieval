#!/usr/bin/env python3
"""Package chem-nq and chem-hotpotqa for Hugging Face upload, matching MTEB layout."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "hf_uploads"

def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def package_nq() -> None:
    src = ROOT / "nq" / "mteb_clean"
    dst = OUT / "chem-nq"
    if dst.exists():
        shutil.rmtree(dst)
    # MTEB uses split subfolders even when only one split exists
    copy(src / "corpus.jsonl", dst / "corpus" / "test" / "corpus.jsonl")
    copy(src / "queries.jsonl", dst / "queries" / "test" / "queries.jsonl")
    copy(src / "qrels" / "test.jsonl", dst / "default" / "test" / "qrels.jsonl")
    summary = json.loads((src / "summary.json").read_text(encoding="utf-8"))
    print("packaged chem-nq")
    return summary

def package_hotpotqa() -> None:
    src = ROOT / "hotpotqa" / "mteb_clean"
    dst = OUT / "chem-hotpotqa"
    if dst.exists():
        shutil.rmtree(dst)
    copy(src / "corpus.jsonl", dst / "corpus" / "test" / "corpus.jsonl")
    for split in ("train", "dev", "test"):
        copy(src / "queries" / f"{split}.jsonl", dst / "queries" / split / "queries.jsonl")
        copy(src / "qrels" / f"{split}.jsonl", dst / "default" / split / "qrels.jsonl")
    summary = json.loads((src / "summary.json").read_text(encoding="utf-8"))
    print("packaged chem-hotpotqa")
    return summary

if __name__ == "__main__":
    package_nq()
    package_hotpotqa()