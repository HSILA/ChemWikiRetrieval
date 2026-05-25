#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import sys
import tarfile
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import produce_matches as nq

ROOT = Path(__file__).resolve().parents[2]
MATCHES = ROOT / "nq/matches.jsonl"
DECISIONS = ROOT / "nq/topic_filter_gemma4.jsonl"
OUT_DIR = ROOT / "nq/html_pages"
OUT_ARCHIVE = ROOT / "nq/html_pages.tar.gz"


def safe_part(text: str, max_len: int = 140) -> str:
    text = text.replace("/", "_").replace("\\", "_").strip()
    text = re.sub(r"[^A-Za-z0-9._()\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return (text or "untitled")[:max_len]


def oldid_from_url(url: str) -> str:
    qs = parse_qs(urlparse(url.replace("&amp;", "&")).query)
    oldid = qs.get("oldid", [""])[0]
    return oldid or "no_oldid"


def html_filename(title: str, url: str) -> str:
    return f"{safe_part(title)}__oldid_{safe_part(oldid_from_url(url), 40)}.html"


def load_targets() -> dict[str, dict[str, str]]:
    matches = {}
    with MATCHES.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            matches[str(row["id"])] = row

    targets = {}
    with DECISIONS.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rid = str(row.get("id") or "")
            if row.get("chemistry") is True and rid in matches:
                m = matches[rid]
                targets[rid] = {
                    "split": m["split"],
                    "document_title": m["document_title"],
                    "document_url": m["document_url"],
                }
    return targets


def maybe_write_html(doc: dict, saved_urls: set[str]) -> bool:
    title = doc.get("title") or ""
    url = doc.get("url") or ""
    html = doc.get("html") or ""
    if not url or not html or url in saved_urls:
        return False
    path = OUT_DIR / html_filename(title, url)
    if path.exists():
        saved_urls.add(url)
        return False
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(html, encoding="utf-8")
    os.replace(tmp, path)
    saved_urls.add(url)
    return True


def write_html_archive() -> int:
    html_files = sorted(OUT_DIR.glob("*.html"))
    tmp = OUT_ARCHIVE.with_suffix(OUT_ARCHIVE.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with tarfile.open(tmp, "w:gz") as tar:
        for path in html_files:
            tar.add(path, arcname=Path("html_pages") / path.name)
    os.replace(tmp, OUT_ARCHIVE)
    return len(html_files)


def scan_reader(reader, target_ids: set[str], saved_urls: set[str], split: str) -> tuple[int, int, int]:
    seen = 0
    matched = 0
    written = 0
    for batch in nq.iter_batches(reader):
        cols = {name: batch.column(name) for name in ["id", "document"]}
        for i in range(batch.num_rows):
            seen += 1
            rid = str(cols["id"][i].as_py())
            if rid not in target_ids:
                continue
            matched += 1
            doc = cols["document"][i].as_py()
            if maybe_write_html(doc, saved_urls):
                written += 1
        if seen % 25000 == 0:
            print(f"{split}_progress seen={seen} matched={matched} written={written}", flush=True)
    return seen, matched, written


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = load_targets()
    saved_urls = set()
    for path in OUT_DIR.glob("*.html"):
        # Existing files are accepted, but URL recovery from filename is not exact.
        # This is fine because deterministic target filenames prevent overwrites.
        pass

    by_split = {
        "validation": {rid for rid, row in targets.items() if row["split"] == "validation"},
        "train": {rid for rid, row in targets.items() if row["split"] == "train"},
    }
    print(f"targets={len(targets)} validation={len(by_split['validation'])} train={len(by_split['train'])} out={OUT_DIR}", flush=True)

    total_seen = total_matched = total_written = 0
    started = time.time()

    if by_split["validation"]:
        reader, source, resp = nq.open_stream(nq.VAL_ARROW, timeout=300)
        try:
            seen, matched, written = scan_reader(reader, by_split["validation"], saved_urls, "validation")
            total_seen += seen; total_matched += matched; total_written += written
            print(f"validation_done seen={seen} matched={matched} written={written}", flush=True)
        finally:
            nq.close_stream(reader, source, None)

    if by_split["train"]:
        reader = source = tmp = None
        try:
            reader, source, tmp = nq.open_train_reader(nq.TRAIN_SHARDS[0], timeout=300)
            seen, matched, written = scan_reader(reader, by_split["train"], saved_urls, "train")
            total_seen += seen; total_matched += matched; total_written += written
            print(f"train_done seen={seen} matched={matched} written={written}", flush=True)
        finally:
            nq.close_stream(reader, source, tmp)

    elapsed = time.time() - started
    file_count = sum(1 for _ in OUT_DIR.glob("*.html"))
    archived_count = write_html_archive()
    archive_mb = OUT_ARCHIVE.stat().st_size / (1024 * 1024)
    print(
        f"done matched={total_matched}/{len(targets)} newly_written={total_written} "
        f"html_files={file_count} archived_files={archived_count} "
        f"archive={OUT_ARCHIVE} archive_mb={archive_mb:.1f} elapsed_sec={elapsed:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
