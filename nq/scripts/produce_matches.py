#!/usr/bin/env python3
"""Stream-filter NaturalQuestionsV2 from HF for chemistry matches.

One RecordBatchStreamReader per shard (each gzipped shard is a self-contained
IPC stream). Writes matches as JSONL and checkpoints after every shard.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import tempfile
import time
import urllib.request
import urllib.error
from pathlib import Path
from urllib.parse import unquote

import pyarrow as pa
import pyarrow.ipc as ipc

BASE_URL = "https://huggingface.co/datasets/rongzhangibm/NaturalQuestionsV2/resolve/main"
TRAIN_SHARDS = [f"{BASE_URL}/train/dataset.arrow.{i:02d}.gz" for i in range(20)]
VAL_ARROW = f"{BASE_URL}/validation/dataset.arrow"


class ChainedGzipStream:
    """Read decompressed bytes across the train .gz parts in URL order."""

    def __init__(self, urls: list[str], timeout: int):
        self.urls = urls
        self.timeout = timeout
        self.index = -1
        self.resp = None
        self.gz = None
        self.closed = False
        self._open_next()

    def _open_next(self) -> bool:
        self.close_current()
        self.index += 1
        if self.index >= len(self.urls):
            return False
        req = urllib.request.Request(self.urls[self.index], headers={"User-Agent": "ChemWikiRetrieval-nq-producer"})
        self.resp = urllib.request.urlopen(req, timeout=self.timeout)
        self.gz = gzip.GzipFile(fileobj=self.resp, mode="rb")
        print(f"opened_train_part={self.urls[self.index]}", file=sys.stderr, flush=True)
        return True

    def close_current(self) -> None:
        for obj in (self.gz, self.resp):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass
        self.gz = None
        self.resp = None

    def readable(self):
        return True

    def read(self, size=-1):
        if self.closed:
            return b""
        if size is None or size < 0:
            chunks = []
            while True:
                chunk = self.read(8 * 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks)
        chunks = []
        remaining = size
        while remaining > 0 and self.gz is not None:
            chunk = self.gz.read(remaining)
            if chunk:
                chunks.append(chunk)
                remaining -= len(chunk)
                continue
            if not self._open_next():
                break
        return b"".join(chunks)

    def close(self):
        self.closed = True
        self.close_current()


class ExactReadWrapper:
    """File-like wrapper whose read(n) keeps reading until n bytes or EOF.

    PyArrow's IPC reader expects file.read(n) to return the full requested
    message body when data remains. gzip.GzipFile over an HTTPResponse may
    legally return a short read for very large Arrow message bodies, causing
    errors such as "Expected to be able to read ... bytes ... got ...".
    """

    def __init__(self, raw):
        self.raw = raw

    @property
    def closed(self):
        return getattr(self.raw, "closed", False)

    def readable(self):
        return True

    def read(self, size=-1):
        if size is None or size < 0:
            return self.raw.read(size)
        chunks = []
        remaining = size
        while remaining > 0:
            chunk = self.raw.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def close(self):
        return self.raw.close()


def normalize_title(title: str) -> str:
    return title.replace(" ", "_").strip()


def extract_title_from_url(url: str) -> str:
    """Extract title=... from en.wikipedia URL query string."""
    # e.g. oldid=...&title=Lithium
    from urllib.parse import urlparse, parse_qs
    qs = urlparse(url).query
    title = parse_qs(qs).get("title", [""])[0]
    return unquote(title.replace("_", " ")).strip()


def load_chem_titles(path: str) -> set[str]:
    titles: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw = row.get("article_title") or row.get("title") or ""
            if not raw:
                continue
            raw = str(raw).strip()
            titles.add(raw)
            titles.add(normalize_title(raw))
            titles.add(raw.replace("_", " ").strip())
    return titles


def iter_batches(reader):
    """Yield batches from either Arrow IPC stream or file readers."""
    if hasattr(reader, "num_record_batches"):
        for idx in range(reader.num_record_batches):
            yield reader.get_batch(idx)
        return
    while True:
        try:
            yield reader.read_next_batch()
        except StopIteration:
            return


def open_stream(url: str, timeout: int = 180) -> pa.ipc.RecordBatchStreamReader:
    """Return an Arrow IPC stream reader for an uncompressed stream URL."""
    req = urllib.request.Request(url, headers={"User-Agent": "ChemWikiRetrieval-nq-producer"})
    resp = urllib.request.urlopen(req, timeout=timeout)
    source = pa.PythonFile(ExactReadWrapper(resp), mode="r")
    reader = ipc.RecordBatchStreamReader(source)
    return reader, source, resp


def open_train_reader(url: str, timeout: int = 180):
    """Open the multipart train Arrow stream.

    The train dataset.arrow.00.gz..19.gz files are gzip-compressed byte ranges
    of one Arrow IPC stream; parts 01..19 start in the middle of Arrow message
    bodies and cannot be opened independently.  Start at part 00 and chain all
    decompressed parts in order.
    """
    if url != TRAIN_SHARDS[0]:
        raise OSError("train shards are multipart Arrow stream chunks; restart from shard-start 0")
    chained = ChainedGzipStream(TRAIN_SHARDS, timeout)
    source = pa.PythonFile(ExactReadWrapper(chained), mode="r")
    reader = ipc.RecordBatchStreamReader(source)
    return reader, source, None


def close_stream(reader, source=None, tmp_path=None) -> None:
    try:
        reader.close()
    except Exception:
        pass
    try:
        if source is not None:
            source.close()
    except Exception:
        pass
    try:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)
    except Exception:
        pass


def is_trailing_ipc_footer_error(exc: Exception) -> bool:
    msg = str(exc)
    trailers = (
        "Expected to be able to read",
        "Invalid flatbuffers message",
        "Invalid IPC stream: negative continuation token",
        "Tried reading schema message, was null or length 0",
    )
    return any(token in msg for token in trailers)


def load_existing_output_keys(path: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not path.exists():
        return keys
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except Exception:
                continue
            split = str(row.get("split", ""))
            rec_id = str(row.get("id", ""))
            if split and rec_id:
                keys.add((split, rec_id))
    return keys


def has_annotated_answer(annotations) -> bool:
    """Return True when an NQ row has usable answer evidence."""
    if not isinstance(annotations, dict):
        return False

    for item in annotations.get("long_answer") or []:
        if isinstance(item, dict) and item.get("start_token") not in (-1, None):
            return True

    for item in annotations.get("short_answers") or []:
        if isinstance(item, dict) and item.get("text"):
            return True

    yes_no = annotations.get("yes_no_answer")
    if isinstance(yes_no, list):
        return any(str(x).upper() in {"YES", "NO"} for x in yes_no)
    return str(yes_no).upper() in {"YES", "NO"}


def process_validation(
    chem_titles: set[str],
    out_handle,
    timeout: int = 300,
    existing_keys: set[tuple[str, str]] | None = None,
):
    """Validation is one large .arrow file."""
    req = urllib.request.Request(
        VAL_ARROW,
        headers={"User-Agent": "ChemWikiRetrieval-nq-producer"},
    )
    print(f"fetching_validation={VAL_ARROW}", file=sys.stderr, flush=True)
    resp = urllib.request.urlopen(req, timeout=timeout)
    print("downloading_validation...", file=sys.stderr, flush=True)
    data = resp.read()
    print(f"validation_downloaded_bytes={len(data)}", file=sys.stderr, flush=True)

    reader = ipc.RecordBatchStreamReader(pa.BufferReader(data))
    seen = 0
    matched = 0
    started = time.time()

    try:
        for batch in iter_batches(reader):
            cols = {
                "document": batch.column("document"),
                "question": batch.column("question"),
                "id": batch.column("id"),
                "annotations": batch.column("annotations"),
            }
            for i in range(batch.num_rows):
                seen += 1
                doc = cols["document"][i].as_py()
                title = doc.get("title", "")
                norm = normalize_title(title)
                if title not in chem_titles and norm not in chem_titles:
                    continue
                q = cols["question"][i].as_py()
                ann = cols["annotations"][i].as_py()
                if not has_annotated_answer(ann):
                    continue
                key = ("validation", str(cols["id"][i].as_py()))
                if existing_keys is not None and key in existing_keys:
                    matched += 1
                    continue
                entry = {
                    "id": key[1],
                    "question": q.get("text", "") if isinstance(q, dict) else str(q),
                    "document_title": title,
                    "document_url": doc.get("url", ""),
                    "annotations": ann,
                    "split": "validation",
                }
                out_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                if existing_keys is not None:
                    existing_keys.add(key)
                matched += 1
    except StopIteration:
        pass

    elapsed = time.time() - started
    print(f"validation_done seen={seen} matched={matched} elapsed_sec={elapsed:.1f}", file=sys.stderr, flush=True)
    return seen, matched


def process_train_shard(url: str, chem_titles: set[str], out_handle, timeout: int = 180, progress_every: int = 5000, existing_keys: set[tuple[str, str]] | None = None):
    seen = 0
    matched = 0
    reader = None
    source = None
    tmp_path = None
    started = time.time()

    try:
        reader, source, tmp_path = open_train_reader(url, timeout=timeout)
        for batch in iter_batches(reader):
            cols = {
                "document": batch.column("document"),
                "question": batch.column("question"),
                "id": batch.column("id"),
                "annotations": batch.column("annotations"),
            }
            for i in range(batch.num_rows):
                seen += 1
                doc = cols["document"][i].as_py()
                title = doc.get("title", "")
                norm = normalize_title(title)
                if title not in chem_titles and norm not in chem_titles:
                    continue
                q = cols["question"][i].as_py()
                ann = cols["annotations"][i].as_py()
                if not has_annotated_answer(ann):
                    continue
                key = ("train", str(cols["id"][i].as_py()))
                if existing_keys is not None and key in existing_keys:
                    matched += 1
                    continue
                entry = {
                    "id": key[1],
                    "question": q.get("text", "") if isinstance(q, dict) else str(q),
                    "document_title": title,
                    "document_url": doc.get("url", ""),
                    "annotations": ann,
                    "split": "train",
                }
                out_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                if existing_keys is not None:
                    existing_keys.add(key)
                matched += 1

            if seen % progress_every == 0:
                elapsed = time.time() - started
                print(f"shard_progress url={url} seen={seen} matched={matched} elapsed_sec={elapsed:.1f}", file=sys.stderr, flush=True)
    except StopIteration:
        pass
    except Exception as exc:
        elapsed = time.time() - started
        if seen > 0 and is_trailing_ipc_footer_error(exc):
            print(
                f"shard_trailing_ipc_footer_warning url={url} seen={seen} matched={matched} "
                f"warning={type(exc).__name__}: {exc} elapsed_sec={elapsed:.1f}",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(f"shard_error url={url} error={type(exc).__name__}: {exc} elapsed_sec={elapsed:.1f}", file=sys.stderr, flush=True)
            raise
    finally:
        close_stream(reader, source, tmp_path)

    elapsed = time.time() - started
    print(f"shard_done url={url} seen={seen} matched={matched} elapsed_sec={elapsed:.1f}", file=sys.stderr, flush=True)
    return seen, matched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chem-titles",
        default="graph/data/chemistry_bfs_depth5/pruned_hybrid/final_chemistry_articles.jsonl",
    )
    parser.add_argument("--output", default="nq/matches.jsonl")
    parser.add_argument("--state-dir", default="nq/state")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--shard-end", type=int, default=20)
    args = parser.parse_args(argv)

    chem_titles = load_chem_titles(args.chem_titles)
    print(f"chem_titles_loaded={len(chem_titles)}", file=sys.stderr, flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing_keys = load_existing_output_keys(out_path)
    f_out = open(out_path, "a", encoding="utf-8")

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    done_file = state_dir / "nq_done_shards.json"
    error_log = state_dir / "nq_errors.jsonl"

    done = set()
    if done_file.exists():
        with open(done_file, "r", encoding="utf-8") as fh:
            done = set(json.load(fh))

    total_seen = 0
    total_matched = 0

    # Train shards
    shard_urls = TRAIN_SHARDS[args.shard_start:args.shard_end]
    for url in shard_urls:
        if url in done:
            print(f"skip_done url={url}", file=sys.stderr, flush=True)
            continue
        try:
            seen, matched = process_train_shard(url, chem_titles, f_out, timeout=args.timeout, existing_keys=existing_keys)
        except Exception as exc:
            err_msg = f"error_shard={url} error={type(exc).__name__}: {exc}"
            print(err_msg, file=sys.stderr, flush=True)
            with open(error_log, "a", encoding="utf-8") as eh:
                eh.write(json.dumps({"timestamp": time.strftime("%Y%m%d-%H:%M"), "shard": url, "error": str(exc)}) + "\n")
            f_out.close()
            return 1  # Exit so caller (cron agent) knows to retry

        total_seen += seen
        total_matched += matched
        processed_urls = TRAIN_SHARDS if url == TRAIN_SHARDS[0] else [url]
        done.update(processed_urls)
        with open(done_file, "w", encoding="utf-8") as sf:
            json.dump(list(done), sf)

    # Validation
    if VAL_ARROW not in done:
        try:
            seen, matched = process_validation(chem_titles, f_out, timeout=args.timeout, existing_keys=existing_keys)
        except Exception as exc:
            err_msg = f"error_validation error={type(exc).__name__}: {exc}"
            print(err_msg, file=sys.stderr, flush=True)
            with open(error_log, "a", encoding="utf-8") as eh:
                eh.write(json.dumps({"timestamp": time.strftime("%Y%m%d-%H:%M"), "shard": "validation", "error": str(exc)}) + "\n")
            f_out.close()
            return 1

        total_seen += seen
        total_matched += matched
        done.add(VAL_ARROW)
        with open(done_file, "w", encoding="utf-8") as sf:
            json.dump(list(done), sf)

    f_out.close()
    print(f"DONE total_seen={total_seen} total_matched={total_matched} output={out_path}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
