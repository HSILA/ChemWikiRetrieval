#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures as futures
import fcntl
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HOT = ROOT / "hotpotqa"
INPUT_FILES = [
    HOT / "mteb_train_matches.json",
    HOT / "mteb_dev_matches.json",
    HOT / "mteb_test_matches.json",
]
MATCHES = HOT / "matches.jsonl"
OUTPUT = HOT / "topic_filter_gemma4.jsonl"
SUMMARY = HOT / "topic_filter_gemma4_summary.json"
LOCK = OUTPUT.with_suffix(OUTPUT.suffix + ".lock")
ENV_FILE = Path.home() / ".hermes" / ".env"
MODEL = "gemma4:31b"
WORKERS = 16
NUM_PREDICT = 1024
TIMEOUT = 180
MAX_RETRIES = 3
OLLAMA_HOST = "https://ollama.com"

SYSTEM_PROMPT = """Return only JSON with exactly two keys: chemistry and reason.
Classify by topic/domain only. Ignore whether the question is factually correct, well-formed, answerable, useful, or phrased awkwardly.
chemistry=true if the question or qrel-linked Wikipedia title is about a chemistry-domain entity or concept: chemistry, biochemistry, molecular biology, materials chemistry, metallurgy, geochemistry, environmental chemistry, compounds, elements, ions, reactions, molecular properties, chemical processes, spectroscopy, molecular/material structure, enzymatic digestion, or metabolism.
Keep chemistry-domain questions even when they ask about history, naming, company ownership, location, source, use, or impact of a chemical substance, material, process, or concept.
chemistry=false only if the topic is mainly outside chemistry, such as pure geography, entertainment, biography, politics, general medicine without molecular/chemical focus, general biology without molecular focus, or generic organization/company questions where the chemistry title is incidental.
If a question is wrong but is about a chemistry concept, chemistry=true.
reason must be one short sentence about the topic only."""


def api_key() -> str:
    if os.environ.get("OLLAMA_API_KEY"):
        return os.environ["OLLAMA_API_KEY"]
    text = ENV_FILE.read_text(encoding="utf-8", errors="ignore") if ENV_FILE.exists() else ""
    m = re.search(r"^\s*(?:export\s+)?OLLAMA_API_KEY\s*=\s*(.*)\s*$", text, re.M)
    if not m:
        raise RuntimeError("OLLAMA_API_KEY not found")
    return m.group(1).strip().strip("'\"")


def load_source_rows() -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for path in INPUT_FILES:
        rows = json.loads(path.read_text(encoding="utf-8"))
        split = path.name.replace("mteb_", "").replace("_matches.json", "")
        for row in rows:
            row_id = str(row.get("id") or row.get("_id") or "")
            if not row_id or row_id in by_id:
                continue
            by_id[row_id] = {
                "id": row_id,
                "question": row.get("question"),
                "split": row.get("split") or split,
                "matching_titles": row.get("matching_corpus_titles") or row.get("matching_titles") or [],
                "relevant_doc_ids": row.get("relevant_doc_ids") or [],
            }
    return [by_id[k] for k in sorted(by_id)]


def write_matches(rows: list[dict[str, Any]]) -> None:
    MATCHES.parent.mkdir(parents=True, exist_ok=True)
    with MATCHES.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def source_rows() -> list[dict[str, Any]]:
    rows = load_source_rows()
    write_matches(rows)
    return rows


def done_ids() -> set[str]:
    done: set[str] = set()
    if not OUTPUT.exists():
        return done
    with OUTPUT.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("id") and isinstance(obj.get("chemistry"), bool):
                done.add(str(obj["id"]))
    return done


def parse_json(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start:end + 1])
        raise


def classify(row: dict[str, Any], key: str) -> dict[str, Any]:
    user_prompt = (
        "Classify this MTEB HotpotQA row by topic only.\n\n"
        f"Question: {row.get('question') or ''}\n"
        f"Qrel-linked chemistry-matched titles: {json.dumps(row.get('matching_titles') or [], ensure_ascii=False)}\n"
        f"Split: {row.get('split') or ''}"
    )
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": "low",
        "options": {"temperature": 0, "num_predict": NUM_PREDICT},
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "ChemWikiRetrieval-hotpotqa-gemma4/1.0"}
    last_error = "unknown"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(f"{OLLAMA_HOST}/api/chat", data=body, headers=headers, method="POST")
            started = time.time()
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            latency = time.time() - started
            outer = json.loads(raw)
            msg = outer.get("message") or {}
            content = msg.get("content") or ""
            thinking = msg.get("thinking") or ""
            if not content:
                raise ValueError(f"empty content done_reason={outer.get('done_reason')} eval_count={outer.get('eval_count')} thinking_len={len(thinking)}")
            decision = parse_json(content)
            if not isinstance(decision.get("chemistry"), bool):
                raise ValueError(f"bad chemistry={decision.get('chemistry')!r}")
            reason = str(decision.get("reason") or "").strip()
            if not reason:
                raise ValueError("empty reason")
            return {
                "chemistry": decision["chemistry"],
                "reason": reason,
                "model_reasoning": thinking,
                "latency_s": round(latency, 3),
                "done_reason": outer.get("done_reason"),
                "eval_count": outer.get("eval_count"),
            }
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < MAX_RETRIES:
                time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(last_error)


def task(row: dict[str, Any], key: str) -> dict[str, Any]:
    obj = {
        "id": str(row.get("id") or ""),
        "question": row.get("question"),
        "split": row.get("split"),
        "matching_titles": row.get("matching_titles") or [],
        "relevant_doc_ids": row.get("relevant_doc_ids") or [],
        "model": MODEL,
    }
    try:
        obj.update(classify(row, key))
        obj["error"] = None
    except Exception as exc:
        obj.update({"chemistry": None, "reason": "", "model_reasoning": "", "error": f"{type(exc).__name__}: {exc}"})
    return obj


def latest_results() -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not OUTPUT.exists():
        return latest
    with OUTPUT.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("id"):
                latest[str(obj["id"])] = obj
    return latest


def write_summary(total_rows: int) -> dict[str, Any]:
    latest = latest_results()
    valid = [r for r in latest.values() if isinstance(r.get("chemistry"), bool)]
    errors = [r for r in latest.values() if r.get("chemistry") is None]
    true_rows = [r for r in valid if r.get("chemistry") is True]
    false_rows = [r for r in valid if r.get("chemistry") is False]
    summary = {
        "input_rows": total_rows,
        "processed_valid": len(valid),
        "chemistry_true": len(true_rows),
        "chemistry_false": len(false_rows),
        "errors_latest": len(errors),
        "model": MODEL,
        "workers": WORKERS,
        "output": str(OUTPUT.relative_to(ROOT)),
        "matches": str(MATCHES.relative_to(ROOT)),
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    HOT.mkdir(parents=True, exist_ok=True)
    key = api_key()
    with LOCK.open("w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"another run is active: {LOCK}", file=sys.stderr)
            return 3
        rows = source_rows()
        already = done_ids()
        pending = [row for row in rows if str(row.get("id") or "") not in already]
        print(f"matches={MATCHES} output={OUTPUT} total={len(rows)} already_done={len(already)} pending={len(pending)} workers={WORKERS} model={MODEL} think=low", flush=True)
        started = time.time()
        with OUTPUT.open("a", encoding="utf-8") as out:
            with futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
                future_map = {ex.submit(task, row, key): row for row in pending}
                for n, fut in enumerate(futures.as_completed(future_map), 1):
                    out.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
                    out.flush()
                    os.fsync(out.fileno())
                    if n % 25 == 0 or n == len(pending):
                        elapsed = max(time.time() - started, 0.001)
                        print(f"completed={n}/{len(pending)} rate={n / elapsed:.2f}/s", flush=True)
        summary = write_summary(len(rows))
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
