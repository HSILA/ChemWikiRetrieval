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
INPUT = ROOT / "nq/matches.jsonl"
OUTPUT = ROOT / "nq/topic_filter_gemma4.jsonl"
LOCK = OUTPUT.with_suffix(OUTPUT.suffix + ".lock")
ENV_FILE = Path.home() / ".hermes" / ".env"
MODEL = "gemma4:31b"
WORKERS = 16
NUM_PREDICT = 1024
TIMEOUT = 180
MAX_RETRIES = 3
OLLAMA_HOST = "https://ollama.com"

SYSTEM_PROMPT = """Return only JSON with exactly two keys: chemistry and reason.
Classify by topic/domain only. Ignore whether the question is factually correct, well-formed, answerable, useful, exam-like, or phrased as a false statement.
chemistry=true if the question or Wikipedia title is about a chemistry-domain entity or concept: chemistry, biochemistry, molecular biology, materials chemistry, metallurgy, geochemistry, environmental chemistry, compounds, elements, ions, reactions, molecular properties, chemical processes, spectroscopy, molecular/material structure, enzymatic digestion, or metabolism.
Keep chemistry-domain questions even when they ask about history, naming, location, source, use, or impact of a chemical substance, material, process, or concept.
chemistry=false only if the topic is mainly outside chemistry, such as pure geography, etymology of non-chemical terms, general physics, hydrology, medicine without molecular/chemical focus, agriculture, statistics, or anatomy.
If a question is wrong but is about a chemistry concept, chemistry=true.
reason must be one short sentence about the topic only."""


def api_key() -> str:
    if os.environ.get("OLLAMA_API_KEY"):
        return os.environ["OLLAMA_API_KEY"]
    m = re.search(r"^\s*(?:export\s+)?OLLAMA_API_KEY\s*=\s*(.*)\s*$", ENV_FILE.read_text(encoding="utf-8", errors="ignore"), re.M)
    if not m:
        raise RuntimeError("OLLAMA_API_KEY not found")
    return m.group(1).strip().strip("'\"")


def short_answers(row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for item in (row.get("annotations") or {}).get("short_answers") or []:
        if isinstance(item, dict):
            text = item.get("text") or []
            if isinstance(text, list):
                out.extend(str(x) for x in text if x)
            elif text:
                out.append(str(text))
    return out


def source_rows():
    with INPUT.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                yield line_no, json.loads(line)


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
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Classify this NQ row by topic only.\n\nQuestion: " + str(row.get("question") or "") + "\nWikipedia title: " + str(row.get("document_title") or "") + "\nShort answers: " + json.dumps(short_answers(row), ensure_ascii=False)},
        ],
        "stream": False,
        "think": "low",
        "options": {"temperature": 0, "num_predict": NUM_PREDICT},
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "ChemWikiRetrieval-gemma4-full/1.0"}
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


def task(item: tuple[int, dict[str, Any]], key: str) -> dict[str, Any]:
    line_no, row = item
    obj = {
        "id": str(row.get("id") or ""),
        "line_no": line_no,
        "question": row.get("question"),
        "document_title": row.get("document_title"),
        "model": MODEL,
    }
    try:
        obj.update(classify(row, key))
        obj["error"] = None
    except Exception as exc:
        obj.update({"chemistry": None, "reason": "", "model_reasoning": "", "error": f"{type(exc).__name__}: {exc}"})
    return obj


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    key = api_key()
    with LOCK.open("w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"another run is active: {LOCK}", file=sys.stderr)
            return 3
        already = done_ids()
        tasks = [(line_no, row) for line_no, row in source_rows() if str(row.get("id") or "") not in already]
        print(f"output={OUTPUT} already_done={len(already)} pending={len(tasks)} workers={WORKERS} model={MODEL} think=low num_predict={NUM_PREDICT}", flush=True)
        started = time.time()
        with OUTPUT.open("a", encoding="utf-8") as out:
            with futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
                future_map = {ex.submit(task, item, key): item for item in tasks}
                for n, fut in enumerate(futures.as_completed(future_map), 1):
                    out.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
                    out.flush()
                    os.fsync(out.fileno())
                    if n % 100 == 0 or n == len(tasks):
                        elapsed = time.time() - started
                        print(f"completed={n}/{len(tasks)} rate={n / elapsed:.2f}/s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
