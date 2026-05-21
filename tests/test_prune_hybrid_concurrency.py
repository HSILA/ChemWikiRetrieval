from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from graph import prune_hybrid as ph


def test_threadsafe_append_jsonl_preserves_all_rows_under_concurrent_writes(tmp_path):
    path = tmp_path / "decisions.jsonl"
    lock = ph.threading.Lock()

    def write_one(i: int) -> None:
        ph.append_jsonl_threadsafe(path, {"i": i, "payload": "x" * 50}, lock)

    with ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(write_one, range(200)))

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 200
    assert sorted(row["i"] for row in rows) == list(range(200))


def test_normalize_usage_preserves_prompt_completion_and_reasoning_tokens():
    usage = {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "completion_tokens_details": {"reasoning_tokens": 3},
    }

    normalized = ph.normalize_usage(usage)

    assert normalized["prompt_tokens"] == 11
    assert normalized["completion_tokens"] == 7
    assert normalized["total_tokens"] == 18
    assert normalized["reasoning_tokens"] == 3
    assert normalized["raw_usage"] == usage
