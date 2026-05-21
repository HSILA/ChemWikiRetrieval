from __future__ import annotations

from graph import build_bfs


def test_existing_output_files_includes_gzipped_jsonl(tmp_path):
    (tmp_path / "article_members.jsonl.gz").write_bytes(b"\x1f\x8b")
    (tmp_path / "categories.jsonl").write_text("{}\n", encoding="utf-8")

    paths = {path.name for path in build_bfs.existing_output_files(tmp_path)}

    assert paths == {"article_members.jsonl.gz", "categories.jsonl"}
