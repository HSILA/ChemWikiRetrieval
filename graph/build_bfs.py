#!/usr/bin/env python3
"""Bounded Wikipedia category BFS extraction into JSONL files.

This is intentionally not a permanent graph database. It streams the raw
Wikimedia SQL dumps depth-by-depth and writes small, inspectable JSONL artifacts
for a seed category such as Category:Chemistry.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import quote

ARTICLE_NS = 0
CATEGORY_NS = 14


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_category_title(title: str) -> str:
    title = title.strip()
    if title.startswith("Category:"):
        title = title[len("Category:") :]
    return title.replace(" ", "_")


def display_title(title: str) -> str:
    return title.replace("_", " ")


def wiki_url(title: str, namespace: int) -> str:
    if namespace == CATEGORY_NS:
        path = "Category:" + title
    else:
        path = title
    # Keep underscores/readable punctuation where safe; percent-encode unicode.
    return "https://en.wikipedia.org/wiki/" + quote(path, safe="/:_()")


def json_dump(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "at", encoding="utf-8") as f:
        for row in rows:
            f.write(json_dump(row) + "\n")
            count += 1
    return count


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def mysql_unescape(ch: str) -> str:
    return {
        "0": "\0",
        "'": "'",
        '"': '"',
        "b": "\b",
        "n": "\n",
        "r": "\r",
        "t": "\t",
        "Z": "\x1a",
        "\\": "\\",
        "%": "%",
        "_": "_",
    }.get(ch, ch)


def mysql_unescape_byte(b: int) -> bytes:
    return {
        ord("0"): b"\0",
        ord("'"): b"'",
        ord('"'): b'"',
        ord("b"): b"\b",
        ord("n"): b"\n",
        ord("r"): b"\r",
        ord("t"): b"\t",
        ord("Z"): b"\x1a",
        ord("\\"): b"\\",
        ord("%"): b"%",
        ord("_"): b"_",
    }.get(b, bytes([b]))


def mysql_escape_for_match(value: str) -> bytes:
    """Return MySQL-escaped bytes for matching an already-quoted SQL string body."""
    raw = value.encode("utf-8")
    out = bytearray()
    for b in raw:
        if b == 0:
            out.extend(b"\\0")
        elif b == ord("\\"):
            out.extend(b"\\\\")
        elif b == ord("'"):
            out.extend(b"\\'")
        elif b == ord('"'):
            out.extend(b'\\"')
        elif b == ord("\n"):
            out.extend(b"\\n")
        elif b == ord("\r"):
            out.extend(b"\\r")
        elif b == ord("\t"):
            out.extend(b"\\t")
        else:
            out.append(b)
    return bytes(out)


def find_sql_tuple_end(data: bytes, start: int) -> int:
    """Find closing ')' for a SQL tuple starting at data[start] == '(' ."""
    in_string = False
    escape = False
    i = start + 1
    while i < len(data):
        b = data[i]
        if in_string:
            if escape:
                escape = False
            elif b == ord("\\"):
                escape = True
            elif b == ord("'"):
                in_string = False
        else:
            if b == ord("'"):
                in_string = True
            elif b == ord(")"):
                return i
        i += 1
    raise ValueError("Unterminated SQL tuple")


def parse_sql_tuple_values_bytes(data: bytes, start: int, end: int) -> List[Any]:
    """Parse a single SQL tuple from bytes[start:end+1]."""
    row: List[Any] = []
    field = bytearray()
    in_string = False
    escape = False
    quoted_field = False

    def finish_field() -> Any:
        raw = bytes(field)
        if quoted_field:
            return raw.decode("utf-8", errors="replace")
        token = raw.decode("ascii", errors="replace").strip()
        if token == "NULL":
            return None
        return token

    for b in data[start + 1 : end]:
        if in_string:
            if escape:
                field.extend(mysql_unescape_byte(b))
                escape = False
            elif b == ord("\\"):
                escape = True
            elif b == ord("'"):
                in_string = False
            else:
                field.append(b)
            continue
        if b == ord("'"):
            in_string = True
            quoted_field = True
            continue
        if b == ord(","):
            row.append(finish_field())
            field = bytearray()
            quoted_field = False
            continue
        field.append(b)
    row.append(finish_field())
    return row


CATEGORYLINKS_ROW_RE = re.compile(rb"\((\d+),'((?:\\.|[^'\\])*)',")
PAGE_ROW_START_RE = re.compile(rb"(?:^|,)\((\d+),")


def convert_unquoted_token(token: str) -> Any:
    token = token.strip()
    if token == "NULL":
        return None
    if token == "":
        return ""
    # Keep parsing conservative. The caller casts fields it needs.
    return token


def iter_sql_insert_rows(path: Path, table_name: str) -> Iterator[List[Any]]:
    """Yield rows from `INSERT INTO `table_name` VALUES ...` SQL dump lines.

    This parser only needs MySQL value-list syntax, not full SQL. It handles
    quoted strings and backslash escapes, and streams row-by-row from each line.
    """

    prefix = f"INSERT INTO `{table_name}` VALUES "
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.startswith(prefix):
                continue
            s = line[len(prefix) :].rstrip("\n")
            if s.endswith(";"):
                s = s[:-1]

            row: Optional[List[Any]] = None
            field: List[str] = []
            in_string = False
            escape = False
            quoted_field = False

            def finish_field() -> Any:
                val = "".join(field)
                return val if quoted_field else convert_unquoted_token(val)

            for ch in s:
                if in_string:
                    if escape:
                        field.append(mysql_unescape(ch))
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == "'":
                        in_string = False
                    else:
                        field.append(ch)
                    continue

                if ch == "'":
                    in_string = True
                    quoted_field = True
                    continue
                if ch == "(":
                    row = []
                    field = []
                    quoted_field = False
                    continue
                if ch == ",":
                    if row is not None:
                        row.append(finish_field())
                        field = []
                        quoted_field = False
                    continue
                if ch == ")":
                    if row is None:
                        continue
                    row.append(finish_field())
                    yield row
                    row = None
                    field = []
                    quoted_field = False
                    continue

                if row is not None:
                    field.append(ch)

            if in_string or row is not None:
                raise ValueError(f"Unfinished SQL row while parsing {path} line {line_no}")


def iter_category_metadata(category_sql: Path) -> Iterator[Tuple[str, Dict[str, int]]]:
    for row in iter_sql_insert_rows(category_sql, "category"):
        # (cat_id, cat_title, cat_pages, cat_subcats, cat_files)
        if len(row) < 5:
            continue
        title = str(row[1])
        cat_pages = int(row[2])
        cat_subcats = int(row[3])
        cat_files = int(row[4])
        yield title, {
            "cat_pages": cat_pages,
            "cat_subcats": cat_subcats,
            "cat_files": cat_files,
            "cat_article_members_estimate": cat_pages - cat_subcats - cat_files,
        }


def resolve_page_ids(page_sql: Path, wanted_ids: set[int]) -> Dict[int, Dict[str, Any]]:
    """Resolve selected page IDs by scanning page.sql.gz with a fast row-start regex.

    We only parse rows whose page_id is wanted, avoiding full SQL tuple parsing for
    every Wikipedia page.
    """
    if not wanted_ids:
        return {}
    remaining = set(wanted_ids)
    resolved: Dict[int, Dict[str, Any]] = {}
    prefix = b"INSERT INTO `page` VALUES "
    with gzip.open(page_sql, "rb") as f:
        for raw_line in f:
            if not raw_line.startswith(prefix):
                continue
            data = raw_line[len(prefix) :].rstrip(b"\n")
            if data.endswith(b";"):
                data = data[:-1]
            for match in PAGE_ROW_START_RE.finditer(data):
                page_id = int(match.group(1))
                if page_id not in remaining:
                    continue
                row_start = data.find(b"(", match.start(), match.end())
                if row_start < 0:
                    continue
                row_end = find_sql_tuple_end(data, row_start)
                row = parse_sql_tuple_values_bytes(data, row_start, row_end)
                if len(row) < 12:
                    continue
                namespace = int(row[1])
                title = str(row[2])
                is_redirect = bool(int(row[5]))
                page_len = int(row[11])
                resolved[page_id] = {
                    "page_id": page_id,
                    "namespace": namespace,
                    "title": title,
                    "display_title": display_title(title),
                    "url": wiki_url(title, namespace),
                    "is_redirect": is_redirect,
                    "page_len": page_len,
                }
                remaining.remove(page_id)
                if not remaining:
                    return resolved
    return resolved


def collect_members_for_frontier(
    categorylinks_sql: Path,
    frontier: set[str],
) -> Tuple[Dict[int, List[Tuple[str, str]]], Counter]:
    """Return cl_from -> [(parent_category, cl_type), ...] for frontier categories.

    Fast path: regex only captures the first two fields of each categorylinks row
    (cl_from and cl_to). We only parse enough tail to read cl_type for matching
    frontier categories.
    """
    by_source_id: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
    counts: Counter = Counter()
    if not frontier:
        return by_source_id, counts

    escaped_to_title = {mysql_escape_for_match(title): title for title in frontier}
    prefix = b"INSERT INTO `categorylinks` VALUES "
    with gzip.open(categorylinks_sql, "rb") as f:
        for raw_line in f:
            if not raw_line.startswith(prefix):
                continue
            data = raw_line[len(prefix) :].rstrip(b"\n")
            if data.endswith(b";"):
                data = data[:-1]
            for match in CATEGORYLINKS_ROW_RE.finditer(data):
                parent_title = escaped_to_title.get(match.group(2))
                if parent_title is None:
                    continue
                cl_from = int(match.group(1))
                row_end = find_sql_tuple_end(data, match.start())
                tail = data[max(match.start(), row_end - 16) : row_end]
                if tail.endswith(b",'page'"):
                    cl_type = "page"
                elif tail.endswith(b",'subcat'"):
                    cl_type = "subcat"
                elif tail.endswith(b",'file'"):
                    counts["ignored_file"] += 1
                    continue
                else:
                    # Fallback parse for unusual rows.
                    row = parse_sql_tuple_values_bytes(data, match.start(), row_end)
                    if len(row) < 7:
                        continue
                    cl_type = str(row[6])
                    if cl_type not in {"page", "subcat"}:
                        counts[f"ignored_{cl_type}"] += 1
                        continue
                by_source_id[cl_from].append((parent_title, cl_type))
                counts[cl_type] += 1
    return by_source_id, counts


def load_metadata_for_categories(category_sql: Path, wanted_titles: set[str]) -> Dict[str, Dict[str, int]]:
    wanted = set(wanted_titles)
    out: Dict[str, Dict[str, int]] = {}
    if not wanted:
        return out
    for title, meta in iter_category_metadata(category_sql):
        if title in wanted:
            out[title] = meta
            wanted.remove(title)
            if not wanted:
                break
    return out


def build_seed_record(seed: str, meta: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "category_title": seed,
        "category_display_title": display_title(seed),
        "category_url": wiki_url(seed, CATEGORY_NS),
        "depth": 0,
        "first_seen_parent_category": None,
        "first_seen_parent_url": None,
        "path": [seed],
        "path_urls": [wiki_url(seed, CATEGORY_NS)],
    }
    record.update(meta.get(seed, {}))
    return record


def run_bfs(args: argparse.Namespace) -> None:
    graph_dir = Path(args.graph_dir).resolve()
    dump_dir = Path(args.dump_dir)
    if not dump_dir.is_absolute():
        dump_dir = graph_dir / dump_dir
    page_sql = dump_dir / args.page_sql
    categorylinks_sql = dump_dir / args.categorylinks_sql
    category_sql = dump_dir / args.category_sql
    out_dir = Path(args.output_dir).resolve() if args.output_dir else graph_dir / "data" / f"{normalize_category_title(args.seed).lower()}_bfs_depth{args.max_depth}"
    frontiers_dir = out_dir / "frontiers"

    for required in [page_sql, categorylinks_sql, category_sql]:
        if not required.exists():
            raise FileNotFoundError(required)

    if out_dir.exists() and args.reset:
        for child in out_dir.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir() and child.name == "frontiers":
                shutil.rmtree(child)
    existing_jsonl = list(out_dir.glob("*.jsonl")) if out_dir.exists() else []
    if existing_jsonl and not args.reset and not args.append:
        raise FileExistsError(f"Output dir already has JSONL files: {out_dir}. Use --reset or --append.")
    out_dir.mkdir(parents=True, exist_ok=True)
    frontiers_dir.mkdir(parents=True, exist_ok=True)

    seed = normalize_category_title(args.seed)
    started_at = utc_now()

    paths: Dict[str, List[str]] = {seed: [seed]}
    seen_categories: set[str] = {seed}
    seen_articles: set[str] = set()
    category_records: Dict[str, Dict[str, Any]] = {}
    article_records: Dict[str, Dict[str, Any]] = {}
    depth_summaries: List[Dict[str, Any]] = []

    seed_meta = load_metadata_for_categories(category_sql, {seed})
    seed_record = build_seed_record(seed, seed_meta)
    category_records[seed] = seed_record
    append_jsonl(out_dir / "categories.jsonl", [seed_record])
    append_jsonl(frontiers_dir / "frontier_depth_0.jsonl", [seed_record])

    frontier = {seed}

    completed_depth = 0
    for depth in range(1, args.max_depth + 1):
        completed_depth = depth
        input_frontier_size = len(frontier)
        t0 = time.time()
        print(f"[{utc_now()}] depth={depth} frontier_categories={input_frontier_size}", file=sys.stderr, flush=True)
        by_source_id, link_counts = collect_members_for_frontier(categorylinks_sql, frontier)
        resolved = resolve_page_ids(page_sql, set(by_source_id.keys()))

        new_categories: Dict[str, Dict[str, Any]] = {}
        new_articles: Dict[str, Dict[str, Any]] = {}
        category_edges: List[Dict[str, Any]] = []
        article_members: List[Dict[str, Any]] = []
        unresolved_count = 0
        skipped_redirects = 0
        skipped_namespace = 0

        for page_id, parent_links in by_source_id.items():
            page = resolved.get(page_id)
            if page is None:
                unresolved_count += len(parent_links)
                continue
            if page["is_redirect"] and not args.include_redirects:
                skipped_redirects += len(parent_links)
                continue

            for parent_category, cl_type in parent_links:
                parent_path = paths.get(parent_category, [parent_category])
                parent_url = wiki_url(parent_category, CATEGORY_NS)

                if cl_type == "subcat":
                    if page["namespace"] != CATEGORY_NS:
                        skipped_namespace += 1
                        continue
                    child = str(page["title"])
                    child_path = parent_path + [child]
                    edge = {
                        "parent_category": parent_category,
                        "parent_category_display_title": display_title(parent_category),
                        "parent_category_url": parent_url,
                        "child_category": child,
                        "child_category_display_title": display_title(child),
                        "child_category_url": wiki_url(child, CATEGORY_NS),
                        "child_page_id": page_id,
                        "depth": depth,
                        "path": child_path,
                        "path_urls": [wiki_url(t, CATEGORY_NS) for t in child_path],
                    }
                    category_edges.append(edge)
                    if child not in seen_categories and child not in new_categories:
                        record = {
                            "category_title": child,
                            "category_display_title": display_title(child),
                            "category_url": wiki_url(child, CATEGORY_NS),
                            "page_id": page_id,
                            "depth": depth,
                            "first_seen_parent_category": parent_category,
                            "first_seen_parent_url": parent_url,
                            "path": child_path,
                            "path_urls": [wiki_url(t, CATEGORY_NS) for t in child_path],
                            "is_redirect": page["is_redirect"],
                            "page_len": page["page_len"],
                        }
                        new_categories[child] = record
                        paths[child] = child_path

                elif cl_type == "page":
                    if page["namespace"] != ARTICLE_NS:
                        skipped_namespace += 1
                        continue
                    article = str(page["title"])
                    article_path = parent_path + [article]
                    member = {
                        "parent_category": parent_category,
                        "parent_category_display_title": display_title(parent_category),
                        "parent_category_url": parent_url,
                        "article_title": article,
                        "article_display_title": display_title(article),
                        "article_url": wiki_url(article, ARTICLE_NS),
                        "page_id": page_id,
                        "depth": depth,
                        "path": article_path,
                        "path_urls": [wiki_url(t, CATEGORY_NS) for t in parent_path] + [wiki_url(article, ARTICLE_NS)],
                        "page_len": page["page_len"],
                        "is_redirect": page["is_redirect"],
                    }
                    article_members.append(member)
                    if article not in seen_articles and article not in new_articles:
                        new_articles[article] = {
                            "article_title": article,
                            "article_display_title": display_title(article),
                            "article_url": wiki_url(article, ARTICLE_NS),
                            "page_id": page_id,
                            "depth": depth,
                            "first_seen_parent_category": parent_category,
                            "first_seen_parent_url": parent_url,
                            "path": article_path,
                            "path_urls": [wiki_url(t, CATEGORY_NS) for t in parent_path] + [wiki_url(article, ARTICLE_NS)],
                            "page_len": page["page_len"],
                            "is_redirect": page["is_redirect"],
                        }

        # Add aggregate category metadata after we know the new categories.
        new_meta = load_metadata_for_categories(category_sql, set(new_categories.keys()))
        for title, meta in new_meta.items():
            new_categories[title].update(meta)

        append_jsonl(out_dir / "category_edges.jsonl", category_edges)
        append_jsonl(out_dir / "article_members.jsonl.gz", article_members)
        append_jsonl(out_dir / "categories.jsonl", new_categories.values())
        append_jsonl(out_dir / "articles.jsonl", new_articles.values())
        append_jsonl(frontiers_dir / f"frontier_depth_{depth}.jsonl", new_categories.values())

        seen_categories.update(new_categories.keys())
        seen_articles.update(new_articles.keys())
        category_records.update(new_categories)
        article_records.update(new_articles)
        frontier = set(new_categories.keys())

        summary = {
            "event": "depth_completed",
            "depth": depth,
            "started_at": started_at,
            "completed_at": utc_now(),
            "seconds": round(time.time() - t0, 2),
            "input_frontier_categories": input_frontier_size,
            "raw_link_counts": dict(link_counts),
            "resolved_source_ids": len(resolved),
            "new_categories": len(new_categories),
            "new_articles": len(new_articles),
            "category_edges_written": len(category_edges),
            "article_members_written": len(article_members),
            "seen_categories_total": len(seen_categories),
            "seen_articles_total": len(seen_articles),
            "unresolved_memberships": unresolved_count,
            "skipped_redirect_memberships": skipped_redirects,
            "skipped_namespace_memberships": skipped_namespace,
        }
        depth_summaries.append(summary)
        append_jsonl(out_dir / "progress.jsonl", [summary])
        write_json(
            out_dir / "summary.json",
            {
                "seed": seed,
                "max_depth": args.max_depth,
                "include_redirects": args.include_redirects,
                "started_at": started_at,
                "last_updated_at": utc_now(),
                "completed_depth": depth,
                "output_dir": str(out_dir),
                "files": {
                    "categories": "categories.jsonl",
                    "articles": "articles.jsonl",
                    "category_edges": "category_edges.jsonl",
                    "article_members": "article_members.jsonl.gz",
                    "frontiers": "frontiers/frontier_depth_<n>.jsonl",
                    "progress": "progress.jsonl",
                },
                "counts": {
                    "seen_categories_total": len(seen_categories),
                    "seen_articles_total": len(seen_articles),
                    "last_frontier_categories": len(frontier),
                },
                "depths": depth_summaries,
            },
        )
        print(
            f"[{utc_now()}] depth={depth} done new_categories={len(new_categories)} new_articles={len(new_articles)} seconds={summary['seconds']}",
            file=sys.stderr,
            flush=True,
        )

        if not frontier:
            break

    write_json(
        out_dir / "summary.json",
        {
            "seed": seed,
            "max_depth": args.max_depth,
            "include_redirects": args.include_redirects,
            "started_at": started_at,
            "last_updated_at": utc_now(),
            "completed_depth": completed_depth,
            "output_dir": str(out_dir),
            "files": {
                "categories": "categories.jsonl",
                "articles": "articles.jsonl",
                "category_edges": "category_edges.jsonl",
                "article_members": "article_members.jsonl.gz",
                "frontiers": "frontiers/frontier_depth_<n>.jsonl",
                "progress": "progress.jsonl",
            },
            "counts": {
                "seen_categories_total": len(seen_categories),
                "seen_articles_total": len(seen_articles),
                "last_frontier_categories": len(frontier),
            },
            "depths": depth_summaries,
        },
    )
    print(f"Output: {out_dir}", file=sys.stderr)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded category BFS from Wikimedia SQL dumps into JSONL.")
    parser.add_argument("--graph-dir", default="graph", help="Graph workspace directory. Outputs default under graph/data/.")
    parser.add_argument("--dump-dir", default="dumps", help="Directory containing SQL dump .gz files. Relative paths resolve under --graph-dir.")
    parser.add_argument("--page-sql", default="enwiki-20171001-page.sql.gz")
    parser.add_argument("--categorylinks-sql", default="enwiki-20171001-categorylinks.sql.gz")
    parser.add_argument("--category-sql", default="enwiki-20171001-category.sql.gz")
    parser.add_argument("--seed", default="Chemistry", help="Seed category title, with or without Category: prefix.")
    parser.add_argument("--max-depth", type=int, default=5, help="BFS depth to extract. Depth 0 is the seed category.")
    parser.add_argument("--output-dir", default=None, help="Default: graph/data/<seed>_bfs_depth<max-depth>")
    parser.add_argument("--include-redirects", action="store_true", help="Include redirect article/category pages. Default skips them.")
    parser.add_argument("--reset", action="store_true", help="Delete existing files in output dir before writing.")
    parser.add_argument("--append", action="store_true", help="Append to existing JSONL files instead of refusing. Use carefully.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_bfs(parse_args())
