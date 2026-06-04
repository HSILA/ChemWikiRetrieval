#!/usr/bin/env python3
"""Extract clean article sections from saved NQ Wikipedia HTML pages.

Merges H3-H6 subsections into their parent H2 section.
Drops: References, Further reading, External links, See also.
Input: nq/html_pages. Output: nq/article_sections.jsonl.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
import shutil
import sys
import time
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
CANONICAL_IN = ROOT / "nq/html_pages"
LEGACY_IN = ROOT / "nq/html_pages"
FALLBACK_IN = ROOT / "nq/html_pages"
OUT = ROOT / "nq/article_sections.jsonl"

STOP_SECTIONS = {
    "references", "notes", "footnotes", "bibliography", "sources", "further reading",
    "external links", "see also", "navigation", "related pages", "works cited",
}
DROP_SECTION_PREFIXES = (
    "references", "notes", "footnotes", "bibliography", "sources", "further reading",
    "external links", "see also",
)
SKIP_TAGS = {"script", "style", "table", "math", "figure", "form", "noscript", "select", "textarea"}
SKIP_CLASS_SUBSTRINGS = (
    "infobox", "navbox", "metadata", "ambox", "mbox", "toc", "thumb", "gallery",
    "reflist", "references", "reference", "hatnote", "dablink", "noprint", "portal",
    "sisterproject", "vertical-navbox", "sidebar", "catlinks", "printfooter",
    "mw-editsection", "mw-jump", "mw-indicators", "collapsible", "nomobile",
)
SKIP_IDS = {
    "toc", "catlinks", "siteNotice", "jump-to-nav", "contentSub", "siteSub",
    "mw-navigation", "mw-head", "mw-panel", "footer", "mw-page-base", "mw-head-base",
}
VOID_ELEMENTS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"
}


def attrs_dict(attrs: list[tuple[str, str | None]]) -> dict[str, str]:
    return {k.lower(): (v or "") for k, v in attrs}


def attr_has(attrs: dict[str, str], needles: Iterable[str]) -> bool:
    hay = " ".join([attrs.get("class", ""), attrs.get("role", ""), attrs.get("id", "")]).lower()
    return any(n in hay for n in needles)


def normalize_heading(text: str) -> str:
    return clean_text(text).strip()


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"</?ref\b[^>]*>", " ", text, flags=re.I)
    text = re.sub(r"\[\s*(?:\d+(?:\s*[,-]\s*\d+)*|note\s+\d+|citation needed|clarification needed|dead link|failed verification)\s*\]", " ", text, flags=re.I)
    text = re.sub(r"\[\s*edit\s*\]", " ", text, flags=re.I)
    text = re.sub(r"\bedit\s*$", "", text, flags=re.I)
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\s+([_^]\{)", r"\1", text)
    text = re.sub(r"([_^]\{)\s+", r"\1", text)
    text = re.sub(r"\s+\}", "}", text)
    text = re.sub(r"\b([A-Za-z])\s+th\b", r"\1th", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def latex_alt_to_text(text: str) -> str:
    text = html.unescape(text or "").strip()
    if not text:
        return ""
    text = re.sub(r"^\{\\displaystyle\s*(.*?)\}$", r"\1", text)
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text or len(text) > 160:
        return ""
    return text


def is_bad_section(path: list[str]) -> bool:
    for item in path:
        low = item.strip().lower()
        if low in STOP_SECTIONS or any(low.startswith(p) for p in DROP_SECTION_PREFIXES):
            return True
    return False





class WikiBlockParser(HTMLParser):
    def __init__(self, article_id: str):
        super().__init__(convert_charrefs=True)
        self.article_id = article_id
        self.in_content = False
        self.content_depth = 0
        self.skip_depth = 0
        self.in_title = False
        self.in_h1 = False
        self.title_parts: list[str] = []
        self.h1_parts: list[str] = []
        self.heading_tag: str | None = None
        self.heading_parts: list[str] = []
        self.path: list[str] = ["Lead"]
        self.section_root: list[str] = ["Lead"]  # H2-level path used for output
        self.block_tag: str | None = None
        self.block_parts: list[str] = []
        self.section_parts: list[str] = []
        self.subsection_names: list[str] = []
        self.blocks: list[dict] = []
        self.block_counter = 0
        self.stopped = False

    def handle_starttag(self, tag: str, attrs):
        tag = tag.lower()
        attrs = attrs_dict(attrs)

        if tag == "title":
            self.in_title = True
        if tag == "h1" and attrs.get("id") == "firstHeading":
            self.in_h1 = True

        if not self.in_content:
            if tag == "div" and "mw-parser-output" in attrs.get("class", "").lower():
                self.in_content = True
                self.content_depth = 1
            return

        is_void = tag in VOID_ELEMENTS
        if not is_void:
            self.content_depth += 1

        if self.stopped:
            return

        if self.skip_depth:
            if not is_void:
                self.skip_depth += 1
            return

        if tag in SKIP_TAGS or attrs.get("id") in SKIP_IDS or attr_has(attrs, SKIP_CLASS_SUBSTRINGS):
            if not is_void:
                self.skip_depth = 1
            return
        if tag == "sup" and ("reference" in attrs.get("class", "").lower() or attrs.get("id", "").startswith("cite_ref")):
            if not is_void:
                self.skip_depth = 1
            return

        if tag == "sub":
            self.add_text(" _{")
            return
        if tag == "sup":
            self.add_text(" ^{")
            return

        if tag == "img" and "mwe-math" in attrs.get("class", "").lower():
            symbol = latex_alt_to_text(attrs.get("alt", ""))
            if symbol:
                self.add_text(f" {symbol} ")
            return

        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            if tag == "h1" and attrs.get("id") == "firstHeading":
                self.in_h1 = True
                return
            self.flush_block()
            self.heading_tag = tag
            self.heading_parts = []
        elif tag in {"p", "li"} and self.block_tag is None and not is_bad_section(self.path):
            self.block_tag = tag
            self.block_parts = []
        elif tag in {"br", "hr"}:
            self.add_text(" ")

    def handle_startendtag(self, tag: str, attrs):
        tag = tag.lower()
        attrs = attrs_dict(attrs)
        if not self.in_content or self.skip_depth or self.stopped:
            return
        if tag == "img" and "mwe-math" in attrs.get("class", "").lower():
            symbol = latex_alt_to_text(attrs.get("alt", ""))
            if symbol:
                self.add_text(f" {symbol} ")
        elif tag in {"br", "hr"}:
            self.add_text(" ")

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag == "title":
            self.in_title = False
        if tag == "h1" and self.in_h1:
            self.in_h1 = False

        if not self.in_content:
            return

        if self.skip_depth:
            self.skip_depth -= 1
            self.content_depth -= 1
            return

        if self.stopped:
            self.content_depth -= 1
            return

        if self.heading_tag == tag:
            heading = normalize_heading(" ".join(self.heading_parts))
            if heading:
                low = heading.lower()
                if low in STOP_SECTIONS or any(low.startswith(p) for p in DROP_SECTION_PREFIXES):
                    self.flush_block()
                    if tag == "h2":
                        # H2 stop section (References, See also, ...) ends the body.
                        self.flush_section()
                        self.stopped = True
                    else:
                        # A stop-prefixed H3-H6 inside a kept section (e.g. an
                        # H3 "Sources of hydrogen" under "Process"): drop only this
                        # subsection's own content via the now-bad path. Do NOT
                        # flush_section, or the parent H2 would split into two rows
                        # with the same section name.
                        self.path = self.path[:1] + [heading]
                else:
                    if tag == "h2":
                        # H2 is the section boundary: flush accumulated text
                        self.flush_section()
                        self.path = [heading]
                        self.section_root = [heading]
                    else:
                        # H3-H6 merge into current section text; keep the
                        # subsection name only as metadata, never inline.
                        if heading not in self.subsection_names:
                            self.subsection_names.append(heading)
                        if tag == "h3":
                            self.path = (self.path[:1] if self.path else []) + [heading]
                        elif tag in {"h4", "h5", "h6"}:
                            self.path = (self.path[:2] if len(self.path) >= 2 else self.path[:1]) + [heading]
            self.heading_tag = None
            self.heading_parts = []
        elif self.block_tag == tag:
            self.flush_block()
        elif tag == "sub":
            self.add_text("} ")
        elif tag == "sup":
            self.add_text("} ")
        elif tag in {"p", "li", "div", "section", "dd", "dt", "br"}:
            self.add_text(" ")

        self.content_depth -= 1
        if self.content_depth <= 0:
            self.in_content = False

    def handle_data(self, data: str):
        if not data:
            return
        if self.in_title:
            self.title_parts.append(data)
        if self.in_h1:
            self.h1_parts.append(data)
        if not self.in_content or self.skip_depth or self.stopped:
            return
        if self.heading_tag:
            self.heading_parts.append(data)
        elif self.block_tag:
            self.block_parts.append(data)

    def add_text(self, text: str):
        if self.heading_tag:
            self.heading_parts.append(text)
        elif self.block_tag:
            self.block_parts.append(text)

    def flush_block(self):
        if not self.block_tag:
            return
        text = clean_text(" ".join(self.block_parts))
        block_type = self.block_tag
        if text and not is_bad_section(self.path):
            low = text.lower()
            junk_bits = (
                "retrieved from", "categories:", "hidden categories:", "wikimedia commons",
                "international standard", "isbn", "authority control", "vte", "doi:",
                "this page was last edited", "privacy policy", "terms of use",
            )
            if not any(j in low for j in junk_bits) and not re.fullmatch(r"[\W\d_]+", text):
                if block_type == "li":
                    self.section_parts.append(f"- {text}")
                else:
                    self.section_parts.append(text)
        self.block_tag = None
        self.block_parts = []

    def flush_section(self):
        text = clean_text(" ".join(self.section_parts))
        if text and not is_bad_section(self.section_root):
            low = text.lower()
            junk_bits = (
                "retrieved from", "categories:", "hidden categories:", "wikimedia commons",
                "international standard", "isbn", "authority control", "vte", "doi:",
                "this page was last edited", "privacy policy", "terms of use",
            )
            if (
                len(text) >= 45
                and not (len(text) < 150 and text.endswith(":") and "- " not in text)
                and not any(j in low for j in junk_bits)
                and not re.fullmatch(r"[\W\d_]+", text)
            ):
                root = list(self.section_root) if self.section_root else ["Lead"]
                section_name = "summary" if (root and root[0] == "Lead") else root[0]
                self.blocks.append({
                    "article_id": self.article_id,
                    "section": section_name,
                    "section_path": root,
                    "subsections": list(self.subsection_names),
                    "block_index": self.block_counter,
                    "block_type": "section",
                    "text": text,
                })
                self.block_counter += 1
        self.section_parts = []
        self.subsection_names = []

    def close(self):
        self.flush_block()
        self.flush_section()
        super().close()

    @property
    def title(self) -> str:
        h1 = clean_text(" ".join(self.h1_parts))
        if h1:
            return h1
        raw = clean_text(" ".join(self.title_parts))
        raw = re.sub(r"\s+-\s+Wikipedia\s*$", "", raw)
        return raw


def parse_filename(path: Path) -> tuple[str, str, str]:
    article_id = path.stem
    if "__oldid_" in article_id:
        title_part, oldid = article_id.rsplit("__oldid_", 1)
    else:
        title_part, oldid = article_id, ""
    return article_id, title_part.replace("_", " "), oldid


def dedup_latest_by_title(files: Iterable[Path]) -> list[Path]:
    """Keep one snapshot per article title (the highest/latest oldid).

    Multiple oldid snapshots of the same article must not each contribute a
    full set of sections to the corpus, so the title contributes exactly one
    set of sections, taken from its most recent revision.
    """
    best: dict[str, tuple[int, str, Path]] = {}
    for p in files:
        stem = p.stem
        if "__oldid_" in stem:
            title_part, oldid = stem.rsplit("__oldid_", 1)
        else:
            title_part, oldid = stem, ""
        try:
            oldid_val = int(oldid)
        except ValueError:
            oldid_val = -1
        cur = best.get(title_part)
        if cur is None or (oldid_val, str(p)) > (cur[0], cur[1]):
            best[title_part] = (oldid_val, str(p), p)
    return [best[t][2] for t in sorted(best)]


def extract_file(path: Path) -> list[dict]:
    article_id, title_guess, oldid = parse_filename(path)
    parser = WikiBlockParser(article_id)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), ""):
            parser.feed(chunk)
    parser.close()
    title = parser.title or title_guess
    article_key = article_id.split("__oldid_")[0]
    rows = []
    for block in parser.blocks:
        row = {
            "id": f"{article_id}::b{block['block_index']:04d}",
            "article_id": article_id,
            "article_key": article_key,
            "title": title,
            "oldid": oldid,
            "section": block["section"],
            "section_path": block["section_path"],
            "subsections": block["subsections"],
            "block_index": block["block_index"],
            "block_type": block["block_type"],
            "text": block["text"],
        }
        rows.append(row)
    return rows


def choose_input_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    for candidate in (CANONICAL_IN, LEGACY_IN, FALLBACK_IN):
        if candidate.exists() and any(candidate.glob("*.html")):
            return candidate
    return CANONICAL_IN


def backup_existing(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = path.with_suffix(path.suffix + f".{stamp}.bak")
        shutil.copy2(path, bak)
        print(f"backed_up_existing={bak}", flush=True)


def validate_output(path: Path) -> dict:
    stats = {
        "rows": 0, "unique_ids": 0, "article_ids": 0, "empty_text": 0, "bad_fragments": 0,
        "reference_like": 0, "external_like": 0, "min_len": None, "max_len": 0,
    }
    ids = set(); article_ids = set()
    bad_re = re.compile(r"<\s*/?\s*(?:a|table|tr|td|th|div|span|p|li|ul|ol|ref)\b[^>]*>|href=|(?-i:\bmw-[a-z])|\[\s*edit\s*\]", re.I)
    ref_re = re.compile(r"^(references|external links|see also|navigation)\b", re.I)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            row = json.loads(line)
            stats["rows"] += 1
            ids.add(row["id"]); article_ids.add(row["article_id"])
            text = row.get("text", "")
            if not text:
                stats["empty_text"] += 1
            if bad_re.search(text):
                stats["bad_fragments"] += 1
            if any(ref_re.search(str(x)) for x in row.get("section_path", [])):
                stats["reference_like"] += 1
            if ref_re.search(text):
                stats["external_like"] += 1
            stats["min_len"] = len(text) if stats["min_len"] is None else min(stats["min_len"], len(text))
            stats["max_len"] = max(stats["max_len"], len(text))
    stats["unique_ids"] = len(ids)
    stats["article_ids"] = len(article_ids)
    stats["duplicate_ids"] = stats["rows"] - stats["unique_ids"]
    return stats


def completed_article_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            article_id = row.get("article_id")
            if article_id:
                done.add(str(article_id))
    return done


def print_samples(input_dir: Path, names: list[str], random_count: int = 0) -> None:
    deduped = dedup_latest_by_title(input_dir.glob("*.html"))
    by_title = {p.stem.split("__oldid_")[0]: p for p in deduped}
    files = [by_title[n] for n in names if n in by_title]
    if random_count and deduped:
        rng = random.Random(20260524)
        files.extend(rng.sample(deduped, min(random_count, len(deduped))))
    seen = set()
    for path in files:
        if path in seen:
            continue
        seen.add(path)
        rows = extract_file(path)
        print(f"\nSAMPLE file={path.name} blocks={len(rows)}", flush=True)
        for row in rows[:8]:
            print(json.dumps({k: row[k] for k in ["id", "title", "section", "subsections", "text"]}, ensure_ascii=False)[:1200], flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir")
    ap.add_argument("--output", default=str(OUT))
    ap.add_argument("--limit", type=int, default=0, help="limit files for canary")
    ap.add_argument("--resume", action="store_true", help="append missing articles, deriving progress from output JSONL")
    ap.add_argument("--max-seconds", type=float, default=0, help="stop cleanly after this many seconds")
    ap.add_argument("--sample", action="store_true", help="print sample extracted blocks instead of writing corpus")
    ap.add_argument("--validate", action="store_true", help="validate existing output and exit")
    args = ap.parse_args(argv)

    input_dir = choose_input_dir(args.input_dir)
    out = Path(args.output)
    if args.validate:
        stats = validate_output(out)
        print(json.dumps(stats, indent=2, sort_keys=True))
        return 0
    if args.sample:
        print_samples(input_dir, ["Lithium", "Helium", "MDMA", "Abiogenesis", "Atmospheric_pressure", "Loperamide"], random_count=4)
        return 0

    files = dedup_latest_by_title(input_dir.glob("*.html"))
    if args.limit:
        by_title = {p.stem.split("__oldid_")[0]: p for p in files}
        preferred = [by_title[n] for n in
                     ["Lithium", "Helium", "MDMA", "Abiogenesis", "Atmospheric_pressure", "Loperamide"]
                     if n in by_title]
        seen = set(preferred)
        files = preferred + [p for p in files if p not in seen]
        files = files[:args.limit]
    if not files:
        print(f"no_html_files input_dir={input_dir}", file=sys.stderr)
        return 2

    out.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        done = completed_article_ids(out)
        files = [p for p in files if p.stem not in done]
        print(f"resume output={out} already_done_articles={len(done)} remaining_files={len(files)}", flush=True)
    elif not args.limit:
        backup_existing(out)
    started = time.time()
    total_rows = 0
    target = out if args.resume else out.with_suffix(out.suffix + ".tmp")
    mode = "a" if args.resume else "w"
    processed = 0
    with target.open(mode, encoding="utf-8") as f:
        for i, path in enumerate(files, 1):
            rows = extract_file(path)
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            total_rows += len(rows)
            processed = i
            if i % 250 == 0 or i == len(files):
                print(f"progress files={i}/{len(files)} rows={total_rows} elapsed_sec={time.time()-started:.1f}", flush=True)
            if args.max_seconds and time.time() - started >= args.max_seconds:
                print(f"stopping_cleanly_after_max_seconds processed_files={processed}/{len(files)} rows_added={total_rows}", flush=True)
                break
    if not args.resume:
        os.replace(target, out)
    stats = validate_output(out)
    print(f"wrote={out} input_dir={input_dir} files_processed={processed}/{len(files)} elapsed_sec={time.time()-started:.1f}")
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
