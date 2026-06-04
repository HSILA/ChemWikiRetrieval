#!/usr/bin/env python3
"""Build the final MTEB/BEIR-style NQ chemistry retrieval dataset.

The corpus is de-duplicated by article title: each title contributes exactly
one set of sections (taken from the latest snapshot during extraction), so a
title with many oldid snapshots no longer appears many times.

Each NQ question is mapped to a section of its article by locating the section
heading that encloses the long-answer byte span in the question's own snapshot,
then joining to the de-duplicated corpus by (article title, section).

Outputs match the mteb/nq Hugging Face layout, plus a `section` field:
- corpus.jsonl: {"_id", "title", "section", "text"}
- queries.jsonl: {"_id", "text"}
- qrels/test.jsonl: {"query-id", "corpus-id", "score"}
"""
from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NQ_DIR = ROOT / "nq"
DEFAULT_OUT_DIR = DEFAULT_NQ_DIR / "mteb_clean"

STOP_SECTIONS = {
    "references", "notes", "footnotes", "bibliography", "sources", "further reading",
    "external links", "see also", "navigation", "related pages", "works cited",
    "contents",
}
DROP_SECTION_PREFIXES = (
    "references", "notes", "footnotes", "bibliography", "sources", "further reading",
    "external links", "see also",
)

BAD_INTRO_RE = re.compile(
    r"""
(?:
    \b(?:thus|therefore|hence|where|whereby)\s*$
  | \b(?:as|is|are|was|were|be|being)\s*$
  | \b(?:given|described|defined|written|expressed|represented|shown|calculated|computed|estimated|approximated|obtained|derived|modeled|modelled)\s+(?:as|by|with)?\s*$
  | \b(?:is|are|was|were|can be|may be|could be)\s+(?:given|described|defined|written|expressed|represented|shown|calculated|computed|estimated|approximated|obtained|derived|modeled|modelled)\s+(?:as|by|with)?\s*$
  | \b(?:the\s+)?(?:equation|formula|expression|reaction|relationship|relation|law)\s+(?:is|are|becomes|can be written|can be expressed)\s*$
  | \b(?:as follows|it reads)\s*$
)
""",
    re.I | re.X,
)


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
    text = re.sub(r"\s+", " ", text.replace("\n", " ").replace("\t", " ")).strip()
    if len(text) > 160:
        return ""
    return text


def is_bad_intro(text: str) -> bool:
    return bool(BAD_INTRO_RE.search((text or "").strip()))


def safe_part(text: str, max_len: int = 140) -> str:
    text = text.replace("/", "_").replace("\\", "_").strip()
    text = re.sub(r"[^A-Za-z0-9._()\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return (text or "untitled")[:max_len]


def oldid_from_url(url: str) -> str:
    qs = parse_qs(urlparse(url.replace("&amp;", "&")).query)
    return qs.get("oldid", [""])[0] or "no_oldid"


def article_key_for(row: dict) -> str:
    return safe_part(row.get("document_title") or "")


def article_id_for(row: dict) -> str:
    oldid = oldid_from_url(row.get("document_url") or "")
    return f"{article_key_for(row)}__oldid_{safe_part(oldid, 40)}"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                yield line_no, json.loads(line)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def match_key(text: str) -> str:
    text = html.unescape(text or "").lower()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\s+([_^]\{)", r"\1", text)
    text = re.sub(r"([_^]\{)\s+", r"\1", text)
    text = re.sub(r"\s+\}", "}", text)
    return text


def normalize_heading_text(raw_inner_html: str) -> str:
    """Clean the heading text from the inner HTML of an <h2> element."""
    m = re.search(
        r'<span[^>]*class="[^"]*mw-headline[^"]*"[^>]*>(.*?)</span>',
        raw_inner_html, re.I | re.S,
    )
    inner = m.group(1) if m else raw_inner_html
    inner = re.sub(r"<[^>]+>", " ", inner)
    return clean_text(inner)


def section_for_byte(html_bytes: bytes, start_byte: int) -> str | None:
    """Return the H2 section name enclosing ``start_byte``.

    ``summary`` for the lead (no preceding section heading), or ``None`` when
    the byte span sits inside a section that the corpus intentionally drops
    (References, See also, ...).
    """
    prefix = html_bytes[: max(0, start_byte)].decode("utf-8", errors="ignore")
    headings = re.findall(r"<h2\b[^>]*>(.*?)</h2>", prefix, re.I | re.S)
    if not headings:
        return "summary"
    heading = normalize_heading_text(headings[-1])
    low = heading.lower()
    if not low or low == "contents":
        return "summary"
    if low in STOP_SECTIONS or any(low.startswith(p) for p in DROP_SECTION_PREFIXES):
        return None
    return heading


def raw_la_text_no_table(la: dict, html_bytes: bytes) -> str | None:
    """Cleaned text of the long-answer byte span, or None if it holds a table."""
    start = la["start_byte"]
    end = la["end_byte"]
    frag = html_bytes[start:end].decode("utf-8", errors="replace")
    if re.search(r"<\s*(table|tbody|thead|tr)\b", frag, re.I):
        return None
    frag = re.sub(r"<style[^>]*>.*?</style>|<script[^>]*>.*?</script>|<!--.*?-->", " ", frag, flags=re.I | re.S)
    frag = re.sub(r"<sup[^>]*class=\"[^\"]*reference[^\"]*\".*?</sup>", " ", frag, flags=re.I | re.S)
    frag = re.sub(r"<span[^>]*class=\"[^\"]*mw-editsection[^\"]*\".*?</span>", " ", frag, flags=re.I | re.S)
    frag = re.sub(
        r"<img[^>]*?alt=\"([^\"]*)\"[^>]*>",
        lambda m: f" {latex_alt_to_text(m.group(1)) or m.group(1)} ",
        frag,
        flags=re.I,
    )
    frag = re.sub(r"<[^>]+>", " ", frag)
    txt = html.unescape(re.sub(r"\s+", " ", frag)).strip()
    txt = re.sub(
        r"\[\s*(?:citation needed|clarification needed|page needed|dead link|failed verification|not specific enough to verify|unreliable source)\s*\]",
        " ", txt, flags=re.I,
    )
    txt = re.sub(r"(?-i:\bmw-[a-z])|edit\s*\]|edit\b|\bster\b|\bsect\b", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    if not txt or len(txt) < 45:
        return None
    if is_bad_intro(txt):
        return None
    return txt


def valid_long_answers(row: dict) -> list[dict]:
    out = []
    for la in (row.get("annotations") or {}).get("long_answer") or []:
        if not isinstance(la, dict):
            continue
        start = la.get("start_byte")
        end = la.get("end_byte")
        cand = la.get("candidate_index")
        if isinstance(start, int) and isinstance(end, int) and start >= 0 and end > start and cand not in (-1, None):
            out.append(la)
    return out


def doc_by_text_containment(article_key: str, answer_text: str,
                            by_key: dict[str, list[tuple[str, str]]]) -> str | None:
    """Fallback: a corpus section of this title whose text contains the answer."""
    bk = match_key(answer_text)
    if len(bk) < 40:
        return None
    hits = {doc_id for doc_id, doc_text in by_key.get(article_key, []) if bk in match_key(doc_text)}
    return next(iter(hits)) if len(hits) == 1 else None


def build_dataset(nq_dir: Path = DEFAULT_NQ_DIR, out_dir: Path = DEFAULT_OUT_DIR) -> dict:
    nq_dir = Path(nq_dir)
    out_dir = Path(out_dir)
    matches_path = nq_dir / "matches.jsonl"
    decisions_path = nq_dir / "topic_filter_gemma4.jsonl"
    sections_path = nq_dir / "article_sections.jsonl"

    decisions = {str(row["id"]): row for _, row in read_jsonl(decisions_path) if row.get("id") is not None}
    matches = [row for _, row in read_jsonl(matches_path)]
    chemistry_rows = [row for row in matches if decisions.get(str(row.get("id")), {}).get("chemistry") is True]

    # --- corpus: one set of sections per de-duplicated article title ---
    corpus_rows: list[dict] = []
    section_index: dict[tuple[str, str], str] = {}
    by_key: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)

    stats = Counter({
        "removed_bad_intro_corpus_rows": 0,
        "removed_without_long_answer": 0,
        "removed_no_corpus_title": 0,
        "removed_missing_html": 0,
        "removed_dropped_section_only": 0,
        "removed_unmapped": 0,
        "matched_by_section": 0,
        "matched_by_text_fallback": 0,
    })

    for _, para in read_jsonl(sections_path):
        text = para.get("text") or ""
        if is_bad_intro(text):
            stats["removed_bad_intro_corpus_rows"] += 1
            continue
        doc_id = para.get("id") or f"doc{len(corpus_rows)}"
        section = para.get("section") or "summary"
        article_key = str(para.get("article_key") or str(para.get("article_id")).split("__oldid_")[0])
        corpus_rows.append({
            "_id": doc_id,
            "title": para.get("title") or "",
            "section": section,
            "text": text,
        })
        by_key[article_key].append((doc_id, text))
        section_index.setdefault((article_key, section), doc_id)

    # --- queries + qrels: map each question to one section of its article ---
    query_rows: list[dict] = []
    qrel_rows: list[dict] = []

    for row in chemistry_rows:
        article_key = article_key_for(row)
        if article_key not in by_key:
            stats["removed_no_corpus_title"] += 1
            continue
        longs = valid_long_answers(row)
        if not longs:
            stats["removed_without_long_answer"] += 1
            continue
        html_path = nq_dir / "html_pages" / f"{article_id_for(row)}.html"
        if not html_path.exists():
            stats["removed_missing_html"] += 1
            continue
        html_bytes = html_path.read_bytes()

        doc_id: str | None = None
        match_kind: str | None = None
        saw_only_dropped = True
        for la in longs:
            section = section_for_byte(html_bytes, la["start_byte"])
            if section is None:
                continue
            saw_only_dropped = False
            cand = section_index.get((article_key, section))
            if cand:
                doc_id, match_kind = cand, "section"
                break
            answer = raw_la_text_no_table(la, html_bytes)
            if answer:
                cand = doc_by_text_containment(article_key, answer, by_key)
                if cand:
                    doc_id, match_kind = cand, "text"
                    break

        if not doc_id:
            # last resort: text containment regardless of detected heading
            for la in longs:
                answer = raw_la_text_no_table(la, html_bytes)
                if answer:
                    cand = doc_by_text_containment(article_key, answer, by_key)
                    if cand:
                        doc_id, match_kind = cand, "text"
                        break

        if not doc_id:
            if saw_only_dropped:
                stats["removed_dropped_section_only"] += 1
            else:
                stats["removed_unmapped"] += 1
            continue

        stats["matched_by_section" if match_kind == "section" else "matched_by_text_fallback"] += 1
        query_id = f"test{len(query_rows)}"
        query_rows.append({"_id": query_id, "text": row.get("question") or ""})
        qrel_rows.append({"query-id": query_id, "corpus-id": doc_id, "score": "1"})

    write_jsonl(out_dir / "corpus.jsonl", corpus_rows)
    write_jsonl(out_dir / "queries.jsonl", query_rows)
    write_jsonl(out_dir / "qrels" / "test.jsonl", qrel_rows)

    summary = {
        "source_matches": len(matches),
        "chemistry_true_questions": len(chemistry_rows),
        "kept_queries": len(query_rows),
        "corpus_rows": len(corpus_rows),
        "corpus_titles": len(by_key),
        **dict(stats),
        "output_files": {
            "corpus": str(out_dir / "corpus.jsonl"),
            "queries": str(out_dir / "queries.jsonl"),
            "qrels": str(out_dir / "qrels" / "test.jsonl"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nq-dir", type=Path, default=DEFAULT_NQ_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = build_dataset(args.nq_dir, args.out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
