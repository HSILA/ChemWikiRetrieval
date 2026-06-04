import json
from pathlib import Path

from nq.scripts.build_mteb_dataset import build_dataset


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_row(qid, question, title="Rate equation", oldid="1", long_answer=None):
    annotations = {"long_answer": [], "short_answers": [], "yes_no_answer": []}
    if long_answer is not None:
        annotations["long_answer"] = [long_answer]
    return {
        "id": qid,
        "question": question,
        "document_title": title,
        "document_url": f"https://en.wikipedia.org/w/index.php?title={title.replace(' ', '_')}&oldid={oldid}",
        "annotations": annotations,
    }


def test_build_dataset_keeps_only_single_clean_long_answer_blocks(tmp_path):
    nq_dir = tmp_path / "nq"
    html_dir = nq_dir / "html_pages"
    html_dir.mkdir(parents=True)

    good_html = '<div class="mw-parser-output"><p>This is a complete chemistry paragraph with enough content to be retained as a clean positive answer.</p></div>'
    bad_html = '<div class="mw-parser-output"><p>The rate law for a reaction that is first order with respect to a reactant A is</p></div>'
    table_html = '<div class="mw-parser-output"><table><tr><td>answer only in table</td></tr></table></div>'
    multi_html = '<div class="mw-parser-output"><p>First complete chemistry paragraph with enough content to be retained.</p><p>Second complete chemistry paragraph with enough content to be retained.</p></div>'

    cases = [
        ("keep", "kept question", "Good", "11", good_html),
        ("bad", "bad intro question", "Bad", "22", bad_html),
        ("zero", "zero block question", "Zero", "33", table_html),
        ("multi", "multi block question", "Multi", "44", multi_html),
    ]
    matches = []
    decisions = []
    paragraphs = []
    for qid, question, title, oldid, html in cases:
        html_path = html_dir / f"{title}__oldid_{oldid}.html"
        html_path.write_text(html, encoding="utf-8")
        data = html.encode("utf-8")
        matches.append(make_row(qid, question, title=title, oldid=oldid, long_answer={"start_byte": 0, "end_byte": len(data), "candidate_index": 0}))
        decisions.append({"id": qid, "chemistry": True})

    paragraphs.extend([
        {"id": "Good__oldid_11::b0000", "article_id": "Good__oldid_11", "title": "Good", "text": "This is a complete chemistry paragraph with enough content to be retained as a clean positive answer."},
        {"id": "Bad__oldid_22::b0000", "article_id": "Bad__oldid_22", "title": "Bad", "text": "The rate law for a reaction that is first order with respect to a reactant A is"},
        {"id": "Multi__oldid_44::b0000", "article_id": "Multi__oldid_44", "title": "Multi", "text": "First complete chemistry paragraph with enough content to be retained."},
    ])
    matches.append(make_row("nolonga", "question without long answer", title="Good", oldid="11"))
    decisions.append({"id": "nolonga", "chemistry": True})

    write_jsonl(nq_dir / "matches.jsonl", matches)
    write_jsonl(nq_dir / "topic_filter_gemma4.jsonl", decisions)
    write_jsonl(nq_dir / "article_sections.jsonl", paragraphs)

    summary = build_dataset(nq_dir=nq_dir, out_dir=nq_dir / "mteb_clean")

    assert summary["kept_queries"] == 2
    assert summary["removed_bad_intro_corpus_rows"] == 1
    assert summary["removed_no_corpus_title"] == 2
    assert summary["removed_without_long_answer"] == 1

    queries = [json.loads(x) for x in (nq_dir / "mteb_clean" / "queries.jsonl").read_text().splitlines()]
    corpus = [json.loads(x) for x in (nq_dir / "mteb_clean" / "corpus.jsonl").read_text().splitlines()]
    qrels = [json.loads(x) for x in (nq_dir / "mteb_clean" / "qrels" / "test.jsonl").read_text().splitlines()]

    assert len(queries) == 2
    assert len(corpus) == 2  # Good + Multi (both matched from article_sections)
    assert len(qrels) == 2


def test_build_dataset_requires_article_to_exist_in_corpus(tmp_path):
    nq_dir = tmp_path / "nq"
    html_dir = nq_dir / "html_pages"
    html_dir.mkdir(parents=True)
    html = '<div class="mw-parser-output"><p>This is a complete chemistry paragraph with enough content to be retained as a clean positive answer.</p></div>'
    (html_dir / "Good__oldid_11.html").write_text(html, encoding="utf-8")
    write_jsonl(nq_dir / "matches.jsonl", [make_row("keep", "kept question", title="Good", oldid="11", long_answer={"start_byte": 0, "end_byte": len(html.encode("utf-8")), "candidate_index": 0})])
    write_jsonl(nq_dir / "topic_filter_gemma4.jsonl", [{"id": "keep", "chemistry": True}])
    write_jsonl(nq_dir / "article_sections.jsonl", [])

    summary = build_dataset(nq_dir=nq_dir, out_dir=nq_dir / "mteb_clean")

    assert summary["kept_queries"] == 0
    assert summary["removed_no_corpus_title"] == 1
    corpus = [json.loads(x) for x in (nq_dir / "mteb_clean" / "corpus.jsonl").read_text().splitlines()]
    qrels = [json.loads(x) for x in (nq_dir / "mteb_clean" / "qrels" / "test.jsonl").read_text().splitlines()]
    assert corpus == []
    assert qrels == []
