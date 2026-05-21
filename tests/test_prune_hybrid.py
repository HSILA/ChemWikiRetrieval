from __future__ import annotations

import gzip
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from graph import prune_hybrid as ph


def tiny_graph() -> ph.Graph:
    categories = {
        "Chemistry": {"category_title": "Chemistry", "depth": 0, "path": ["Chemistry"]},
        "Analytical_chemistry": {
            "category_title": "Analytical_chemistry",
            "depth": 1,
            "path": ["Chemistry", "Analytical_chemistry"],
        },
        "Materials_science": {
            "category_title": "Materials_science",
            "depth": 1,
            "path": ["Chemistry", "Materials_science"],
        },
        "Chemists": {"category_title": "Chemists", "depth": 1, "path": ["Chemistry", "Chemists"]},
        "Women_chemists": {
            "category_title": "Women_chemists",
            "depth": 2,
            "path": ["Chemistry", "Chemists", "Women_chemists"],
        },
    }
    articles = {
        "Titration": {"article_title": "Titration", "article_url": "https://en.wikipedia.org/wiki/Titration"},
        "Composite_material": {"article_title": "Composite_material"},
        "Marie_Curie": {"article_title": "Marie_Curie"},
        "Polymer": {"article_title": "Polymer"},
    }
    parent_to_children = {
        "Chemistry": ["Analytical_chemistry", "Materials_science", "Chemists"],
        "Chemists": ["Women_chemists"],
    }
    child_to_parents = {
        "Analytical_chemistry": ["Chemistry"],
        "Materials_science": ["Chemistry"],
        "Chemists": ["Chemistry"],
        "Women_chemists": ["Chemists"],
    }
    category_to_articles = {
        "Analytical_chemistry": ["Titration", "Polymer"],
        "Materials_science": ["Composite_material", "Polymer"],
        "Chemists": ["Marie_Curie"],
        "Women_chemists": ["Marie_Curie"],
    }
    article_to_categories = {
        "Titration": ["Analytical_chemistry"],
        "Composite_material": ["Materials_science"],
        "Marie_Curie": ["Chemists", "Women_chemists"],
        "Polymer": ["Analytical_chemistry", "Materials_science"],
    }
    return ph.Graph(
        categories=categories,
        articles=articles,
        parent_to_children=parent_to_children,
        child_to_parents=child_to_parents,
        category_to_articles=category_to_articles,
        article_to_categories=article_to_categories,
    )


def decisions() -> dict[str, dict]:
    return {
        "Chemistry": {
            "label": "keep",
            "confidence_score": 3,
            "expand_descendants": True,
            "include_direct_articles": True,
            "reason": "root",
        },
        "Analytical_chemistry": {
            "label": "keep",
            "confidence_score": 3,
            "expand_descendants": True,
            "include_direct_articles": True,
            "reason": "lab methods",
        },
        "Materials_science": {
            "label": "mixed",
            "confidence_score": 2,
            "expand_descendants": True,
            "include_direct_articles": False,
            "reason": "mixed materials and non-chemistry items",
        },
        "Chemists": {
            "label": "drop",
            "confidence_score": 0,
            "expand_descendants": False,
            "include_direct_articles": False,
            "reason": "people branch",
        },
        "Women_chemists": {
            "label": "drop",
            "confidence_score": 0,
            "expand_descendants": False,
            "include_direct_articles": False,
            "reason": "people branch",
        },
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else type(path).open
    with opener(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_load_graph_reads_gzipped_article_members(tmp_path):
    write_jsonl(tmp_path / "categories.jsonl", [{"category_title": "Chemistry"}, {"category_title": "Analytical_chemistry"}])
    write_jsonl(tmp_path / "articles.jsonl", [{"article_title": "Titration"}])
    write_jsonl(
        tmp_path / "category_edges.jsonl",
        [{"parent_category": "Chemistry", "child_category": "Analytical_chemistry"}],
    )
    write_jsonl(
        tmp_path / "article_members.jsonl.gz",
        [{"parent_category": "Analytical_chemistry", "article_title": "Titration"}],
    )

    graph = ph.load_graph(tmp_path)

    assert graph.category_to_articles["Analytical_chemistry"] == ["Titration"]
    assert graph.article_to_categories["Titration"] == ["Analytical_chemistry"]


def test_resolve_jsonl_path_rejects_plain_and_gz_side_by_side(tmp_path):
    plain = tmp_path / "article_members.jsonl"
    gz = tmp_path / "article_members.jsonl.gz"
    plain.write_text('{"plain":true}\n', encoding="utf-8")
    with gzip.open(gz, "wt", encoding="utf-8") as f:
        f.write('{"gz":true}\n')

    with pytest.raises(FileExistsError, match="both plain and gzipped JSONL"):
        ph.resolve_jsonl_path(plain)


def test_context_cache_coalesces_concurrent_first_fetch(tmp_path):
    cache = ph.JsonlContextCache(tmp_path / "context_cache.jsonl", kind="article")
    calls = []
    lock = threading.Lock()

    def fetch_fn(title):
        with lock:
            calls.append(title)
        time.sleep(0.05)
        return {"intro": title}

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: cache.fetch("Titration", fetch_fn), range(8)))

    assert results == [{"intro": "Titration"}] * 8
    assert calls == ["Titration"]
    assert len(list(ph.read_jsonl(tmp_path / "context_cache.jsonl"))) == 1


def test_normalize_category_decision_accepts_mixed_and_forces_article_review():
    raw = {
        "label": "mixed",
        "confidence_score": 2,
        "expand_descendants": True,
        "include_direct_articles": True,
        "reason": "contains both chemistry and non-chemistry pages",
    }

    normalized = ph.normalize_category_decision(raw)

    assert normalized["label"] == "mixed"
    assert normalized["include_direct_articles"] is False
    assert normalized["expand_descendants"] is True


def test_normalize_category_decision_only_strong_keep_includes_direct_articles():
    strong = ph.normalize_category_decision({"label": "keep", "confidence_score": 2, "reason": "strong keep"})
    weak = ph.normalize_category_decision({"label": "keep", "confidence_score": 1, "reason": "weak keep"})

    assert strong["expand_descendants"] is True
    assert strong["include_direct_articles"] is True
    assert weak["expand_descendants"] is True
    assert weak["include_direct_articles"] is False


def test_branch_status_closes_drop_branch_but_keeps_mixed_branch_reachable():
    status = ph.compute_branch_status(tiny_graph(), decisions())

    assert status["Chemistry"].reachable is True
    assert status["Analytical_chemistry"].reachable is True
    assert status["Materials_science"].reachable is True
    assert status["Chemists"].reachable is True
    assert status["Women_chemists"].reachable is False
    assert status["Materials_science"].opens_descendants is True
    assert status["Chemists"].opens_descendants is False


def test_stage2_candidate_plan_uses_option_c_strong_keep_wins_over_mixed():
    graph = tiny_graph()
    status = ph.compute_branch_status(graph, decisions())

    candidates = {c["article_title"]: c for c in ph.build_article_candidates(graph, decisions(), status)}

    assert candidates["Titration"]["candidate_status"] == "auto_keep_candidate"
    assert candidates["Composite_material"]["candidate_status"] == "needs_article_review"
    assert candidates["Polymer"]["candidate_status"] == "auto_keep_candidate"
    assert candidates["Marie_Curie"]["candidate_status"] == "auto_drop_context"

    summary = ph.summarize_candidate_plan(candidates.values())
    assert summary["needs_article_review"] == 1
    assert summary["auto_keep_candidate"] == 2
    assert summary["auto_drop_context"] == 1


def test_stage2_candidate_plan_reviews_weak_keep_even_without_mixed_parent():
    graph = tiny_graph()
    graph.categories["Weak_keep"] = {"category_title": "Weak_keep", "depth": 1, "path": ["Chemistry", "Weak_keep"]}
    graph.articles["Ambiguous_article"] = {"article_title": "Ambiguous_article"}
    graph.parent_to_children.setdefault("Chemistry", []).append("Weak_keep")
    graph.child_to_parents["Weak_keep"] = ["Chemistry"]
    graph.category_to_articles["Weak_keep"] = ["Ambiguous_article"]
    graph.article_to_categories["Ambiguous_article"] = ["Weak_keep"]
    stage1 = decisions()
    stage1["Weak_keep"] = ph.normalize_category_decision(
        {"label": "keep", "confidence_score": 1, "reason": "weak category-level keep"}
    )

    status = ph.compute_branch_status(graph, stage1)
    candidates = {c["article_title"]: c for c in ph.build_article_candidates(graph, stage1, status)}

    assert status["Weak_keep"].opens_descendants is True
    assert status["Weak_keep"].contributes_direct_articles is False
    assert candidates["Ambiguous_article"]["candidate_status"] == "needs_article_review"


def test_stage2_article_prompt_uses_minimal_confidence_schema():
    assert '"confidence_score": 0 | 1 | 2 | 3' in ph.ARTICLE_SYSTEM_PROMPT
    assert '"thinking_process"' not in ph.ARTICLE_SYSTEM_PROMPT
    assert "score_0_to_3" not in ph.ARTICLE_SYSTEM_PROMPT
    assert "article_type" not in ph.ARTICLE_SYSTEM_PROMPT
    assert "weak-keep" not in ph.ARTICLE_SYSTEM_PROMPT
    assert "reachable MIXED" not in ph.ARTICLE_SYSTEM_PROMPT
    assert "hard science of chemistry" in ph.ARTICLE_SYSTEM_PROMPT
    assert "category graph traversal" in ph.ARTICLE_SYSTEM_PROMPT
    assert "supporting context for disambiguation only" in ph.ARTICLE_SYSTEM_PROMPT
    assert "Base your decision primarily on the article title and the Wikipedia intro extract" in ph.ARTICLE_SYSTEM_PROMPT
    assert "fundamental chemistry-science knowledge" in ph.ARTICLE_SYSTEM_PROMPT
    assert "Chemically focused aspects of geochemistry or environmental chemistry" in ph.ARTICLE_SYSTEM_PROMPT
    assert "Established chemistry subfield overview articles" in ph.ARTICLE_SYSTEM_PROMPT
    assert "chemical processes, reactions, separations, catalysis, reactors" in ph.ARTICLE_SYSTEM_PROMPT
    assert "Drop broad non-chemistry or engineering/instrumentation discipline overviews" in ph.ARTICLE_SYSTEM_PROMPT
    assert "Drop list, outline, index, timeline, or topical-guide pages" in ph.ARTICLE_SYSTEM_PROMPT
    assert "Biographies, individual people" in ph.ARTICLE_SYSTEM_PROMPT


def test_stage2_article_payload_is_minimal_and_article_focused():
    graph = tiny_graph()
    graph.articles["Titration"].update(
        {
            "article_display_title": "Titration",
            "article_url": "https://en.wikipedia.org/wiki/Titration",
            "depth": 1,
            "first_seen_parent_category": "Analytical_chemistry",
            "path": ["Chemistry", "Analytical_chemistry", "Titration"],
        }
    )
    candidate = {
        "article_title": "Titration",
        "candidate_status": "needs_article_review",
        "reachable_categories": [{"category_title": "Analytical_chemistry", "reason": "lab methods"}],
        "review_categories": [{"category_title": "Analytical_chemistry", "reason": "lab methods"}],
        "keep_categories": [],
    }
    wiki_context = {
        "article_title": "Titration",
        "article_url": "https://en.wikipedia.org/wiki/Titration",
        "page_id": 123,
        "wikipedia_intro_extract": "Titration is a chemical analysis method.",
    }

    payload = ph.build_article_user_payload(candidate, graph, wiki_context)
    compact = json.loads(payload.split("\n", 1)[1])

    assert compact["article"] == {
        "article_title": "Titration",
        "bfs_depth": 1,
        "path": ["Chemistry", "Analytical_chemistry", "Titration"],
    }
    assert compact["fetched_wikipedia_context"] == {
        "wikipedia_intro_extract": "Titration is a chemical analysis method."
    }
    assert "candidate_evidence" not in compact
    assert "article_url" not in payload
    assert "first_seen_parent_category" not in payload


def test_fetch_article_context_uses_intro_extract_capped_at_1000_chars(monkeypatch):
    seen_params = {}

    def fake_wiki_get(params):
        seen_params.update(params)
        return {"query": {"pages": [{"extract": "x" * 1200, "pageid": 123}]}}

    monkeypatch.setattr(ph, "wiki_get", fake_wiki_get)

    context = ph.fetch_article_context("Titration")

    assert seen_params["exintro"] == 1
    assert seen_params["explaintext"] == 1
    assert seen_params["exchars"] == 1000
    assert seen_params["prop"] == "extracts"
    assert "inprop" not in seen_params
    assert context == {"wikipedia_intro_extract": "x" * 1000}


def test_category_evidence_uses_current_stage1_schema_only():
    evidence = ph.category_evidence(tiny_graph(), "Materials_science", decisions()["Materials_science"])

    assert evidence["label"] == "mixed"
    assert evidence["confidence_score"] == 2
    assert evidence["reason"] == "mixed materials and non-chemistry items"
    assert "score_0_to_3" not in evidence
    assert "category_type" not in evidence


def test_branch_summary_does_not_count_removed_category_type_field():
    status = ph.compute_branch_status(tiny_graph(), decisions())
    summary = ph.summarize_branch_status(status, decisions())

    assert "label_counts" in summary
    assert "category_type_counts" not in summary


def test_final_article_rows_includes_reviewed_keep_using_confidence_score():
    graph = tiny_graph()
    status = ph.compute_branch_status(graph, decisions())
    candidates = ph.build_article_candidates(graph, decisions(), status)
    article_decisions = {
        "Composite_material": {"label": "keep", "confidence_score": 2, "reason": "chemistry material"},
        "Polymer": {"label": "keep", "confidence_score": 1, "reason": "too uncertain"},
    }

    final_rows = {row["article_title"]: row for row in ph.final_article_rows(graph, candidates, article_decisions)}

    assert "Composite_material" in final_rows
    assert final_rows["Composite_material"]["inclusion_source"] == "stage2_article_keep"


def test_decision_rows_are_flat_without_nested_classification_or_removed_fields():
    args = SimpleNamespace(model="deepseek/deepseek-v4-flash", reasoning_effort="medium")
    response = {
        "usage": {"prompt_tokens": 1},
        "response_id": "resp-1",
        "model_returned": "deepseek/deepseek-v4-flash",
        "raw_response": '{"label":"keep","confidence_score":2,"reason":"chemistry"}',
    }
    article_candidate = {
        "article_title": "Polymer",
        "article_display_title": "Polymer",
        "article_url": "https://en.wikipedia.org/wiki/Polymer",
        "candidate_status": "needs_article_review",
        "review_categories": [],
    }

    row = ph.build_article_decision_row(
        article_candidate,
        args,
        {"label": "keep", "confidence_score": 2, "reason": "chemistry material"},
        response,
    )

    assert row["label"] == "keep"
    assert row["confidence_score"] == 2
    assert row["reason"] == "chemistry material"
    assert row["prompt_version"] == ph.ARTICLE_PROMPT_VERSION
    assert "classification" not in row
    assert "stage" not in row
    assert "kind" not in row
    assert "article_type" not in row
    assert "score_0_to_3" not in row


def test_pending_review_candidates_skips_existing_before_applying_limit():
    review_rows = [
        {"article_title": "A", "candidate_status": "needs_article_review"},
        {"article_title": "B", "candidate_status": "needs_article_review"},
        {"article_title": "C", "candidate_status": "needs_article_review"},
    ]
    existing = {"A": {"label": "keep", "confidence_score": 2}}

    pending = ph.pending_review_candidates(review_rows, existing, limit=1)

    assert [row["article_title"] for row in pending] == ["B"]


def test_stage2_uses_thread_pool_and_skips_existing_before_submit(tmp_path, monkeypatch):
    output_dir = tmp_path / "pruned_hybrid"
    output_dir.mkdir()
    ph.append_jsonl(
        output_dir / "article_decisions.jsonl",
        {
            "article_title": "A",
            "label": "keep",
            "confidence_score": 3,
            "reason": "already done",
        },
    )
    candidates = [
        {"article_title": "A", "candidate_status": "needs_article_review", "review_categories": []},
        {"article_title": "B", "candidate_status": "needs_article_review", "review_categories": []},
        {"article_title": "C", "candidate_status": "needs_article_review", "review_categories": []},
    ]
    submitted = []
    max_workers_seen = []

    class FakeFuture:
        def __init__(self, fn, row):
            self.fn = fn
            self.row = row

        def result(self):
            return self.fn(self.row)

    class FakePool:
        def __init__(self, max_workers):
            max_workers_seen.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, row):
            submitted.append(row["article_title"])
            return FakeFuture(fn, row)

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(ph, "ThreadPoolExecutor", FakePool)
    monkeypatch.setattr(ph, "as_completed", lambda futures: futures)
    monkeypatch.setattr(ph, "load_graph", lambda _path: tiny_graph())
    monkeypatch.setattr(ph, "load_category_decisions", lambda _path: decisions())
    monkeypatch.setattr(ph, "compute_branch_status", lambda _graph, _decisions: {})
    monkeypatch.setattr(ph, "build_article_candidates", lambda _graph, _decisions, _status: candidates)
    monkeypatch.setattr(ph, "summarize_candidate_plan", lambda _candidates: {"needs_article_review": 3})
    monkeypatch.setattr(ph, "summarize_branch_status", lambda _status, _decisions: {})
    monkeypatch.setattr(ph, "fetch_article_context", lambda title: {"wikipedia_intro_extract": f"{title} chemistry"})
    monkeypatch.setattr(
        ph,
        "openrouter_chat_json",
        lambda **_kwargs: {
            "parsed": {"label": "keep", "confidence_score": 2, "reason": "chemistry"},
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            "raw_response": '{"label":"keep","confidence_score":2,"reason":"chemistry"}',
        },
    )
    args = SimpleNamespace(
        input_dir=tmp_path,
        limit=None,
        dry_run=False,
        model="google/gemma-4-26b-a4b-it",
        reasoning_effort="medium",
        max_tokens=2048,
        estimate_cost_per_call_usd=0.0001,
        concurrency=7,
    )

    assert ph.run_stage2(args) == 0

    assert max_workers_seen == [7]
    assert submitted == ["B", "C"]
    assert not (output_dir / "branch_status.jsonl").exists()
    assert not (output_dir / "article_candidates.jsonl").exists()
    rows = list(ph.read_jsonl(output_dir / "article_decisions.jsonl"))
    assert [row["article_title"] for row in rows] == ["A", "B", "C"]
    summary = json.loads((output_dir / "stage2_summary.json").read_text())
    assert summary["processed"] == 2
    assert summary["skipped_existing"] == 1
    assert summary["failed"] == 0
    assert summary["concurrency"] == 7


def test_openrouter_retries_transport_and_response_json_decode_errors(monkeypatch):
    attempts = []

    class FakeResponse:
        status_code = 200
        text = "not-json"

        def __init__(self, payload=None, *, json_error=None):
            self.payload = payload
            self.json_error = json_error

        def json(self):
            if self.json_error:
                raise self.json_error
            return self.payload

    good_payload = {
        "id": "ok-1",
        "choices": [{"message": {"content": '{"label":"drop","confidence_score":2,"reason":"ok"}'}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }

    def fake_post(*_args, **_kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("Response ended prematurely")
        if len(attempts) == 2:
            return FakeResponse(json_error=ValueError("Expecting value: line 1 column 1"))
        return FakeResponse(payload=good_payload)

    monkeypatch.setattr(ph.requests, "post", fake_post)
    monkeypatch.setattr(ph.time, "sleep", lambda _seconds: None)

    result = ph.openrouter_chat_json(
        api_key="test-key",
        model="google/gemma-4-26b-a4b-it",
        system_prompt="Return JSON.",
        user_payload="{}",
        reasoning_effort="medium",
        max_tokens=1024,
    )

    assert result["parsed"]["label"] == "drop"
    assert len(attempts) == 3


def test_openrouter_retries_length_empty_content_with_low_reasoning(monkeypatch):
    efforts = []

    class FakeResponse:
        status_code = 200
        text = "{}"

        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    empty_payload = {
        "id": "truncated-1",
        "choices": [
            {
                "finish_reason": "length",
                "native_finish_reason": "length",
                "message": {"content": None, "reasoning": "long hidden reasoning"},
            }
        ],
    }
    good_payload = {
        "id": "ok-2",
        "choices": [{"message": {"content": '{"label":"keep","confidence_score":2,"reason":"chemistry"}'}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }

    def fake_post(*_args, **kwargs):
        efforts.append(kwargs["json"]["reasoning"]["effort"])
        return FakeResponse(empty_payload if len(efforts) == 1 else good_payload)

    monkeypatch.setattr(ph.requests, "post", fake_post)
    monkeypatch.setattr(ph.time, "sleep", lambda _seconds: None)

    result = ph.openrouter_chat_json(
        api_key="test-key",
        model="google/gemma-4-26b-a4b-it",
        system_prompt="Return JSON.",
        user_payload="{}",
        reasoning_effort="medium",
        max_tokens=1024,
    )

    assert result["parsed"]["label"] == "keep"
    assert efforts == ["medium", "low"]


def test_openrouter_fatal_credit_error_raises_without_retry(monkeypatch):
    attempts = []

    class FakeResponse:
        status_code = 402
        text = '{"error":{"message":"Insufficient credits","code":402}}'

    def fake_post(*_args, **_kwargs):
        attempts.append(1)
        return FakeResponse()

    monkeypatch.setattr(ph.requests, "post", fake_post)

    with pytest.raises(ph.FatalOpenRouterError):
        ph.openrouter_chat_json(
            api_key="test-key",
            model="google/gemma-4-26b-a4b-it",
            system_prompt="Return JSON.",
            user_payload="{}",
            reasoning_effort="medium",
            max_tokens=1024,
        )

    assert len(attempts) == 1


def test_stage2_records_single_fatal_provider_error_and_skips_queued_work(tmp_path, monkeypatch):
    output_dir = tmp_path / "pruned_hybrid"
    output_dir.mkdir()
    candidates = [
        {"article_title": "B", "candidate_status": "needs_article_review", "review_categories": []},
        {"article_title": "C", "candidate_status": "needs_article_review", "review_categories": []},
        {"article_title": "D", "candidate_status": "needs_article_review", "review_categories": []},
    ]

    class FakeFuture:
        def __init__(self, fn, row):
            self.fn = fn
            self.row = row

        def result(self):
            return self.fn(self.row)

    class FakePool:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, row):
            return FakeFuture(fn, row)

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(ph, "ThreadPoolExecutor", FakePool)
    monkeypatch.setattr(ph, "as_completed", lambda futures: futures)
    monkeypatch.setattr(ph, "load_graph", lambda _path: tiny_graph())
    monkeypatch.setattr(ph, "load_category_decisions", lambda _path: decisions())
    monkeypatch.setattr(ph, "compute_branch_status", lambda _graph, _decisions: {})
    monkeypatch.setattr(ph, "build_article_candidates", lambda _graph, _decisions, _status: candidates)
    monkeypatch.setattr(ph, "summarize_candidate_plan", lambda _candidates: {"needs_article_review": 3})
    monkeypatch.setattr(ph, "summarize_branch_status", lambda _status, _decisions: {})
    monkeypatch.setattr(ph, "fetch_article_context", lambda title: {"wikipedia_intro_extract": f"{title} chemistry"})
    monkeypatch.setattr(
        ph,
        "openrouter_chat_json",
        lambda **_kwargs: (_ for _ in ()).throw(ph.FatalOpenRouterError("HTTP 402: insufficient credits")),
    )
    args = SimpleNamespace(
        input_dir=tmp_path,
        limit=None,
        dry_run=False,
        model="google/gemma-4-26b-a4b-it",
        reasoning_effort="medium",
        max_tokens=1024,
        estimate_cost_per_call_usd=0.0001,
        concurrency=7,
    )

    assert ph.run_stage2(args) == 1

    error_rows = list(ph.read_jsonl(output_dir / "article_errors.jsonl"))
    assert len(error_rows) == 1
    assert error_rows[0]["article_title"] == "B"
    summary = json.loads((output_dir / "stage2_summary.json").read_text())
    assert summary["processed"] == 0
    assert summary["failed"] == 1
    assert summary["fatal_error"] == "HTTP 402: insufficient credits"


def test_resume_flag_is_not_exposed():
    parser = ph.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["stage1", "--resume"])
    with pytest.raises(SystemExit):
        parser.parse_args(["stage2", "--no-resume"])
    with pytest.raises(SystemExit):
        parser.parse_args(["stage1", "--titles", "Chemistry"])


def test_output_path_flags_are_not_exposed_to_keep_bundle_canonical():
    parser = ph.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["stage1", "--output-dir", "/tmp/other"])
    with pytest.raises(SystemExit):
        parser.parse_args(["stage2", "--output-dir", "/tmp/other"])
    with pytest.raises(SystemExit):
        parser.parse_args(["stage2", "--category-decisions", "/tmp/category_decisions.jsonl"])


def test_stage_defaults_keep_category_model_but_use_gemma_for_articles():
    parser = ph.build_parser()

    stage1 = parser.parse_args(["stage1"])
    stage2 = parser.parse_args(["stage2"])

    assert stage1.model == "deepseek/deepseek-v4-flash"
    assert stage1.max_tokens == 1024
    assert stage2.model == "google/gemma-4-26b-a4b-it"
    assert stage2.reasoning_effort == "medium"
    assert stage2.max_tokens == 2048
    assert stage2.estimate_cost_per_call_usd == 0.0001
