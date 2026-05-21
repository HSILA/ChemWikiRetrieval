#!/usr/bin/env python3
"""Hybrid Wikipedia category/article pruning for ChemWikiRetrieval.

Single-script pipeline:

1. stage1: classify raw BFS categories as keep / mixed / drop.
2. stage2: use stage1 decisions plus graph memberships to plan article review,
   optionally dry-run the review count, then classify only articles touched by
   reachable mixed categories.

The script intentionally writes JSONL artifacts for resumability and auditability.
It fetches Wikipedia context in Python before OpenRouter calls instead of relying
on model-side tools.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import textwrap
import threading
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

try:  # Keep deterministic planning/test imports working even if requests is absent.
    import requests
except ImportError:  # pragma: no cover - exercised only in minimal envs
    requests = None  # type: ignore[assignment]

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
DEFAULT_INPUT_DIR = Path("graph/data/chemistry_bfs_depth5")
DEFAULT_CATEGORY_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_ARTICLE_MODEL = "google/gemma-4-26b-a4b-it"
CATEGORY_PROMPT_VERSION = "category_hybrid_v1_keep_mixed_drop"
ARTICLE_PROMPT_VERSION = "article_hybrid_v4_applied_chemistry_subfields_no_outlines"
USER_AGENT = "ChemWikiRetrieval/0.1 (https://github.com/HSILA/ChemWikiRetrieval; research script)"
FATAL_OPENROUTER_STATUS_CODES = {401, 402, 403}
TRANSIENT_OPENROUTER_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


class FatalOpenRouterError(RuntimeError):
    """Run-level OpenRouter failure; retrying individual rows cannot fix it."""

CATEGORY_SYSTEM_PROMPT = """
You classify a Wikipedia category node from a BFS rooted at Category:Chemistry.
The downstream goal is a high-precision retrieval benchmark about chemistry as a
science: concepts, substances, reactions, properties, methods, instruments,
materials/composition, molecular mechanisms, and applied chemical science.

Return a CATEGORY decision using exactly one of these labels:
- keep: the category is primarily chemistry-science content and its direct
  article members can be included without article-by-article LLM review.
- mixed: the category/branch may contain useful chemistry-science pages but also
  substantial non-chemistry, people, institution, company, history, pop-culture,
  or other drift. Direct articles MUST be reviewed in stage 2.
- drop: the category is not suitable and the branch should normally be closed.

Make two separate branch decisions:
1) expand_descendants: true only when child categories may contain useful
   chemistry-science concepts.
2) include_direct_articles: true only for keep categories whose direct article
   members are safe to auto-keep. For mixed or drop categories this MUST be false.

Positive scope:
- core chemistry branches: organic, inorganic, physical, analytical, nuclear,
  theoretical, environmental chemistry
- chemical substances/classes, reactions, bonding, stoichiometry,
  thermodynamics/kinetics when chemical, spectroscopy/analytical methods,
  lab techniques, instruments
- chemically focused applied areas: materials/composition, metallurgy when about
  materials/processes, chemical engineering process science,
  biochemistry/geochemistry/medicinal chemistry/pharmacology when focused on
  molecular/chemical mechanisms

Negative scope even with a chemistry flavor:
- people/biographies/groups of people: Chemists, American chemists, women
  chemists, Nobel laureates in Chemistry, alchemists
- history/social/institutional categories: History of chemistry, awards,
  societies, organizations, universities, professorships, museums
- publications/admin/list containers: books, journals, templates, stubs,
  portals, Wikipedia books; lists unless specifically of chemical
  substances/reactions
- companies, brands, facilities, legal/regulatory entities, government agencies
- fiction/pop culture
- broad neighboring sciences where chemistry is only incidental: generic atomic
  physics, subatomic particles, generic biology/medicine/geology

Calibration:
- Analytical chemistry => keep, expand_descendants=true, include_direct_articles=true
- Chemical reactions => keep, expand_descendants=true, include_direct_articles=true
- Materials science => mixed, expand_descendants=true, include_direct_articles=false
- Chemical engineering => mixed, expand_descendants=true, include_direct_articles=false
- Pharmacology under Medicinal chemistry => mixed, expand_descendants=true, include_direct_articles=false
- Chemists / Nobel laureates / History of chemistry / Chemical companies /
  Fictional chemists => drop, expand_descendants=false, include_direct_articles=false

Return ONLY valid JSON, no markdown, using exactly these keys:
{
  "label": "keep" | "mixed" | "drop",
  "confidence_score": 0 | 1 | 2 | 3,
  "reason": "one short sentence grounded in the title/path/extract/members"
}
"""

ARTICLE_SYSTEM_PROMPT = """
Your explicit task is to evaluate a Wikipedia article and determine whether its core subject is the hard science of chemistry, or if it is only tangentially related, such as a biography, company, or historical event.

You are filtering these articles for a high-precision chemistry retrieval benchmark. This article was flagged by a category graph traversal. The category path and depth are supporting context for disambiguation only. Base your decision primarily on the article title and the Wikipedia intro extract.

Inclusion criteria (label: "keep"):
Keep articles focusing strictly on fundamental chemistry-science knowledge:
- Chemical substances, molecular/atomic structures, chemical classes, reactions, and chemical bonding.
- Chemical properties, analytical methods, spectroscopy, laboratory techniques, and scientific instruments.
- Materials science compositions and fundamental chemical processes.
- Molecular-level mechanisms in biochemistry, medicinal chemistry, or pharmacology.
- Chemically focused aspects of geochemistry or environmental chemistry.
- Established chemistry subfield overview articles when the title and intro explicitly define the subject as chemistry or chemical processes, such as analytical chemistry, biochemistry, geochemistry, or environmental chemistry.
- Applied chemistry and chemical-engineering articles when the article focuses on chemical processes, reactions, separations, catalysis, reactors, materials transformation, or molecular/chemical mechanisms.

Exclusion criteria (label: "drop"):
Drop articles where chemistry is absent, historical, or incidental:
- Biographies, individual people, organizations, institutions, academic awards, or companies/brands/facilities.
- Pure history, social contexts, legal regulations, government agencies, publications, books, or journals.
- Drop broad non-chemistry or engineering/instrumentation discipline overviews, companies, facilities, management, or industrial history where chemistry is only a context rather than the core subject.
- Drop list, outline, index, timeline, or topical-guide pages unless they are specifically about chemical substances, chemical classes, reactions, or analytical methods.
- Fiction, pop culture, or generic medicine/biology/physics/geology where molecular chemistry is not the core focus.

Return ONLY a single, well-formed JSON object with no markdown or conversational text. Ensure internal quotes within strings are properly escaped. Use exactly these keys:
{
  "label": "keep" | "drop",
  "confidence_score": 0 | 1 | 2 | 3,
  "reason": "one short sentence strictly explaining the decision based on the title and intro extract"
}
"""


@dataclass
class Graph:
    categories: dict[str, dict[str, Any]]
    articles: dict[str, dict[str, Any]]
    parent_to_children: dict[str, list[str]]
    child_to_parents: dict[str, list[str]]
    category_to_articles: dict[str, list[str]]
    article_to_categories: dict[str, list[str]]


@dataclass(frozen=True)
class BranchStatus:
    category_title: str
    reachable: bool
    opens_descendants: bool
    contributes_direct_articles: bool
    label: str
    reason: str = ""


def local_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H:%M")


def normalize_title(title: str, *, category: bool = False) -> str:
    title = (title or "").strip()
    if category and title.startswith("Category:"):
        title = title[len("Category:") :]
    return title.replace(" ", "_")


def display_title(title: str, *, category: bool = False) -> str:
    return normalize_title(title, category=category).replace("_", " ")


def category_url(title: str) -> str:
    return "https://en.wikipedia.org/wiki/Category:" + quote(normalize_title(title, category=True), safe="/:_()")


def article_url(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + quote(normalize_title(title), safe="/:_()")


def resolve_jsonl_path(path: Path) -> Path:
    if path.suffix == ".jsonl":
        gz_path = path.with_suffix(path.suffix + ".gz")
        if path.exists() and gz_path.exists():
            raise FileExistsError(
                f"Found both plain and gzipped JSONL for {path}; delete one so the canonical input is unambiguous."
            )
        if gz_path.exists():
            return gz_path
    if path.exists():
        return path
    return path


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} line {line_no}: {exc}") from exc


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        f.flush()


def append_jsonl_threadsafe(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    """Append exactly one JSONL row while holding a process-local lock."""
    with lock:
        append_jsonl(path, row)


def normalize_usage(usage: dict[str, Any] | None) -> dict[str, Any]:
    """Keep OpenRouter token accounting in a consistent per-row shape."""
    raw = usage or {}
    details = raw.get("completion_tokens_details") or raw.get("completion_tokens_detail") or {}
    reasoning_tokens = raw.get("reasoning_tokens")
    if reasoning_tokens is None and isinstance(details, dict):
        reasoning_tokens = details.get("reasoning_tokens")
    return {
        "prompt_tokens": raw.get("prompt_tokens"),
        "completion_tokens": raw.get("completion_tokens"),
        "total_tokens": raw.get("total_tokens"),
        "reasoning_tokens": reasoning_tokens,
        "raw_usage": raw,
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def add_unique(mapping: dict[str, list[str]], key: str, value: str) -> None:
    values = mapping[key]
    if value not in values:
        values.append(value)


def load_graph(input_dir: Path) -> Graph:
    categories_path = resolve_jsonl_path(input_dir / "categories.jsonl")
    articles_path = resolve_jsonl_path(input_dir / "articles.jsonl")
    edges_path = resolve_jsonl_path(input_dir / "category_edges.jsonl")
    memberships_path = resolve_jsonl_path(input_dir / "article_members.jsonl")
    for path in [categories_path, articles_path, edges_path, memberships_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required graph file: {path}")

    categories = {row["category_title"]: row for row in read_jsonl(categories_path)}
    articles = {row["article_title"]: row for row in read_jsonl(articles_path)}
    parent_to_children: dict[str, list[str]] = defaultdict(list)
    child_to_parents: dict[str, list[str]] = defaultdict(list)
    category_to_articles: dict[str, list[str]] = defaultdict(list)
    article_to_categories: dict[str, list[str]] = defaultdict(list)

    for row in read_jsonl(edges_path):
        parent = row["parent_category"]
        child = row["child_category"]
        add_unique(parent_to_children, parent, child)
        add_unique(child_to_parents, child, parent)

    for row in read_jsonl(memberships_path):
        category_title = row["parent_category"]
        article_title = row["article_title"]
        add_unique(category_to_articles, category_title, article_title)
        add_unique(article_to_categories, article_title, category_title)

    return Graph(
        categories=categories,
        articles=articles,
        parent_to_children=dict(parent_to_children),
        child_to_parents=dict(child_to_parents),
        category_to_articles=dict(category_to_articles),
        article_to_categories=dict(article_to_categories),
    )


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def clamp_score(value: Any) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        score = 0
    return max(0, min(3, score))


def normalize_category_decision(raw: dict[str, Any]) -> dict[str, Any]:
    label = str(raw.get("label", "drop")).strip().lower()
    if label == "borderline":
        label = "mixed"
    if label not in {"keep", "mixed", "drop"}:
        label = "drop"
    confidence_score = clamp_score(raw.get("confidence_score"))

    decision = {
        "label": label,
        "confidence_score": confidence_score,
        "reason": str(raw.get("reason") or ""),
        "expand_descendants": label in {"keep", "mixed"},
        "include_direct_articles": label == "keep" and confidence_score >= 2,
    }
    return decision


def normalize_article_decision(raw: dict[str, Any]) -> dict[str, Any]:
    label = str(raw.get("label", "drop")).strip().lower()
    if label not in {"keep", "drop"}:
        label = "drop"
    return {
        "label": label,
        "confidence_score": clamp_score(raw.get("confidence_score")),
        "reason": str(raw.get("reason") or ""),
    }


def default_category_decision() -> dict[str, Any]:
    return {
        "label": "drop",
        "confidence_score": 0,
        "reason": "no stage1 decision found",
        "expand_descendants": False,
        "include_direct_articles": False,
    }


def load_category_decisions(path: Path) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return decisions
    for row in read_jsonl(path):
        title = row.get("category_title")
        if not title:
            continue
        classification = row.get("classification")
        raw = classification if isinstance(classification, dict) else row
        decisions[title] = normalize_category_decision(raw)
    return decisions


def load_article_decisions(path: Path) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return decisions
    for row in read_jsonl(path):
        title = row.get("article_title")
        if not title:
            continue
        classification = row.get("classification")
        raw = classification if isinstance(classification, dict) else row
        decisions[title] = normalize_article_decision(raw)
    return decisions


def parse_json_response(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("OpenRouter response JSON is not an object")
    return data


def wiki_get(params: dict[str, Any], *, retries: int = 3, sleep_s: float = 1.0) -> dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests is not installed; install dependencies with uv pip install requests")
    base = {"format": "json", "formatversion": "2", "origin": "*"}
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(WIKI_API_URL, params={**base, **params}, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001 - retain context for retry wrapper
            last_error = exc
            if attempt < retries - 1:
                time.sleep(sleep_s * (2**attempt))
    raise RuntimeError(f"Wikipedia fetch failed: {last_error}")


def fetch_category_context(title: str, *, member_limit: int = 12) -> dict[str, Any]:
    title = normalize_title(title, category=True)
    cmtitle = f"Category:{title}"
    page = wiki_get(
        {
            "action": "query",
            "titles": cmtitle,
            "prop": "extracts|categoryinfo|info",
            "explaintext": 1,
            "exintro": 1,
            "exchars": 900,
            "redirects": 1,
            "inprop": "url",
        }
    )
    pages = page.get("query", {}).get("pages", [])
    page_obj = pages[0] if pages else {}
    members = wiki_get(
        {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": cmtitle,
            "cmtype": "page|subcat",
            "cmlimit": member_limit,
            "cmsort": "sortkey",
        }
    )
    cm_rows = members.get("query", {}).get("categorymembers", [])
    return {
        "category_title": title,
        "category_display_title": display_title(title, category=True),
        "category_url": category_url(title),
        "wikipedia_extract": (page_obj.get("extract") or "").strip()[:900],
        "categoryinfo": page_obj.get("categoryinfo") or {},
        "sample_members": [
            {
                "title": member.get("title"),
                "namespace": member.get("ns"),
                "kind": "subcategory" if member.get("ns") == 14 else "page",
            }
            for member in cm_rows
        ],
    }


def fetch_article_context(title: str) -> dict[str, Any]:
    title = normalize_title(title)
    page = wiki_get(
        {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": 1,
            "exintro": 1,
            "exchars": 1000,
            "redirects": 1,
        }
    )
    pages = page.get("query", {}).get("pages", [])
    page_obj = pages[0] if pages else {}
    return {
        "wikipedia_intro_extract": (page_obj.get("extract") or "").strip()[:1000],
    }


class JsonlContextCache:
    def __init__(self, path: Path, *, kind: str) -> None:
        self.path = path
        self.kind = kind
        self.lock = threading.Lock()
        self.records: dict[str, dict[str, Any]] = {}
        self.inflight: dict[str, threading.Event] = {}
        if path.exists():
            for row in read_jsonl(path):
                if row.get("kind") == kind and row.get("title"):
                    self.records[row["title"]] = row

    def get(self, title: str) -> dict[str, Any] | None:
        with self.lock:
            record = self.records.get(title)
        if record and record.get("status") == "ok":
            return record.get("context") or {}
        return None

    def fetch(self, title: str, fetch_fn: Any) -> dict[str, Any]:
        while True:
            cached = self.get(title)
            if cached is not None:
                return cached
            with self.lock:
                event = self.inflight.get(title)
                if event is None:
                    event = threading.Event()
                    self.inflight[title] = event
                    break
            event.wait()

        try:
            try:
                context = fetch_fn(title)
                record = {
                    "title": title,
                    "kind": self.kind,
                    "fetched_at": local_timestamp(),
                    "status": "ok",
                    "context": context,
                    "error": None,
                }
            except Exception as exc:  # noqa: BLE001 - keep the run resumable/auditable
                context = {"title": title, "error": str(exc)}
                record = {
                    "title": title,
                    "kind": self.kind,
                    "fetched_at": local_timestamp(),
                    "status": "error",
                    "context": context,
                    "error": str(exc),
                }
            with self.lock:
                existing = self.records.get(title)
                if existing and existing.get("status") == "ok":
                    return existing.get("context") or {}
                append_jsonl(self.path, record)
                self.records[title] = record
            return context
        finally:
            with self.lock:
                event = self.inflight.pop(title, None)
                if event is not None:
                    event.set()


def openrouter_chat_json(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_payload: str,
    reasoning_effort: str,
    max_tokens: int,
    retries: int = 3,
) -> dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests is not installed; install dependencies with uv pip install requests")
    base_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": textwrap.dedent(system_prompt).strip()},
            {"role": "user", "content": user_payload},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "reasoning": {"effort": reasoning_effort},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/HSILA/ChemWikiRetrieval",
        "X-Title": "ChemWikiRetrieval hybrid pruning",
    }
    last_error: str | None = None
    force_low_reasoning = False
    for attempt in range(retries):
        payload = dict(base_payload)
        if force_low_reasoning and reasoning_effort != "low":
            payload["reasoning"] = {"effort": "low"}
        try:
            response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
        except Exception as exc:  # noqa: BLE001 - transport/provider stream failures are retryable
            last_error = str(exc)
        else:
            if response.status_code < 400:
                try:
                    data = response.json()
                except Exception as exc:  # noqa: BLE001 - provider returned non-JSON/partial JSON
                    last_error = f"Could not decode OpenRouter JSON response: {exc}; body={response.text[:800]}"
                else:
                    choice = (data.get("choices") or [{}])[0]
                    message = choice.get("message") or {}
                    content = message.get("content")
                    finish_reason = choice.get("finish_reason") or choice.get("native_finish_reason")
                    if not content:
                        last_error = f"Empty OpenRouter message content: {json.dumps(data)[:800]}"
                        if finish_reason == "length":
                            force_low_reasoning = True
                    else:
                        try:
                            parsed = parse_json_response(content)
                        except Exception as exc:  # noqa: BLE001 - retry transient/truncated JSON
                            last_error = f"Could not parse JSON response: {exc}; content={content[:800]!r}"
                            if finish_reason == "length":
                                force_low_reasoning = True
                        else:
                            return {
                                "raw_response": content,
                                "parsed": parsed,
                                "usage": data.get("usage") or {},
                                "response_id": data.get("id"),
                                "model_returned": data.get("model"),
                                "created": data.get("created"),
                            }
            else:
                last_error = f"HTTP {response.status_code}: {response.text[:800]}"
                if response.status_code in FATAL_OPENROUTER_STATUS_CODES:
                    raise FatalOpenRouterError(last_error)
                if response.status_code not in TRANSIENT_OPENROUTER_STATUS_CODES:
                    break
        if attempt < retries - 1:
            time.sleep(2**attempt)
    raise RuntimeError(f"OpenRouter request failed: {last_error}")


def build_local_category_context(graph: Graph, title: str, *, sample_limit: int = 12) -> dict[str, Any]:
    row = graph.categories.get(title, {"category_title": title})
    child_titles = graph.parent_to_children.get(title, [])[:sample_limit]
    article_titles = graph.category_to_articles.get(title, [])[:sample_limit]
    parent_titles = graph.child_to_parents.get(title, [])[:sample_limit]
    return {
        "category_title": title,
        "category_display_title": row.get("category_display_title") or display_title(title, category=True),
        "category_url": row.get("category_url") or category_url(title),
        "bfs_depth": row.get("depth"),
        "path": row.get("path"),
        "bfs_counts": {
            "cat_pages": row.get("cat_pages"),
            "cat_subcats": row.get("cat_subcats"),
            "cat_files": row.get("cat_files"),
            "cat_article_members_estimate": row.get("cat_article_members_estimate"),
        },
        "parent_categories": [
            {
                "category_title": parent,
                "category_display_title": graph.categories.get(parent, {}).get("category_display_title")
                or display_title(parent, category=True),
            }
            for parent in parent_titles
        ],
        "child_category_sample": [
            {
                "category_title": child,
                "category_display_title": graph.categories.get(child, {}).get("category_display_title")
                or display_title(child, category=True),
            }
            for child in child_titles
        ],
        "direct_article_sample": [
            {
                "article_title": article,
                "article_display_title": graph.articles.get(article, {}).get("article_display_title") or display_title(article),
            }
            for article in article_titles
        ],
    }


def build_category_user_payload(local_context: dict[str, Any], wiki_context: dict[str, Any]) -> str:
    compact = {
        "task": "classify_wikipedia_category_for_chemistry_pruning",
        "local_graph_context": local_context,
        "fetched_wikipedia_context": wiki_context,
    }
    return "Classify this category for chemistry graph pruning:\n" + json.dumps(compact, ensure_ascii=False, indent=2)


def build_article_user_payload(candidate: dict[str, Any], graph: Graph, wiki_context: dict[str, Any]) -> str:
    title = candidate["article_title"]
    row = graph.articles.get(title, {"article_title": title})
    compact = {
        "task": "classify_wikipedia_article_for_chemistry_subset",
        "article": {
            "article_title": title,
            "bfs_depth": row.get("depth"),
            "path": row.get("path"),
        },
        "fetched_wikipedia_context": {
            "wikipedia_intro_extract": (wiki_context.get("wikipedia_intro_extract") or wiki_context.get("wikipedia_extract") or "")[:1000]
        },
    }
    return "Classify this article for the chemistry retrieval subset:\n" + json.dumps(compact, ensure_ascii=False, indent=2)


def root_category(graph: Graph) -> str:
    if "Chemistry" in graph.categories:
        return "Chemistry"
    depth_zero = [title for title, row in graph.categories.items() if row.get("depth") == 0]
    return sorted(depth_zero)[0] if depth_zero else sorted(graph.categories)[0]


def compute_branch_status(graph: Graph, decisions: dict[str, dict[str, Any]]) -> dict[str, BranchStatus]:
    root = root_category(graph)
    status: dict[str, BranchStatus] = {}

    def make_status(title: str, *, reachable: bool) -> BranchStatus:
        decision = decisions.get(title, default_category_decision())
        label = decision["label"]
        opens = bool(reachable and decision.get("expand_descendants") and label != "drop")
        contributes = bool(reachable and label == "keep" and decision.get("include_direct_articles"))
        return BranchStatus(
            category_title=title,
            reachable=reachable,
            opens_descendants=opens,
            contributes_direct_articles=contributes,
            label=label,
            reason=decision.get("reason", ""),
        )

    queue: deque[str] = deque([root])
    status[root] = make_status(root, reachable=True)
    while queue:
        parent = queue.popleft()
        parent_status = status[parent]
        if not parent_status.opens_descendants:
            continue
        for child in graph.parent_to_children.get(parent, []):
            existing_child_status = status.get(child)
            child_was_reachable = existing_child_status.reachable if existing_child_status else False
            if not child_was_reachable:
                status[child] = make_status(child, reachable=True)
                queue.append(child)

    for title in graph.categories:
        if title not in status:
            status[title] = make_status(title, reachable=False)
    return status


def category_evidence(graph: Graph, category_title: str, decision: dict[str, Any]) -> dict[str, Any]:
    row = graph.categories.get(category_title, {})
    return {
        "category_title": category_title,
        "category_display_title": row.get("category_display_title") or display_title(category_title, category=True),
        "label": decision.get("label"),
        "confidence_score": decision.get("confidence_score"),
        "reason": decision.get("reason"),
        "include_direct_articles": decision.get("include_direct_articles"),
        "expand_descendants": decision.get("expand_descendants"),
        "path": row.get("path"),
    }


def build_article_candidates(
    graph: Graph,
    decisions: dict[str, dict[str, Any]],
    branch_status: dict[str, BranchStatus],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for article_title in graph.articles:
        member_categories = graph.article_to_categories.get(article_title, [])
        reachable_categories: list[dict[str, Any]] = []
        keep_categories: list[dict[str, Any]] = []
        review_categories: list[dict[str, Any]] = []
        for category_title in member_categories:
            status = branch_status.get(category_title)
            if not status or not status.reachable:
                continue
            decision = decisions.get(category_title, default_category_decision())
            evidence = category_evidence(graph, category_title, decision)
            reachable_categories.append(evidence)
            if decision["label"] == "keep" and decision.get("include_direct_articles"):
                keep_categories.append(evidence)
            elif decision["label"] in {"mixed", "keep"}:
                review_categories.append(evidence)

        if keep_categories:
            candidate_status = "auto_keep_candidate"
            reasons = ["article appears under at least one reachable strong keep/direct-include category"]
        elif review_categories:
            candidate_status = "needs_article_review"
            reasons = ["article appears under a reachable mixed or weak-keep category and lacks a strong keep category"]
        else:
            candidate_status = "auto_drop_context"
            reasons = ["article has no reachable category requiring keep or article review"]

        article_row = graph.articles.get(article_title, {})
        candidates.append(
            {
                "article_title": article_title,
                "article_display_title": article_row.get("article_display_title") or display_title(article_title),
                "article_url": article_row.get("article_url") or article_url(article_title),
                "candidate_status": candidate_status,
                "reachable_categories": reachable_categories,
                "keep_categories": keep_categories,
                "review_categories": review_categories,
                "all_member_categories": member_categories,
                "reasons": reasons,
            }
        )
    return candidates


def summarize_candidate_plan(candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(candidates)
    counts = Counter(row.get("candidate_status") for row in rows)
    return {
        "total_unique_articles": len(rows),
        "auto_keep_candidate": counts.get("auto_keep_candidate", 0),
        "needs_article_review": counts.get("needs_article_review", 0),
        "auto_drop_context": counts.get("auto_drop_context", 0),
    }


def summarize_branch_status(status: dict[str, BranchStatus], decisions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    label_counts = Counter(decision.get("label") for decision in decisions.values())
    reachable = sum(1 for item in status.values() if item.reachable)
    open_count = sum(1 for item in status.values() if item.opens_descendants)
    direct_count = sum(1 for item in status.values() if item.contributes_direct_articles)
    return {
        "category_decisions": len(decisions),
        "label_counts": dict(label_counts),
        "reachable_categories": reachable,
        "open_descendant_categories": open_count,
        "direct_article_categories": direct_count,
    }


def print_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def selected_categories(graph: Graph, limit: int | None) -> list[dict[str, Any]]:
    rows = list(graph.categories.values())
    rows.sort(key=lambda row: (row.get("depth") if row.get("depth") is not None else 10**9, row.get("category_title", "")))
    if limit is not None:
        rows = rows[:limit]
    return rows


def run_stage1(args: argparse.Namespace) -> int:
    input_dir: Path = args.input_dir
    output_dir: Path = input_dir / "pruned_hybrid"
    decisions_path = output_dir / "category_decisions.jsonl"
    errors_path = output_dir / "category_errors.jsonl"
    cache = JsonlContextCache(output_dir / "category_context_cache.jsonl", kind="category")
    graph = load_graph(input_dir)
    existing = load_category_decisions(decisions_path)
    rows = selected_categories(graph, args.limit)

    if args.dry_run:
        print_json(
            {
                "stage": "stage1",
                "dry_run": True,
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "selected_categories": len(rows),
                "already_classified": sum(1 for row in rows if row["category_title"] in existing),
                "would_call_llm": sum(1 for row in rows if row["category_title"] not in existing),
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "concurrency": args.concurrency,
            }
        )
        return 0

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY is not set; pass --env-file or export it", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    skipped = 0
    failed = 0
    run_started = time.monotonic()
    total_rows = len(rows)
    write_lock = threading.Lock()
    print_lock = threading.Lock()
    usage_totals: dict[str, float] = defaultdict(float)
    token_usage_totals: dict[str, float] = defaultdict(float)
    pending_rows: list[dict[str, Any]] = []
    for row in rows:
        title = row["category_title"]
        if title in existing:
            skipped += 1
        else:
            pending_rows.append(row)

    existing_selected = sum(1 for row in rows if row["category_title"] in existing)

    def progress_fields() -> dict[str, Any]:
        attempted_this_run = processed + failed
        completed_total = existing_selected + processed
        remaining_total = max(total_rows - completed_total, 0)
        elapsed_s = max(time.monotonic() - run_started, 0.001)
        sec_per_attempt = elapsed_s / attempted_this_run if attempted_this_run else None
        eta_s = remaining_total * sec_per_attempt if sec_per_attempt is not None else None
        return {
            "processed_this_run": processed,
            "failed_this_run": failed,
            "skipped_existing": skipped,
            "completed_total": completed_total,
            "total_categories": total_rows,
            "percent_total": round((completed_total / total_rows) * 100, 2) if total_rows else 100.0,
            "concurrency": args.concurrency,
            "elapsed_s": round(elapsed_s, 1),
            "eta_s": round(eta_s, 1) if eta_s is not None else None,
            "eta_hours": round(eta_s / 3600, 2) if eta_s is not None else None,
        }

    def classify_category(row: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
        title = row["category_title"]
        try:
            local_context = build_local_category_context(graph, title)
            wiki_context = cache.fetch(title, fetch_category_context)
            user_payload = build_category_user_payload(local_context, wiki_context)
            response = openrouter_chat_json(
                api_key=api_key,
                model=args.model,
                system_prompt=CATEGORY_SYSTEM_PROMPT,
                user_payload=user_payload,
                reasoning_effort=args.reasoning_effort,
                max_tokens=args.max_tokens,
            )
            classification = normalize_category_decision(response["parsed"])
            usage = response.get("usage") or {}
            token_usage = normalize_usage(usage)
            decision_row = build_category_decision_row(row, args, classification, response)
            return "success", decision_row, {"usage": usage, "token_usage": token_usage}
        except Exception as exc:  # noqa: BLE001 - log and continue batch
            error_row = {
                "stage": 1,
                "kind": "category_error",
                "created_at": local_timestamp(),
                "category_title": title,
                "error": str(exc),
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
            }
            return "error", error_row, {}

    with ThreadPoolExecutor(max_workers=max(args.concurrency, 1)) as pool:
        futures = [pool.submit(classify_category, row) for row in pending_rows]
        for future in as_completed(futures):
            status, result_row, meta = future.result()
            if status == "success":
                append_jsonl_threadsafe(decisions_path, result_row, write_lock)
                processed += 1
                for key, value in (meta.get("usage") or {}).items():
                    if isinstance(value, (int, float)):
                        usage_totals[key] += float(value)
                for key, value in (meta.get("token_usage") or {}).items():
                    if isinstance(value, (int, float)):
                        token_usage_totals[key] += float(value)
                message = {
                    "category_title": result_row["category_title"],
                    "label": result_row["label"],
                    **progress_fields(),
                }
                with print_lock:
                    print(json.dumps(message, ensure_ascii=False), flush=True)
            else:
                append_jsonl_threadsafe(errors_path, result_row, write_lock)
                failed += 1
                message = {"category_title": result_row["category_title"], "error": result_row["error"], **progress_fields()}
                with print_lock:
                    print(json.dumps(message, ensure_ascii=False), file=sys.stderr, flush=True)

    summary = {
        "stage": "stage1",
        "created_at": local_timestamp(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "selected_categories": len(rows),
        "processed": processed,
        "skipped_existing": skipped,
        "failed": failed,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "concurrency": args.concurrency,
        "prompt_version": CATEGORY_PROMPT_VERSION,
        "usage_totals": dict(usage_totals),
        "token_usage_totals": dict(token_usage_totals),
    }
    (output_dir / "stage1_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_json(summary)
    return 0 if failed == 0 else 1


def review_candidates(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in candidates if row.get("candidate_status") == "needs_article_review"]


def pending_review_candidates(
    review_rows: list[dict[str, Any]],
    existing_article_decisions: dict[str, dict[str, Any]],
    limit: int | None,
) -> list[dict[str, Any]]:
    pending = [row for row in review_rows if row["article_title"] not in existing_article_decisions]
    if limit is not None:
        return pending[:limit]
    return pending


def build_category_decision_row(
    row: dict[str, Any],
    args: argparse.Namespace,
    classification: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    title = row["category_title"]
    return {
        "category_title": title,
        "category_display_title": row.get("category_display_title") or display_title(title, category=True),
        "category_url": row.get("category_url") or category_url(title),
        **classification,
        "depth": row.get("depth"),
        "path": row.get("path"),
        "created_at": local_timestamp(),
        "model": args.model,
        "model_returned": response.get("model_returned"),
        "reasoning_effort": args.reasoning_effort,
        "response_id": response.get("response_id"),
        "raw_response": response.get("raw_response"),
        "usage": response.get("usage") or {},
    }


def build_article_decision_row(
    candidate: dict[str, Any],
    args: argparse.Namespace,
    classification: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    title = candidate["article_title"]
    return {
        "article_title": title,
        "article_display_title": candidate.get("article_display_title") or display_title(title),
        "article_url": candidate.get("article_url") or article_url(title),
        "candidate_status": candidate["candidate_status"],
        **classification,
        "created_at": local_timestamp(),
        "model": args.model,
        "model_returned": response.get("model_returned"),
        "prompt_version": ARTICLE_PROMPT_VERSION,
        "reasoning_effort": args.reasoning_effort,
        "response_id": response.get("response_id"),
        "raw_response": response.get("raw_response"),
        "usage": response.get("usage") or {},
        "review_categories": candidate.get("review_categories", []),
    }


def final_article_rows(
    graph: Graph,
    candidates: Iterable[dict[str, Any]],
    article_decisions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    final_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        title = candidate["article_title"]
        include = False
        inclusion_source = ""
        if candidate["candidate_status"] == "auto_keep_candidate":
            include = True
            inclusion_source = "auto_keep_category"
        elif candidate["candidate_status"] == "needs_article_review":
            decision = article_decisions.get(title)
            if decision and decision.get("label") == "keep" and int(decision.get("confidence_score", 0)) >= 2:
                include = True
                inclusion_source = "stage2_article_keep"
        if not include:
            continue
        article = graph.articles.get(title, {})
        final_rows.append(
            {
                "article_title": title,
                "article_display_title": article.get("article_display_title") or display_title(title),
                "article_url": article.get("article_url") or article_url(title),
                "inclusion_source": inclusion_source,
                "candidate_status": candidate["candidate_status"],
                "supporting_categories": candidate.get("keep_categories") or candidate.get("review_categories"),
            }
        )
    return final_rows


def run_stage2(args: argparse.Namespace) -> int:
    run_started = time.monotonic()
    input_dir: Path = args.input_dir
    output_dir: Path = input_dir / "pruned_hybrid"
    decisions_path: Path = output_dir / "category_decisions.jsonl"
    article_decisions_path = output_dir / "article_decisions.jsonl"
    final_path = output_dir / "final_chemistry_articles.jsonl"
    errors_path = output_dir / "article_errors.jsonl"
    graph = load_graph(input_dir)
    category_decisions = load_category_decisions(decisions_path)
    if not category_decisions:
        print(f"No category decisions found at {decisions_path}; run stage1 first", file=sys.stderr)
        return 2

    branch_status = compute_branch_status(graph, category_decisions)
    candidates = build_article_candidates(graph, category_decisions, branch_status)
    plan_summary = summarize_candidate_plan(candidates)
    branch_summary = summarize_branch_status(branch_status, category_decisions)
    missing_decisions = len(set(graph.categories) - set(category_decisions))
    review_rows = review_candidates(candidates)
    existing_article_decisions = load_article_decisions(article_decisions_path)
    already_reviewed = sum(1 for row in review_rows if row["article_title"] in existing_article_decisions)
    pending_review_rows = pending_review_candidates(review_rows, existing_article_decisions, limit=None)
    review_rows_to_call = pending_review_candidates(review_rows, existing_article_decisions, args.limit)

    dry_run_summary = {
        "stage": "stage2",
        "dry_run": bool(args.dry_run),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "category_decisions_path": str(decisions_path),
        "model": args.model,
        "prompt_version": ARTICLE_PROMPT_VERSION,
        "reasoning_effort": args.reasoning_effort,
        "concurrency": args.concurrency,
        "missing_category_decisions": missing_decisions,
        "branch_summary": branch_summary,
        "candidate_summary": plan_summary,
        "stage2_llm_review_articles_total": len(review_rows),
        "stage2_llm_review_articles_already_done": already_reviewed,
        "stage2_llm_review_articles_pending": len(pending_review_rows),
        "stage2_llm_review_articles_this_run": len(review_rows_to_call),
        "estimated_stage2_cost_usd_this_run": round(len(review_rows_to_call) * args.estimate_cost_per_call_usd, 6),
        "estimated_stage2_cost_usd_pending": round(len(pending_review_rows) * args.estimate_cost_per_call_usd, 6),
        "estimated_stage2_cost_usd_all_review_articles": round(len(review_rows) * args.estimate_cost_per_call_usd, 6),
        "planning_elapsed_s": round(time.monotonic() - run_started, 3),
        "sample_review_articles": [row["article_title"] for row in review_rows_to_call[:20]],
    }
    if args.dry_run:
        print_json(dry_run_summary)
        return 0

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY is not set; pass --env-file or export it", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    cache = JsonlContextCache(output_dir / "article_context_cache.jsonl", kind="article")

    processed = 0
    skipped = already_reviewed
    failed = 0
    write_lock = threading.Lock()
    print_lock = threading.Lock()
    fatal_lock = threading.Lock()
    fatal_stop = threading.Event()
    fatal_error: str | None = None
    usage_totals: dict[str, float] = defaultdict(float)
    token_usage_totals: dict[str, float] = defaultdict(float)
    total_review_rows = len(review_rows)
    existing_selected = already_reviewed

    def progress_fields() -> dict[str, Any]:
        attempted_this_run = processed + failed
        completed_total = existing_selected + processed
        remaining_total = max(total_review_rows - completed_total, 0)
        elapsed_s = max(time.monotonic() - run_started, 0.001)
        sec_per_attempt = elapsed_s / attempted_this_run if attempted_this_run else None
        eta_s = remaining_total * sec_per_attempt if sec_per_attempt is not None else None
        return {
            "processed_this_run": processed,
            "failed_this_run": failed,
            "skipped_existing": skipped,
            "completed_total": completed_total,
            "total_article_reviews": total_review_rows,
            "percent_total": round((completed_total / total_review_rows) * 100, 2) if total_review_rows else 100.0,
            "concurrency": args.concurrency,
            "elapsed_s": round(elapsed_s, 1),
            "eta_s": round(eta_s, 1) if eta_s is not None else None,
            "eta_hours": round(eta_s / 3600, 2) if eta_s is not None else None,
        }

    def classify_article(candidate: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
        nonlocal fatal_error
        title = candidate["article_title"]
        if fatal_stop.is_set():
            return "skipped_fatal", {"article_title": title}, {}
        try:
            wiki_context = cache.fetch(title, fetch_article_context)
            user_payload = build_article_user_payload(candidate, graph, wiki_context)
            response = openrouter_chat_json(
                api_key=api_key,
                model=args.model,
                system_prompt=ARTICLE_SYSTEM_PROMPT,
                user_payload=user_payload,
                reasoning_effort=args.reasoning_effort,
                max_tokens=args.max_tokens,
            )
            classification = normalize_article_decision(response["parsed"])
            usage = response.get("usage") or {}
            token_usage = normalize_usage(usage)
            decision_row = build_article_decision_row(candidate, args, classification, response)
            return "success", decision_row, {"usage": usage, "token_usage": token_usage}
        except FatalOpenRouterError as exc:
            with fatal_lock:
                if fatal_stop.is_set():
                    return "skipped_fatal", {"article_title": title}, {}
                fatal_error = str(exc)
                fatal_stop.set()
            error_row = {
                "stage": 2,
                "kind": "article_error",
                "created_at": local_timestamp(),
                "article_title": title,
                "error": str(exc),
                "fatal": True,
                "model": args.model,
                "prompt_version": ARTICLE_PROMPT_VERSION,
                "reasoning_effort": args.reasoning_effort,
            }
            return "fatal", error_row, {}
        except Exception as exc:  # noqa: BLE001 - log and continue batch
            error_row = {
                "stage": 2,
                "kind": "article_error",
                "created_at": local_timestamp(),
                "article_title": title,
                "error": str(exc),
                "model": args.model,
                "prompt_version": ARTICLE_PROMPT_VERSION,
                "reasoning_effort": args.reasoning_effort,
            }
            return "error", error_row, {}

    pending_rows = [row for row in review_rows_to_call if row["article_title"] not in existing_article_decisions]
    skipped += len(review_rows_to_call) - len(pending_rows)

    with ThreadPoolExecutor(max_workers=max(args.concurrency, 1)) as pool:
        futures = [pool.submit(classify_article, candidate) for candidate in pending_rows]
        for future in as_completed(futures):
            status, result_row, meta = future.result()
            if status == "success":
                append_jsonl_threadsafe(article_decisions_path, result_row, write_lock)
                existing_article_decisions[result_row["article_title"]] = {
                    "label": result_row["label"],
                    "confidence_score": result_row["confidence_score"],
                    "reason": result_row.get("reason", ""),
                }
                processed += 1
                for key, value in (meta.get("usage") or {}).items():
                    if isinstance(value, (int, float)):
                        usage_totals[key] += float(value)
                for key, value in (meta.get("token_usage") or {}).items():
                    if isinstance(value, (int, float)):
                        token_usage_totals[key] += float(value)
                message = {"article_title": result_row["article_title"], "label": result_row["label"], **progress_fields()}
                with print_lock:
                    print(json.dumps(message, ensure_ascii=False), flush=True)
            elif status == "skipped_fatal":
                continue
            elif status == "fatal":
                append_jsonl_threadsafe(errors_path, result_row, write_lock)
                failed += 1
                message = {"article_title": result_row["article_title"], "error": result_row["error"], **progress_fields()}
                with print_lock:
                    print(json.dumps(message, ensure_ascii=False), file=sys.stderr, flush=True)
            else:
                append_jsonl_threadsafe(errors_path, result_row, write_lock)
                failed += 1
                message = {"article_title": result_row["article_title"], "error": result_row["error"], **progress_fields()}
                with print_lock:
                    print(json.dumps(message, ensure_ascii=False), file=sys.stderr, flush=True)

    final_written = False
    final_count = None
    if args.limit is None:
        final_rows = final_article_rows(graph, candidates, existing_article_decisions)
        final_count = write_jsonl(final_path, final_rows)
        final_written = True

    summary = {
        **dry_run_summary,
        "dry_run": False,
        "processed": processed,
        "skipped_existing": skipped,
        "failed": failed,
        "fatal_error": fatal_error,
        "usage_totals": dict(usage_totals),
        "token_usage_totals": dict(token_usage_totals),
        "wrote_final_articles": str(final_path) if final_written else None,
        "final_chemistry_articles": final_count,
        "note": "final file is only written for full stage2 runs without --limit",
    }
    (output_dir / "stage2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_json(summary)
    return 0 if failed == 0 else 1


def add_common_args(parser: argparse.ArgumentParser, *, model_default: str, max_tokens_default: int) -> None:
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--model", default=model_default)
    parser.add_argument("--env-file", type=Path, default=Path.home() / ".hermes" / ".env")

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent OpenRouter workers for LLM calls")
    parser.add_argument("--max-tokens", type=int, default=max_tokens_default)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage1 = subparsers.add_parser("stage1", help="Classify categories as keep/mixed/drop")
    add_common_args(stage1, model_default=DEFAULT_CATEGORY_MODEL, max_tokens_default=1024)
    stage1.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high", "xhigh"])
    stage1.add_argument("--dry-run", action="store_true", help="Plan selected category calls without OpenRouter")
    stage1.set_defaults(func=run_stage1)

    stage2 = subparsers.add_parser("stage2", help="Plan/classify article reviews from mixed categories")
    add_common_args(stage2, model_default=DEFAULT_ARTICLE_MODEL, max_tokens_default=2048)
    stage2.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high", "xhigh"])
    stage2.add_argument("--dry-run", action="store_true", help="Count article-review calls without OpenRouter or writes")
    stage2.add_argument(
        "--estimate-cost-per-call-usd",
        type=float,
        default=0.0001,
        help="Rough stage2 per-article cost estimate used only in summaries",
    )
    stage2.set_defaults(func=run_stage2)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.env_file:
        load_env_file(args.env_file)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
