# Wikipedia category graph pipeline

This folder builds and prunes a Wikipedia category graph rooted at `Category:Chemistry`.

## Raw dumps

Raw Wikimedia SQL dumps are **not committed**. Place them under `graph/dumps/`:

```bash
mkdir -p graph/dumps
curl -L -C - -o graph/dumps/enwiki-20171001-page.sql.gz \
  https://archive.org/download/enwiki-20171001/enwiki-20171001-page.sql.gz
curl -L -C - -o graph/dumps/enwiki-20171001-categorylinks.sql.gz \
  https://archive.org/download/enwiki-20171001/enwiki-20171001-categorylinks.sql.gz
curl -L -C - -o graph/dumps/enwiki-20171001-category.sql.gz \
  https://archive.org/download/enwiki-20171001/enwiki-20171001-category.sql.gz
```

Expected source files:

- `enwiki-20171001-page.sql.gz`
- `enwiki-20171001-categorylinks.sql.gz`
- `enwiki-20171001-category.sql.gz`

## Build the high-recall graph

```bash
uv run python graph/build_bfs.py \
  --graph-dir graph \
  --dump-dir dumps \
  --seed Chemistry \
  --max-depth 5
```

Outputs default to `graph/data/chemistry_bfs_depth5/`.

Core graph files:

- `categories.jsonl`: unique reached categories
- `category_edges.jsonl`: parent to child category edges
- `articles.jsonl`: unique reached article pages
- `article_members.jsonl.gz`: gzipped category to direct article memberships
- `summary.json`: build metadata

The `frontiers/` files and `progress.jsonl` are local debug artifacts. They are useful for inspecting BFS expansion, but pruning reads only the core graph files above. `prune_hybrid.py` reads `article_members.jsonl.gz` directly, so decompression is optional.

To inspect or restore the plain JSONL locally:

```bash
gzip -dk graph/data/chemistry_bfs_depth5/article_members.jsonl.gz
```

## Hybrid pruning

```bash
uv run python graph/prune_hybrid.py stage1 --input-dir graph/data/chemistry_bfs_depth5
uv run python graph/prune_hybrid.py stage2 --input-dir graph/data/chemistry_bfs_depth5
```

Pruning outputs live in:

```text
graph/data/chemistry_bfs_depth5/pruned_hybrid/
```

Key final artifact:

- `final_chemistry_articles.jsonl`: retained high-precision chemistry article set

Other pruning artifacts:

- `category_decisions.jsonl`: raw Stage 1 LLM decisions for every category: label, confidence, reason, model metadata, and token usage.
- `article_decisions.jsonl`: Stage 2 LLM decisions for the `needs_article_review` articles.

`branch_status` is computed in memory from `category_decisions.jsonl` and the graph topology during Stage 2. It is not written as a separate file because it is fully derived.
`article_candidates` is also computed in memory during Stage 2. It is derived from the graph, category decisions, and branch status.
The context cache files are local rerun caches and are ignored by git.

Current canonical result:

- Stage 2 reviewed articles: `29,926`
- Stage 2 canonical keeps: `8,446`
- Stage 2 drops: `21,471`
- Auto-kept from strong category branches: `32,614`
- Final retained articles: `41,060`
