# NQ pipeline

This folder contains the Natural Questions subset and the scripts that generated it.

The goal is to start from the repo's chemistry article list, stream Natural Questions from Hugging Face, keep rows whose Wikipedia page is in scope, clean the topic matches, save the source HTML pages, and extract clean paragraph blocks for retrieval.

## Current outputs

- `matches.jsonl`
  - Final answerable NQ question rows kept for this subset.
  - Current count: 3,001 rows.

- `topic_filter_gemma4.jsonl`
  - Gemma topic-filter decisions for the matched NQ rows.
  - Current count: 3,001 rows.

- `html_pages/`
  - Saved Wikipedia HTML snapshots from the NQ `document.html` field.
  - Current count: 2,369 HTML files.
  - This folder is ignored and not included in the Git repository because it is large.
  - `extract_html_pages.py` also creates a local `nq/html_pages.tar.gz` archive, but that archive is ignored too because it is over GitHub's normal file-size limit.

- `article_paragraphs.jsonl`
  - Clean paragraph/list blocks extracted from the saved HTML files.
  - Current count: 75,880 rows.

- `scripts/`
  - The active scripts for this NQ pipeline.

## Input article list

The pipeline starts from the chemistry article list created by the Wikipedia category pruning step:

```text
graph/data/chemistry_bfs_depth5/pruned_hybrid/final_chemistry_articles.jsonl
```

This file is outside `nq/` because it is shared graph/category data, not Natural Questions data.

## Step 1: Stream NQ and match by Wikipedia title

Script:

```text
nq/scripts/produce_matches.py
```

What it does:

- Streams `rongzhangibm/NaturalQuestionsV2` from Hugging Face.
- Uses the train Arrow shard URLs and the validation Arrow URL embedded in the script.
- Does not locally cache the full NQ dataset.
- Loads article titles from `final_chemistry_articles.jsonl`.
- Matches each NQ row by `document.title`.
- Removes rows with no annotated answer evidence.
- Writes matching question rows to `nq/matches.jsonl`.

Command:

```bash
cd /home/spark/projects/research/ChemWikiRetrieval
.venv/bin/python nq/scripts/produce_matches.py \
  --chem-titles graph/data/chemistry_bfs_depth5/pruned_hybrid/final_chemistry_articles.jsonl \
  --output nq/matches.jsonl \
  --state-dir nq/state
```

If this script is rerun, runtime checkpoints go under `nq/state/`. That folder is not part of the final dataset and can be removed after a successful run.

## Step 2: Clean topic matches with Gemma

Script:

```text
nq/scripts/filter_topic_gemma4.py
```

What it does:

- Reads `nq/matches.jsonl`.
- Uses `gemma4:31b` through Ollama Cloud.
- Classifies whether each question/page is actually in the chemistry scope.
- Keeps interdisciplinary chemistry, such as biochemistry, materials, metallurgy, geochemistry, molecular biology, environmental chemistry, and molecular processes.
- Drops unrelated or too-general matches, such as pure geography, non-chemical etymology, or general non-chemical physics.
- Writes decisions to `nq/topic_filter_gemma4.jsonl`.

Command:

```bash
cd /home/spark/projects/research/ChemWikiRetrieval
.venv/bin/python nq/scripts/filter_topic_gemma4.py
```

Historical note for this run:

- The broad title-match stage found 7,011 NQ rows.
- Gemma kept 6,609 as topic-relevant and rejected 399 as noise.
- After removing rows without answer annotations, the active answerable title-match file has 3,001 rows.
- The active topic filter has 2,816 `chemistry=true` rows and 185 `chemistry=false` rows.

## Step 3: Use only rows with annotated answers

For retrieval benchmark construction, rows without usable NQ answer annotations are removed by `nq/scripts/produce_matches.py`.

Reason:

- NQ can contain questions whose page title matches the target topic but whose annotations do not include a usable answer span.
- Those rows are poor retrieval examples because there is no positive answer evidence to anchor the query.

Current kept answerable rows:

- 3,001 total rows in `nq/matches.jsonl`.
- 1,910 rows have both short and long answer evidence.
- 1,091 rows have long answer evidence only.

Rows with no annotated answer are not written to `nq/matches.jsonl`. This keeps the first NQ file focused on answerable retrieval candidates before the Gemma topic filter is applied.

## Step 4: Save the NQ Wikipedia HTML snapshots

Script:

```text
nq/scripts/extract_html_pages.py
```

What it does:

- Reads `nq/matches.jsonl` and `nq/topic_filter_gemma4.jsonl`.
- Keeps rows marked in-scope by the Gemma topic filter.
- Streams the corresponding NQ records again from Hugging Face.
- Saves the embedded `document.html` snapshots from NQ, not live Wikipedia.
- Writes HTML files to `nq/html_pages/`.
- Creates a local gzip archive at `nq/html_pages.tar.gz` after the HTML files are saved.
- Both `nq/html_pages/` and `nq/html_pages.tar.gz` are ignored by Git because they are large local source snapshots.

Why embedded HTML is used:

- It preserves the exact Wikipedia snapshot bundled with NQ.
- It avoids drift from current live Wikipedia pages.
- It keeps the retrieval corpus faithful to the source benchmark.

Command:

```bash
cd /home/spark/projects/research/ChemWikiRetrieval
.venv/bin/python nq/scripts/extract_html_pages.py
```

## Step 5: Extract clean article paragraphs

Script:

```text
nq/scripts/extract_article_paragraphs.py
```

What it does:

- Reads HTML files from `nq/html_pages/`.
- Extracts paragraph and list-like text blocks.
- Removes navigation, references, tables, infoboxes, citations, edit labels, and other page chrome.
- Preserves compact raw LaTeX from math image alt text when useful.
- Drops long display equations and boilerplate fragments that make poor retrieval text.
- Writes clean blocks to `nq/article_paragraphs.jsonl`.

Command:

```bash
cd /home/spark/projects/research/ChemWikiRetrieval
.venv/bin/python nq/scripts/extract_article_paragraphs.py
```
