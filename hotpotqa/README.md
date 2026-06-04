# HotpotQA chemistry subset

This folder contains the HotpotQA chemistry subset built from the Hugging Face dataset `mteb/hotpotqa`.

The pipeline is simpler than NQ because HotpotQA in MTEB already has query, corpus, and qrel files. We do not extract Wikipedia HTML. We only match qrel-linked corpus document titles against the pruned chemistry Wikipedia title set, then run a small Gemma topic filter to remove incidental matches.

## Current counts

Source: `mteb/hotpotqa`

Chemistry title set: 41,060 Wikipedia articles

## MTEB HotpotQA source check

`mteb/hotpotqa` was checked against the local original HotpotQA files before using it as the rebuild source.

Original HotpotQA counts:

- train: 90,447 questions
- dev fullwiki: 7,405 questions
- dev distractor: 7,405 questions
- train plus one dev split: 97,852 unique question IDs

`mteb/hotpotqa` counts:

- queries config: 97,852 total query IDs
- qrel query IDs:
  - train: 85,000
  - dev: 5,447
  - test: 7,405

Verification results:

- MTEB query IDs exactly match original train plus one dev split.
- MTEB qrel train plus dev query IDs exactly match original train IDs.
- MTEB qrel test query IDs exactly match original dev IDs.
- Every qrel query ID exists in the MTEB queries config.

Conclusion: unlike `mteb/nq`, `mteb/hotpotqa` is a representative retrieval-formatted version of original HotpotQA. It is safe to build the chemistry subset directly from `mteb/hotpotqa`.

Initial qrel-title matches:

- train: 258 / 85,000 queries
- dev: 15 / 5,447 queries
- test: 28 / 7,405 queries
- unique matched questions: 301

Gemma topic filter:

- model: `gemma4:31b`
- workers: 16
- valid processed rows: 301
- chemistry true: 274
- chemistry false: 27
- errors: 0

Note: an earlier validation against original HotpotQA gave the same 301 unique gold-evidence questions. Those original-HotpotQA validation artifacts are archived under the root `backups/` folder, not kept in this active directory.

## Temporary broad-corpus rebuild variants

The current uploaded new HotpotQA dataset used only qrel-positive documents as corpus, which produced a positive-only corpus of 464 documents. That is too narrow for a retrieval benchmark because it has no chemistry distractor pool.

A temporary broad-corpus rebuild was created under:

`hotpotqa/tmp_mteb_broad_chemistry/`

Both variants use the same corpus strategy:

- include all `mteb/hotpotqa` corpus rows whose title matches the 41,060 chemistry Wikipedia title set
- force-add any selected qrel-positive documents that do not pass the chemistry title filter
- keep all HotpotQA qrels for selected queries, which gives exactly 2 positives per query
- verify that no qrel-positive document is missing from the corpus

The variants differ only in query selection.

### `title_match`

This is the pre-cleaning mechanical filter. A query is kept when at least one qrel-linked gold document title matches the chemistry title set.

Stats:

- queries: 301
  - train: 258
  - dev: 15
  - test: 28
- qrels: 602
  - train: 516
  - dev: 30
  - test: 56
- qrels per query: exactly 2
- unique qrel-positive corpus docs: 507
- corpus docs: 40,289
- corpus docs from chemistry title filter: 40,070
- qrel-positive docs force-added to corpus: 219
- unreferenced corpus docs: 39,782
- missing qrel-positive docs: 0

Interpretation: higher recall, slightly noisier query set.

### `gemma_true`

This starts from `title_match`, then keeps only rows where the Gemma cleanup pass marked the query as actually chemistry-related.

Stats:

- queries: 274
  - train: 234
  - dev: 14
  - test: 26
- qrels: 548
  - train: 468
  - dev: 28
  - test: 52
- qrels per query: exactly 2
- unique qrel-positive corpus docs: 464
- corpus docs: 40,262
- corpus docs from chemistry title filter: 40,070
- qrel-positive docs force-added to corpus: 192
- unreferenced corpus docs: 39,798
- missing qrel-positive docs: 0

Interpretation: higher precision, fewer queries. This is the preferred candidate unless the removed queries are manually reviewed and rescued.

## Active files

- `mteb_train_matches.json`: train split questions whose qrel-linked corpus title matches the chemistry title set.
- `mteb_dev_matches.json`: dev split matches.
- `mteb_test_matches.json`: test split matches.
- `matches.jsonl`: the same 301 matched questions flattened into one JSONL file.
- `topic_filter_gemma4.jsonl`: Gemma decisions for each matched question.
- `topic_filter_gemma4_summary.json`: final Gemma counts.
- `mteb_filter_summary.json`: first-stage title-match counts.
- `scripts/extract_mteb_matches.py`: rebuilds the MTEB title-match files from Hugging Face.
- `scripts/filter_topic_gemma4.py`: reruns the Gemma topic filter with resume support.

## Rebuild

From the repository root:

```bash
.venv/bin/python hotpotqa/scripts/extract_mteb_matches.py
.venv/bin/python hotpotqa/scripts/filter_topic_gemma4.py
```

The Gemma script is append/resume safe. It will skip rows that already have a boolean `chemistry` decision in `topic_filter_gemma4.jsonl`.

## Data shape

`matches.jsonl` rows contain:

```json
{
  "id": "hotpotqa question id",
  "question": "question text",
  "split": "train | dev | test",
  "matching_titles": ["qrel-linked chemistry corpus titles"],
  "relevant_doc_ids": ["MTEB corpus ids"]
}
```

`topic_filter_gemma4.jsonl` adds:

```json
{
  "chemistry": true,
  "reason": "short model reason",
  "model": "gemma4:31b",
  "error": null
}
```
