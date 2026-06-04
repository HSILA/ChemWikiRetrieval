# ChemWikiRetrieval Dashboard Design Brief

## Project Overview

ChemWikiRetrieval is a chemistry-focused benchmark dataset for evaluating embedding/retrieval models. It filters Wikipedia chemistry content from two well-known QA datasets (Natural Questions and HotpotQA) to create MTEB/BEIR-style retrieval tasks.

The project has two completed datasets on HuggingFace:

- **hsila/chem-nq** - Natural Questions subset: 7,874 corpus passages (Wikipedia section-level), 2,745 queries, ~2,745 qrels. Corpus fields: `id`, `title`, `section`, `text`. Queries: `id`, `text`.
- **hsila/chem-hotpotqa** - HotpotQA subset: 464 corpus passages, 274 queries across train/dev/test splits. Corpus fields: `id`, `title`, `text`. Queries: `id`, `text`.

The evaluation sweep script (`scripts/run_chemwiki_mteb_sweep.py`) runs 140+ embedding models (ChEmbed variants, nomic-embed-text-v1) through MTEB's `AbsTaskRetrieval` on both tasks and writes results to a `results/` directory.

## What the Dashboard Should Show

A results dashboard that visualizes how each embedding model performs on chemistry retrieval. The target audience is ML researchers evaluating chemistry-aware vs general-purpose embeddings.

### Data Available per Model per Task

From MTEB's result cache (JSON files in `results/`), each model run produces:

- **nDCG@10** (primary metric)
- **Recall@1, Recall@3, Recall@10, Recall@100**
- **Precision@1, Precision@3, Precision@10**
- **MRR@10**
- **MAP@100**

Each metric is available for both tasks: `ChemNQRetrieval` and `ChemHotpotQARetrieval`.

The models come in these families:

- **chembed-vanilla** (20 seeds): Baseline, nochemistry training, BERT tokenizer
- **ChEmbed-vanilla-e5** (10 seeds): Baseline with E5-style prompt fine-tuning
- **chembed-plug** (20 seeds): Chemistry vocabulary plugged in, no chemistry training
- **chembed-plug-e5** (10 seeds): Chemistry vocab + E5 prompts, no chemistry training
- **chembed-full** (3 seeds): Full chemistry training
- **ChEmbed-full-e6** (10 seeds): Full chemistry training + E5-style
- **chembed-prog2-e6** (20 seeds): Progressive training variant
- **nomic-embed-text-v1** (1): External baseline

### Dashboard Sections I Want

1. **Leaderboard / Comparison Table**
   - Rows = models (or model families aggregated)
   - Columns = metrics (nDCG@10, Recall@10, MRR@10, etc.)
   - Two views: one per-task, one cross-task aggregate
   - Highlight best-in-family and overall best
   - Sortable columns

2. **Model Family Comparison**
   - Grouped bar chart or dot plot comparing families (vanilla vs plug vs full vs prog2 vs nomic)
   - Show mean ± stddev across seeds for each family
   - Separate panels for each metric or a selectable metric dropdown

3. **Per-Model Detail Drilldown**
   - Click a model name to see its full result breakdown
   - Show individual seed results vs family mean
   - Compare any two models side by side

4. **Dataset Comparison**
   - How does NQ performance correlate with HotpotQA performance for each model?
   - Scatter plot: NQ metric on X, HotpotQA metric on Y, one dot per model (or per family mean)
   - Helps identify models that are strong on one dataset but not the other

5. **Training Progress / Ablation View**
   - For families with multiple training stages (vanilla → plug → full), show the incremental gain
   - Waterfall or step chart: baseline → +vocab → +training

### Style and Tone

- Chemistry research, not SaaS startup
- Dark theme (the Glot app uses CSS custom properties: `--bg`, `--surface`, `--fg`, `--accent`, `--muted`, etc.)
- Clean, data-dense, not flashy
- Monospace for numbers and labels
- Serif for headlines (the Glot app uses `serif` class for headings, `mono` class for labels/metrics)
- The Glot app's color system: `--accent` (primary action color), `--warn` (yellow/orange), `--bad` (red), `--info` (blue), `--good` (green)
- No em dashes anywhere in the copy or UI text

### Technical Constraints

- Must be a single self-contained HTML file
- No external API calls or backend needed; all data will be embedded as JSON
- Can use CDN libraries (Chart.js, D3, etc.)
- Responsive but desktop-first (researchers will view on large screens)

### Data Format Reference

MTEB result cache stores one JSON file per model per task. Example structure:

```json
{
  "model_name": "hsila/chembed-vanilla-1",
  "model_revision": "main",
  "task_name": "ChemNQRetrieval",
  "scores": {
    "test": {
      "ndcg_at_10": 0.423,
      "recall_at_1": 0.185,
      "recall_at_3": 0.301,
      "recall_at_10": 0.512,
      "recall_at_100": 0.789,
      "precision_at_1": 0.185,
      "precision_at_3": 0.112,
      "precision_at_10": 0.069,
      "mrr_at_10": 0.287,
      "map_at_100": 0.356
    }
  }
}
```

The dashboard should be designed to load a directory of these files and render all sections from the embedded data.