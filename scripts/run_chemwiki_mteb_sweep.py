#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets==4.8.5",
#   "huggingface-hub==1.17.0",
#   "mteb==2.12.30",
#   "pyarrow==24.0.0",
#   "sentence-transformers==5.5.1",
#   "torch==2.12.0",
#   "transformers==5.9.0",
# ]
# ///
"""Run a ChemWiki retrieval sweep over the Hugging Face datasets.

This is a single-file version of the Chempile sweep runner adapted for:
- hsila/chem-nq
- hsila/chem-hotpotqa

It intentionally does not import local task definitions from any repository.
It defines the MTEB retrieval tasks and the ChEmbed/Nomic wrapper inline.
"""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import mteb
import torch
from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.cache import ResultCache
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
from mteb.types import PromptType

logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path("results")
LOG_PATH = Path("run.log")
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 2048
NUM_PROC = 4

MODELS = [
    "hsila/chembed-vanilla-1",
    "hsila/chembed-vanilla-2",
    "hsila/chembed-vanilla-3",
    "hsila/chembed-vanilla-4",
    "hsila/chembed-vanilla-5",
    "hsila/chembed-vanilla-6",
    "hsila/chembed-vanilla-7",
    "hsila/chembed-vanilla-8",
    "hsila/chembed-vanilla-9",
    "hsila/chembed-vanilla-10",
    "hsila/chembed-vanilla-11",
    "hsila/chembed-vanilla-12",
    "hsila/chembed-vanilla-13",
    "hsila/chembed-vanilla-14",
    "hsila/chembed-vanilla-15",
    "hsila/chembed-vanilla-16",
    "hsila/chembed-vanilla-17",
    "hsila/chembed-vanilla-18",
    "hsila/chembed-vanilla-19",
    "hsila/chembed-vanilla-20",
    "hsila/chembed-full-1",
    "hsila/chembed-full-2",
    "hsila/chembed-full-3",
    "hsila/chembed-full-4",
    "hsila/chembed-full-5",
    "hsila/chembed-full-6",
    "hsila/chembed-full-7",
    "hsila/chembed-full-8",
    "hsila/chembed-full-9",
    "hsila/chembed-full-10",
    "hsila/chembed-full-11",
    "hsila/chembed-full-12",
    "hsila/chembed-full-13",
    "hsila/chembed-full-14",
    "hsila/chembed-full-15",
    "hsila/chembed-full-16",
    "hsila/chembed-full-17",
    "hsila/chembed-full-18",
    "hsila/chembed-full-19",
    "hsila/chembed-full-20",
    "hsila/chembed-plug-1",
    "hsila/chembed-plug-2",
    "hsila/chembed-plug-3",
    "hsila/chembed-plug-4",
    "hsila/chembed-plug-5",
    "hsila/chembed-plug-6",
    "hsila/chembed-plug-7",
    "hsila/chembed-plug-8",
    "hsila/chembed-plug-9",
    "hsila/chembed-plug-10",
    "hsila/chembed-plug-11",
    "hsila/chembed-plug-12",
    "hsila/chembed-plug-13",
    "hsila/chembed-plug-14",
    "hsila/chembed-plug-15",
    "hsila/chembed-plug-16",
    "hsila/chembed-plug-17",
    "hsila/chembed-plug-18",
    "hsila/chembed-plug-19",
    "hsila/chembed-plug-20",
    "hsila/ChEmbed-vanilla-e5-1",
    "hsila/ChEmbed-vanilla-e5-2",
    "hsila/ChEmbed-vanilla-e5-3",
    "hsila/ChEmbed-vanilla-e5-4",
    "hsila/ChEmbed-vanilla-e5-5",
    "hsila/ChEmbed-vanilla-e5-6",
    "hsila/ChEmbed-vanilla-e5-7",
    "hsila/ChEmbed-vanilla-e5-8",
    "hsila/ChEmbed-vanilla-e5-9",
    "hsila/ChEmbed-vanilla-e5-10",
    "hsila/chembed-plug-e5-1",
    "hsila/chembed-plug-e5-2",
    "hsila/chembed-plug-e5-3",
    "hsila/chembed-plug-e5-4",
    "hsila/chembed-plug-e5-5",
    "hsila/chembed-plug-e5-6",
    "hsila/chembed-plug-e5-7",
    "hsila/chembed-plug-e5-8",
    "hsila/chembed-plug-e5-9",
    "hsila/chembed-plug-e5-10",
    "hsila/chembed-plug-e6-1",
    "hsila/chembed-plug-e6-2",
    "hsila/chembed-plug-e6-3",
    "hsila/chembed-plug-e6-4",
    "hsila/chembed-plug-e6-5",
    "hsila/chembed-plug-e6-6",
    "hsila/chembed-plug-e6-7",
    "hsila/chembed-plug-e6-8",
    "hsila/chembed-plug-e6-9",
    "hsila/chembed-plug-e6-10",
    "hsila/chembed-prog1-5-1",
    "hsila/chembed-prog1-5-2",
    "hsila/chembed-prog1-5-3",
    "hsila/chembed-prog1-5-4",
    "hsila/chembed-prog1-5-5",
    "hsila/chembed-prog1-5-6",
    "hsila/chembed-prog1-5-7",
    "hsila/chembed-prog1-5-8",
    "hsila/chembed-prog1-5-9",
    "hsila/chembed-prog1-5-10",
    "hsila/chembed-prog1-e6-1",
    "hsila/chembed-prog1-e6-2",
    "hsila/chembed-prog1-e6-3",
    "hsila/chembed-prog1-e6-4",
    "hsila/chembed-prog1-e6-5",
    "hsila/chembed-prog1-e6-6",
    "hsila/chembed-prog1-e6-7",
    "hsila/chembed-prog1-e6-8",
    "hsila/chembed-prog1-e6-9",
    "hsila/chembed-prog1-e6-10",
    "hsila/chembed-prog2-e5-11",
    "hsila/chembed-prog2-e5-12",
    "hsila/chembed-prog2-e5-13",
    "hsila/chembed-prog2-e5-14",
    "hsila/chembed-prog2-e5-15",
    "hsila/chembed-prog2-e5-16",
    "hsila/chembed-prog2-e5-17",
    "hsila/chembed-prog2-e5-18",
    "hsila/chembed-prog2-e5-19",
    "hsila/chembed-prog2-e5-20",
    "hsila/chembed-prog2-e6-11",
    "hsila/chembed-prog2-e6-12",
    "hsila/chembed-prog2-e6-13",
    "hsila/chembed-prog2-e6-14",
    "hsila/chembed-prog2-e6-15",
    "hsila/chembed-prog2-e6-16",
    "hsila/chembed-prog2-e6-17",
    "hsila/chembed-prog2-e6-18",
    "hsila/chembed-prog2-e6-19",
    "hsila/chembed-prog2-e6-20",
    "hsila/ChEmbed-full-e6-1",
    "hsila/ChEmbed-full-e6-2",
    "hsila/ChEmbed-full-e6-3",
    "hsila/ChEmbed-full-e6-4",
    "hsila/ChEmbed-full-e6-5",
    "hsila/ChEmbed-full-e6-6",
    "hsila/ChEmbed-full-e6-7",
    "hsila/ChEmbed-full-e6-8",
    "hsila/ChEmbed-full-e6-9",
    "hsila/ChEmbed-full-e6-10",
    "nomic-ai/nomic-embed-text-v1",
]

_NOMIC_PROMPTS = {
    "Classification": "classification: ",
    "MultilabelClassification": "classification: ",
    "Clustering": "clustering: ",
    "PairClassification": "classification: ",
    "Reranking": "classification: ",
    "STS": "classification: ",
    "Summarization": "classification: ",
    PromptType.query.value: "search_query: ",
    PromptType.document.value: "search_document: ",
}

class ChEmbedWrapper(SentenceTransformerEncoderWrapper):
    """Inline wrapper copied from chempile-retrieval, without local imports."""

    def __init__(self, model_name: str, **kwargs: Any):
        if "model_prompts" not in kwargs:
            kwargs["model_prompts"] = _NOMIC_PROMPTS
        super().__init__(model_name, **kwargs)
        self.model_name = model_name

    def to(self, device: str) -> None:
        self.model.to(device)

    def encode(
        self,
        inputs,
        *,
        task_metadata,
        hf_split,
        hf_subset,
        prompt_type=None,
        batch_size=8,
        **kwargs,
    ):
        prompt_name = (
            self.get_prompt_name(task_metadata, prompt_type)
            or PromptType.document.value
        )
        sentences = [text for batch in inputs for text in batch["text"]]
        emb = self.model.encode(
            sentences,
            prompt_name=prompt_name,
            batch_size=batch_size,
            **kwargs,
        )
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().detach().float().numpy()
        return emb


def make_metadata(
    *,
    name: str,
    dataset_path: str,
    description: str,
    eval_splits: list[str],
    task_subtypes: list[str],
) -> TaskMetadata:
    return TaskMetadata(
        name=name,
        dataset={"path": dataset_path, "revision": "main"},
        description=description,
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=eval_splits,
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        domains=["Chemistry"],
        task_subtypes=task_subtypes,
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        sample_creation="found",
        bibtex_citation="",
        reference=f"https://huggingface.co/datasets/{dataset_path}",
    )


class ChemNQRetrievalHF(AbsTaskRetrieval):
    metadata = make_metadata(
        name="ChemNQRetrievalHF",
        dataset_path="hsila/chem-nq",
        description=(
            "Chemistry subset of Natural Questions as a retrieval task. Queries are "
            "answerable NQ questions whose source Wikipedia title matched the curated "
            "chemistry article set. Corpus documents are cleaned section-level blocks "
            "from the NQ-embedded Wikipedia HTML snapshots, preserving snapshot faithfulness."
        ),
        eval_splits=["test"],
        task_subtypes=["Question answering"],
    )


class ChemHotpotQARetrievalHF(AbsTaskRetrieval):
    metadata = make_metadata(
        name="ChemHotpotQARetrievalHF",
        dataset_path="hsila/chem-hotpotqa",
        description=(
            "Chemistry subset of HotpotQA as a retrieval task. Queries are HotpotQA "
            "questions whose qrel-linked Wikipedia evidence titles matched the curated "
            "chemistry article set and survived a topic filter. Corpus documents are "
            "the linked HotpotQA Wikipedia passages."
        ),
        eval_splits=["train", "dev", "test"],
        task_subtypes=["Question answering", "Reasoning as Retrieval"],
    )


TASK_CLASSES = [
    ChemNQRetrievalHF,
    ChemHotpotQARetrievalHF,
]


def pick_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"Using device: cuda ({name})")
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        print("Using device: mps")
        return "mps"
    print("Using device: cpu")
    return "cpu"


def load_model(model_name: str, device: str):
    """Load a model by its Hugging Face id.

    SentenceTransformer fetches and caches the snapshot on first use, so no explicit
    snapshot download is needed here.
    """
    model = ChEmbedWrapper(model_name, trust_remote_code=True)
    if hasattr(model, "to"):
        model.to(device)
    return model


def set_max_seq_length(model, max_seq_length: int) -> None:
    inner = getattr(model, "model", model)
    if hasattr(inner, "max_seq_length"):
        inner.max_seq_length = max_seq_length


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def configure_logging() -> None:
    logging.getLogger().handlers.clear()
    logging.getLogger().setLevel(logging.WARNING)

    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    file_handler = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )

    root = logging.getLogger()
    root.addHandler(console)
    root.addHandler(file_handler)

    logger.setLevel(logging.INFO)
    logger.info("Starting ChemWiki sweep")

    for noisy_logger in [
        "datasets",
        "filelock",
        "httpx",
        "huggingface_hub",
        "mteb",
        "sentence_transformers",
        "transformers",
        "urllib3",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def run_one_model(
    *,
    model_id: str,
    task_classes: list[type[AbsTaskRetrieval]],
    batch_size: int,
    max_seq_length: int,
    num_proc: int,
    cache: ResultCache,
    device: str,
) -> None:
    print(f"\n=== model: {model_id} ===")
    logger.info("Starting model: %s", model_id)
    model = load_model(model_id, device=device)
    set_max_seq_length(model, max_seq_length)
    try:
        for task_cls in task_classes:
            print(f"--- task: {task_cls.metadata.name} ---")
            logger.info("Starting task: %s", task_cls.metadata.name)
            task = task_cls()
            mteb.evaluate(
                model,
                tasks=[task],
                cache=cache,
                encode_kwargs={"batch_size": batch_size},
                num_proc=num_proc,
            )
            del task
            clear_memory()
            logger.info("Finished task: %s", task_cls.metadata.name)
    finally:
        del model
        clear_memory()
        logger.info("Finished model: %s", model_id)


def main() -> None:
    configure_logging()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    cache = ResultCache(cache_path=OUTPUT_ROOT)

    print(f"Models: {len(MODELS)}")
    print("Tasks:", ", ".join(task.metadata.name for task in TASK_CLASSES))
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Log file: {LOG_PATH}")
    logger.info("Device: %s", device)
    logger.info("Models: %s", len(MODELS))
    logger.info("Tasks: %s", ", ".join(task.metadata.name for task in TASK_CLASSES))
    logger.info("Output root: %s", OUTPUT_ROOT)

    for model_id in MODELS:
        run_one_model(
            model_id=model_id,
            task_classes=TASK_CLASSES,
            batch_size=BATCH_SIZE,
            max_seq_length=MAX_SEQ_LENGTH,
            num_proc=NUM_PROC,
            cache=cache,
            device=device,
        )


if __name__ == "__main__":
    main()
