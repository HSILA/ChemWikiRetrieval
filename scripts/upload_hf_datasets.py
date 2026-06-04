#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
UPLOADS = ROOT / "hf_uploads"
OWNER = "hsila"

def push(folder_name: str, repo_id: str) -> None:
    local = UPLOADS / folder_name
    api = HfApi()
    print(f"uploading {local} to {repo_id}")
    api.upload_folder(
        folder_path=str(local),
        repo_id=repo_id,
        repo_type="dataset",
        token=os.environ["HF_TOKEN"],
        commit_message="Section-level rebuild: one block per Wikipedia section; recovered list/paragraph raw long-answer spans; table answers excluded",
    )
    print(f"done: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    push("chem-nq", f"{OWNER}/chem-nq")
    push("chem-hotpotqa", f"{OWNER}/chem-hotpotqa")
