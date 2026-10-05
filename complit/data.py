"""Load CompLit from the Hugging Face Hub as Inspect samples."""
from __future__ import annotations

from datasets import load_dataset
from inspect_ai.dataset import MemoryDataset, Sample

HF_REPO = "dayoon/CompLit"
SETTINGS = ("standard", "implicit", "cumulative", "unmet")
PAPER_TYPE = {"content": "content", "figural": "vision", "comparative": "multi_doc", "metadata": "metadata"}


def load_setting(setting: str, domain: str | None = None, repo: str = HF_REPO) -> MemoryDataset:
    """One CompLit setting as an Inspect dataset.

    Args:
        setting: standard, implicit, cumulative or unmet.
        domain: keep only one arXiv domain (cs, econ, eess, math, physics, q-bio, q-fin, stat).
        repo: Hugging Face dataset repo.
    """
    if setting not in SETTINGS:
        raise ValueError(f"setting must be one of {SETTINGS}, got {setting!r}")
    rows = load_dataset(repo, setting, split="test")
    samples = []
    for r in rows:
        if domain and r["domain"] != domain:
            continue
        paper = r["paper"] or {}
        metadata = {
            "setting": setting,
            "domain": r["domain"],
            "answerable": bool(r["answerable"]),
            "gold_arxiv": paper.get("arxiv_id") or "",
        }
        if setting == "cumulative":
            conv = r["conversation"]
            metadata["intention"] = conv["intention"]
            metadata["turns"] = [{"utterance": t["utterance"], "revealed": t["revealed_constraints"]}
                                 for t in conv["turns"]]
            # the simulator's prompts name constraint types as in the paper's code
            metadata["constraints"] = [{"type": PAPER_TYPE[c["source"]], "field": c["field"],
                                        "value": c["value"]} for c in r["constraints"]]
        samples.append(Sample(
            id=r["id"],
            input=r["question"],
            target=paper.get("title") or "",
            metadata=metadata,
        ))
    return MemoryDataset(samples, name=f"complit-{setting}")
