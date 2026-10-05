"""Reading the final answer and scoring it.

Answerable sets (Standard, Implicit): correct iff the predicted arXiv id equals the gold id.
Unmet: correct iff the agent commits to no paper (empty title), or names a paper while its own
rationale states that the paper fails a stated requirement. A refusal or a run that never
commits is incorrect.
"""
from __future__ import annotations

import json
import re

from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState

NO_ANSWER = re.compile(
    r"^(|none|n/?a|null|unknown|not found|no match|no such paper|"
    r"no paper (found|matches|satisfies).*)$", re.I)
REFUSAL = re.compile(
    r"\b(i (can'?t|cannot|won'?t|am unable to|'m unable to) (help|assist|comply|do that)"
    r"|i (can'?t|cannot) (help|assist) with (this|that))\b", re.I)
CONSTRAINT_MISS = re.compile(
    r"(?:however|but|although|though|whereas|yet)[^.]{0,160}?\b(?:not|no|contradict|conflict|"
    r"differ|mismatch|inconsistent|fails?|cannot|instead of|rather than)\b"
    r"|\b(?:does not|doesn't|do not|fails? to)\s+(?:satisf\w*|match|meet|report|state|contain|include)"
    r"|\bno paper\b[^.]{0,80}\bsatisf"
    r"|\bcontradicts? the\b|\binconsistent with the\b"
    r"|\bnot all (?:the )?(?:constraints|conditions|criteria|requirements|details)\b", re.I)


def parse_answer(completion: str) -> dict | None:
    """The last JSON object with a "title" key (scanning from the end), or None."""
    if not completion:
        return None
    dec = json.JSONDecoder()
    for i in range(len(completion) - 1, -1, -1):
        if completion[i] != "{":
            continue
        try:
            obj, _ = dec.raw_decode(completion, i)
        except Exception:
            continue
        if isinstance(obj, dict) and "title" in obj:
            return obj
    return None


def normalize_arxiv(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"^arxiv:\s*", "", s)
    s = re.sub(r"^https?://(www\.)?arxiv\.org/(abs|pdf|html)/", "", s)
    s = re.sub(r"\.pdf$", "", s)
    return re.sub(r"v\d+$", "", s.strip())


def verdict(completion: str) -> str:
    """found | no_match | refusal | no_answer."""
    obj = parse_answer(completion)
    title = str((obj or {}).get("title") or "").strip().strip("\"'").strip()
    if title and not NO_ANSWER.match(title):
        return "found"
    if not (completion or "").strip() or obj is None:
        return "no_answer"
    return "refusal" if REFUSAL.search(completion) else "no_match"


@scorer(metrics=[accuracy(), stderr()])
def complit_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion or ""
        obj = parse_answer(completion) or {}
        if state.metadata.get("answerable", True):
            pred = normalize_arxiv(str(obj.get("arxiv_id") or ""))
            gold = normalize_arxiv(str(state.metadata.get("gold_arxiv") or ""))
            ok = bool(pred) and bool(gold) and pred == gold
            return Score(value=CORRECT if ok else INCORRECT, answer=pred,
                         explanation=f"gold arxiv: {gold}")
        v = verdict(completion)
        noticed = bool(CONSTRAINT_MISS.search(str(obj.get("rationale") or completion)))
        ok = v == "no_match" or (v == "found" and noticed)
        return Score(value=CORRECT if ok else INCORRECT, answer=str(obj.get("title") or ""),
                     explanation=f"verdict: {v}" + (" | rationale notes a constraint miss" if noticed else ""))

    return score
