# CompLit — evaluation code

Run any LLM agent on **CompLit**, a scientific literature search benchmark, with [Inspect](https://inspect.aisi.org.uk/).
The data loads from Hugging Face automatically: [`dayoon/CompLit`](https://huggingface.co/datasets/dayoon/CompLit).

**Paper:** coming soon · **Project page:** coming soon

---

## Run it in 3 steps

```bash
# 1. install
git clone https://github.com/dayoon-ko/CompLit.git && cd CompLit
pip install -e .

# 2. keys
cp .env.example .env          # then fill in the keys you need (table below)

# 3. run  (--model = any model Inspect supports)
inspect eval complit/implicit --model openai/gpt-5.4
```

Results print at the end. Open the full logs with `inspect view`.

---

## Pick a setting

| Setting | Command | N | Correct when the agent … |
|---|---|---:|---|
| Standard | `inspect eval complit/standard --model …` | 350 | returns the target paper |
| Implicit | `inspect eval complit/implicit --model …` | 220 | returns the target paper (every requirement must be inferred) |
| Cumulative | `inspect eval complit/cumulative --model …` | 220 | returns the target paper after the last turn (requirements arrive over 3–5 turns) |
| Unmet | `inspect eval complit/unmet --model …` | 300 | abstains (no paper satisfies every requirement) |

---

## Keys

| Key | Needed for | Where to get it |
|---|---|---|
| your model's key (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, …) | the agent | your provider |
| `SERPER_API_KEY` | the `web_search` tool | [serper.dev](https://serper.dev) (or run with `-T web_search=false`) |
| `OPENROUTER_API_KEY` | **Cumulative only**: the simulated researcher (`openai/gpt-5.4-mini`) | [openrouter.ai](https://openrouter.ai) |

`arxiv_search` and `read_paper` use public arXiv / ar5iv pages and need no key.

---

## I want to …

| … | Do this |
|---|---|
| try it on a few queries first | add `--limit 5` |
| run all four settings | `inspect eval complit/standard complit/implicit complit/cumulative complit/unmet --model …` |
| evaluate one field only | `-T domain=physics` (`cs`, `econ`, `eess`, `math`, `physics`, `q-bio`, `q-fin`, `stat`) |
| give the agent the paper figures | `-T read_paper_images=4` (vision models; the paper used 4 for Kimi) |
| run without web search | `-T web_search=false` |
| run more queries in parallel | `--max-connections 8` |
| use a local model | `--model vllm/<name>` or `--model openai-api/<name>` with `OPENAI_BASE_URL` (see Inspect docs) |
| fix `HTTP Error 429` from arXiv | arXiv rate-limits; lower `--max-connections` or set `ARXIV_MIN_INTERVAL=1` (seconds between requests, default 0.5) |
| look at what the agent did | `inspect view` |
| load the data myself | `load_dataset("dayoon/CompLit", "implicit", split="test")` |

---

## Options (`-T name=value`)

| Option | Default | Settings | Meaning |
|---|---|---|---|
| `domain` | all | all | keep one arXiv domain |
| `message_limit` | `30` | single-turn | messages per query |
| `forced_answer` | `true` | single-turn | if the agent never commits, ask once more, with no tools, for its final answer |
| `per_turn_tool_calls` | `10` | Cumulative | tool calls per user turn |
| `per_turn_messages` | `30` | Cumulative | messages per user turn |
| `read_paper_images` | `0` | all | figures attached by `read_paper` |
| `web_search` | `true` | all | attach the `web_search` tool |
| `sim_model` | `openai/gpt-5.4-mini` | Cumulative | model behind the simulated researcher |

The defaults are the settings used in the paper.

---

## How it works

```
inspect eval complit/implicit --model <model>
        │
        ▼
complit/tasks.py   implicit()  →  Task = data + agent + scorer
        │
        ├─ data     complit/data.py        loads the queries from Hugging Face
        │
        ├─ agent    (the model is called here)
        │    ├─ single-turn: complit/tasks.py  paper_finder()
        │    │     ReAct loop: model → tools (arxiv_search, read_paper, web_search) → model …
        │    │     if it never answers, one forced final answer
        │    └─ Cumulative: complit/cumulative.py  conversation_agent()
        │          each turn: complit/simulator.py writes the researcher's message → same ReAct loop
        │
        └─ scorer   complit/scoring.py     arXiv id match (Unmet: correct if it abstains)
```

1. **Prompt.** The agent gets one user message with the task, its three tools, and the researcher's request. There is no system message.
2. **Answer.** The last line of its reply is JSON: `{"title", "arxiv_id", "rationale", "confidence"}`. An empty title means "no paper satisfies the request".
3. **Cumulative.** Each turn's message is fixed in the dataset and sent as written. If the agent asked something or proposed a paper, `gpt-5.4-mini` sends the same message with one added sentence that answers it. Tool history is dropped between turns, and the agent's replies are kept.

The model is whatever you pass to `--model`; no model name is hard-coded.

---

## Files

| File | What it does |
|---|---|
| `complit/tasks.py` | the four Inspect tasks and the single-turn agent |
| `complit/cumulative.py` | the multi-turn loop for Cumulative |
| `complit/simulator.py` | the simulated researcher (prompts and checks) |
| `complit/prompts.py` | every prompt the agent sees |
| `complit/scoring.py` | answer parsing and scoring |
| `complit/data.py` | loads the dataset from Hugging Face |
| `complit/tools/` | `arxiv_search`, `read_paper`, `web_search` |

---

## Citation

```bibtex
@article{ko2026complit,
  title   = {CompLit: Scientific Literature Search Benchmarks Must Cover Implicit, Cumulative, and Unmet Needs},
  author  = {Ko, Dayoon and Kim, Jihyuk and Jeong, Soyeong and Lee, Young-Jun and Lee, Dahyun and Kim, Juyeon and Kim, Gunhee and Lee, Moontae and Lee, Kyungjae},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

Code: MIT License. Data: CC BY 4.0 (see the dataset card).
