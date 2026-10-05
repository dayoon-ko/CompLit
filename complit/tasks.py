"""Inspect tasks, one per CompLit setting.

    inspect eval complit/implicit --model openrouter/deepseek/deepseek-v4-pro

The agent is a ReAct loop with no system message and three tools (arxiv_search, read_paper,
web_search). It stops when it replies without a tool call or reaches `message_limit` messages
(30 in the paper). Its final reply must end with the JSON answer in `complit/prompts.py`.
"""
from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.agent import react, run
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, GenerateConfig, get_model
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.util import message_limit as message_limit_ctx

from complit.cumulative import conversation_agent
from complit.data import load_setting
from complit.prompts import FORCED_ANSWER, build_prompt
from complit.scoring import complit_scorer, parse_answer
from complit.tools import arxiv_search, read_paper, web_search_serper


def make_agent(use_web_search: bool, read_paper_images: int):
    tools = [arxiv_search()]
    if use_web_search:
        tools.append(web_search_serper())
    tools.append(read_paper(max_images=read_paper_images))
    return react(name="paper_finder", description="Finds an academic paper by searching arXiv",
                 prompt=None, submit=False, tools=tools)


def committed(completion: str, answerable: bool) -> bool:
    """Named a paper; on Unmet, any answer JSON (an empty title is an abstention)."""
    ans = parse_answer(completion or "")
    if not answerable:
        return ans is not None
    return bool(ans and (str(ans.get("title") or "").strip() or str(ans.get("arxiv_id") or "").strip()))


def forced_answer_messages(messages: list) -> list:
    """The conversation up to the agent's last reply (tool calls stripped) + FORCED_ANSWER."""
    last = max((i for i, m in enumerate(messages) if isinstance(m, ChatMessageAssistant)), default=None)
    msgs = list(messages[: last + 1]) if last is not None else list(messages)
    if msgs and isinstance(msgs[-1], ChatMessageAssistant) and msgs[-1].tool_calls:
        out = msgs.pop()
        if (out.text or "").strip():
            msgs.append(out.model_copy(update={"tool_calls": None}))
    return msgs + [ChatMessageUser(content=FORCED_ANSWER)]


@solver
def paper_finder(use_web_search: bool, read_paper_images: int, budget: int, forced_answer: bool):
    agent = make_agent(use_web_search, read_paper_images)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        prompt = build_prompt(state.input_text, use_read_paper=True, use_web_search=use_web_search,
                              read_paper_images=read_paper_images)
        agent_state, _ = await run(agent, prompt, limits=[message_limit_ctx(budget)])
        completion = agent_state.output.completion or ""
        state.messages = list(agent_state.messages)
        state.metadata["forced_answer"] = False
        if forced_answer and not committed(completion, state.metadata["answerable"]):
            # the run ended without committing: one tool-free call, as in the paper
            out = await get_model().generate(forced_answer_messages(agent_state.messages), tools=[],
                                             config=GenerateConfig(max_tokens=8000))
            completion = out.completion or ""
            state.messages.append(ChatMessageUser(content=FORCED_ANSWER))
            state.messages.append(ChatMessageAssistant(content=completion))
            state.metadata["forced_answer"] = True
        state.output.completion = completion
        return state

    return solve


def single_turn(setting: str, domain: str, message_limit: int, read_paper_images: int,
                web_search: bool, forced_answer: bool) -> Task:
    return Task(
        dataset=load_setting(setting, domain or None),
        solver=paper_finder(web_search, read_paper_images, message_limit, forced_answer),
        scorer=complit_scorer(),
    )


@task
def standard(domain: str = "", message_limit: int = 30, read_paper_images: int = 0,
             web_search: bool = True, forced_answer: bool = True) -> Task:
    """Standard: constraints at every evidence level, given at once (350 queries)."""
    return single_turn("standard", domain, message_limit, read_paper_images, web_search, forced_answer)


@task
def implicit(domain: str = "", message_limit: int = 30, read_paper_images: int = 0,
             web_search: bool = True, forced_answer: bool = True) -> Task:
    """Implicit: every constraint must be inferred from the paper (220 queries)."""
    return single_turn("implicit", domain, message_limit, read_paper_images, web_search, forced_answer)


@task
def unmet(domain: str = "", message_limit: int = 30, read_paper_images: int = 0,
          web_search: bool = True, forced_answer: bool = True) -> Task:
    """Unmet: no paper satisfies every constraint; the correct answer is to abstain (300 queries)."""
    return single_turn("unmet", domain, message_limit, read_paper_images, web_search, forced_answer)


@task
def cumulative(domain: str = "", per_turn_tool_calls: int = 10, per_turn_messages: int = 30,
               read_paper_images: int = 0, web_search: bool = True,
               sim_model: str = "openai/gpt-5.4-mini") -> Task:
    """Cumulative: the Implicit queries revealed over 3-5 turns by a simulated researcher (220).

    The simulator calls `sim_model` through OpenRouter (needs OPENROUTER_API_KEY).
    """
    return Task(
        dataset=load_setting("cumulative", domain or None),
        solver=conversation_agent(web_search, read_paper_images, per_turn_tool_calls,
                                  per_turn_messages, sim_model),
        scorer=complit_scorer(),
    )
