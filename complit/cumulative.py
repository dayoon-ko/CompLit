"""Cumulative: the requirements arrive over 3-5 user turns (ported from the paper's code).

Per turn: the simulated researcher speaks (complit/simulator.py), then the agent gets up to
10 tool calls and 30 new messages. A turn that never produces text gets one tool-free forced
reply. Tool calls and results are dropped from the agent's context between turns; its prose
replies stay. The last reply that holds the answer JSON is scored (arXiv id match).
"""
from __future__ import annotations

import asyncio
import functools
import inspect as pyinspect
import os
from typing import Any

from inspect_ai.agent import react, run
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ChatMessageUser, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import ToolDef
from inspect_ai.util import message_limit

from complit import simulator as sim
from complit.prompts import CONTRACT, TOOL_BUDGET_MSG, TURN_NUDGE, researcher_turn
from complit.scoring import parse_answer
from complit.tools import arxiv_search, read_paper, web_search_serper

ANSWER_CONTRACT = "\n\n" + CONTRACT


def _capped_tools(tools: list, budget: dict) -> list:
    """Same tools (name, description, schema), but call 11+ in a turn returns TOOL_BUDGET_MSG."""
    out = []
    for t in tools:
        td = ToolDef(t)

        def make(orig):
            @functools.wraps(orig)
            async def execute(*args, **kwargs):
                if budget["left"] <= 0:
                    return TOOL_BUDGET_MSG
                budget["left"] -= 1
                return await orig(*args, **kwargs)

            execute.__signature__ = pyinspect.signature(orig)
            return execute

        out.append(ToolDef(make(td.tool), name=td.name, description=td.description,
                           parameters=td.parameters, parallel=td.parallel).as_tool())
    return out


def _sim_client(base_url: str, api_key_env: str):
    from openai import OpenAI

    key = os.environ.get(api_key_env, "")
    if not key:
        raise RuntimeError(f"{api_key_env} is not set; the simulated researcher needs it")
    return OpenAI(base_url=base_url, api_key=key)


def _last_assistant_text(messages: list[Any]) -> str:
    for m in reversed(messages):
        if isinstance(m, ChatMessageAssistant):
            text = m.text.strip() if m.text else ""
            if text:
                return text
    return ""


def _without_dangling_tool_calls(messages: list[Any]) -> list[Any]:
    msgs = list(messages)
    while msgs:
        provided = {m.tool_call_id for m in msgs if isinstance(m, ChatMessageTool)}
        pending = [i for i, m in enumerate(msgs)
                   if isinstance(m, ChatMessageAssistant)
                   and any(tc.id not in provided for tc in (m.tool_calls or []))]
        if not pending:
            return msgs
        msgs = msgs[: pending[-1]]
    return msgs


@solver
def conversation_agent(use_web_search: bool = True, read_paper_images: int = 0,
                       per_turn_tool_calls: int = 10, per_turn_messages: int = 30,
                       sim_model: str = sim.SIM_MODEL,
                       sim_base_url: str = "https://openrouter.ai/api/v1",
                       sim_api_key_env: str = "OPENROUTER_API_KEY") -> Solver:
    def base_tools():
        tools = [arxiv_search()]
        if use_web_search:
            tools.append(web_search_serper())
        tools.append(read_paper(max_images=read_paper_images))
        return tools

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        budget = {"left": per_turn_tool_calls}          # per sample: samples run concurrently
        finder = react(name="paper_finder", description="Finds an academic paper by searching arXiv",
                       prompt=None, submit=False, tools=_capped_tools(base_tools(), budget))
        sim.SIM_MODEL = sim_model
        client = _sim_client(sim_base_url, sim_api_key_env)

        md = state.metadata
        constraints = sim.canonical_constraints({"constraints": [dict(c) for c in md["constraints"]]})
        for c in constraints:
            c.setdefault("rendered_value", c.get("value"))
        intention = md["intention"]
        turns = md["turns"]
        n_turns = len(turns)

        sim_msgs: list[dict[str, str]] = []     # simulator's view, roles inverted
        ledger: list[tuple[int, str]] = []
        agent_messages: list[Any] = []
        transcript: list[dict[str, Any]] = []
        agent_texts: list[str] = []

        for turn in range(1, n_turns + 1):
            spec = turns[turn - 1]
            picked = [constraints[i] for i in spec["revealed"] if 0 <= i < len(constraints)]
            ref = (spec["utterance"] or "").strip()
            last_agent = sim_msgs[-1]["content"] if sim_msgs else ""
            needs_resp = bool(last_agent) and bool(sim.ADAPTER_RESPONSE_RE.search(last_agent))

            # 1. the researcher speaks: the prepared utterance, or the adapter's delivery of it
            if ref and not needs_resp:
                stage, user_msg, rev = "verbatim", ref, list(range(len(picked)))
            else:
                stage = "adapter"
                instr = sim.adapter_instruction(ref, turn == n_turns)
                turn_sys = sim.SIM_SYS_SCHEDULED_NEED.format(
                    intention=intention,
                    all_facts="\n".join([v for _t, v in ledger] + [sim.fact_line(c) for c in picked])
                    or "(nothing specific yet -- only the broad context of your work)")
                user_msg, rev, _att, _ti, _to = await asyncio.to_thread(
                    sim.naturalize_turn, client, turn_sys, sim_msgs, instr, picked)
            sim_msgs.append({"role": "assistant", "content": user_msg})
            ledger += [(turn, sim.fact_line(picked[i])) for i in rev]
            transcript.append({"turn": turn, "role": "researcher", "stage": stage, "text": user_msg})

            # 2. the agent replies
            body = researcher_turn(user_msg, turn=turn, n_turns=n_turns,
                                   use_web_search=use_web_search, read_paper_images=read_paper_images)
            messages = list(agent_messages) + [ChatMessageUser(content=body)]
            budget["left"] = per_turn_tool_calls
            turn_start = len(messages)
            agent_state, _ = await run(finder, messages,
                                       limits=[message_limit(len(messages) + per_turn_messages)])
            agent_messages = _without_dangling_tool_calls(agent_state.messages)
            agent_msg = _last_assistant_text(agent_messages[turn_start:])
            forced = False
            if not agent_msg:
                nudge = TURN_NUDGE + (ANSWER_CONTRACT if turn == n_turns else "")
                out = await get_model().generate(
                    list(agent_messages) + [ChatMessageUser(content=nudge)], tools=[])
                agent_msg = (out.completion or "").strip()
                if agent_msg:
                    agent_messages.append(ChatMessageAssistant(content=agent_msg))
                    forced = True
            agent_msg = agent_msg or "<no reply>"
            sim_msgs.append({"role": "user", "content": agent_msg})
            agent_texts.append(agent_msg)
            transcript.append({"turn": turn, "role": "agent", "forced": forced, "text": agent_msg})
            # next turn starts from the conversation only: drop tool calls and tool results
            agent_messages = [m for m in agent_messages
                              if m.role == "user"
                              or (m.role == "assistant" and not getattr(m, "tool_calls", None))]

        # 3. the answer: the last reply that holds the answer JSON
        answer = agent_texts[-1] if agent_texts else ""
        for txt in reversed(agent_texts):
            if parse_answer(txt):
                answer = txt
                break
        state.messages = agent_messages
        state.output.completion = answer
        state.metadata["transcript"] = transcript
        return state

    return solve
