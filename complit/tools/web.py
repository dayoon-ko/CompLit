"""Web search tool (Google via Serper) for the CompLit agent. Requires SERPER_API_KEY."""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from collections import OrderedDict

from inspect_ai.tool import Tool, tool

SERPER_URL = "https://google.serper.dev/search"
_TIMEOUT = 30
_MAX_SNIPPET = 300

_CACHE_MAX = 4096
_cache: "OrderedDict[str, str]" = OrderedDict()
_cache_lock = threading.Lock()


def _serper(query: str, num: int) -> dict:
    key = os.getenv("SERPER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("SERPER_API_KEY is not set")
    ck = f"{num}|{query}"
    with _cache_lock:
        hit = _cache.get(ck)
        if hit is not None:
            _cache.move_to_end(ck)
            return json.loads(hit)
    payload = json.dumps({"q": query, "num": num}).encode()
    req = urllib.request.Request(
        SERPER_URL,
        data=payload,
        headers={"X-API-KEY": key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        body = resp.read().decode("utf-8")
    with _cache_lock:
        _cache[ck] = body
        if len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return json.loads(body)


def _format(data: dict, max_results: int) -> str:
    out: list[str] = []
    kg = data.get("knowledgeGraph") or {}
    if kg.get("title"):
        desc = (kg.get("description") or "")[:_MAX_SNIPPET]
        out.append(f"[knowledge panel] {kg['title']}: {desc}")
    for i, r in enumerate((data.get("organic") or [])[:max_results], 1):
        title = (r.get("title") or "").strip()
        link = (r.get("link") or "").strip()
        snippet = (r.get("snippet") or "").strip()[:_MAX_SNIPPET]
        out.append(f"{i}. {title}\n   url: {link}\n   {snippet}")
    if not out:
        return (
            "No web results for that query (the search ran and returned nothing). "
            "Try different wording."
        )
    return "\n".join(out)


@tool
def web_search_serper() -> Tool:
    async def execute(query: str, max_results: int = 8) -> str:
        """Search the web (Google) for information about a paper.

        Use this when arXiv search is not finding the paper: search for the described
        method, benchmark numbers, venue, or author details as they would be written
        about anywhere on the web, then confirm the title you find.

        Args:
            query: Free-text web search query, exactly as you would type it in Google.
            max_results: Number of results to return (default 8, max 15).

        Returns:
            A numbered list of web results with title, url and snippet.
        """
        max_results = max(1, min(int(max_results), 15))
        try:
            data = _serper(query, max_results)
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8")[:200]
            except Exception:
                pass
            return (
                f"Web search could not be completed (HTTP {e.code}: {detail}). This is "
                "a transport error, NOT an empty result set: the query never ran. "
                "Retry it or use arxiv_search."
            )
        except Exception as e:
            return (
                f"Web search could not be completed ({type(e).__name__}: {e}). This is "
                "a transport error, NOT an empty result set: the query never ran. "
                "Retry it or use arxiv_search."
            )
        return _format(data, max_results)

    return execute
