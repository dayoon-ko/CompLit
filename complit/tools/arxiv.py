"""arXiv tools for the CompLit agent.

- ``arxiv_search``: queries the public arXiv API and returns a compact list of hits
  (title, arXiv id, year, authors, truncated abstract).
- ``read_paper``: returns the full text of one paper (ar5iv HTML, truncated when long);
  with ``max_images > 0`` it also attaches the paper's figures as images.

Requests to arXiv are paced and retried on 429/5xx so that a rate-limited stretch is not
mistaken for an empty result.
"""

from __future__ import annotations

import base64
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import OrderedDict
from itertools import zip_longest
from pathlib import Path

import anyio.to_thread
from inspect_ai.model import ContentImage, ContentText
from inspect_ai.tool import Tool, ToolResult, tool

ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
AR5IV_URL = "https://ar5iv.labs.arxiv.org/html/{}"
MAX_PAPER_CHARS = 24000

FIG_MAX_IMAGES = 8
FIG_MAX_BYTES = int(os.getenv("READ_PAPER_FIG_MAX_BYTES", 4 * 1024 * 1024))
FIG_MAX_TOTAL_BYTES = int(os.getenv("READ_PAPER_FIG_MAX_TOTAL_BYTES", 12 * 1024 * 1024))
FIG_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
_FIG_CACHE_DIR = Path(
    os.getenv("READ_PAPER_FIG_CACHE", Path(__file__).resolve().parent.parent / ".fig_cache")
)
_fig_inflight = threading.Semaphore(int(os.getenv("READ_PAPER_FIG_INFLIGHT", "4")))

_MIN_REQUEST_INTERVAL = float(os.getenv("ARXIV_MIN_INTERVAL", "0.5"))
_MAX_INFLIGHT = int(os.getenv("ARXIV_MAX_INFLIGHT", "4"))
_RETRY_DELAYS = (5, 15, 30, 60)
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

_throttle_lock = threading.Lock()
_inflight = threading.Semaphore(_MAX_INFLIGHT)
_last_request_at = 0.0

_CACHE_MAX = 4096
_cache: "OrderedDict[str, str]" = OrderedDict()
_cache_lock = threading.Lock()


_SHARED_THROTTLE = os.getenv("ARXIV_SHARED_THROTTLE", "").strip()


def _shared_throttle() -> None:
    if not _SHARED_THROTTLE:
        return
    import fcntl
    with open(_SHARED_THROTTLE, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            raw = f.read().strip()
            last = float(raw) if raw else 0.0
            wait = _MIN_REQUEST_INTERVAL - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            f.seek(0)
            f.truncate()
            f.write(repr(time.time()))
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _throttled_get(url: str, timeout: int) -> str:
    global _last_request_at
    with _cache_lock:
        hit = _cache.get(url)
        if hit is not None:
            _cache.move_to_end(url)
            return hit
    with _inflight:
        with _throttle_lock:
            wait = _MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request_at)
            if wait > 0:
                time.sleep(wait)
            _last_request_at = time.monotonic()
        _shared_throttle()
        req = urllib.request.Request(
            url, headers={"User-Agent": "paper-search-eval/1.0 (inspect_ai)"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    with _cache_lock:
        _cache[url] = body
        if len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return body


def _fetch(query: str, max_results: int) -> str:
    return _fetch_raw(f"all:{query}", max_results)


def _fetch_raw(search_query: str, max_results: int) -> str:
    params = urllib.parse.urlencode(
        {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
        }
    )
    url = f"{ARXIV_API}?{params}"
    last_exc: Exception | None = None
    for delay in (*_RETRY_DELAYS, None):
        try:
            return _throttled_get(url, timeout=30)
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code not in _RETRYABLE_STATUS or delay is None:
                raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_exc = e
            if delay is None:
                raise
        time.sleep(delay)
    raise last_exc if last_exc else RuntimeError("arXiv fetch failed")


DUAL_QUERY = True
AND_TERMS = 4

_QUERY_STOPWORDS = frozenset("""a an the of for and or to in on with using via based by from that
this these those is are was were be been we our their its it as at into over under between within
can could should would may might will shall do does did not no than then there here which who whom
whose what when where why how paper method approach model models study work research propose
proposed novel new results result show shows shown also more most such other others only very""".split())


def _and_query(query: str, k: int = AND_TERMS) -> str:
    seen, terms = set(), []
    for t in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", query.lower()):
        if t in _QUERY_STOPWORDS or len(t) <= 2 or t in seen:
            continue
        seen.add(t)
        terms.append(t)
        if len(terms) == k:
            break
    return " AND ".join(f"all:{t}" for t in terms) if len(terms) >= 2 else ""


def _id_from_url(entry_id: str) -> str:
    return re.sub(r"^https?://arxiv\.org/abs/", "", (entry_id or "").strip())


def _entries(xml_text: str) -> list:
    return ET.fromstring(xml_text).findall(f"{ATOM}entry")


def _entry_id(e) -> str:
    return _id_from_url(e.findtext(f"{ATOM}id") or "")


def _interleave(primary: list, secondary: list, limit: int) -> list:
    merged, seen = [], set()
    for a, b in zip_longest(primary, secondary):
        for e in (a, b):
            if e is None:
                continue
            eid = _entry_id(e)
            if eid in seen:
                continue
            seen.add(eid)
            merged.append(e)
            if len(merged) >= limit:
                return merged
    return merged


def _format_entries(entries: list) -> str:
    if not entries:
        return "No results found. Try different or broader search terms."

    blocks: list[str] = []
    for idx, e in enumerate(entries, 1):
        title = " ".join((e.findtext(f"{ATOM}title") or "").split())
        published = e.findtext(f"{ATOM}published") or ""
        year = published[:4]
        authors = [
            (a.findtext(f"{ATOM}name") or "").strip()
            for a in e.findall(f"{ATOM}author")
        ]
        authors_str = ", ".join(authors[:6]) + (" et al." if len(authors) > 6 else "")
        summary = " ".join((e.findtext(f"{ATOM}summary") or "").split())
        if len(summary) > 500:
            summary = summary[:500] + "..."
        arxiv_id = _id_from_url(e.findtext(f"{ATOM}id") or "")
        blocks.append(
            f"[{idx}] {title}\n"
            f"    arxiv_id: {arxiv_id} | year: {year}\n"
            f"    authors: {authors_str}\n"
            f"    abstract: {summary}"
        )
    return "\n\n".join(blocks)


def _format(xml_text: str) -> str:
    return _format_entries(_entries(xml_text))


@tool
def arxiv_search() -> Tool:
    async def execute(query: str, max_results: int = 10) -> str:
        """Search arXiv for papers matching a query.

        Args:
            query: Free-text search terms (keywords, concepts, author names, etc.).
                Searches titles, abstracts, authors and other fields.
            max_results: Number of results to return (default 10, keep modest).

        Returns:
            A numbered list of matching papers with title, arxiv_id, year, authors
            and a truncated abstract.
        """
        max_results = max(1, min(int(max_results), 25))
        try:
            xml_text = _fetch(query, max_results)
            strict = _and_query(query) if DUAL_QUERY else ""
            if strict:
                try:
                    broad_entries = _entries(xml_text)
                    strict_entries = _entries(_fetch_raw(strict, max_results))
                    if strict_entries:
                        return _format_entries(
                            _interleave(strict_entries, broad_entries, max_results)
                        )
                except Exception:
                    pass
        except Exception as e:
            return (
                f"arXiv search could not be completed ({type(e).__name__}: {e}). "
                "This is a transport error, NOT an empty result set: the query was "
                "never executed. Retry the same query rather than reformulating it."
            )
        try:
            return _format(xml_text)
        except ET.ParseError as e:
            return f"Failed to parse arXiv response ({e}). Try again."

    return execute


def _normalize_arxiv_id(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id.strip())


def _fetch_paper_html(arxiv_id: str) -> str:
    req = urllib.request.Request(
        AR5IV_URL.format(_normalize_arxiv_id(arxiv_id)),
        headers={"User-Agent": "paper-search-eval/1.0 (inspect_ai)"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        if resp.status != 200:
            return ""
        ctype = resp.headers.get("content-type", "").lower()
        html = resp.read().decode("utf-8", errors="replace")
    if "html" not in ctype:
        return ""
    head = html[:2000]
    if "This paper does not exist" in head or "Conversion failed" in head:
        return ""
    return html


def _html_to_text(html: str) -> str:
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        root = (
            soup.find("article")
            or soup.find("div", class_="ltx_document")
            or soup.body
            or soup
        )
        text = " ".join(root.get_text(" ").split())
    except Exception:
        text = " ".join(re.sub(r"<[^>]+>", " ", html).split())
    return text


def _fetch_paper_text(arxiv_id: str) -> str:
    html = _fetch_paper_html(arxiv_id)
    return _html_to_text(html) if html else ""


def _figure_sources(html: str, page_url: str, limit: int) -> tuple[list[tuple[str, str]], int]:
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return [], 0
    soup = BeautifulSoup(html, "html.parser")
    out: list[tuple[str, str]] = []
    skipped = 0
    for fig in soup.find_all("figure"):
        if "ltx_table" in (fig.get("class") or []):
            continue
        cap_el = fig.find("figcaption")
        caption = " ".join(cap_el.get_text(" ").split())[:300] if cap_el else ""
        refs = ([(i.get("src") or "").strip() for i in fig.find_all("img")]
                + [(o.get("data") or "").strip() for o in fig.find_all("object")]
                + [(e.get("src") or "").strip() for e in fig.find_all("embed")])
        for src in refs:
            if not src:
                continue
            ext = os.path.splitext(urllib.parse.urlparse(src).path)[1].lower()
            if ext not in FIG_MIME:
                skipped += 1
                continue
            url = src if src.startswith("http") else urllib.parse.urljoin(page_url + "/", src)
            if len(out) < limit:
                out.append((url, caption))
        skipped += len(fig.find_all("svg"))
    return out[:limit], skipped


def _download_figure(url: str, dest: Path) -> bytes | None:
    with _fig_inflight:
        req = urllib.request.Request(
            url, headers={"User-Agent": "paper-search-eval/1.0 (inspect_ai)"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            if resp.status != 200:
                return None
            data = resp.read(FIG_MAX_BYTES + 1)
    if not data or len(data) > FIG_MAX_BYTES:
        return None
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(data)
        tmp.replace(dest)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
    return data


def _decodable(data: bytes) -> bool:
    try:
        import io

        from PIL import Image
    except ImportError:
        return True
    try:
        Image.open(io.BytesIO(data)).load()
        return True
    except Exception:
        return False


def _fetch_figures(arxiv_id: str, html: str, limit: int) -> tuple[list[tuple[bytes, str, str]], int]:
    if limit <= 0:
        return [], 0
    aid = _normalize_arxiv_id(arxiv_id)
    srcs, skipped = _figure_sources(html, AR5IV_URL.format(aid), limit)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", aid)
    out: list[tuple[bytes, str, str]] = []
    total = 0
    for i, (url, caption) in enumerate(srcs):
        path = urllib.parse.urlparse(url).path
        ext = os.path.splitext(path)[1].lower()
        name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(path))[-60:]
        dest = _FIG_CACHE_DIR / safe / f"{i:02d}_{name}"
        try:
            data = dest.read_bytes() if dest.exists() else _download_figure(url, dest)
        except Exception:
            data = None
        if not data:
            continue
        if not _decodable(data):
            try:
                dest.unlink(missing_ok=True)
            except Exception:
                pass
            skipped += 1
            continue
        if total + len(data) > FIG_MAX_TOTAL_BYTES:
            break
        total += len(data)
        out.append((data, FIG_MIME[ext], caption))
    return out, skipped


_NO_ID = "No arxiv_id provided. Pass an id from arxiv_search results."


def _read_error(arxiv_id: str, e: Exception) -> str:
    return (
        f"Failed to read paper {arxiv_id} ({type(e).__name__}: {e}). "
        "Try again or pick another candidate."
    )


def _unavailable(arxiv_id: str) -> str:
    return (
        f"Full text for {arxiv_id} is unavailable on ar5iv. Rely on the "
        "abstract from arxiv_search, or try another candidate."
    )


def _truncate(text: str) -> str:
    if len(text) > MAX_PAPER_CHARS:
        return text[:MAX_PAPER_CHARS] + "\n...[truncated]"
    return text


def _as_inspect_would_truncate(text: str) -> str:
    try:
        from inspect_ai.model._call_tools import truncate_tool_output

        truncated = truncate_tool_output("read_paper", text, None)
        return truncated.output if truncated else text
    except Exception:
        limit = 16 * 1024
        data = text.encode("utf-8")
        if len(data) <= limit:
            return text
        head = data[:limit].decode("utf-8", errors="ignore")
        return (
            "\nThe output of your call to read_paper was too long to be displayed.\n"
            "Here is a truncated version:\n<START_TOOL_OUTPUT>\n"
            f"{head}\n<END_TOOL_OUTPUT>\n"
        )


def _read_multimodal(arxiv_id: str, max_images: int) -> ToolResult:
    html = _fetch_paper_html(arxiv_id)
    if not html:
        return _unavailable(arxiv_id)
    text = _truncate(_html_to_text(html))
    if not text:
        return _unavailable(arxiv_id)
    figures, skipped = _fetch_figures(arxiv_id, html, max_images)
    if not figures:
        return text
    listing = "\n".join(
        f"[image {i + 1}] {cap or '(no caption)'}" for i, (_, _, cap) in enumerate(figures)
    )
    note = (f"\n({skipped} further figure(s) in this paper could not be attached (SVG or "
            f"undecodable image file). Their captions are in the text below.)" if skipped else "")
    header = (
        f"{len(figures)} figure(s) from this paper are attached as images after the text, "
        f"in this order:\n{listing}{note}\n\n"
    )
    content: list[ContentText | ContentImage] = [
        ContentText(text=header + _as_inspect_would_truncate(text))
    ]
    for data, mime, _cap in figures:
        b64 = base64.b64encode(data).decode("utf-8")
        content.append(ContentImage(image=f"data:{mime};base64,{b64}"))
    return content


@tool
def read_paper(max_images: int = 0) -> Tool:
    n_images = max(0, min(int(max_images or 0), FIG_MAX_IMAGES))

    async def execute(arxiv_id: str) -> str:
        """Read the full text of one arXiv paper to verify it against the description.

        Use this once you have a promising candidate from arxiv_search: fetch its
        full text and check whether it actually satisfies the constraints in the
        description before reporting it.

        Args:
            arxiv_id: The arXiv id of the paper (e.g. "2106.09063" or "2106.09063v4"),
                as shown in arxiv_search results.

        Returns:
            The paper's full text (title, abstract, body, tables), truncated if long.
        """
        if not arxiv_id or not str(arxiv_id).strip():
            return _NO_ID
        try:
            text = _fetch_paper_text(str(arxiv_id))
        except Exception as e:
            return _read_error(arxiv_id, e)
        if not text:
            return _unavailable(arxiv_id)
        return _truncate(text)

    async def execute_with_figures(arxiv_id: str) -> ToolResult:
        """Read the full text of one arXiv paper, with its figures, to verify it against the description.

        Use this once you have a promising candidate from arxiv_search: fetch its full
        text and figures and check whether it actually satisfies the constraints in the
        description before reporting it, so figure-derived claims can be checked against
        the figure itself rather than against its caption.

        Args:
            arxiv_id: The arXiv id of the paper (e.g. "2106.09063" or "2106.09063v4"),
                as shown in arxiv_search results.

        Returns:
            The paper's full text (title, abstract, body, tables), truncated if long,
            followed by its figures as images.
        """
        if not arxiv_id or not str(arxiv_id).strip():
            return _NO_ID
        try:
            return await anyio.to_thread.run_sync(
                _read_multimodal, str(arxiv_id), n_images, abandon_on_cancel=True
            )
        except Exception as e:
            return _read_error(arxiv_id, e)

    return execute_with_figures if n_images else execute


if __name__ == "__main__":
    import asyncio

    print(asyncio.run(arxiv_search()("attention is all you need", 2)))
    print("\n--- read_paper ---")
    print(asyncio.run(read_paper()("1706.03762"))[:600])
    print("\n--- read_paper(max_images=3) ---")
    out = asyncio.run(read_paper(max_images=3)("1706.03762"))
    if isinstance(out, list):
        for part in out:
            if part.type == "image":
                print(f"[image] {len(part.image)} chars of data uri: {part.image[:48]}...")
            else:
                print(part.text[:600])
    else:
        print("no figures attached:", str(out)[:300])
