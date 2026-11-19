"""Optional external retrieval from Wikipedia (MediaWiki search + REST page summaries).

Only used when a configuration sets `web_search="wikipedia"`; the question text is then sent to
en.wikipedia.org. Failures (network down, timeouts, HTTP errors) never crash the pipeline: they
are returned as an error string, logged in the trace, and the pipeline continues with whatever
local evidence it has (or abstains if there is none). Each passage carries the page URL and the
page's last-modified timestamp so freshness can be judged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import quote

import httpx

from app.config import get_settings
from app.pipeline.retrieval import Passage

API_URL = "https://en.wikipedia.org/w/api.php"
SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
_UA_HINT = ("Wikipedia refused the request. Its robot policy requires a User-Agent with contact details: "
            "set HG_WEB_USER_AGENT, e.g. 'MyResearchBot/1.0 (https://your-site.example/contact)'.")


@dataclass
class WebSearchResult:
    """Passages found on the web plus any error that occurred."""

    passages: list[Passage] = field(default_factory=list)
    error: str | None = None
    query: str = ""

    def to_dict(self) -> dict:
        return {"query": self.query, "error": self.error, "passages": [p.to_dict() for p in self.passages]}


def wikipedia_search(query: str, k: int = 3, client: httpx.Client | None = None) -> WebSearchResult:
    """Search Wikipedia and return the summaries of the top-k pages as passages."""
    result = WebSearchResult(query=query)
    own = client is None
    if own:
        settings = get_settings()
        if not settings.web_user_agent.strip():
            result.error = "web search not configured: " + _UA_HINT
            return result
        client = httpx.Client(timeout=settings.web_timeout_s, follow_redirects=True,
                              headers={"User-Agent": settings.web_user_agent})
    http = client
    assert http is not None
    try:
        resp = http.get(API_URL, params={
            "action": "query", "list": "search", "srsearch": query, "srlimit": k, "format": "json",
        })
        if resp.status_code in (403, 429):
            result.error = f"HTTP {resp.status_code}: " + (_UA_HINT if resp.status_code == 403 else "rate limited")
            return result
        resp.raise_for_status()
        hits = resp.json().get("query", {}).get("search", [])
        for rank, hit in enumerate(hits[:k]):
            title = hit.get("title", "")
            summary = http.get(SUMMARY_URL.format(title=quote(title.replace(" ", "_"), safe="")))
            if summary.status_code != 200:
                continue
            data = summary.json()
            text = (data.get("extract") or "").strip()
            if not text:
                continue
            url = data.get("content_urls", {}).get("desktop", {}).get("page") or f"https://en.wikipedia.org/wiki/{quote(title)}"
            result.passages.append(Passage(
                chunk_id=-1 - rank, source_id=url, title=f"Wikipedia: {title}", text=text,
                # search rank is the only relevance signal Wikipedia gives; map it to (0, 1]
                score=1.0 / (rank + 1), origin="web", source=url, last_modified=data.get("timestamp"),
            ))
    except (httpx.HTTPError, ValueError) as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        if own:
            http.close()
    return result
