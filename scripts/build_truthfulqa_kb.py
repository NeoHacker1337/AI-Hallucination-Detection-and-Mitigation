"""Build a retrieval corpus for TruthfulQA from the source URLs listed in the dataset.

TruthfulQA ships no passages, only a `source` URL per question (mostly Wikipedia). This script
downloads those pages, keeps the paragraph text, and indexes it into a collection (default
"truthfulqa_sources"). Pages that cannot be fetched are reported and skipped - nothing is invented.

    python -m scripts.build_truthfulqa_kb --sample-size 50 --seed 42

Use the same --sample-size and --seed as the experiment so the sources of every sampled question
are included. Then run the experiment with "kb_collection": "truthfulqa_sources".
"""

from __future__ import annotations

import argparse
import re
import sys
import time

import httpx

from app.db import init_db
from app.evaluation.datasets import DatasetUnavailableError, load_benchmark

USER_AGENT = "HalluGuard-research/1.0 (academic hallucination study)"


def page_text(html_text: str) -> str:
    """Keep the text of <p> elements (falls back to all visible text)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "table", "sup"]):
        tag.decompose()
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = "\n".join(p for p in paragraphs if len(p.split()) >= 8) or soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--collection", default="truthfulqa_sources")
    parser.add_argument("--max-words-per-page", type=int, default=4000)
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between requests")
    args = parser.parse_args()

    try:
        items = load_benchmark("truthfulqa", args.sample_size, args.seed)
    except DatasetUnavailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    urls = list(dict.fromkeys(i.metadata.get("source") for i in items
                              if str(i.metadata.get("source") or "").startswith("http")))
    print(f"{len(items)} questions, {len(urls)} unique source URLs")

    init_db()
    from app.pipeline.retrieval import DocumentInput, get_knowledge_base

    docs: list[DocumentInput] = []
    failed: list[tuple[str, str]] = []
    with httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=30) as client:
        for n, url in enumerate(urls, start=1):
            try:
                resp = client.get(url)
                resp.raise_for_status()
                text = " ".join(page_text(resp.text).split()[: args.max_words_per_page])
                if len(text.split()) < 30:
                    failed.append((url, "too little text"))
                else:
                    docs.append(DocumentInput(title=url.rsplit("/", 1)[-1][:200] or url, text=text, source=url))
            except Exception as exc:  # noqa: BLE001
                failed.append((url, str(exc)[:120]))
            print(f"  [{n}/{len(urls)}] {'ok' if not failed or failed[-1][0] != url else 'FAILED'} {url}")
            time.sleep(args.delay)
    if not docs:
        print("ERROR: no source page could be fetched.", file=sys.stderr)
        return 1
    kb = get_knowledge_base()
    kb.add_documents(args.collection, docs)
    print(f"Indexed {len(docs)} pages into '{args.collection}' ({kb.count(args.collection)} chunks). "
          f"{len(failed)} failed:")
    for url, reason in failed:
        print(f"  - {url}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
