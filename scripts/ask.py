"""Ask one question from the command line and print every stage.

    python -m scripts.ask "Who wrote Pride and Prejudice?" --config C5 --model ollama:llama3
"""

from __future__ import annotations

import argparse
import json
import sys

from app.config import get_settings
from app.db import init_db, session_scope
from app.db.crud import save_result
from app.pipeline.orchestrator import Pipeline
from app.pipeline.settings import build_settings
from app.providers import ProviderError, get_provider


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", default="C5")
    parser.add_argument("--collection", default="global")
    args = parser.parse_args()

    init_db()
    spec = args.model or get_settings().default_model
    try:
        provider = get_provider(spec)
        provider.check_available()
        result = Pipeline(provider, build_settings(args.config), collection=args.collection).run(args.question)
    except (ProviderError, LookupError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for st in result.stages:
        print(f"\n=== {st.order + 1}. {st.stage} ({st.latency_ms:.0f} ms, {st.llm_calls} calls, "
              f"{st.prompt_tokens + st.completion_tokens} tokens)")
        print(json.dumps(st.output, indent=2, default=str)[:3000])
    print(f"\nDECISION: {result.decision.value} (confidence={result.confidence}) - {'; '.join(result.decision_reasons)}")
    print(f"ANSWER: {result.final_answer}")
    print(f"Total: {result.latency_ms / 1000:.1f} s, {result.llm_calls} LLM calls, "
          f"{result.prompt_tokens + result.completion_tokens} tokens")
    with session_scope() as session:
        run = save_result(session, result, question_id="cli", source="cli")
        print(f"Saved as run {run.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
