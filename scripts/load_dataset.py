"""Download (and cache) a benchmark, print a preview, and optionally index its gold passages.

Examples:
    python -m scripts.load_dataset --dataset truthfulqa --sample-size 5
    python -m scripts.load_dataset --dataset halueval_qa --sample-size 100 --index-collection halueval_kb
    python -m scripts.load_dataset --dataset custom_csv --csv-path data/custom_example.csv
"""

from __future__ import annotations

import argparse
import sys

from app.db import init_db
from app.evaluation.datasets import DatasetUnavailableError, load_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["truthfulqa", "halueval_qa", "custom_csv"])
    parser.add_argument("--sample-size", type=int, default=None, help="Number of items (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--index-collection", default=None,
                        help="If set, add the dataset's gold passages to this knowledge-base collection")
    parser.add_argument("--preview", type=int, default=3)
    args = parser.parse_args()

    try:
        items = load_benchmark(args.dataset, args.sample_size, args.seed, csv_path=args.csv_path)
    except DatasetUnavailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    with_ctx = sum(1 for i in items if i.contexts)
    print(f"Loaded {len(items)} items from '{args.dataset}' ({with_ctx} with gold passages).")
    for item in items[: args.preview]:
        print(f"- [{item.id}] {item.question}\n    correct: {item.correct_answers[:2]}\n"
              f"    incorrect: {item.incorrect_answers[:2]}")

    if args.index_collection:
        if not with_ctx:
            print("ERROR: this dataset has no gold passages to index.", file=sys.stderr)
            return 1
        init_db()
        from app.pipeline.retrieval import DocumentInput, get_knowledge_base

        kb = get_knowledge_base()
        texts = list(dict.fromkeys(c for i in items for c in i.contexts))
        kb.add_documents(args.index_collection, [DocumentInput(f"{args.dataset} passage {n}", t, args.dataset)
                                                 for n, t in enumerate(texts)])
        print(f"Indexed {len(texts)} passages into collection '{args.index_collection}' "
              f"({kb.count(args.index_collection)} chunks).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
