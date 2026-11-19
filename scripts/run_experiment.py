"""Run a full experiment from the command line (no server needed).

    python -m scripts.run_experiment --dataset halueval_qa --sample-size 30 \
        --models ollama:llama3 ollama:mistral --configs C0 C1 C2 C3 C4 C5

or from a JSON file with the same fields as POST /experiments:

    python -m scripts.run_experiment --config-file data/experiment_example.json

or continue an interrupted experiment (finished runs are kept, the rest are run):

    python -m scripts.run_experiment --resume 3
"""

from __future__ import annotations

import argparse
import json
import sys

from pydantic import ValidationError

from app.db import init_db
from app.evaluation.experiment_config import ExperimentConfig
from app.evaluation.report import experiment_dir
from app.evaluation.runner import create_experiment, run_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-file", help="JSON file with an ExperimentConfig")
    parser.add_argument("--resume", type=int, metavar="EXPERIMENT_ID",
                        help="Continue an interrupted experiment; all other options are ignored")
    parser.add_argument("--name", default="cli-experiment")
    parser.add_argument("--dataset", default="truthfulqa")
    parser.add_argument("--csv-path")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--configs", nargs="+", default=["C0", "C1", "C2", "C3", "C4", "C5"])
    parser.add_argument("--kb-collection", default="global")
    parser.add_argument("--judge", choices=["nli", "llm", "both"], default="nli")
    parser.add_argument("--judge-model")
    parser.add_argument("--n-samples", type=int, help="Override number of samples for SE/self-consistency")
    args = parser.parse_args()

    if args.resume is not None:
        init_db()
        return _finish(args.resume, resume=True)

    try:
        if args.config_file:
            with open(args.config_file, encoding="utf-8") as fh:
                cfg = ExperimentConfig(**json.load(fh))
        else:
            if not args.models:
                parser.error("--models is required without --config-file")
            overrides = {"n_samples": args.n_samples} if args.n_samples else {}
            cfg = ExperimentConfig(
                name=args.name, dataset=args.dataset, csv_path=args.csv_path, sample_size=args.sample_size,
                seed=args.seed, models=args.models, configurations=args.configs, kb_collection=args.kb_collection,
                judge={"mode": args.judge, "llm_model": args.judge_model}, pipeline_overrides=overrides,
            )
    except ValidationError as exc:
        print(f"Invalid experiment configuration:\n{exc}", file=sys.stderr)
        return 2

    init_db()
    exp_id = create_experiment(cfg)
    print(f"Experiment {exp_id} created: {len(cfg.models)} model(s) x {len(cfg.configurations)} configuration(s) "
          f"x {cfg.sample_size} question(s)")
    return _finish(exp_id, resume=False)


def _finish(exp_id: int, resume: bool) -> int:
    """Run (or resume) the experiment and print the hypothesis statements and artifact location."""
    if resume:
        print(f"Resuming experiment {exp_id}")
    try:
        result = run_experiment(exp_id, resume=resume)
    except Exception as exc:  # noqa: BLE001
        print(f"Experiment {exp_id} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("\n".join(result["hypothesis"]))
    print(f"\nArtifacts: {experiment_dir(exp_id)} (results.csv, summary.csv, significance.csv, *.png, report.html)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
