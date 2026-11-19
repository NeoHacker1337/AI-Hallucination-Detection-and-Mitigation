"""Batch experiment runner: every model x configuration x question, with the same seeds and
decoding settings for every configuration."""

from __future__ import annotations

import json
import logging
import platform
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib import metadata
from typing import Any

from sqlalchemy import select, update

from app.config import get_settings
from app.db import session_scope
from app.db.crud import save_result
from app.db.models import Experiment, Judgement, Run
from app.evaluation.datasets import DatasetItem, load_benchmark
from app.evaluation.experiment_config import ExperimentConfig
from app.evaluation.judge import AnswerJudge, JudgeDecision
from app.evaluation.report import build_report, experiment_dir
from app.pipeline.abstention import is_abstention
from app.pipeline.claims import rule_based_claims
from app.pipeline.evidence_check import evidence_check
from app.pipeline.nli import NLIScorer, get_nli
from app.pipeline.orchestrator import Pipeline, PipelineResult
from app.pipeline.retrieval import DocumentInput, KnowledgeBase, Passage, get_knowledge_base
from app.pipeline.settings import CONFIG_DESCRIPTIONS, build_settings
from app.providers import get_provider
from app.utils import set_global_seed, to_jsonable

log = logging.getLogger("halluguard.runner")
_EXECUTOR: ThreadPoolExecutor | None = None
_EXEC_LOCK = threading.Lock()


class ExperimentError(RuntimeError):
    """Raised for configuration problems detected before or during an experiment."""


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in ("torch", "transformers", "sentence-transformers", "faiss-cpu", "datasets", "scikit-learn", "scipy"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def _model_details(spec: str) -> dict[str, Any]:
    """Record the exact Ollama model digest when possible, for reproducibility."""
    provider = get_provider(spec)
    info: dict[str, Any] = {"spec": provider.spec}
    if provider.prefix == "ollama":
        import httpx

        try:
            resp = httpx.get(f"{provider.base_url}/api/tags", timeout=10)  # type: ignore[attr-defined]
            for m in resp.json().get("models", []):
                if m.get("name") in (provider.model, f"{provider.model}:latest"):
                    info.update(digest=m.get("digest"), modified_at=m.get("modified_at"), details=m.get("details"))
        except Exception as exc:  # noqa: BLE001
            info["digest_error"] = str(exc)
    return info


def faithfulness_score(answer: str, passages: list[Passage], nli: NLIScorer) -> float | None:
    """Share of the answer's (rule-extracted) claims entailed by the retrieved passages.

    Computed identically for every configuration that retrieved passages, independently of
    whether the evidence-check stage was switched on. None when not applicable.
    """
    if not passages or not answer or is_abstention(answer):
        return None
    claims = rule_based_claims(answer)
    if not claims:
        return None
    return evidence_check(claims, passages, nli).supported_fraction


def _store_judgement(session, run: Run, target: str, answer: str, dec: JudgeDecision, item: DatasetItem) -> None:  # type: ignore[no-untyped-def]
    session.add(Judgement(
        run_id=run.id, target=target, answer=answer or "", mode=dec.mode, label=dec.label, nli_label=dec.nli_label,
        nli_correct_score=dec.nli_correct_score, nli_incorrect_score=dec.nli_incorrect_score,
        nli_matched_reference=dec.matched_reference, llm_label=dec.llm_label, llm_raw=dec.llm_raw,
        judges_disagree=dec.judges_disagree, uncertain=dec.uncertain, reason=dec.reason,
        references={"correct": item.correct_answers, "incorrect": item.incorrect_answers},
    ))


def _prepare_knowledge_base(cfg: ExperimentConfig, experiment_id: int, items: list[DatasetItem], kb: KnowledgeBase) -> str:
    """Build a per-experiment collection from dataset passages, or check the chosen collection."""
    contexts = [c for item in items for c in item.contexts]
    if contexts:
        collection = f"exp{experiment_id}_{cfg.dataset}"
        kb.delete_collection(collection)
        unique = list(dict.fromkeys(contexts))
        kb.add_documents(collection, [DocumentInput(title=f"{cfg.dataset} passage {i}", text=t, source=cfg.dataset)
                                      for i, t in enumerate(unique)])
        return collection
    if kb.count(cfg.kb_collection) == 0:
        raise ExperimentError(
            f"Dataset '{cfg.dataset}' has no gold passages and collection '{cfg.kb_collection}' is empty, so "
            "retrieval configurations (C1-C5) cannot run. Upload documents (POST /documents), run "
            "scripts/build_truthfulqa_kb.py, or set kb_collection to a filled collection."
        )
    return cfg.kb_collection


def _changed_settings(recorded: dict, current: dict, prefix: str = "") -> list[str]:
    """Keys whose value differs between recorded and current settings. Keys that exist only in the
    current settings are settings added later; they are ignored (their defaults keep the old
    behaviour for C0-C5)."""
    changed: list[str] = []
    for key, old_value in recorded.items():
        if key not in current:
            changed.append(f"{prefix}{key} (removed)")
        elif isinstance(old_value, dict) and isinstance(current[key], dict):
            changed.extend(_changed_settings(old_value, current[key], f"{prefix}{key}."))
        elif old_value != current[key]:
            changed.append(f"{prefix}{key}")
    return changed


def completed_run_keys(experiment_id: int) -> set[tuple[str, str, str]]:
    """(model, configuration, question id) of every run of the experiment that finished without error."""
    with session_scope() as session:
        rows = session.execute(
            select(Run.model, Run.config_name, Run.question_id)
            .where(Run.experiment_id == experiment_id, Run.error.is_(None))
        ).all()
    return {(m, c, q) for m, c, q in rows}


def delete_failed_runs(experiment_id: int) -> int:
    """Delete the experiment's errored runs (with their stage logs and judgements) so they can be re-run."""
    with session_scope() as session:
        runs = session.scalars(select(Run).where(Run.experiment_id == experiment_id, Run.error.is_not(None))).all()
        for run in runs:
            session.delete(run)
        return len(runs)


def run_experiment(experiment_id: int, resume: bool = False) -> dict[str, Any]:
    """Execute a stored experiment synchronously. Updates status, progress and summary in the DB.

    With `resume=True` the runs that already finished without error are kept and skipped, errored
    runs are deleted and re-run, and the question sample is checked against the one recorded when
    the experiment first started (the experiment must use the same dataset sample to stay valid).
    """
    with session_scope() as session:
        exp = session.get(Experiment, experiment_id)
        if exp is None:
            raise ExperimentError(f"Experiment {experiment_id} not found")
        if resume and exp.status == "completed":
            raise ExperimentError(f"Experiment {experiment_id} is already completed; nothing to resume")
        cfg = ExperimentConfig(**exp.config)
        exp.status, exp.error, exp.finished_at = "running", None, None
        if not resume or exp.started_at is None:
            exp.started_at = datetime.now(timezone.utc)
    try:
        set_global_seed(cfg.seed)
        # --- preflight: fail fast with clear messages, never fake output ------------------------
        for spec in cfg.models + ([cfg.judge.llm_model] if cfg.judge.llm_model else []):
            get_provider(spec).check_available()
        items = load_benchmark(cfg.dataset, cfg.sample_size, cfg.seed, csv_path=cfg.csv_path)
        nli = get_nli()
        needs_retrieval = any(build_settings(c).enable_retrieval for c in cfg.configurations)
        kb = get_knowledge_base() if needs_retrieval else None
        collection = _prepare_knowledge_base(cfg, experiment_id, items, kb) if kb else cfg.kb_collection
        judge = AnswerJudge(
            mode=cfg.judge.mode, nli=nli,
            llm=get_provider(cfg.judge.llm_model) if cfg.judge.llm_model else None,
            entail_threshold=cfg.judge.entail_threshold, seed=cfg.seed,
        )
        overrides = {**cfg.pipeline_overrides, "seed": cfg.seed}
        resolved = {c: build_settings(c, overrides).model_dump() for c in cfg.configurations}
        config_path = experiment_dir(experiment_id) / "config.json"
        if resume and config_path.exists():
            recorded = json.loads(config_path.read_text(encoding="utf-8"))
            if recorded.get("question_ids") != [i.id for i in items]:
                raise ExperimentError(
                    f"Cannot resume experiment {experiment_id}: the dataset sample differs from the one it "
                    "started with (dataset changed or not reproducible). Start a new experiment instead."
                )
            changed = _changed_settings(recorded.get("resolved_pipeline_settings", {}), to_jsonable(resolved))
            if changed:
                raise ExperimentError(
                    f"Cannot resume experiment {experiment_id}: pipeline settings differ from the recorded ones "
                    f"({', '.join(changed[:8])}). Resuming would mix two pipeline versions; start a new experiment."
                )
            recorded.setdefault("resumed_at", []).append(datetime.now(timezone.utc).isoformat())
            recorded["models_at_resume"] = [_model_details(m) for m in cfg.models]
            config_path.write_text(json.dumps(to_jsonable(recorded), indent=2))
        else:
            config_path.write_text(json.dumps(to_jsonable({
            "experiment": cfg.model_dump(), "resolved_pipeline_settings": resolved,
            "configuration_descriptions": {c: CONFIG_DESCRIPTIONS[c] for c in cfg.configurations},
            "knowledge_base_collection": collection, "question_ids": [i.id for i in items],
            "models": [_model_details(m) for m in cfg.models],
            "global_settings": {"embedding_model": get_settings().embedding_model, "nli_model": get_settings().nli_model},
            "versions": _versions(), "started_at": datetime.now(timezone.utc).isoformat(),
            }), indent=2))

        total = len(cfg.models) * len(cfg.configurations) * len(items)
        finished: set[tuple[str, str, str]] = set()
        if resume:
            removed = delete_failed_runs(experiment_id)
            finished = completed_run_keys(experiment_id)
            log.info("experiment %s: resuming with %s finished runs (%s errored runs removed for re-run)",
                     experiment_id, len(finished), removed)
        done = sum(1 for spec in cfg.models for c in cfg.configurations for i in items if (spec, c, i.id) in finished)
        with session_scope() as session:
            session.execute(update(Experiment).where(Experiment.id == experiment_id).values(progress_total=total, progress_done=done))

        for spec in cfg.models:
            provider = get_provider(spec)
            pipelines = {
                c: Pipeline(provider, build_settings(c, overrides), kb=kb, nli=nli, collection=collection)
                for c in cfg.configurations
            }
            for item in items:
                for cname, pipeline in pipelines.items():
                    if (spec, cname, item.id) in finished:
                        continue
                    _run_one(experiment_id, cfg, item, pipeline, judge, nli, spec, cname)
                    done += 1
                    with session_scope() as session:
                        # status is re-asserted because a server started meanwhile marks experiments it
                        # does not own as interrupted, although this process is still running them
                        session.execute(update(Experiment).where(Experiment.id == experiment_id)
                                        .values(progress_done=done, status="running", error=None))
                    log.info("experiment %s: %s/%s", experiment_id, done, total)

        with session_scope() as session:
            summary = build_report(session, experiment_id)
            exp = session.get(Experiment, experiment_id)
            assert exp is not None
            exp.summary, exp.status, exp.finished_at = summary, "completed", datetime.now(timezone.utc)
        return summary
    except Exception as exc:  # noqa: BLE001
        log.error("experiment %s failed: %s", experiment_id, traceback.format_exc())
        with session_scope() as session:
            exp = session.get(Experiment, experiment_id)
            if exp is not None:
                exp.status, exp.error, exp.finished_at = "failed", f"{type(exc).__name__}: {exc}", datetime.now(timezone.utc)
        raise


def _run_one(experiment_id: int, cfg: ExperimentConfig, item: DatasetItem, pipeline: Pipeline,
             judge: AnswerJudge, nli: NLIScorer, spec: str, cname: str) -> None:
    """Run, judge and store one (model, configuration, question). Errors are stored, not hidden."""
    try:
        result: PipelineResult = pipeline.run(item.question)
    except Exception as exc:  # noqa: BLE001
        with session_scope() as session:
            session.add(Run(
                experiment_id=experiment_id, question_id=item.id, question=item.question, model=spec,
                config_name=cname, settings=pipeline.s.model_dump(), error=f"{type(exc).__name__}: {exc}",
            ))
        log.warning("run failed (%s, %s, %s): %s", spec, cname, item.id, exc)
        return
    init_dec = judge.judge(item.question, result.initial_answer, item.correct_answers, item.incorrect_answers)
    cand_dec = init_dec if result.candidate_answer == result.initial_answer else judge.judge(
        item.question, result.candidate_answer, item.correct_answers, item.incorrect_answers)
    faith = faithfulness_score(result.final_answer, result.passages, nli) if not result.abstained else None
    with session_scope() as session:
        run = save_result(session, result, item.id, experiment_id, cfg.queue_escalations, f"experiment:{experiment_id}")
        run.faithfulness = faith
        _store_judgement(session, run, "initial", result.initial_answer, init_dec, item)
        _store_judgement(session, run, "candidate", result.candidate_answer, cand_dec, item)


def submit_experiment(experiment_id: int, resume: bool = False) -> None:
    """Run (or resume) an experiment in a background worker thread (used by the API)."""
    global _EXECUTOR
    with _EXEC_LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = ThreadPoolExecutor(max_workers=get_settings().max_parallel_experiments, thread_name_prefix="exp")
    def _job() -> None:
        try:
            run_experiment(experiment_id, resume=resume)
        except Exception:  # noqa: BLE001  (already recorded in the DB)
            pass
    _EXECUTOR.submit(_job)


def create_experiment(cfg: ExperimentConfig) -> int:
    """Insert a pending experiment row and return its id."""
    with session_scope() as session:
        exp = Experiment(name=cfg.name, status="pending", config=cfg.model_dump())
        session.add(exp)
        session.flush()
        return exp.id


def mark_interrupted_experiments() -> None:
    """Experiments left 'running' or 'pending' by a previous process can never finish; mark them failed."""
    with session_scope() as session:
        session.execute(
            update(Experiment).where(Experiment.status.in_(["running", "pending"]))
            .values(status="failed", error="interrupted: the server stopped before the experiment finished")
        )
