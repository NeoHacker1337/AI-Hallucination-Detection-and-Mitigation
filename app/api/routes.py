"""HTTP endpoints."""

from __future__ import annotations

import io
import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import __version__
from app.api.schemas import (
    AskRequest,
    AskResponse,
    DocumentsJSON,
    DocumentsResponse,
    ExperimentStatus,
    JudgementOverride,
    ReviewAction,
    ReviewOut,
    StageOut,
)
from app.config import get_settings
from app.db import get_db, session_scope
from app.db.crud import resolve_review, save_result
from app.db.models import Experiment, Judgement, ReviewItem, Run
from app.evaluation.datasets import DatasetUnavailableError
from app.evaluation.experiment_config import ExperimentConfig
from app.evaluation.report import build_report, experiment_dir, results_dataframe
from app.evaluation.runner import create_experiment, submit_experiment
from app.pipeline.orchestrator import Pipeline
from app.pipeline.retrieval import DocumentInput, get_knowledge_base
from app.pipeline.settings import CONFIG_DESCRIPTIONS, build_settings
from app.providers import ProviderError, get_provider

router = APIRouter()


# --------------------------------------------------------------------------------------------- health
@router.get("/health")
def health() -> dict[str, Any]:
    """Liveness plus a quick check of the default LLM backend (does not load heavy models)."""
    settings = get_settings()
    llm_status = "ok"
    try:
        get_provider(settings.default_model).check_available()
    except ProviderError as exc:
        llm_status = f"unavailable: {exc}"
    return {
        "status": "ok",
        "version": __version__,
        "default_model": settings.default_model,
        "default_model_status": llm_status,
        "nli_model": settings.nli_model,
        "embedding_model": settings.embedding_model,
        "configurations": CONFIG_DESCRIPTIONS,
    }


# --------------------------------------------------------------------------------------------- ask
def _ask(req: AskRequest) -> AskResponse:
    try:
        settings = build_settings(req.configuration, req.overrides)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    spec = req.model or get_settings().default_model
    try:
        provider = get_provider(spec)
        provider.check_available()
        result = Pipeline(provider, settings, collection=req.collection).run(req.question)
    except ProviderError as exc:
        raise HTTPException(503, str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(409, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc
    with session_scope() as session:
        run = save_result(session, result, question_id="ask", source="ask")
        session.flush()
        review = session.execute(select(ReviewItem.id).where(ReviewItem.run_id == run.id)).scalar()
        run_id = run.id
    return AskResponse(
        run_id=run_id, question=result.question, model=result.model, configuration=result.config_name,
        final_answer=result.final_answer, candidate_answer=result.candidate_answer,
        decision=result.decision.value, decision_reasons=result.decision_reasons, confidence=result.confidence,
        abstained=result.abstained, escalated=result.escalated, review_item_id=review,
        n_regenerations=result.n_regenerations, citations=result.citations, latency_ms=result.latency_ms,
        verification_status=result.assessment.verification_status if result.assessment else None,
        issues=result.assessment.issues if result.assessment else [],
        remaining_uncertainty=result.assessment.remaining_uncertainty if result.assessment else [],
        sources=result.assessment.sources if result.assessment else [],
        llm_calls=result.llm_calls, prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
        stages=[StageOut(stage=s.stage, order=s.order, latency_ms=s.latency_ms, llm_calls=s.llm_calls,
                         prompt_tokens=s.prompt_tokens, completion_tokens=s.completion_tokens,
                         output=json.loads(json.dumps(s.output, default=str))) for s in result.stages],
    )


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Run one question through the chosen configuration and log every stage."""
    return _ask(req)


# --------------------------------------------------------------------------------------------- documents
def _read_upload(name: str, data: bytes) -> str:
    if name.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8", errors="replace")


@router.post("/documents", response_model=DocumentsResponse)
async def upload_documents(request: Request) -> DocumentsResponse:
    """Add documents to a knowledge-base collection.

    Accepts either JSON (`{"collection": "...", "documents": [{"title", "text", "source"}]}`) or
    multipart form data with one or more `files` (.txt, .md, .pdf) and an optional `collection` field.
    """
    content_type = request.headers.get("content-type", "")
    docs: list[DocumentInput] = []
    if content_type.startswith("application/json"):
        try:
            body = DocumentsJSON(**(await request.json()))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(422, f"Invalid JSON body: {exc}") from exc
        collection = body.collection
        docs = [DocumentInput(d.title, d.text, d.source) for d in body.documents]
    elif content_type.startswith("multipart/form-data"):
        form = await request.form()
        collection = str(form.get("collection") or "global")
        for upload in form.getlist("files"):
            if not hasattr(upload, "read"):
                continue
            data = await upload.read()  # type: ignore[union-attr]
            text = _read_upload(upload.filename or "upload", data)  # type: ignore[union-attr]
            if text.strip():
                docs.append(DocumentInput(title=upload.filename or "upload", text=text, source=upload.filename))  # type: ignore[union-attr]
    else:
        raise HTTPException(415, "Use application/json or multipart/form-data")
    if not docs:
        raise HTTPException(400, "No document text received.")
    kb = get_knowledge_base()
    try:
        ids = await run_in_threadpool(kb.add_documents, collection, docs)
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc
    return DocumentsResponse(collection=collection, document_ids=ids, total_chunks_in_collection=kb.count(collection))


@router.get("/documents/collections")
def collections() -> dict[str, int]:
    """Collections and their chunk counts."""
    return get_knowledge_base().list_collections()


# --------------------------------------------------------------------------------------------- experiments
def _status(exp: Experiment) -> ExperimentStatus:
    return ExperimentStatus(
        id=exp.id, name=exp.name, status=exp.status, progress_done=exp.progress_done,
        progress_total=exp.progress_total, error=exp.error, created_at=exp.created_at, started_at=exp.started_at,
        finished_at=exp.finished_at, config=exp.config, summary=exp.summary,
    )


@router.post("/experiments", response_model=ExperimentStatus, status_code=202)
def start_experiment(cfg: ExperimentConfig, db: Session = Depends(get_db)) -> ExperimentStatus:
    """Create an experiment and run it in the background. Poll GET /experiments/{id}."""
    if cfg.dataset not in ("truthfulqa", "halueval_qa", "custom_csv"):
        raise HTTPException(400, str(DatasetUnavailableError(f"Unknown dataset '{cfg.dataset}'")))
    exp_id = create_experiment(cfg)
    submit_experiment(exp_id)
    exp = db.get(Experiment, exp_id)
    assert exp is not None
    return _status(exp)


@router.post("/experiments/{experiment_id}/resume", response_model=ExperimentStatus, status_code=202)
def resume_experiment(experiment_id: int, db: Session = Depends(get_db)) -> ExperimentStatus:
    """Continue a failed or interrupted experiment: finished runs are kept, missing and errored runs are run."""
    exp = _get_exp(db, experiment_id)
    if exp.status != "failed":
        raise HTTPException(
            409, f"Experiment {experiment_id} is '{exp.status}'; only failed or interrupted experiments can be "
            "resumed (an experiment still running in another process must be stopped first)",
        )
    submit_experiment(experiment_id, resume=True)
    db.refresh(exp)
    return _status(exp)


@router.get("/experiments", response_model=list[ExperimentStatus])
def list_experiments(db: Session = Depends(get_db)) -> list[ExperimentStatus]:
    return [_status(e) for e in db.execute(select(Experiment).order_by(Experiment.id.desc())).scalars()]


def _get_exp(db: Session, experiment_id: int) -> Experiment:
    exp = db.get(Experiment, experiment_id)
    if exp is None:
        raise HTTPException(404, f"Experiment {experiment_id} not found")
    return exp


@router.get("/experiments/{experiment_id}", response_model=ExperimentStatus)
def get_experiment(experiment_id: int, db: Session = Depends(get_db)) -> ExperimentStatus:
    """Status, progress and summary metrics."""
    return _status(_get_exp(db, experiment_id))


@router.get("/experiments/{experiment_id}/results")
def get_results(experiment_id: int, format: str = Query("json", pattern="^(json|csv)$"),
                db: Session = Depends(get_db)) -> Any:
    """Per-question results with judge decisions, as JSON or a CSV download."""
    _get_exp(db, experiment_id)
    df = results_dataframe(db, experiment_id)
    if format == "csv":
        path = experiment_dir(experiment_id) / "results.csv"
        df.to_csv(path, index=False)
        return FileResponse(path, media_type="text/csv", filename=f"experiment_{experiment_id}_results.csv")
    return JSONResponse(json.loads(df.to_json(orient="records")))


@router.get("/experiments/{experiment_id}/report", response_class=HTMLResponse)
def get_report(experiment_id: int, rebuild: bool = False, db: Session = Depends(get_db)) -> HTMLResponse:
    """HTML report with tables and charts (rebuilt on request, e.g. after manual judge corrections)."""
    exp = _get_exp(db, experiment_id)
    path = experiment_dir(experiment_id) / "report.html"
    if rebuild or not path.exists():
        if exp.status in ("pending", "running") and not rebuild:
            raise HTTPException(409, f"Experiment is {exp.status}; the report is built when it completes "
                                     "(or pass ?rebuild=true for a partial report).")
        summary = build_report(db, experiment_id)
        if exp.status == "completed":
            exp.summary = summary
    return HTMLResponse(path.read_text(encoding="utf-8"))


@router.post("/runs/{run_id}/judgement")
def override_judgement(run_id: int, body: JudgementOverride, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Record a manual label that replaces the automatic judge label in all metrics."""
    j = db.execute(select(Judgement).where(Judgement.run_id == run_id, Judgement.target == body.target)).scalar()
    if j is None:
        raise HTTPException(404, f"No {body.target} judgement for run {run_id}")
    j.manual_label = body.label
    return {"run_id": run_id, "target": body.target, "automatic_label": j.label, "manual_label": j.manual_label}


@router.get("/runs/{run_id}")
def get_run(run_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Full stored record of one run, including every stage output."""
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(404, f"Run {run_id} not found")
    cols = {c.name: getattr(run, c.name) for c in Run.__table__.columns}
    cols["stages"] = [{"stage": s.stage, "order": s.order, "latency_ms": s.latency_ms, "llm_calls": s.llm_calls,
                       "prompt_tokens": s.prompt_tokens, "completion_tokens": s.completion_tokens,
                       "output": s.output} for s in run.stages]
    cols["judgements"] = [{c.name: getattr(j, c.name) for c in Judgement.__table__.columns} for j in run.judgements]
    return json.loads(json.dumps(cols, default=str))


# --------------------------------------------------------------------------------------------- review
def _review_out(item: ReviewItem) -> ReviewOut:
    return ReviewOut.model_validate(item, from_attributes=True)


@router.get("/review-queue", response_model=list[ReviewOut])
def review_queue(status: str = "pending", db: Session = Depends(get_db)) -> list[ReviewOut]:
    """Escalated answers (status: pending | approved | rejected | edited | all)."""
    query = select(ReviewItem).order_by(ReviewItem.id)
    if status != "all":
        query = query.where(ReviewItem.status == status)
    return [_review_out(i) for i in db.execute(query).scalars()]


@router.post("/review/{item_id}", response_model=ReviewOut)
def review(item_id: int, body: ReviewAction, db: Session = Depends(get_db)) -> ReviewOut:
    """Approve, reject or edit an escalated answer."""
    item = db.get(ReviewItem, item_id)
    if item is None:
        raise HTTPException(404, f"Review item {item_id} not found")
    try:
        resolve_review(db, item, body.action, body.corrected_answer, body.note)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    db.flush()
    return _review_out(item)
