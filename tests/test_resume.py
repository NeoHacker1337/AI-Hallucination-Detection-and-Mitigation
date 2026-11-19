"""Resuming an interrupted experiment: finished runs are kept, errored and missing runs are run."""

from __future__ import annotations

import pytest

from app.evaluation.datasets import DatasetItem

ITEMS = [
    DatasetItem(id=f"q{i}", question=f"What is the capital of country {i}?", correct_answers=[f"City {i}"])
    for i in range(3)
]


class CountingProvider:
    """Fake LLM that answers every question and counts the questions it was asked."""

    prefix = "fake"
    model = "fake"
    spec = "fake:fake"

    def __init__(self) -> None:
        self.questions: list[str] = []

    def check_available(self) -> None:
        return None

    def chat(self, messages, params):  # type: ignore[no-untyped-def]
        from app.providers import Completion

        self.questions.append(messages[-1].content)
        return Completion(text="City 0.", model=self.spec, prompt_tokens=10, completion_tokens=3, latency_ms=1.0)


@pytest.fixture
def runner_env(monkeypatch, keyword_nli):  # type: ignore[no-untyped-def]
    import app.db.database as database
    from app.evaluation import runner

    monkeypatch.setattr(database, "_engine", None)
    monkeypatch.setattr(database, "_SessionLocal", None)
    database.init_db()
    provider = CountingProvider()
    monkeypatch.setattr(runner, "get_provider", lambda spec: provider)
    monkeypatch.setattr(runner, "load_benchmark", lambda *a, **k: list(ITEMS))
    monkeypatch.setattr(runner, "get_nli", lambda: keyword_nli)
    monkeypatch.setattr(runner, "build_report", lambda session, exp_id: {"hypothesis": []})
    yield runner, provider
    if database._engine is not None:
        database._engine.dispose()


def _runs(experiment_id: int):  # type: ignore[no-untyped-def]
    from sqlalchemy import select

    from app.db import session_scope
    from app.db.models import Run

    with session_scope() as session:
        return [(r.question_id, r.config_name, r.error) for r in session.scalars(
            select(Run).where(Run.experiment_id == experiment_id).order_by(Run.id))]


def test_resume_skips_finished_and_reruns_errored(runner_env) -> None:  # type: ignore[no-untyped-def]
    from app.db import session_scope
    from app.db.models import Experiment, Run
    from app.evaluation.experiment_config import ExperimentConfig

    runner, provider = runner_env
    cfg = ExperimentConfig(name="t", dataset="custom_csv", csv_path="unused.csv", sample_size=3, models=["fake:fake"], configurations=["C0"])
    exp_id = runner.create_experiment(cfg)
    runner.run_experiment(exp_id)
    assert len(_runs(exp_id)) == 3 and len(provider.questions) == 3

    # simulate an interruption: q2 never ran, q1 errored
    with session_scope() as session:
        for run in session.query(Run).filter(Run.experiment_id == exp_id, Run.question_id.in_(["q1", "q2"])):
            session.delete(run)
        session.add(Run(experiment_id=exp_id, question_id="q1", question="x", model="fake:fake",
                        config_name="C0", settings={}, error="ProviderError: timeout"))
        exp = session.get(Experiment, exp_id)
        exp.status, exp.progress_done = "failed", 1

    provider.questions.clear()
    runner.run_experiment(exp_id, resume=True)
    assert len(provider.questions) == 2  # only q1 (errored) and q2 (missing) were run again
    runs = _runs(exp_id)
    assert sorted(q for q, _, _ in runs) == ["q0", "q1", "q2"]
    assert all(err is None for _, _, err in runs)
    with session_scope() as session:
        exp = session.get(Experiment, exp_id)
        assert exp.status == "completed" and exp.progress_done == exp.progress_total == 3


def test_resume_refuses_a_different_question_sample(runner_env, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from app.db import session_scope
    from app.db.models import Experiment
    from app.evaluation.experiment_config import ExperimentConfig

    runner, _ = runner_env
    cfg = ExperimentConfig(name="t", dataset="custom_csv", csv_path="unused.csv", sample_size=3, models=["fake:fake"], configurations=["C0"])
    exp_id = runner.create_experiment(cfg)
    runner.run_experiment(exp_id)
    with session_scope() as session:
        session.get(Experiment, exp_id).status = "failed"
    monkeypatch.setattr(runner, "load_benchmark", lambda *a, **k: list(reversed(ITEMS)))
    with pytest.raises(runner.ExperimentError, match="sample differs"):
        runner.run_experiment(exp_id, resume=True)


def test_completed_experiment_is_not_resumed(runner_env) -> None:  # type: ignore[no-untyped-def]
    from app.evaluation.experiment_config import ExperimentConfig

    runner, _ = runner_env
    cfg = ExperimentConfig(name="t", dataset="custom_csv", csv_path="unused.csv", sample_size=3, models=["fake:fake"], configurations=["C0"])
    exp_id = runner.create_experiment(cfg)
    runner.run_experiment(exp_id)
    with pytest.raises(runner.ExperimentError, match="already completed"):
        runner.run_experiment(exp_id, resume=True)
