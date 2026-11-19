"""Behaviour test: the same questions answered without mitigation (Scenario A, C0) and with it
(Scenario B, C6), plus RAG-only (C1) to separate the effect of retrieval from the effect of
detection, verification and abstention.

Question categories (data/behaviour/questions.csv):
  known       - facts the model very likely learned in training
  unknown     - fictional entities / future events: no evidence exists, the right behaviour is to abstain
  outdated    - facts from after the models' training cut-off (only the knowledge base has them)
  ambiguous   - questions that cannot be answered without clarification
  misleading  - questions built on a false premise; the right answer corrects the premise

Every number in the output is computed from the runs made by this script. The judge is automatic
(NLI + lexical match, as in the experiments); all answers are written to the results CSV so they
can be checked by hand.

    python -m scripts.behaviour_test --model ollama:llama3 --configs C0 C1 C6
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.db import init_db, session_scope
from app.db.crud import save_result
from app.evaluation.judge import CORRECT, AnswerJudge
from app.pipeline.abstention import is_abstention
from app.pipeline.claims import rule_based_claims
from app.pipeline.evidence_check import SUPPORTED, evidence_check
from app.pipeline.generation import strip_citations
from app.pipeline.nli import get_nli
from app.pipeline.orchestrator import Pipeline
from app.pipeline.retrieval import DocumentInput, get_knowledge_base
from app.pipeline.settings import build_settings
from app.providers import get_provider
from app.utils import set_global_seed

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "behaviour"
COLLECTION = "behaviour"
DEFAULT_PAPER = ROOT.parent / "Publish Final" / "3-IJCSE-2026-7456.pdf"
PAPER_DATE = "2026-06-30"  # publication date printed on the paper

# An answer to an unanswerable question that explicitly says the thing does not exist / is unknown
# is a correct refusal even when it is not phrased as "I don't know".
_REFUSAL_RE = re.compile(
    r"\b(?:there (?:is|was|are|were) no\b|i (?:could not|couldn't|can't|cannot) find|(?:i'm|i am) not familiar|"
    r"no (?:record|information|evidence|such|known|reliable)|not (?:aware|a real|real|known|"
    r"exist|yet)|does(?: not|n't) exist|fictional|fictitious|hypothetical|has(?: not|n't) (?:taken place|happened|"
    r"been (?:played|held|decided))|(?:is|are) in the future|cannot (?:be )?(?:determine|know|predict)|"
    r"can't (?:determine|know|predict)|unable to (?:find|determine)|could you (?:clarify|specify|provide)|"
    r"(?:please )?(?:clarify|specify) (?:who|what|which)|more context|who (?:are you|you are) referring|"
    r"what (?:are you|you are) referring|\b(?:of|about|by|to) (?:what|whom|who)\s*\?)",
    re.IGNORECASE,
)

# For an *answerable* question, an honest statement of the model's own knowledge limit is a
# decline (counted as abstention), not a hallucination. Claims about the world ("it has not
# happened yet", "it does not exist") are not in this list: for a real event they are false.
_KNOWLEDGE_LIMIT_RE = re.compile(
    r"\b(?:i(?:'m| am) not aware|i (?:do not|don't) have (?:any )?(?:information|data|knowledge)|"
    r"knowledge cut-?off|i (?:could not|couldn't|cannot|can't) find|i(?:'m| am) not familiar|"
    r"i(?:'m| am) unable to (?:find|provide|verify))",
    re.IGNORECASE,
)


def build_kb(paper: Path | None) -> int:
    """(Re)build the behaviour collection from the curated documents and, if found, the paper."""
    kb = get_knowledge_base()
    kb.delete_collection(COLLECTION)
    spec = json.loads((DATA / "kb_documents.json").read_text(encoding="utf-8"))
    docs = [DocumentInput(d["title"], d["text"], d.get("source"), d.get("content_date")) for d in spec["documents"]]
    if paper is not None and paper.exists():
        from pypdf import PdfReader

        text = "\n".join(page.extract_text() or "" for page in PdfReader(str(paper)).pages)
        docs.append(DocumentInput(
            "AI Hallucinations in Large Language Model and Multimodal Models: A Systematic Review "
            "(IJCSE Vol.14 Issue 6, June 2026)", text, f"file:{paper.name}", PAPER_DATE,
        ))
    elif paper is not None:
        print(f"warning: paper not found at {paper}; question O4 has no evidence", file=sys.stderr)
    kb.add_documents(COLLECTION, docs)
    return kb.count(COLLECTION)


def split(cell: object) -> list[str]:
    return [x.strip() for x in str(cell or "").split(";") if x.strip() and str(cell) != "nan"]


def score_outcome(row: pd.Series, shown: str, abstained: bool, clarified: bool, judge: AnswerJudge) -> tuple[str, str]:
    """Outcome of the answer the user sees: correct / hallucinated / over_abstention (answerable
    question) or correct_abstention / hallucinated (unanswerable question)."""
    clean = strip_citations(shown)
    if row.expected == "answer":
        if abstained:
            return "over_abstention", "system abstained on an answerable question"
        dec = judge.judge(row.question, clean, split(row.correct_answers), split(row.incorrect_answers))
        if dec.label == CORRECT:
            return "correct", f"judge: {dec.reason}"
        if _KNOWLEDGE_LIMIT_RE.search(clean[:300]):
            return "over_abstention", "model declined (stated its own knowledge limit)"
        return "hallucinated", f"judge: {dec.reason}"
    if abstained or clarified:
        return "correct_abstention", "abstained or asked for clarification"
    if _REFUSAL_RE.search(clean):
        return "correct_abstention", "answer states the information does not exist / is unknown / asks for context"
    return "hallucinated", "gave a substantive answer to an unanswerable question"


def evaluate(row: pd.Series, result, judge: AnswerJudge, nli, kb) -> dict:  # type: ignore[no-untyped-def]
    """Score the answer the user would see."""
    shown = result.final_answer
    status = result.assessment.verification_status if result.assessment else None
    abstained = bool(result.abstained or is_abstention(shown))
    clarified = status == "clarification_requested"
    clean = strip_citations(shown)
    outcome, method = score_outcome(row, shown, abstained, clarified, judge)
    # evidence support of the shown answer, measured the same way for every configuration
    claims = [] if abstained else rule_based_claims(clean)
    unsupported = 0
    if claims:
        evidence = kb.search(COLLECTION, row.question, 4)
        res = evidence_check(claims, evidence, nli)
        unsupported = sum(1 for c in res.claims if c.label != SUPPORTED)
    retrieval_hit = None
    if result.passages and row.expected == "answer":
        refs = [r.lower() for r in split(row.correct_answers)]
        retrieval_hit = any(r in p.text.lower() for p in result.passages for r in refs)
    return {
        "outcome": outcome, "judge_method": method, "abstained": abstained, "n_claims": len(claims),
        "n_unsupported_claims": unsupported, "retrieval_hit": retrieval_hit, "verification_status": status,
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config, g in df.groupby("config", sort=False):
        ans = g[g.expected == "answer"]
        unans = g[g.expected != "answer"]
        rows.append({
            "config": config,
            "n": len(g),
            "accuracy_answerable": (ans.outcome == "correct").mean() if len(ans) else None,
            "hallucination_rate_all": (g.outcome == "hallucinated").mean(),
            "hallucination_rate_answerable": (ans.outcome == "hallucinated").mean() if len(ans) else None,
            "correct_abstention_rate_unanswerable": (unans.outcome == "correct_abstention").mean() if len(unans) else None,
            "over_abstention_rate_answerable": (ans.outcome == "over_abstention").mean() if len(ans) else None,
            "unsupported_claims_total": int(g.n_unsupported_claims.sum()),
            "claims_total": int(g.n_claims.sum()),
            "retrieval_success_answerable": ans.retrieval_hit.dropna().astype(bool).mean() if ans.retrieval_hit.notna().any() else None,
            "verified_by_evidence": int((g.verification_status == "verified_by_evidence").sum()),
            "latency_mean_s": g.latency_s.mean(),
            "llm_calls_mean": g.llm_calls.mean(),
        })
    return pd.DataFrame(rows)


def by_category(df: pd.DataFrame) -> pd.DataFrame:
    t = df.assign(ok=df.outcome.isin(["correct", "correct_abstention"]), halluc=df.outcome == "hallucinated")
    return (t.groupby(["category", "config"], sort=False)
            .agg(n=("ok", "size"), handled_correctly=("ok", "mean"), hallucinated=("halluc", "mean"),
                 abstained=("abstained", "mean"), latency_mean_s=("latency_s", "mean"))
            .reset_index())


def print_tables(df: pd.DataFrame, tag: str) -> None:
    summary, cats = summarize(df), by_category(df)
    summary.to_csv(DATA / f"summary_{tag}.csv", index=False)
    cats.to_csv(DATA / f"by_category_{tag}.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\nSUMMARY\n" + summary.round(3).to_string(index=False))
    print("\nBY CATEGORY\n" + cats.round(3).to_string(index=False))


def rescore(path: Path, questions_path: Path, seed: int) -> int:
    """Recompute outcomes of a saved run with the current scoring rules; answers are not regenerated."""
    df = pd.read_csv(path).fillna({"final_answer": "", "verification_status": ""})
    questions = pd.read_csv(questions_path, dtype=str).fillna("").set_index("id")
    judge = AnswerJudge(mode="nli", nli=get_nli(), llm=None, entail_threshold=0.5, seed=seed)
    changed = []
    for i, r in df.iterrows():
        q = questions.loc[r.id]
        row = pd.Series({**q.to_dict(), "id": r.id})
        outcome, method = score_outcome(row, r.final_answer, bool(r.abstained),
                                        r.verification_status == "clarification_requested", judge)
        if outcome != r.outcome:
            changed.append((r.id, r.config, r.outcome, outcome))
        df.at[i, "outcome"], df.at[i, "judge_method"] = outcome, method
    out = path.with_name(path.stem + "_rescored.csv")
    df.to_csv(out, index=False)
    print("changed outcomes:", changed or "none")
    print_tables(df, path.stem.replace("results_", "") + "_rescored")
    print(f"\nRescored answers: {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="ollama:llama3")
    parser.add_argument("--configs", nargs="+", default=["C0", "C1", "C6"])
    parser.add_argument("--questions", default=str(DATA / "questions.csv"))
    parser.add_argument("--paper", default=str(DEFAULT_PAPER), help="PDF added to the knowledge base ('' to skip)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", nargs="*", help="Run only these question ids")
    parser.add_argument("--rescore", help="Re-apply the scoring rules to a saved results CSV (no LLM calls)")
    args = parser.parse_args()

    if args.rescore:
        return rescore(Path(args.rescore), Path(args.questions), args.seed)

    init_db()
    set_global_seed(args.seed)
    provider = get_provider(args.model)
    provider.check_available()
    n_chunks = build_kb(Path(args.paper) if args.paper else None)
    print(f"knowledge base '{COLLECTION}': {n_chunks} chunks")
    questions = pd.read_csv(args.questions, dtype=str).fillna("")
    if args.only:
        questions = questions[questions.id.isin(args.only)]
    nli = get_nli()
    kb = get_knowledge_base()
    judge = AnswerJudge(mode="nli", nli=nli, llm=None, entail_threshold=0.5, seed=args.seed)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    source = f"behaviour:{stamp}"
    pipelines = {c: Pipeline(provider, build_settings(c, {"seed": args.seed}), kb=kb, nli=nli, collection=COLLECTION)
                 for c in args.configs}
    records = []
    started = time.perf_counter()
    for _, row in questions.iterrows():
        for config, pipeline in pipelines.items():
            result = pipeline.run(row.question)
            with session_scope() as session:
                run = save_result(session, result, row.id, None, False, source)
                session.flush()
                run_id = run.id
            ev = evaluate(row, result, judge, nli, kb)
            records.append({
                "id": row.id, "category": row.category, "expected": row.expected, "config": config,
                "question": row.question, "final_answer": result.final_answer,
                "candidate_answer": result.candidate_answer, "decision": result.decision.value,
                "decision_reasons": "; ".join(result.decision_reasons),
                "issues": json.dumps(result.assessment.issues if result.assessment else []),
                "remaining_uncertainty": "; ".join(result.assessment.remaining_uncertainty if result.assessment else []),
                "latency_s": result.latency_ms / 1000, "llm_calls": result.llm_calls,
                "n_regenerations": result.n_regenerations, "run_id": run_id, **ev,
            })
            print(f"[{time.perf_counter() - started:7.0f}s] {row.id} {config}: {ev['outcome']:<18} "
                  f"{ev['verification_status'] or '':<32} {result.final_answer[:70]!r}", flush=True)

    df = pd.DataFrame(records)
    tag = f"{args.model.replace(':', '_')}_{stamp}"
    out_csv = DATA / f"results_{tag}.csv"
    df.to_csv(out_csv, index=False)
    print_tables(df, tag)
    print(f"\nAll answers: {out_csv}\nRuns are stored in the database with source '{source}' (stage logs included).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
