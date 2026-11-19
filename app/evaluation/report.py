"""Turn stored runs into results.csv, summary tables, significance tests, charts and an HTML report.

All numbers come from the `runs` and `judgements` tables; nothing is hard-coded.
"""

from __future__ import annotations

import base64
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.config import get_settings
from app.db.models import Experiment, Run
from app.evaluation import charts
from app.evaluation.judge import ABSTAINED, CORRECT, HALLUCINATED
from app.evaluation.metrics import (
    auroc,
    bootstrap_ci,
    expected_calibration_error,
    mcnemar_test,
    mean_or_nan,
    paired_bootstrap_diff_ci,
    precision_recall_f1,
)
from app.pipeline.decision import Decision, is_flagged


def experiment_dir(experiment_id: int) -> Path:
    """Folder holding the artefacts of one experiment."""
    path = get_settings().experiments_dir / f"exp_{experiment_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dataframe(session: Session, experiment_id: int) -> pd.DataFrame:
    """One row per run, with judge decisions for the initial and the candidate answer."""
    runs = session.execute(
        select(Run).where(Run.experiment_id == experiment_id).options(selectinload(Run.judgements)).order_by(Run.id)
    ).scalars().all()
    rows: list[dict[str, Any]] = []
    for r in runs:
        j = {jd.target: jd for jd in r.judgements}
        init, cand = j.get("initial"), j.get("candidate")
        cand_label = (cand.manual_label or cand.label) if cand else None
        init_label = (init.manual_label or init.label) if init else None
        if r.error:
            final_label = None
        elif r.abstained:
            final_label = ABSTAINED
        else:
            final_label = cand_label
        rows.append({
            "run_id": r.id, "model": r.model, "config": r.config_name, "question_id": r.question_id,
            "question": r.question, "initial_answer": r.initial_answer, "candidate_answer": r.candidate_answer,
            "final_answer": r.final_answer, "first_decision": r.first_decision, "decision": r.decision,
            "decision_reason": r.decision_reason, "abstained": r.abstained, "escalated": r.escalated,
            "n_regenerations": r.n_regenerations, "confidence": r.confidence, "first_confidence": r.first_confidence,
            "semantic_entropy": r.semantic_entropy, "semantic_entropy_norm": r.semantic_entropy_norm,
            "selfcheck_score": r.selfcheck_score, "supported_fraction": r.supported_fraction,
            "contradicted_fraction": r.contradicted_fraction, "faithfulness": r.faithfulness,
            "latency_ms": r.latency_ms, "llm_calls": r.llm_calls, "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens, "total_tokens": r.prompt_tokens + r.completion_tokens,
            "initial_label": init_label, "candidate_label": cand_label, "final_label": final_label,
            "correct": final_label == CORRECT if final_label else None,
            "hallucinated": final_label == HALLUCINATED if final_label else None,
            "judge_mode": cand.mode if cand else None,
            "judge_reason": cand.reason if cand else None,
            "judge_uncertain": cand.uncertain if cand else None,
            "judges_disagree": cand.judges_disagree if cand else None,
            "nli_label": cand.nli_label if cand else None,
            "nli_correct_score": cand.nli_correct_score if cand else None,
            "nli_incorrect_score": cand.nli_incorrect_score if cand else None,
            "matched_reference": cand.nli_matched_reference if cand else None,
            "llm_judge_label": cand.llm_label if cand else None,
            "llm_judge_raw": cand.llm_raw if cand else None,
            "manual_label": cand.manual_label if cand else None,
            "initial_judge_reason": init.reason if init else None,
            "correct_references": " | ".join((cand.references or {}).get("correct", [])) if cand else None,
            "incorrect_references": " | ".join((cand.references or {}).get("incorrect", [])) if cand else None,
            "citations": json.dumps(r.citations or []),
            "error": r.error,
        })
    return pd.DataFrame(rows)


def _rate_with_ci(values: pd.Series, seed: int) -> tuple[float, float, float]:
    arr = values.astype(float).to_numpy()
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    lo, hi = bootstrap_ci(arr, seed=seed)
    return float(arr.mean()), lo, hi


def summarize(df: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the summary table and the significance table (each config vs C0, same model)."""
    summary_rows: list[dict[str, Any]] = []
    sig_rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    valid_all = df[df["error"].isna()]
    for (model, config), group in df.groupby(["model", "config"], sort=False):
        g = group[group["error"].isna()]
        n = len(g)
        answered = g[~g["abstained"]]
        hal, hal_lo, hal_hi = _rate_with_ci(g["final_label"] == HALLUCINATED, seed)
        acc, acc_lo, acc_hi = _rate_with_ci(g["final_label"] == CORRECT, seed)
        sel, sel_lo, sel_hi = _rate_with_ci(answered["final_label"] == CORRECT, seed)
        init_hal = (g["initial_label"] == HALLUCINATED).mean() if n else float("nan")

        cand = g[g["candidate_label"].isin([CORRECT, HALLUCINATED]) & g["confidence"].notna()]
        ece = expected_calibration_error(cand["confidence"].astype(float), (cand["candidate_label"] == CORRECT).astype(float))

        det = g[g["initial_label"].isin([CORRECT, HALLUCINATED])]
        y = (det["initial_label"] == HALLUCINATED).astype(float)
        au_se = auroc(det["semantic_entropy_norm"].astype(float), y)
        au_sc = auroc(det["selfcheck_score"].astype(float), y)
        au_conf = auroc(1.0 - det["first_confidence"].astype(float), y)
        with_dec = det[det["first_decision"].notna()]
        if len(with_dec):
            prf = precision_recall_f1(
                with_dec["first_decision"].map(lambda d: is_flagged(Decision(d))).to_numpy(),
                (with_dec["initial_label"] == HALLUCINATED).to_numpy(),
            )
        else:
            prf = {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}

        summary_rows.append({
            "model": model, "config": config, "n": n, "n_errors": int(group["error"].notna().sum()),
            "hallucination_rate": hal, "hallucination_rate_ci_low": hal_lo, "hallucination_rate_ci_high": hal_hi,
            "accuracy": acc, "accuracy_ci_low": acc_lo, "accuracy_ci_high": acc_hi,
            "abstention_rate": float(g["abstained"].mean()) if n else float("nan"),
            "coverage": float(1 - g["abstained"].mean()) if n else float("nan"),
            "selective_accuracy": sel, "selective_accuracy_ci_low": sel_lo, "selective_accuracy_ci_high": sel_hi,
            "hallucination_rate_answered": float((answered["final_label"] == HALLUCINATED).mean()) if len(answered) else float("nan"),
            "escalation_rate": float(g["escalated"].mean()) if n else float("nan"),
            "initial_hallucination_rate": float(init_hal),
            "ece": ece, "n_with_confidence": len(cand),
            "auroc_semantic_entropy": au_se, "auroc_selfcheck": au_sc, "auroc_confidence": au_conf,
            "detector_precision": prf["precision"], "detector_recall": prf["recall"], "detector_f1": prf["f1"],
            "faithfulness": mean_or_nan(g["faithfulness"].astype(float)),
            "latency_mean_s": float(g["latency_ms"].mean() / 1000) if n else float("nan"),
            "latency_median_s": float(g["latency_ms"].median() / 1000) if n else float("nan"),
            "llm_calls_mean": float(g["llm_calls"].mean()) if n else float("nan"),
            "tokens_mean": float(g["total_tokens"].mean()) if n else float("nan"),
            "judge_uncertain_rate": float(g["judge_uncertain"].fillna(False).astype(bool).mean()) if n else float("nan"),
            "judges_disagree_rate": float(g["judges_disagree"].fillna(False).astype(bool).mean()) if n else float("nan"),
        })

        if config == "C0":
            continue
        base = valid_all[(valid_all.model == model) & (valid_all.config == "C0")]
        if base.empty:
            continue
        paired = base.merge(g, on="question_id", suffixes=("_base", "_cfg"))
        if paired.empty:
            continue
        hal_b = (paired["final_label_base"] == HALLUCINATED).to_numpy()
        hal_c = (paired["final_label_cfg"] == HALLUCINATED).to_numpy()
        cor_b = (paired["final_label_base"] == CORRECT).to_numpy()
        cor_c = (paired["final_label_cfg"] == CORRECT).to_numpy()
        mc_h = mcnemar_test(hal_b, hal_c)
        mc_a = mcnemar_test(cor_b, cor_c)
        d_h, d_h_lo, d_h_hi = paired_bootstrap_diff_ci(hal_b.astype(float), hal_c.astype(float), seed=seed)
        d_a, d_a_lo, d_a_hi = paired_bootstrap_diff_ci(cor_b.astype(float), cor_c.astype(float), seed=seed)
        lat_b = paired["latency_ms_base"].mean()
        sig_rows.append({
            "model": model, "config": config, "n_paired": len(paired),
            "hallucination_diff": d_h, "hallucination_diff_ci_low": d_h_lo, "hallucination_diff_ci_high": d_h_hi,
            "mcnemar_hallucination_n01": mc_h["n01"], "mcnemar_hallucination_n10": mc_h["n10"],
            "mcnemar_hallucination_p": mc_h["p_value"], "mcnemar_hallucination_method": mc_h["method"],
            "accuracy_diff": d_a, "accuracy_diff_ci_low": d_a_lo, "accuracy_diff_ci_high": d_a_hi,
            "mcnemar_accuracy_p": mc_a["p_value"], "mcnemar_accuracy_method": mc_a["method"],
            "latency_ratio_vs_c0": float(paired["latency_ms_cfg"].mean() / lat_b) if lat_b else float("nan"),
            "llm_calls_ratio_vs_c0": float(paired["llm_calls_cfg"].mean() / max(paired["llm_calls_base"].mean(), 1e-9)),
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(sig_rows)


def hypothesis_statements(sig: pd.DataFrame, alpha: float = 0.05) -> list[str]:
    """Plain-language reading of the full-pipeline (C5, C6) vs C0 tests, generated only from measured numbers."""
    out: list[str] = []
    if sig.empty or "config" not in sig:
        return ["No configuration could be compared with the C0 baseline (C0 missing or no paired questions)."]
    full = sig[sig.config.isin(["C5", "C6"])]
    if full.empty:
        return ["Neither C5 nor C6 was run, so the main hypothesis (full pipeline vs baseline) was not tested."]
    for _, r in full.iterrows():
        direction = "fewer" if r.hallucination_diff < 0 else ("more" if r.hallucination_diff > 0 else "the same number of")
        signif = r.mcnemar_hallucination_p < alpha
        verdict = (
            "supports the hypothesis" if (signif and r.hallucination_diff < 0)
            else "contradicts the hypothesis" if (signif and r.hallucination_diff > 0)
            else "does not give significant evidence either way"
        )
        out.append(
            f"{r.model}: {r.config} produced {direction} hallucinations than C0 "
            f"(difference {r.hallucination_diff:+.3f}, 95% CI [{r.hallucination_diff_ci_low:+.3f}, "
            f"{r.hallucination_diff_ci_high:+.3f}], McNemar p={r.mcnemar_hallucination_p:.4f}, n={r.n_paired}) at "
            f"{r.latency_ratio_vs_c0:.1f}x the latency and {r.llm_calls_ratio_vs_c0:.1f}x the LLM calls. "
            f"At alpha={alpha} this {verdict}."
        )
    return out


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return "n/a" if math.isnan(value) else f"{value:.3f}"
    return html.escape(str(value))


def _table(df: pd.DataFrame, columns: list[str]) -> str:
    cols = [c for c in columns if c in df.columns]
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{_fmt(row[c])}</td>" for c in cols) + "</tr>" for _, row in df.iterrows())
    return f"<div class='tw'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _img(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f"<img alt='{html.escape(path.stem)}' src='data:image/png;base64,{data}'/>"


JUDGING_TEXT = (
    "Each answer is labelled correct, hallucinated or abstained. Abstentions are detected with fixed "
    "patterns (e.g. \"I don't know\"). With the NLI judge, a short reference (at most 6 words) found "
    "inside the normalised answer counts as a match; otherwise the DeBERTa-MNLI model tests whether "
    "'question + answer' entails 'question + reference'. The answer is correct if it matches a correct "
    "reference better than any incorrect one, and hallucinated otherwise. Answers matching no reference "
    "are counted as hallucinated and flagged 'judge_uncertain'. With the LLM judge, a judge model compares "
    "the answer with the references; in 'both' mode the LLM verdict is used and disagreements with NLI are "
    "flagged. Every decision is in results.csv (judge_reason, nli_* and llm_judge_* columns)."
)

LIMITATIONS_TEXT = [
    "Small samples give wide confidence intervals; check the CI columns before drawing conclusions.",
    "The automatic judge makes errors. Audit rows with judge_uncertain or judges_disagree set.",
    "Thresholds are a-priori defaults unless tuned on a separate split; tuning on the test questions inflates results.",
    "Results depend on the exact model versions (Ollama tags / API snapshots) recorded in config.json.",
    "Hallucination rate is a share of all questions, so abstaining lowers it; read it together with coverage and accuracy.",
]


def build_report(session: Session, experiment_id: int) -> dict[str, Any]:
    """Write results.csv, summary.csv, significance.csv, PNG charts and report.html. Returns the summary."""
    exp = session.get(Experiment, experiment_id)
    if exp is None:
        raise LookupError(f"Experiment {experiment_id} not found")
    out = experiment_dir(experiment_id)
    df = results_dataframe(session, experiment_id)
    df.to_csv(out / "results.csv", index=False)
    seed = int(exp.config.get("seed", 0))
    summary, sig = summarize(df, seed=seed)
    summary.to_csv(out / "summary.csv", index=False)
    sig.to_csv(out / "significance.csv", index=False)

    chart_paths: list[Path] = []
    if not summary.empty:
        chart_paths.append(charts.hallucination_rate_chart(summary, out / "hallucination_rate.png"))
        chart_paths.append(charts.accuracy_coverage_chart(df[df["error"].isna()], summary, out / "accuracy_vs_coverage.png"))
        if "C0" in set(summary["config"]):
            chart_paths.append(charts.gain_latency_chart(summary, out / "gain_vs_latency.png"))
        chart_paths.append(charts.roc_chart(df[df["error"].isna()], out / "roc_curves.png"))

    statements = hypothesis_statements(sig)
    audit = df[(df["judge_uncertain"] == True) | (df["judges_disagree"] == True)].head(50)  # noqa: E712
    main_cols = ["model", "config", "n", "n_errors", "hallucination_rate", "hallucination_rate_ci_low",
                 "hallucination_rate_ci_high", "accuracy", "abstention_rate", "selective_accuracy",
                 "escalation_rate", "ece", "faithfulness", "latency_mean_s", "llm_calls_mean", "tokens_mean"]
    det_cols = ["model", "config", "initial_hallucination_rate", "auroc_semantic_entropy", "auroc_selfcheck",
                "auroc_confidence", "detector_precision", "detector_recall", "detector_f1",
                "judge_uncertain_rate", "judges_disagree_rate"]
    sig_cols = ["model", "config", "n_paired", "hallucination_diff", "hallucination_diff_ci_low",
                "hallucination_diff_ci_high", "mcnemar_hallucination_p", "accuracy_diff", "mcnemar_accuracy_p",
                "latency_ratio_vs_c0", "llm_calls_ratio_vs_c0"]
    audit_cols = ["run_id", "model", "config", "question", "candidate_answer", "correct_references",
                  "nli_label", "llm_judge_label", "judge_reason"]
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HalluGuard Experiment {experiment_id}</title>
<style>
:root{{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--line:#2c2c2a}}}}
body{{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}}
main{{max-width:1200px;margin:0 auto}} h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:18px;margin:32px 0 8px}}
p,li{{color:var(--ink2)}} .tw{{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:8px}}
table{{border-collapse:collapse;font-size:12px;width:100%}} th,td{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
th{{color:var(--muted);font-weight:600;position:sticky;top:0;background:var(--surface)}} td{{font-variant-numeric:tabular-nums}}
img{{max-width:100%;background:#fcfcfb;border-radius:8px;border:1px solid var(--line);margin:8px 0}}
pre{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:12px;overflow-x:auto;font-size:12px}}
.hyp{{background:var(--surface);border-left:3px solid #2a78d6;padding:8px 12px;border-radius:4px}}
</style></head><body><main>
<h1>Experiment {experiment_id}: {html.escape(exp.name)}</h1>
<p>Status: {html.escape(exp.status)}. Dataset: {html.escape(str(exp.config.get('dataset')))}, sample size
{exp.config.get('sample_size')}, seed {exp.config.get('seed')}. Files: results.csv, summary.csv, significance.csv,
config.json in <code>{html.escape(str(out))}</code>.</p>
<h2>Hypothesis check (full pipeline vs C0)</h2>
{''.join(f"<p class='hyp'>{html.escape(s)}</p>" for s in statements)}
<h2>Summary</h2>{_table(summary, main_cols) if not summary.empty else '<p>No runs.</p>'}
<h2>Detection quality (initial answers)</h2>{_table(summary, det_cols) if not summary.empty else ''}
<h2>Paired comparison with the baseline</h2>{_table(sig, sig_cols) if not sig.empty else '<p>No paired comparison available.</p>'}
<h2>Charts</h2>{''.join(_img(p) for p in chart_paths)}
<h2>How correctness is judged</h2><p>{html.escape(JUDGING_TEXT)}</p>
<h2>Judge audit sample (uncertain or disagreeing decisions, up to 50)</h2>
{_table(audit, audit_cols) if not audit.empty else '<p>No uncertain or disagreeing judge decisions.</p>'}
<h2>Limitations</h2><ul>{''.join(f'<li>{html.escape(t)}</li>' for t in LIMITATIONS_TEXT)}</ul>
<h2>Configuration</h2><pre>{html.escape(json.dumps(exp.config, indent=2))}</pre>
</main></body></html>"""
    (out / "report.html").write_text(page, encoding="utf-8")

    def clean(records: list[dict]) -> list[dict]:
        return [{k: (None if isinstance(v, float) and math.isnan(v) else (v.item() if isinstance(v, np.generic) else v))
                 for k, v in rec.items()} for rec in records]

    return {
        "summary": clean(summary.to_dict(orient="records")),
        "significance": clean(sig.to_dict(orient="records")),
        "hypothesis": statements,
        "artifacts_dir": str(out),
    }
