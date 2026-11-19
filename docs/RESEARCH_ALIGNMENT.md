# HalluGuard: alignment with the review paper

Reference: M. Singh and S. Banerjee, "AI Hallucinations in Large Language Model and Multimodal
Models: A Systematic Review of Causes, Types, Detection, Mitigation, and Future Directions",
*IJCSE* 14(6), 26–33, June 2026 (Algorithm 1, Tables 1–3, Section 5).

Research question: when a model is asked about something it did not learn (missing, long-tail,
post-cutoff, poorly documented or ambiguous), can a system *around* the model detect the gap,
avoid fabricating, retrieve evidence, verify the answer and abstain when needed, without
retraining the model?

Status labels used below: **Implemented** (in code, unit-tested, exercised in the behaviour
test), **Partial** (implemented with a stated restriction), **Not implemented** (outside the
system's scope, and not claimed).

---

## 1. Review of the project as it was before this change (C0–C5)

| # | Question | Before | Notes |
|---|---|---|---|
| 1 | Detects insufficient knowledge? | Partial | Semantic entropy (Farquhar et al.) and SelfCheck-NLI measure uncertainty; the RAG prompt asks for "I don't know" without evidence. There was no check that retrieved passages were relevant, so an unrelated passage still counted as "evidence". |
| 2 | Identifies unsupported answers? | Implemented | NLI evidence check per claim (supported / contradicted / not enough evidence). |
| 3 | Retrieval (RAG)? | Partial | Local FAISS knowledge base only. No relevance, reliability or freshness validation; no external source. |
| 4 | Verifies against evidence? | Implemented | Claim-level NLI against every retrieved passage, plus factored Chain-of-Verification. |
| 5 | Prevents fabricated facts and citations? | Partial | Facts: yes (evidence check, CoVe, abstention). Citations: `[S#]` labels were extracted and marked valid/invalid, but invalid labels and invented URLs/DOIs did not affect the decision. |
| 6 | Uncertainty and appropriate response? | Implemented | Abstention, clarification for ambiguous questions, human review after R failed regenerations. The abstention text was "I don't know." |
| 7 | Local models through Ollama? | Implemented | Ollama provider plus an OpenAI-compatible provider. |
| 8 | Works without retraining? | Implemented | Every stage is black-box (prompts, sampling, NLI, retrieval). |
| 9 | Layered workflow of Algorithm 1? | Partial | Retrieve → generate → detect → regenerate ≤ R → escalate was implemented. Re-retrieval re-used the question with a larger k instead of `Re-retrieve(q, r)`. Detection was factual only (no citation, logical, mathematical or code checks). |

Problems found:

* **"VERIFIED" overstated what was checked.** In C2/C3 an answer passed on consistency signals
  alone and was shown as VERIFIED without any evidence check. In C5 an answer with only 50% of
  its claims supported could also be VERIFIED.
* **Retrieved text was trusted.** A passage containing instructions ("ignore previous
  instructions…") reached the model unchanged.
* **No freshness signal.** Nothing recorded how old the evidence was, so a post-cutoff question
  answered from an out-of-date document looked as reliable as any other.
* Two bugs fixed earlier in this project (before this review): NLI "contradiction" scores from
  off-topic passages outweighed a supporting passage, and "According to [S1]," was not removed
  before NLI.

## 2. What was added (configuration C6) and what is still missing

C0–C5 were left unchanged so the recorded experiments stay comparable and resumable (the
runner now refuses to resume an experiment whose settings differ). Everything new is in **C6**.

| Workflow step (your brief) | Paper | Status | Where |
|---|---|---|---|
| 1. Receive question: topic, context, needs | Table 1 (inference) | Implemented | `query_analysis.py`: normalisation, ambiguity, high-risk domains, and a new **time-sensitivity** flag |
| 2. Assess knowledge / uncertainty from practical signals | §5.1.3 | Implemented | semantic entropy, SelfCheck-NLI, evidence support, type checks. The system does **not** ask the model whether a topic was in its training data. |
| 3. Retrieve; validate relevance, reliability, freshness | §5.1.5 RAG | Implemented (local KB); Partial (web) | `evidence_quality.py`: relevance threshold, injection quarantine, staleness report; `web_search.py`: Wikipedia, off by default, needs `HG_WEB_USER_AGENT` |
| 4. Grounded response | Alg. 1 line 2 | Implemented | cited `[S#]` answers; guarded prompt treats passages as data |
| 5. Detect: semantic entropy, self-consistency, evidence, citation, logical, math, code | §5.1.3, Table 3 | Implemented | `type_checks.py` (new): citation/reference validation, logical self-consistency (bidirectional NLI), arithmetic recomputation, static Python checks (never executed) |
| 6. Verify and correct; re-retrieve; bounded retry | Alg. 1 lines 4–9, CoV | Implemented | CoVe, then regeneration with **claim-guided re-retrieval**, at most `max_regenerations` (default 2) |
| 7. Calibrated abstention | §5.2.1, Future Scope | Implemented | final text: "I do not have enough verified information to answer this question reliably." |
| 8. Return answer, sources, status, remaining uncertainty | §5.2.2 | Implemented | `assessment.py`: `verification_status`, typed issues, remaining uncertainty, sources with relevance and date, in the API and UI |
| Human oversight | Alg. 1 line 9 | Implemented | review queue for NEEDS_REVIEW answers |
| Security: retrieved text cannot override instructions | — | Partial | pattern-based quarantine plus guarded prompt. The system has no tools that act, so retrieved text cannot trigger actions. Novel injections can still get through. |
| Logs of retrieval, detection, verification, correction, abstention | — | Implemented | one StageLog row per stage, per run (SQLite) |

Taxonomy coverage (paper Fig. 2):

| Type | Detection in HalluGuard |
|---|---|
| Factual | NLI evidence check, semantic entropy, SelfCheck, CoVe |
| Logical | bidirectional NLI between the answer's own claims (detects self-contradiction only, not invalid multi-step inference) |
| Citation | `[S#]` labels and URLs/DOIs/arXiv ids checked against the retrieved sources (not against bibliographic databases) |
| Mathematical | explicit arithmetic recomputed with a safe evaluator (no symbolic algebra or word problems) |
| Code | static analysis of Python blocks: syntax, missing modules, invented stdlib functions (no execution, no other languages) |
| Visual/object, multimodal, agentic | **Not implemented.** The system answers text questions only; it has no image input and no tool-using agent. |

Benchmarks (paper Table 2): **TruthfulQA** and **HaluEval** loaders exist and HaluEval has been
run. FActScore, FEVER and RAGTruth are not integrated. A `custom_csv` loader accepts any dataset
with question / correct / incorrect / context columns, which is how FEVER-style claim data could
be added.

Still missing (proposed, not done):

1. FEVER / RAGTruth / FActScore loaders and an atomic-fact (FActScore-style) precision metric.
2. Bibliographic lookup (Crossref/DOI resolution) so citations are checked against real records,
   not only against the retrieved passages.
3. Threshold calibration on a held-out split (all thresholds are a-priori).
4. Adaptive retrieval (retrieve only when closed-book uncertainty is high); C6 always retrieves,
   which is what Algorithm 1 specifies.
5. An LLM- or classifier-based injection detector in addition to patterns.

## 3. Behaviour test: Scenario A (no mitigation) vs Scenario B (full workflow)

Setup: `ollama:llama3` (8B, knowledge cut-off before all the "outdated" events), seed 42,
temperature 0 for answers, identical prompts and decoding across configurations.
**Scenario A = C0** (question → model → answer). **Scenario B = C6** (analysis → validated
retrieval → grounded answer → detection → verification/regeneration → assessment).
**C1** (RAG only) is included to separate the effect of retrieval from everything after it.

Questions: `data/behaviour/questions.csv` (19: 4 known, 4 unknown, 5 outdated, 3 ambiguous,
3 misleading) and `data/behaviour/questions_heldout.csv` (10, written *before* C6 v2 was run).
Knowledge base: 11 short curated test documents (`data/behaviour/kb_documents.json`, written for
this test, including one outdated document and one prompt-injection attack) and the full text of
the review paper (54 chunks in total).

Scoring (automatic, then every answer checked by hand): answerable questions are *correct*,
*hallucinated* (wrong or false claim, including "this event has not happened yet" for a past
event) or *over-abstention* (system or model declined); unanswerable questions (unknown,
ambiguous) are *correct abstention* (abstained, asked for clarification, or said the thing does
not exist) or *hallucinated* (gave an answer). Manual review changed 3 automatic C0 verdicts in the
first run (A2 "The height of what?" is a clarification request; O4 "I'm not aware…" is a decline;
M1 corrects the premise). The rescoring is reproducible with `--rescore`; the reference fix for M1
is in the CSV.

### 3.1 Original 19 questions

| Metric | C0 (A) | C1 (RAG) | C6 v1 | C6 v2* |
|---|---|---|---|---|
| Accuracy on answerable (n = 12) | 0.58 (7) | 0.75 (9) | 0.42 (5) | 0.75 (9) |
| Hallucination rate, all questions | 0.21 (4/19) | 0.05 (1/19) | **0.00** | **0.00** |
| Hallucination rate, answerable | 0.33 (4/12) | 0.00 | 0.00 | 0.00 |
| Correct abstention, unanswerable (n = 7) | 1.00 | 0.86 | 1.00 | 1.00 |
| Over-abstention, answerable | 0.08 | 0.25 | 0.58 | 0.25 |
| Claims of shown answers not supported by the KB | 37 of 45 | 1 of 11 | 0 of 5 | 1 of 11 |
| Answers marked `verified_by_evidence` | – | – | 5 | 8 |
| Retrieval contained the right answer (answerable) | – | 12/12 | 12/12 | 12/12 |
| Mean latency per question | 9.1 s | 7.0 s | 24.5 s | 32.6 s |
| Mean LLM calls | 1.0 | 1.0 | 3.6 | 5.6 |

\* C6 v2 was changed after inspecting v1 on these same questions (see 3.3), so its numbers on this
set are optimistic. Use the held-out set for C6 v2.

By category (handled correctly):

| Category | C0 | C1 | C6 v1 | C6 v2* |
|---|---|---|---|---|
| known (4) | 4 | 4 | 4 | 4 |
| unknown (4) | 4 | 4 | 4 | 4 |
| outdated (5) | **0** | 5 | 1 | 3 |
| ambiguous (3) | 3 | 2 | 3 | 3 |
| misleading (3) | 3 | 0 | 0 | 2 |

### 3.2 Held-out 10 questions (C6 v2, not used for any change)

| Metric | C0 (A) | C1 (RAG) | C6 v2 (B) |
|---|---|---|---|
| Accuracy on answerable (n = 7) | 0.57 (4) | 0.57 (4) | **0.71 (5)** |
| Hallucination rate, all questions | 0.30 (3/10) | 0.00 | **0.00** |
| Correct abstention, unanswerable (n = 3) | 1.00 | 1.00 | 1.00 |
| Over-abstention, answerable | 0.00 | 0.43 | 0.29 |
| Claims not supported by the KB | 26 of 28 | 0 of 4 | 1 of 6 |
| `verified_by_evidence` / `partially_supported` | – | – | 4 / 1 |
| Mean latency | 9.3 s | 5.0 s | 21.4 s (2.3× C0) |
| Mean LLM calls | 1.0 | 1.0 | 3.8 |

All three C0 hallucinations are the post-cut-off failure the research question is about: the
model states that Euro 2024 / Paris 2024 "have not yet taken place" (H5, H6, H10).

### 3.3 What went wrong and what was changed

* **Semantic-entropy false alarms (v1).** On O2 all five samples said "Spain", but strict
  bidirectional entailment put each in its own cluster (normalised entropy 1.0), because the
  samples added different details. C6 now uses the existing `loose` equivalence (no
  contradiction in either direction and entailment in one). C0–C5 still use `strict`.
* **False-premise questions (v1).** The grounding prompt says to answer "I don't know" when the
  passages lack the information, and llama3 applied it to "Why did Einstein get the Nobel Prize
  for relativity?". C6's prompt now tells the model to correct a premise the passages contradict.
  This fixed 2 of 3 on the original set and 1 of 3 on the held-out set; the model still answers
  "I don't know" on the others, so the instruction is followed only part of the time.
* **Bug found by the test.** The time-sensitivity regex was written with literal backspace
  characters instead of `\b`, so the freshness check never fired in v1. Fixed, with a test. A
  later fix (after all runs; it changes only the uncertainty notes, not decisions) stopped a year
  such as "2024" from marking a question as time-sensitive.
* **Remaining over-abstention.** O2 v2 was escalated to human review after SelfCheck stayed at
  0.53 (≥ 0.5) through two regenerations, although the answer was right. O5 ("current European
  champion"; one 2021 and one 2024 document) ended in "I don't know", and the stale-evidence note
  was shown. The injection document was quarantined on every Australia question (K1, M2).

## 4. Final assessment

**Does the project answer the research question?** Partly, and measurably.

* *Detect insufficient knowledge / prevent fabrication:* on these 29 questions, C6 produced no
  hallucinated answer, against 7 for C0. Questions about fictional entities, the future, and
  ambiguous references were all declined.
* *Retrieve and use external knowledge:* retrieval is what fixed the post-cut-off failures
  (C0 0 of 7 outdated questions correct; C1 7 of 7; C6 v2 5 of 7).
* *Verify:* C6 releases an answer as `verified_by_evidence` only when every claim is entailed by a
  passage. Two premise-correcting answers were released as `partially_supported` because the
  "not Sydney / not Edison" clause is not stated in the passages; the status says so.
* *Cost:* C6 v2 takes 2.3–3.6× C0's latency and 3.8–5.6× the LLM calls. RAG alone (C1) was as
  fast as C0, and on these questions it hallucinated once (an ambiguous question answered from an
  unrelated passage of the paper).
* *Trade-off:* the paper's helpfulness-vs-honesty trade-off appears directly. C6 trades
  hallucinations for abstentions (2 of 7 answerable held-out questions declined, all of them
  false-premise questions).

**Limitations of this evidence.** 29 questions, one 8B model, one run each, an automatic judge
checked by one reviewer, and a small curated knowledge base in which retrieval always found the
right passage. No significance test is meaningful at this size; the differences are descriptive.
Real deployments have noisier retrieval, where RAG itself can introduce errors. The 100-question
HaluEval experiment (exp 3) is the larger-sample test of C0–C5 and is 141 of 1,200 runs complete;
C6 has not yet been run on HaluEval.

**Not claimed.** Hallucination is not eliminated: the system reduced it on these questions and
made the remaining uncertainty visible. Visual, multimodal and agentic hallucinations are not
addressed.

### Reproduce

```bash
python -m scripts.behaviour_test --model ollama:llama3 --configs C0 C1 C6
python -m scripts.behaviour_test --model ollama:llama3 --configs C0 C1 C6 --questions data/behaviour/questions_heldout.csv
python -m scripts.behaviour_test --rescore data/behaviour/results_<run>.csv
```

Result files used above (all in `data/behaviour/`):
`results_ollama_llama3_20261119T125857Z_rescored.csv` (C0, C1, C6 v1),
`results_ollama_llama3_20261119T131714Z.csv` (C6 v2, original set),
`results_ollama_llama3_20261119T132750Z.csv` (held-out). Every run, with all stage logs, is in the
database under source `behaviour:<timestamp>`.
