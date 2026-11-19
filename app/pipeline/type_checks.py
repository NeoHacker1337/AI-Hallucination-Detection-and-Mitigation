"""Type-specific hallucination checks (paper, Table 3): citation, logical, mathematical, code.

Factual hallucinations are handled by the NLI evidence check (evidence_check.py). The checks here
cover the types whose detection the paper rates as easy (citation, mathematical, code: checked
deterministically) or hard (logical: approximated with NLI between the answer's own claims).
Visual/object, multimodal and agentic hallucinations are out of scope: this system handles text
question answering only and has no image inputs or tool-using agents.

All checks are deterministic and never execute model-generated code.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import operator
import re
import sys
from dataclasses import asdict, dataclass, field

from app.pipeline.generation import strip_citations
from app.pipeline.nli import CONTRADICT, NLIScorer
from app.pipeline.retrieval import Passage

FACTUAL, LOGICAL, CITATION, MATHEMATICAL, CODE = "factual", "logical", "citation", "mathematical", "code"
NOT_SUPPORTED_TYPES = ("visual/object", "multimodal", "agentic")


@dataclass
class Issue:
    """One detected problem. `hard` issues fail the answer; soft ones are reported as uncertainty."""

    type: str
    detail: str
    hard: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TypeCheckResult:
    """All issues found plus what was checked."""

    issues: list[Issue] = field(default_factory=list)
    checked: dict[str, int] = field(default_factory=dict)

    @property
    def hard_issues(self) -> list[Issue]:
        return [i for i in self.issues if i.hard]

    def to_dict(self) -> dict:
        return {"issues": [i.to_dict() for i in self.issues], "checked": self.checked,
                "not_supported_types": list(NOT_SUPPORTED_TYPES)}


# ------------------------------------------------------------------------------------------ citation
_CITE_RE = re.compile(r"\[S(\d+)\]")
_URL_RE = re.compile(r"https?://[^\s)\]>\"']+", re.IGNORECASE)
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>]+", re.IGNORECASE)
_ARXIV_RE = re.compile(r"\barxiv[:\s]*(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)


def _norm_ref(ref: str) -> str:
    return ref.lower().rstrip(".,;:)").replace("https://", "").replace("http://", "").replace("www.", "")


def check_citations(answer: str, passages: list[Passage] | None) -> tuple[list[Issue], int]:
    """Citation-type checks.

    * [S#] labels that point to no retrieved passage are fabricated citations (hard).
    * URLs, DOIs and arXiv ids in the answer that appear in no passage text or source are
      unverifiable references (hard when evidence was provided, since the model was told to use
      only the passages; soft without retrieval, where they cannot be checked).
    * A grounded answer that cites nothing is reported as a soft issue.
    """
    issues: list[Issue] = []
    n_passages = len(passages or [])
    labels = {int(m) for m in _CITE_RE.findall(answer or "")}
    for n in sorted(labels):
        if not 1 <= n <= n_passages:
            issues.append(Issue(CITATION, f"citation [S{n}] refers to no retrieved passage (only {n_passages} given)"))
    haystack = " ".join(_norm_ref(f"{p.text} {p.source or ''} {p.source_id}") for p in passages or [])
    refs = _URL_RE.findall(answer or "") + _DOI_RE.findall(answer or "") + [f"arxiv:{a}" for a in _ARXIV_RE.findall(answer or "")]
    for ref in dict.fromkeys(refs):
        key = _norm_ref(ref).replace("arxiv:", "")
        if key not in haystack:
            issues.append(Issue(CITATION, f"reference '{ref}' does not appear in any retrieved source",
                                hard=passages is not None))
    if passages is not None and passages and not labels and answer and len(answer.split()) > 3:
        issues.append(Issue(CITATION, "grounded answer cites no passage", hard=False))
    return issues, len(labels) + len(refs)


# ------------------------------------------------------------------------------------------ logical
def check_logical_consistency(claims: list[str], nli: NLIScorer, threshold: float) -> tuple[list[Issue], int]:
    """Flag pairs of claims from the same answer that contradict each other (NLI in both
    directions must reach `threshold`, which keeps false alarms from one-sided NLI errors low)."""
    pairs = [(i, j) for i in range(len(claims)) for j in range(i + 1, len(claims))]
    if not pairs:
        return [], 0
    batch = [(claims[i], claims[j]) for i, j in pairs] + [(claims[j], claims[i]) for i, j in pairs]
    probs = nli.predict(batch)
    issues = []
    for n, (i, j) in enumerate(pairs):
        forward, backward = float(probs[n, CONTRADICT]), float(probs[n + len(pairs), CONTRADICT])
        if min(forward, backward) >= threshold:
            issues.append(Issue(LOGICAL, f"claims contradict each other: '{claims[i]}' vs '{claims[j]}' "
                                         f"(P(contradict) {forward:.2f}/{backward:.2f})"))
    return issues, len(pairs)


# ------------------------------------------------------------------------------------------ math
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}
_NUM = r"-?\d+(?:,\d{3})*(?:\.\d+)?"
_EQ_RE = re.compile(
    rf"(?<![\w.])((?:{_NUM}|\()\s*(?:[-+*/×x÷^%]\s*\(?\s*{_NUM}\s*\)?\s*)+\)?)\s*=\s*({_NUM})(?!\w|\.\d)"
)
_WORD_OPS = [
    (re.compile(rf"({_NUM})\s+(?:plus|added to)\s+({_NUM})\s+(?:is|equals|=)\s+({_NUM})", re.I), operator.add),
    (re.compile(rf"({_NUM})\s+minus\s+({_NUM})\s+(?:is|equals|=)\s+({_NUM})", re.I), operator.sub),
    (re.compile(rf"({_NUM})\s+(?:times|multiplied by)\s+({_NUM})\s+(?:is|equals|=)\s+({_NUM})", re.I), operator.mul),
    (re.compile(rf"({_NUM})\s+divided by\s+({_NUM})\s+(?:is|equals|=)\s+({_NUM})", re.I), operator.truediv),
    (re.compile(rf"({_NUM})\s*%\s+of\s+({_NUM})\s+(?:is|equals|=)\s+({_NUM})", re.I), lambda a, b: a / 100 * b),
]


def _num(text: str) -> float:
    return float(text.replace(",", ""))


def safe_eval(expr: str) -> float:
    """Evaluate an arithmetic expression of numbers and + - * / ** % // only (no names, no calls)."""
    expr = expr.replace(",", "").replace("×", "*").replace("÷", "/").replace("^", "**")
    expr = re.sub(r"(?<=\d)\s*x\s*(?=[\d(])", "*", expr)

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -ev(node.operand) if isinstance(node.op, ast.USub) else ev(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](left, right)
        raise ValueError("unsupported expression")

    return ev(ast.parse(expr, mode="eval"))


def _matches(computed: float, stated_text: str) -> bool:
    stated = _num(stated_text)
    decimals = len(stated_text.split(".")[1]) if "." in stated_text else 0
    tolerance = max(0.5 * 10 ** (-decimals), abs(computed) * 1e-9)  # the stated value may be rounded
    return math.isclose(computed, stated, abs_tol=tolerance)


def check_math(answer: str) -> tuple[list[Issue], int]:
    """Recompute every explicit arithmetic statement ("12 * 7 = 84", "15% of 80 is 12")."""
    text = strip_citations(answer or "")
    issues: list[Issue] = []
    checked = 0
    for m in _EQ_RE.finditer(text):
        try:
            value = safe_eval(m.group(1))
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
            continue
        checked += 1
        if not _matches(value, m.group(2)):
            issues.append(Issue(MATHEMATICAL, f"'{m.group(0).strip()}' is wrong: the expression equals {value:g}"))
    for pattern, op in _WORD_OPS:
        for m in pattern.finditer(text):
            try:
                value = op(_num(m.group(1)), _num(m.group(2)))
            except ZeroDivisionError:
                continue
            checked += 1
            if not _matches(value, m.group(3)):
                issues.append(Issue(MATHEMATICAL, f"'{m.group(0).strip()}' is wrong: it equals {value:g}"))
    return issues, checked


# ------------------------------------------------------------------------------------------ code
_FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.DOTALL)


_NO_IMPORT = {"antigravity", "this", "idlelib", "turtle", "turtledemo", "tkinter", "webbrowser", "pydoc"}


def _stdlib_attribute_issues(tree: ast.AST) -> list[Issue]:
    """Calls such as `json.loadz(...)` on imported *standard-library* modules whose attribute does
    not exist. Only stdlib modules are imported for the lookup (importing arbitrary third-party
    modules could run their code)."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name in sys.stdlib_module_names and a.name not in _NO_IMPORT:
                    aliases[a.asname or a.name] = a.name
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases:
            module_name = aliases[node.value.id]
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            if not hasattr(module, node.attr):
                issues.append(Issue(CODE, f"'{node.value.id}.{node.attr}' does not exist in the standard "
                                          f"library module '{module_name}' (invented API)"))
    return issues


def check_code(answer: str) -> tuple[list[Issue], int]:
    """Static checks of Python code blocks: syntax, imports of modules that are not installed or do
    not exist (soft: they may exist elsewhere), and invented standard-library functions. The code
    is parsed, never executed."""
    issues: list[Issue] = []
    checked = 0
    for lang, body in _FENCE_RE.findall(answer or ""):
        if lang and lang.lower() not in ("python", "py", "python3"):
            continue
        checked += 1
        try:
            tree = ast.parse(body)
        except SyntaxError as exc:
            if lang:  # an unlabelled block may simply not be Python
                issues.append(Issue(CODE, f"Python code does not parse: {exc.msg} (line {exc.lineno})"))
            continue
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 else [])
            for name in names:
                top = name.split(".")[0]
                if top in sys.stdlib_module_names:
                    continue
                if importlib.util.find_spec(top) is None:
                    issues.append(Issue(CODE, f"imports '{top}', which is not installed here and may not exist",
                                        hard=False))
        issues.extend(_stdlib_attribute_issues(tree))
    return issues, checked


# ------------------------------------------------------------------------------------------ all
def run_type_checks(
    answer: str, claims: list[str], passages: list[Passage] | None, nli: NLIScorer | None,
    logic_threshold: float,
) -> TypeCheckResult:
    """Run every type-specific check on one answer."""
    result = TypeCheckResult()
    for name, (issues, n) in (
        (CITATION, check_citations(answer, passages)),
        (MATHEMATICAL, check_math(answer)),
        (CODE, check_code(answer)),
    ):
        result.issues.extend(issues)
        result.checked[name] = n
    if nli is not None:
        issues, n = check_logical_consistency(claims, nli, logic_threshold)
        result.issues.extend(issues)
        result.checked[LOGICAL] = n
    return result
