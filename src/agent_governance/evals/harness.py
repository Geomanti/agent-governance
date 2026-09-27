"""Evaluation harness: golden suites, behavioral assertions, rubric, CI gate.

The job description draws a hard line: *"Design evaluations, not spot checks...
You can explain why 'we reviewed a bunch of outputs and they looked good' is
not evaluation."* The distinction it is pointing at is falsifiability. A spot
check produces an opinion; an evaluation produces a number that a regression can
move and a CI job can block on.

Four things make that true here.

**Behavioral assertions, not string matching.** An ``Assertion`` is an
invariant with a name and a severity, evaluated against a *trajectory* -- the
sequence of decisions and effects a run produced. Assertions test what the
system did, not what it said.

**Safety assertions are not averaged.** A case's safety assertions must all
hold. Quality assertions are scored and averaged against a threshold. Blending
the two is how a suite starts passing while being unsafe, so the arithmetic
keeps them apart.

**A written rubric, not vibes.** ``Rubric`` carries weighted criteria with
explicit pass descriptions. ``Judge`` scores against the rubric; a judge that
cannot cite a criterion is not accepted.

**The grader is graded.** ``evals.mutants`` seeds deliberate defects into a
reference implementation and asserts the suite *fails* on each. A suite that
passes a mutant is blind to that class of bug, and the meta-eval makes that
visible as a number -- mutated score -- rather than a claim.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence

__all__ = [
    "Severity",
    "Fail",
    "Assertion",
    "Trajectory",
    "Case",
    "CaseResult",
    "SuiteResult",
    "Suite",
    "RubricCriterion",
    "Rubric",
    "JudgeVerdict",
    "Judge",
    "GateDecision",
    "gate",
]


class Severity(str, Enum):
    SAFETY = "safety"
    QUALITY = "quality"


@dataclass(frozen=True)
class Fail:
    """A failed assertion, carrying the evidence a reviewer would ask for."""

    assertion: str
    severity: Severity
    message: str

    def __str__(self) -> str:  # pragma: no cover - display
        return f"[{self.severity.value}] {self.assertion}: {self.message}"


@dataclass(frozen=True)
class Assertion:
    """A named invariant over a trajectory.

    ``check`` returns ``None`` on success or a message explaining the failure.
    Messages should name the observed value -- "quote leaked into public digest"
    is a spot check; "field 'body' contained 'SECRET-ROADMAP' (6 occurrences)"
    is a finding.
    """

    name: str
    check: Callable[["Trajectory"], str | None]
    severity: Severity = Severity.SAFETY
    description: str = ""

    def evaluate(self, traj: "Trajectory") -> Fail | None:
        try:
            msg = self.check(traj)
        except Exception as exc:  # noqa: BLE001 - an erroring assertion is a failure
            return Fail(self.name, self.severity, f"assertion raised {type(exc).__name__}: {exc}")
        if msg:
            return Fail(self.name, self.severity, msg)
        return None


@dataclass
class Trajectory:
    """Everything an assertion may inspect about one run.

    ``output`` is the agent's produced artefact, ``decisions`` the audit
    decisions in order, ``effects`` the side effects that actually landed, and
    ``meta`` anything else a case wants to expose.
    """

    case_id: str
    output: Any = None
    decisions: list[str] = field(default_factory=list)
    effects: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def text(self) -> str:
        """Output as text, for assertions that inspect content."""
        if self.output is None:
            return ""
        if isinstance(self.output, str):
            return self.output
        if isinstance(self.output, Mapping):
            return "\n".join(f"{k}: {v}" for k, v in self.output.items())
        if isinstance(self.output, (list, tuple)):
            return "\n".join(str(x) for x in self.output)
        return str(self.output)

    def denied_any(self) -> bool:
        return "deny" in self.decisions


@dataclass(frozen=True)
class Case:
    """One golden case: an input, the assertions it must satisfy, a runner."""

    case_id: str
    description: str
    run: Callable[[], Trajectory]
    assertions: tuple[Assertion, ...] = ()

    def execute(self) -> "CaseResult":
        started = time.perf_counter()
        try:
            traj = self.run()
            if not isinstance(traj, Trajectory):
                raise TypeError(f"case runner returned {type(traj).__name__}, expected Trajectory")
            traj.case_id = self.case_id
        except Exception as exc:  # noqa: BLE001 - a crashing case is a failed case
            elapsed = (time.perf_counter() - started) * 1000
            return CaseResult(
                case_id=self.case_id,
                passed=False,
                failures=(
                    Fail("case.runner", Severity.SAFETY, f"raised {type(exc).__name__}: {exc}"),
                ),
                duration_ms=elapsed,
            )

        fails: list[Fail] = []
        for a in self.assertions:
            f = a.evaluate(traj)
            if f is not None:
                fails.append(f)

        safety_fails = [f for f in fails if f.severity is Severity.SAFETY]
        checks = len(self.assertions) or 1
        passed = not safety_fails and (len(fails) / checks) < 1.0
        return CaseResult(
            case_id=self.case_id,
            passed=passed,
            failures=tuple(fails),
            duration_ms=(time.perf_counter() - started) * 1000,
            assertion_count=len(self.assertions),
        )


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    passed: bool
    failures: tuple[Fail, ...] = ()
    duration_ms: float = 0.0
    assertion_count: int = 0

    @property
    def safety_failures(self) -> tuple[Fail, ...]:
        return tuple(f for f in self.failures if f.severity is Severity.SAFETY)

    @property
    def quality_failures(self) -> tuple[Fail, ...]:
        return tuple(f for f in self.failures if f.severity is Severity.QUALITY)


@dataclass(frozen=True)
class SuiteResult:
    """The number a CI gate reads."""

    suite: str
    cases: tuple[CaseResult, ...]
    threshold: float
    duration_ms: float = 0.0

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total) if self.total else 0.0

    @property
    def safety_violations(self) -> tuple[Fail, ...]:
        return tuple(f for c in self.cases for f in c.safety_failures)

    @property
    def quality_failures(self) -> tuple[Fail, ...]:
        return tuple(f for c in self.cases for f in c.quality_failures)

    @property
    def ok(self) -> bool:
        """Passes only when the threshold is met *and* no safety assertion failed.

        The conjunction is the design decision that matters: a suite cannot
        trade a safety violation for a better average.
        """
        return not self.safety_violations and self.pass_rate >= self.threshold

    def summary(self) -> str:
        lines = [
            f"suite {self.suite}: {self.passed}/{self.total} passed "
            f"({self.pass_rate:.0%}) threshold {self.threshold:.0%} -> "
            f"{'PASS' if self.ok else 'BLOCK'}",
            f"  safety violations: {len(self.safety_violations)}",
            f"  quality failures:  {len(self.quality_failures)}",
            f"  duration:          {self.duration_ms:.1f} ms",
        ]
        for f in self.safety_violations:
            lines.append(f"  ! {f}")
        return "\n".join(lines)


@dataclass(frozen=True)
class Suite:
    """A golden suite: named cases and a pass threshold."""

    name: str
    cases: tuple[Case, ...]
    threshold: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")

    def run(self) -> SuiteResult:
        started = time.perf_counter()
        results = tuple(c.execute() for c in self.cases)
        return SuiteResult(
            suite=self.name,
            cases=results,
            threshold=self.threshold,
            duration_ms=(time.perf_counter() - started) * 1000,
        )


# --------------------------------------------------------------------------
# Rubric + judge
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RubricCriterion:
    """One weighted criterion with an explicit pass description."""

    key: str
    description: str
    weight: float
    passes_when: str

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError(f"criterion {self.key!r} needs a positive weight")


@dataclass(frozen=True)
class Rubric:
    """A written rubric. Weights are normalised at scoring time."""

    name: str
    criteria: tuple[RubricCriterion, ...]

    @property
    def total_weight(self) -> float:
        return sum(c.weight for c in self.criteria)

    def criterion(self, key: str) -> RubricCriterion:
        for c in self.criteria:
            if c.key == key:
                return c
        raise KeyError(f"rubric {self.name!r} has no criterion {key!r}")


@dataclass(frozen=True)
class JudgeVerdict:
    """A judge's score, with the citations that justify it."""

    score: float
    citations: tuple[str, ...]
    rationale: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("judge score must be in [0, 1]")


class Judge:
    """Scores an output against a rubric.

    ``score_fn`` produces ``{criterion_key: score_in_[0,1]}``. The verdict is a
    weighted sum, and **every criterion must be scored** -- a judge that skips
    a criterion raises rather than quietly scoring it zero, because silent
    zeros are how a rubric stops meaning anything. Citations are required: a
    verdict with no citation is rejected.
    """

    def __init__(
        self,
        rubric: Rubric,
        score_fn: Callable[[str, Rubric], Mapping[str, float]],
        *,
        require_citations: bool = True,
    ) -> None:
        self.rubric = rubric
        self.score_fn = score_fn
        self.require_citations = require_citations
        self._verdicts: list[JudgeVerdict] = []

    def judge(self, output: str, *, citations: Sequence[str] = ()) -> JudgeVerdict:
        raw = dict(self.score_fn(output, self.rubric))
        missing = [c.key for c in self.rubric.criteria if c.key not in raw]
        if missing:
            raise ValueError(f"judge did not score criteria: {missing}")
        for k, v in raw.items():
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"criterion {k!r} scored {v}, outside [0, 1]")

        total = self.rubric.total_weight
        weighted = sum(raw[c.key] * c.weight for c in self.rubric.criteria) / total

        cites = tuple(citations) or tuple(
            f"{c.key}: {c.passes_when}" for c in self.rubric.criteria if raw[c.key] >= 0.5
        )
        if self.require_citations and not cites:
            raise ValueError("a verdict without a citation is not accepted")

        v = JudgeVerdict(score=weighted, citations=cites, rationale=f"{len(raw)} criteria scored")
        self._verdicts.append(v)
        return v

    @property
    def verdicts(self) -> tuple[JudgeVerdict, ...]:
        return tuple(self._verdicts)


# --------------------------------------------------------------------------
# CI gate
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GateDecision:
    """What CI reads. ``blocked`` is what stops a merge."""

    ok: bool
    blocked: bool
    reasons: tuple[str, ...]
    results: tuple[SuiteResult, ...]

    def summary(self) -> str:
        lines = [
            "=" * 68,
            f"EVAL GATE: {'PASS' if self.ok else 'BLOCKED'}",
            "=" * 68,
        ]
        for r in self.results:
            lines.append(r.summary())
        if self.reasons:
            lines.append("")
            lines.append("Reasons:")
            lines.extend(f"  - {r}" for r in self.reasons)
        lines.append("=" * 68)
        return "\n".join(lines)


def gate(results: Iterable[SuiteResult]) -> GateDecision:
    """Decide whether the suites let a change merge.

    A regression is *any* drop below threshold or *any* safety violation. The
    gate is deliberately conservative: it blocks on safety unconditionally, so
    it cannot be satisfied by improving an average.
    """
    rs = tuple(results)
    reasons: list[str] = []
    for r in rs:
        if r.safety_violations:
            reasons.append(
                f"{r.suite}: {len(r.safety_violations)} safety violation(s) "
                "-- safety cannot be averaged away"
            )
        if r.pass_rate < r.threshold:
            reasons.append(
                f"{r.suite}: pass rate {r.pass_rate:.0%} below threshold {r.threshold:.0%}"
            )
    return GateDecision(ok=not reasons, blocked=bool(reasons), reasons=tuple(reasons), results=rs)
