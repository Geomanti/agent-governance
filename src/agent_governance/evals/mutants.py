"""Meta-evaluation: the thing that grades the has to be graded too.

A suite that has never failed is not evidence of anything. This module seeds
deliberate defects -- mutants -- into the governance core and asserts the
suite **catches each one**. The output is a mutation score: the fraction of
seeded defects the suite detected.

**The mutation is suite-level, not predicate-level.** An earlier version of
this file probed a single predicate per mutant ("does ``grants_for`` filter?").
That is weaker than it looks: it re-tests the mechanism rather than the suite,
so a mutant could be reported as killed while the *suite* was blind to the same
defect through a different route. This version mutates a live method on the
shared class, runs the **entire suite**, and counts the mutant as killed only
if the gate actually blocks. That is the property CI depends on -- it is the
suite that must fail, not a probe.

The mutants, one per way the model can break:

* ``forbidden-scope-accepted`` -- construction-time rejection is removed.
* ``surface-unfiltered`` -- tools register regardless of credential scopes.
* ``budget-ceiling-ignored`` -- the interruption ceiling stops applying.
* ``tier-ceiling-ignored`` -- a credential may claim autonomy above its tier.
* ``review-gate-removed`` -- a propose-tier effect applies without a human.
* ``rollback-disabled`` -- a failed run leaves its effects in place.
* ``audit-chain-broken`` -- the hash chain stops detecting a rewritten record.

A suite scoring 1.0 kills every mutant. Any survivor is named in the report,
because a survivor is a class of bug the suite cannot see -- and that is
something a green light must not hide.
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence

from ..capabilities import (
    AgentCredential,
    AgentCredential as _Cred,
    CapabilityError,
    FORBIDDEN_SCOPES,
    RiskTier,
    ToolGrant,
    audit_credential,
)

__all__ = [
    "Mutant",
    "MutantResult",
    "MutationScore",
    "MUTANTS",
    "score_mutation",
    "audit_forbidden_paths",
    "ForbiddenPathAudit",
]


@dataclass(frozen=True)
class Mutant:
    """A seeded defect: apply it, run the suite, expect the gate to block.

    ``apply`` mutates the live system and returns an undo callable. ``undo`` is
    invoked in a ``finally`` so a mutant can never leak into a later run.
    """

    name: str
    description: str
    apply: Callable[[], Callable[[], None]]
    guards: str


@dataclass(frozen=True)
class MutantResult:
    mutant: str
    detected: bool
    detail: str
    guards: str


@dataclass(frozen=True)
class MutationScore:
    results: tuple[MutantResult, ...]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def killed(self) -> int:
        return sum(1 for r in self.results if r.detected)

    @property
    def score(self) -> float:
        return (self.killed / self.total) if self.total else 0.0

    @property
    def survivors(self) -> tuple[MutantResult, ...]:
        return tuple(r for r in self.results if not r.detected)

    def summary(self) -> str:
        lines = [f"mutation score: {self.killed}/{self.total} ({self.score:.0%})"]
        for r in self.results:
            mark = "killed " if r.detected else "SURVIVED"
            lines.append(f"  [{mark}] {r.mutant}: {r.detail}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Mutation operators. Each returns an undo callable.
# --------------------------------------------------------------------------


def _mutate_forbidden_scope() -> Callable[[], None]:
    """Remove the construction-time rejection of the forbidden scope class."""
    original = _Cred.__post_init__

    def neutered(self) -> None:
        if not self.agent_id:
            raise CapabilityError("agent_id must be set")

    _Cred.__post_init__ = neutered
    return lambda: setattr(_Cred, "__post_init__", original)


def _mutate_surface_unfiltered() -> Callable[[], None]:
    """Register every tool regardless of the credential's scopes."""
    original = _Cred.grants_for
    _Cred.grants_for = lambda self, available: tuple(available)  # type: ignore[assignment]
    return lambda: setattr(_Cred, "grants_for", original)


def _mutate_budget_ceiling() -> Callable[[], None]:
    """Stop refusing contact to an exhausted recipient."""
    from ..budget import BudgetLedger

    original = BudgetLedger.is_exhausted
    BudgetLedger.is_exhausted = lambda self, recipient: False  # type: ignore[assignment]
    return lambda: setattr(BudgetLedger, "is_exhausted", original)


def _mutate_tier_ceiling() -> Callable[[], None]:
    """Let any tier claim any autonomy class."""
    original = _Cred.__post_init__
    from ..capabilities import FORBIDDEN_SCOPES as FS

    def ceilingless(self) -> None:
        if not self.agent_id:
            raise CapabilityError("agent_id must be set")
        # Keep only the forbidden-scope check so this mutant isolates the
        # tier ceiling rather than duplicating the forbidden-scope mutant.
        if self.scopes & FS:
            raise CapabilityError("forbidden scope")

    _Cred.__post_init__ = ceilingless
    return lambda: setattr(_Cred, "__post_init__", original)


def _mutate_review_gate() -> Callable[[], None]:
    """Let a propose-tier effect apply without human approval."""
    from ..runtime import RunState

    original = RunState.requires_review
    RunState.requires_review = property(lambda self: False)  # type: ignore[assignment]
    return lambda: setattr(RunState, "requires_review", original)


def _mutate_rollback_disabled() -> Callable[[], None]:
    """Make rollback a no-op that still reports success."""
    from ..runtime import TaskQueue, RunStatus

    def fake_rollback(self, state, *, reason: str = ""):
        state.status = RunStatus.ROLLED_BACK
        self.store.save(state)
        return state

    original = TaskQueue.rollback
    TaskQueue.rollback = fake_rollback  # type: ignore[assignment]
    return lambda: setattr(TaskQueue, "rollback", original)


def _mutate_audit_chain() -> Callable[[], None]:
    """Stop the chain from detecting content tampering."""
    from .. import audit as audit_mod

    original = audit_mod.AuditRecord.compute_hash
    audit_mod.AuditRecord.compute_hash = lambda self: self.record_hash or "0"  # type: ignore[assignment]
    return lambda: setattr(audit_mod.AuditRecord, "compute_hash", original)


MUTANTS: tuple[Mutant, ...] = (
    Mutant(
        "forbidden-scope-accepted",
        "construction-time rejection of the forbidden scope class is removed",
        _mutate_forbidden_scope,
        "capabilities.AgentCredential.__post_init__",
    ),
    Mutant(
        "surface-unfiltered",
        "tools register regardless of the credential's scopes",
        _mutate_surface_unfiltered,
        "capabilities.AgentCredential.grants_for",
    ),
    Mutant(
        "budget-ceiling-ignored",
        "the interruption ceiling stops refusing exhausted recipients",
        _mutate_budget_ceiling,
        "budget.BudgetLedger.is_exhausted",
    ),
    Mutant(
        "tier-ceiling-ignored",
        "a credential may claim autonomy above its tier ceiling",
        _mutate_tier_ceiling,
        "capabilities.AgentCredential.__post_init__",
    ),
    Mutant(
        "review-gate-removed",
        "a propose-tier effect applies without human approval",
        _mutate_review_gate,
        "runtime.RunState.requires_review",
    ),
    Mutant(
        "rollback-disabled",
        "a failed run leaves its effects applied",
        _mutate_rollback_disabled,
        "runtime.TaskQueue.rollback",
    ),
    Mutant(
        "audit-chain-broken",
        "the hash chain stops detecting a rewritten record",
        _mutate_audit_chain,
        "audit.AuditRecord.compute_hash",
    ),
)


def score_mutation(mutants: Sequence[Mutant] | None = None) -> MutationScore:
    """Apply each mutant, run the full suite, and record whether it was caught.

    Detection is the gate blocking. A mutant that the suite still passes is a
    survivor: a failure class the suite is blind to, reported by name.
    """
    from .runner import run_all  # local import: runner imports this module

    ms = mutants if mutants is not None else MUTANTS
    out: list[MutantResult] = []
    for m in ms:
        undo = m.apply()
        try:
            with tempfile.TemporaryDirectory() as td:
                # include_meta=False: we are asking whether the *suites* block
                # this mutant, so we must not re-enter the meta-eval.
                rep = run_all(td, include_meta=False)
            blocked = not rep["gate"]["ok"]
            if blocked:
                det = f"gate blocked: {rep['gate']['reasons'][0]}"
            else:
                det = "SURVIVED -- the suite passed with this defect live"
        except Exception as exc:  # noqa: BLE001 - a crashing mutant is a detection
            blocked = True
            det = f"suite raised {type(exc).__name__}: {exc}"
        finally:
            undo()
        out.append(
            MutantResult(mutant=m.name, detected=blocked, detail=det, guards=m.guards)
        )
    return MutationScore(tuple(out))


# --------------------------------------------------------------------------
# Forbidden-path audit
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ForbiddenPathAudit:
    """Proof that the dangerous autonomy class has no implementing code path."""

    forbidden_autonomy: str
    implemented: bool
    reachable_tools: tuple[str, ...]
    forbidden_scopes: tuple[str, ...]
    verdict: str

    @property
    def ok(self) -> bool:
        return not self.implemented and not self.reachable_tools


def audit_forbidden_paths(grants: Sequence[ToolGrant]) -> ForbiddenPathAudit:
    """Audit the registry for any path to the commitment-write class.

    This is the check the job description asks for in its "what success looks
    like" section: *"an audit of any agent's credentials shows it cannot perform
    the writes it is not allowed to perform."*
    """
    from ..capabilities import AUTONOMY_IMPLEMENTED, AutonomyClass

    reachable = tuple(
        g.name
        for g in grants
        if g.autonomy is AutonomyClass.COMMITMENT_WRITE or (g.scopes & FORBIDDEN_SCOPES)
    )
    implemented = AUTONOMY_IMPLEMENTED[AutonomyClass.COMMITMENT_WRITE]
    ok = not implemented and not reachable
    return ForbiddenPathAudit(
        forbidden_autonomy=AutonomyClass.COMMITMENT_WRITE.value,
        implemented=implemented,
        reachable_tools=reachable,
        forbidden_scopes=tuple(sorted(FORBIDDEN_SCOPES)),
        verdict=(
            "no code path: the commitment-write class is unreachable by construction"
            if ok
            else "REACHABLE -- governance hole"
        ),
    )
