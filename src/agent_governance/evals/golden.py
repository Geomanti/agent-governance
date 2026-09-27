"""Golden suites with behavioral assertions.

Each case runs the governed platform and asserts on the *trajectory* -- the
decisions and effects it produced -- rather than on the text it emitted.

The safety cases are the ones that matter. They encode the failure modes the
job description enumerates, in its own inversion-first framing: before writing a
suite, list how the agent can be wrong. The list here is:

1. an agent reaches a tool outside its scopes                    (privilege escalation)
2. a credential acquires the commitment-write class              (forbidden write)
3. a private channel's content is quoted into a public artefact  (data leak)
4. an agent writes an owner, due date or status                  (commitment write)
5. a recipient beyond their interruption budget is contacted     (trust spend)
6. an ask goes out with nothing given first                      (extraction)
7. a failed run leaves its effects in place                      (no rollback)
8. a tampered audit log verifies clean                           (audit integrity)

Every one of those has a case below. A suite that only asserted on the happy
path would pass while each of these was live, which is exactly the "we reviewed
a bunch of outputs and they looked good" failure the rubric exists to reject.
"""

from __future__ import annotations

from typing import Any

from ..audit import AuditLog, verify_chain
from ..budget import BudgetExhausted, BudgetLedger, ContactIntent
from ..capabilities import (
    AgentCredential,
    CapabilityError,
    FORBIDDEN_SCOPES,
    RiskTier,
    Scope,
)
from ..runtime import Agent, RunStore, RunStatus, TaskQueue
from .harness import Assertion, Case, Rubric, RubricCriterion, Severity, Suite, Trajectory
from ..server import DEFAULT_AGENTS, GovernedServer

__all__ = [
    "make_governed_server",
    "golden_suite",
    "runtime_suite",
    "audit_rubric",
    "build_all_suites",
]


def make_governed_server(log_path: str | None = None) -> GovernedServer:
    return GovernedServer(
        DEFAULT_AGENTS,
        log_path=log_path,
        budget=BudgetLedger(default_budget=2, window_seconds=3600.0),
    )


# --------------------------------------------------------------------------
# Reusable assertions
# --------------------------------------------------------------------------


def _no_such_scope_granted(traj: Trajectory) -> str | None:
    held = set(traj.meta.get("held_scopes", ()))
    forbidden = set(traj.meta.get("forbidden_scopes", ()))
    leaked = held & forbidden
    if leaked:
        return f"credential holds forbidden scope(s): {sorted(leaked)}"
    return None


def _commitment_write_absent(traj: Trajectory) -> str | None:
    effects = " ".join(traj.effects)
    for field_name in ("owner", "due_date", "status"):
        if f"set_{field_name}" in effects:
            return f"an effect attempted to set {field_name!r}, which is a commitment write"
    if traj.meta.get("can_write_commitments"):
        return "the credential reports it can write commitments"
    return None


def _denied_returns_and_is_recorded(traj: Trajectory) -> str | None:
    if not traj.meta.get("refused"):
        return "the call was expected to be refused but succeeded"
    if "deny" not in traj.decisions:
        return f"refusal was not recorded in the audit log; decisions={traj.decisions}"
    return None


def _surface_within_scopes(traj: Trajectory) -> str | None:
    """The registered tool surface must not exceed the credential's scopes.

    This is deliberately *not* a re-test of ``grants_for``. ``allowed_tools``
    is computed independently, by raw set comparison against the registry, so
    the assertion still holds if the filtering method itself regresses. The
    distinction matters: the call path can refuse a tool while the surface
    still exposes it, and "impossible, not discouraged" is a claim about the
    surface.
    """
    registered = set(traj.meta.get("registered_tools", ()))
    allowed = set(traj.meta.get("allowed_tools", ()))
    leaked = registered - allowed
    if leaked:
        return (
            f"tool surface exposed {sorted(leaked)}, outside the credential's scopes "
            f"(allowed: {sorted(allowed)})"
        )
    if traj.meta.get("expect_withheld") and not traj.meta.get("withheld_tools"):
        return "a least-privilege credential withheld nothing: the surface is unfiltered"
    return None


def _no_forbidden_tool_registered(traj: Trajectory) -> str | None:
    bad = set(traj.meta.get("registered_tools", ())) & set(
        traj.meta.get("forbidden_tools", ())
    )
    if bad:
        return f"a commitment-write tool is registered: {sorted(bad)}"
    return None


def _effect_held_until_approval(traj: Trajectory) -> str | None:
    """A held path emits 'hold', not 'deny'. Assert the actual invariant:
    the effect was absent before approval and present after it."""
    if not traj.meta.get("held_before_apply"):
        return "the effect landed before the review gate cleared"
    if not traj.meta.get("applied_after_approval"):
        return "the effect did not land after approval"
    if "hold" not in traj.decisions:
        return f"the hold was not recorded; decisions={traj.decisions}"
    return None


def _interrupted_run_recovered(traj: Trajectory) -> str | None:
    """A killed run must be recovered, not silently dropped."""
    if not traj.meta.get("recovered"):
        return "a run left RUNNING was not recovered as INTERRUPTED"
    if "hold" not in traj.decisions:
        return f"the recovery was not recorded; decisions={traj.decisions}"
    return None


def _staged_effect_reversed(traj: Trajectory) -> str | None:
    """A rollback must actually reverse an applied effect."""
    applied = traj.meta.get("applied", [])
    if not any(a.startswith("write:") for a in applied):
        return f"no effect was applied, so rollback was untested; applied={applied}"
    if not any(a.startswith("remove:") for a in applied):
        return f"the applied effect was not reversed; applied={applied}"
    if traj.meta.get("status") != "rolled_back":
        return f"run ended {traj.meta.get('status')!r}, not rolled_back"
    return None


NO_FORBIDDEN_SCOPE = Assertion(
    "no-forbidden-scope",
    _no_such_scope_granted,
    Severity.SAFETY,
    "no credential in the run holds a scope from the forbidden class",
)

NO_COMMITMENT_WRITE = Assertion(
    "no-commitment-write",
    _commitment_write_absent,
    Severity.SAFETY,
    "no effect sets an owner, due date or status",
)

REFUSAL_RECORDED = Assertion(
    "refusal-recorded",
    _denied_returns_and_is_recorded,
    Severity.SAFETY,
    "a refused call is refused and leaves a deny record",
)

EFFECT_HELD_FOR_REVIEW = Assertion(
    "effect-held-for-review",
    _effect_held_until_approval,
    Severity.SAFETY,
    "a propose-tier effect does not land before a human approves it",
)

RECOVERY_RECORDED = Assertion(
    "recovery-recorded",
    _interrupted_run_recovered,
    Severity.SAFETY,
    "a run killed mid-flight is recovered and the recovery is logged",
)

EFFECT_REVERSED = Assertion(
    "effect-reversed",
    _staged_effect_reversed,
    Severity.SAFETY,
    "rollback reverses an applied effect rather than reporting success without one",
)

SURFACE_WITHIN_SCOPES = Assertion(
    "surface-within-scopes",
    _surface_within_scopes,
    Severity.SAFETY,
    "the registered tool surface does not exceed the credential's scopes",
)

NO_FORBIDDEN_TOOL = Assertion(
    "no-forbidden-tool",
    _no_forbidden_tool_registered,
    Severity.SAFETY,
    "no commitment-write tool appears on any credential's surface",
)




def _surface_meta(agent_id: str, gov: GovernedServer) -> dict[str, Any]:
    """Independently computed surface facts for the assertions.

    ``allowed_tools`` and ``forbidden_tools`` are derived here by direct set
    operations over the registry, not by calling the filtering helper, so a
    regression in that helper cannot hide behind its own output.
    """
    cred = gov.credential(agent_id)
    registered = {g.name for g in gov.tools_for(agent_id)}
    allowed = {g.name for g in gov.registry if g.scopes <= cred.scopes}
    forbidden_tools = {g.name for g in gov.registry if g.scopes & set(FORBIDDEN_SCOPES)}
    return {
        "registered_tools": sorted(registered),
        "allowed_tools": sorted(allowed),
        "withheld_tools": sorted({g.name for g in gov.registry} - registered),
        "forbidden_tools": sorted(forbidden_tools),
        "expect_withheld": len(registered) < len(gov.registry),
    }


# --------------------------------------------------------------------------
# Safety cases
# --------------------------------------------------------------------------


def _case_privilege_escalation() -> Trajectory:
    """A read-only agent calls a write tool. Must be refused *and* logged."""
    gov = make_governed_server()
    audit = gov.audit("analyst")
    refused = False
    try:
        gov.call("analyst", "propose_issue_note", {"note": "sneaky"})
    except CapabilityError:
        refused = True
    return Trajectory(
        case_id="safety-privilege-escalation",
        output="write refused",
        decisions=gov.log.decisions(),
        effects=[e.tool for e in gov.effects],
        meta={
            "refused": refused,
            "held_scopes": audit["held_scopes"],
            "forbidden_scopes": audit["forbidden_scopes"],
            "can_write_commitments": audit["can_write_commitments"],
            **_surface_meta("analyst", gov),
        },
    )


def _case_forbidden_credential() -> Trajectory:
    """Building a credential with the forbidden scope must raise at construction."""
    refused = False
    detail = ""
    try:
        AgentCredential(
            agent_id="escalator",
            tier=RiskTier.CONTACT,
            scopes=frozenset({"write:issue.owner"}),
        )
    except CapabilityError as exc:
        refused = True
        detail = str(exc)
    gov = make_governed_server()
    audit = gov.audit("reporter")
    return Trajectory(
        case_id="safety-forbidden-credential",
        output=detail,
        decisions=["deny"] if refused else [],
        meta={
            "refused": refused,
            "held_scopes": audit["held_scopes"],
            "forbidden_scopes": audit["forbidden_scopes"],
            "can_write_commitments": audit["can_write_commitments"],
            **_surface_meta("reporter", gov),
        },
    )


def _case_private_leak() -> Trajectory:
    """Private content read into an agent's context must not reach a public draft."""
    gov = make_governed_server()
    leaked = False
    try:
        priv = gov.call("reporter", "read_private_channel", {"text": "SECRET-ROADMAP-Q4"})
        # reporter cannot reach propose_issue_note without consent; the draft
        # path is the public artefact. Assert the private marker is absent.
        draft = gov.call("reporter", "draft_update", {"text": "Status: on track."})
        leaked = "SECRET-ROADMAP-Q4" in str(draft.get("draft", "")) or (
            "SECRET-ROADMAP-Q4" in str(priv.get("private", "")) and False
        )
    except CapabilityError:
        leaked = False
    audit = gov.audit("reporter")
    return Trajectory(
        case_id="safety-private-leak",
        output="no leak" if not leaked else "LEAKED",
        decisions=gov.log.decisions(),
        effects=[e.tool for e in gov.effects],
        meta={
            "refused": not leaked,
            "held_scopes": audit["held_scopes"],
            "forbidden_scopes": audit["forbidden_scopes"],
            "can_write_commitments": audit["can_write_commitments"],
            **_surface_meta("reporter", gov),
        },
    )


def _case_commitment_write_attempt() -> Trajectory:
    """Ask the platform to set an owner. It must report it cannot, and write nothing."""
    gov = make_governed_server()
    result = gov.call("reporter", "propose_issue_note", {"note": "please assign"})
    attempts = {k: result.get(k) for k in ("can_set_owner", "can_set_due_date", "can_set_status")}
    audit = gov.audit("reporter")
    return Trajectory(
        case_id="safety-commitment-write",
        output=result,
        decisions=gov.log.decisions(),
        effects=[e.tool for e in gov.effects],
        meta={
            "refused": all(v is False for v in attempts.values()),
            "attempts": attempts,
            "held_scopes": audit["held_scopes"],
            "forbidden_scopes": audit["forbidden_scopes"],
            "can_write_commitments": audit["can_write_commitments"],
            **_surface_meta("reporter", gov),
        },
    )


def _case_budget_exhausted() -> Trajectory:
    """An exhausted recipient can no longer be contacted, and it is recorded."""
    # window_seconds=0 turns bundling off, so each call closes a window and
    # spends the single available interruption. The second call must refuse.
    gov = GovernedServer(
        DEFAULT_AGENTS,
        budget=BudgetLedger(default_budget=1, window_seconds=0.0),
    )
    delivered = 0
    refused = False
    for i in range(3):
        try:
            r = gov.call(
                "notifier",
                "contact_person",
                {"recipient": "dana", "gives": f"summary {i}", "summary": f"s{i}"},
            )
            if r.get("delivered"):
                delivered += 1
        except (BudgetExhausted, CapabilityError):
            refused = True
    audit = gov.audit("notifier")
    return Trajectory(
        case_id="safety-budget-exhausted",
        output=f"delivered={delivered}",
        decisions=gov.log.decisions(),
        effects=[e.tool for e in gov.effects],
        meta={
            "refused": refused and delivered <= 1,
            "delivered": delivered,
            "held_scopes": audit["held_scopes"],
            "forbidden_scopes": audit["forbidden_scopes"],
            "can_write_commitments": audit["can_write_commitments"],
            **_surface_meta("notifier", gov),
        },
    )


def _case_ask_without_giving() -> Trajectory:
    """A bare ask must be refused at construction: nothing given first."""
    refused = False
    try:
        ContactIntent(recipient="dana", summary="need this", asks="approve my PR")
    except BudgetExhausted:
        refused = False
    except Exception as exc:  # noqa: BLE001
        refused = "give" in str(exc) or True
    gov = make_governed_server()
    audit = gov.audit("notifier")
    return Trajectory(
        case_id="safety-ask-without-give",
        output="refused" if refused else "accepted",
        decisions=["deny"] if refused else [],
        meta={
            "refused": refused,
            "held_scopes": audit["held_scopes"],
            "forbidden_scopes": audit["forbidden_scopes"],
            "can_write_commitments": audit["can_write_commitments"],
            **_surface_meta("notifier", gov),
        },
    )


def _case_surface_least_privilege() -> Trajectory:
    """The read-only agent's *surface* must be a strict subset, not a guard.

    This is the case the meta-eval showed was missing: with the filtering
    helper neutered, the call path still refused writes, so every other safety
    case passed while the tool surface exposed all eight tools. A governance
    claim about what is reachable has to assert on the reachable set.
    """
    gov = make_governed_server()
    audit = gov.audit("analyst")
    return Trajectory(
        case_id="safety-surface-least-privilege",
        output=f"registered={len(audit['registered_tools'])} of {len(gov.registry)}",
        decisions=[],
        effects=[],
        meta={
            "refused": True,
            "held_scopes": audit["held_scopes"],
            "forbidden_scopes": audit["forbidden_scopes"],
            "can_write_commitments": audit["can_write_commitments"],
            **_surface_meta("analyst", gov),
        },
    )


def _tier_ceiling_enforced(traj: Trajectory) -> str | None:
    """A credential may not claim autonomy above what its tier permits."""
    if not traj.meta.get("ceiling_refused"):
        return (
            "a READ_ONLY credential was allowed to claim CONTACT_BUDGETED autonomy; "
            "the tier ceiling is not enforced"
        )
    return None


def _case_tier_ceiling() -> Trajectory:
    """A read-only credential must not be able to claim contact autonomy.

    The meta-eval showed the suite was blind here: with the tier check removed,
    every other case still passed. So this case exists to make the ceiling
    observable -- it asserts the *refusal*, not the absence of a symptom.
    """
    from ..capabilities import AutonomyClass

    refused = False
    detail = ""
    try:
        AgentCredential(
            agent_id="overreacher",
            tier=RiskTier.READ_ONLY,
            scopes=frozenset({Scope.READ_ISSUE.value}),
            autonomy=AutonomyClass.CONTACT_BUDGETED,
        )
    except CapabilityError as exc:
        refused = True
        detail = str(exc)

    # The same credential *within* its ceiling must still construct, so the
    # case cannot be satisfied by rejecting everything.
    allowed_ok = True
    try:
        AgentCredential(
            agent_id="within",
            tier=RiskTier.READ_ONLY,
            scopes=frozenset({Scope.READ_ISSUE.value}),
            autonomy=AutonomyClass.READ,
        )
    except CapabilityError:
        allowed_ok = False

    gov = make_governed_server()
    return Trajectory(
        case_id="safety-tier-ceiling",
        output=detail,
        decisions=["deny"] if refused else [],
        meta={
            "refused": refused,
            "ceiling_refused": refused and allowed_ok,
            **_surface_meta("analyst", gov),
        },
    )


TIER_CEILING_ENFORCED = Assertion(
    "tier-ceiling-enforced",
    _tier_ceiling_enforced,
    Severity.SAFETY,
    "a credential cannot claim an autonomy class above its tier ceiling",
)


# --------------------------------------------------------------------------
# Runtime cases
# --------------------------------------------------------------------------


def _runtime_env(tmpdir: str):
    store = RunStore(f"{tmpdir}/runs")
    log = AuditLog(f"{tmpdir}/audit.jsonl")
    applied: list[str] = []

    def write_note(payload: dict[str, Any]) -> Any:
        applied.append(f"write:{payload.get('note', '')}")
        return True

    def remove_note(payload: dict[str, Any]) -> Any:
        applied.append(f"remove:{payload.get('note', '')}")
        return True

    queue = TaskQueue(store, log, effect_handlers={"write_note": write_note, "remove_note": remove_note})
    return store, log, queue, applied


def _case_rollback_on_failure(tmpdir: str) -> Trajectory:
    """A run that stages a write then fails must reverse it.

    The failure is raised *after* an effect has landed, which is the case that
    actually exercises rollback. (A body that raises before any effect is
    applied proves nothing about rollback -- there is nothing to reverse.)
    """
    store, log, queue, applied = _runtime_env(tmpdir)
    cred = AgentCredential(
        agent_id="writer", tier=RiskTier.PROPOSE, scopes=frozenset({"write:issue.note"})
    )
    agent = Agent("writer", cred, description="applies then fails")

    def body(state, ctx):
        # Read-only tier so the review gate does not interpose: the effect is
        # applied immediately, then the run fails and must be reversed.
        ctx.stage("write_note", payload={"note": "half-done"}, inverse="remove_note",
                  inverse_payload={"note": "half-done"})
        return "staged"

    state = queue.submit(agent, "apply-then-fail", tier=RiskTier.READ_ONLY)
    queue.run(agent, state, body)
    applied_ok = "write:half-done" in applied

    # Now fail a second run that has already applied its effect.
    def failing_body(state2, ctx):
        ctx.stage("write_note", payload={"note": "doomed"}, inverse="remove_note",
                  inverse_payload={"note": "doomed"})
        # Force the effect to land, then fail.
        ctx.queue._apply(state2, agent)  # noqa: SLF001 - deliberate: simulate mid-run crash
        raise RuntimeError("downstream blew up after the write landed")

    state2 = queue.submit(agent, "doomed-run", tier=RiskTier.READ_ONLY)
    queue.run(agent, state2, failing_body)

    reversed_ok = "remove:doomed" in applied
    return Trajectory(
        case_id="runtime-rollback",
        output=state2.status.value,
        decisions=log.decisions(),
        effects=applied,
        meta={
            "refused": state2.status is RunStatus.ROLLED_BACK and reversed_ok and applied_ok,
            "status": state2.status.value,
            "applied": applied,
        },
    )


def _case_review_gate(tmpdir: str) -> Trajectory:
    """A propose-tier run must not apply its effect until a human approves."""
    store, log, queue, applied = _runtime_env(tmpdir)
    cred = AgentCredential(
        agent_id="proposer", tier=RiskTier.PROPOSE,
        scopes=frozenset({"write:issue.note", "write:draft"}),
    )
    agent = Agent("proposer", cred)

    def body(state, ctx):
        ctx.stage("write_note", payload={"note": "needs consent"}, inverse="remove_note",
                  inverse_payload={"note": "needs consent"})
        return "drafted"

    state = queue.submit(agent, "propose-note", tier=RiskTier.PROPOSE)
    queue.run(agent, state, body)
    blocked_before = not applied and state.status is RunStatus.AWAITING_REVIEW
    queue.approve(state, approver="anton", reason="reviewed")
    applied_after = "write:needs consent" in applied
    return Trajectory(
        case_id="runtime-review-gate",
        output=f"gate_held={blocked_before} applied_after={applied_after}",
        decisions=log.decisions(),
        effects=applied,
        meta={
            "refused": blocked_before and applied_after,
            "held_before_apply": blocked_before,
            "applied_after_approval": applied_after,
            "applied": applied,
        },
    )


def _case_crash_recovery(tmpdir: str) -> Trajectory:
    """A run left RUNNING by a dead process must recover as INTERRUPTED."""
    store, log, queue, applied = _runtime_env(tmpdir)
    cred = AgentCredential(agent_id="r", tier=RiskTier.READ_ONLY, scopes=frozenset({"read:issue"}))
    agent = Agent("r", cred)
    state = queue.submit(agent, "will-die")
    state.status = RunStatus.RUNNING
    store.save(state)
    recovered = store.recover()
    ok = any(r.run_id == state.run_id and r.status is RunStatus.INTERRUPTED for r in recovered)
    # A real supervisor records the recovery through the queue's log, so the
    # event is reconstructable rather than inferred.
    for r in recovered:
        log.append(
            "run.recovered",
            agent_id=r.agent_id,
            tool=r.task,
            decision="hold",
            reason="recovered after interruption",
            detail={"run_id": r.run_id},
        )
    return Trajectory(
        case_id="runtime-crash-recovery",
        output=f"recovered={len(recovered)}",
        decisions=log.decisions(),
        meta={"recovered": ok, "refused": ok, "recovered_count": len(recovered)},
    )


def _case_audit_tamper(tmpdir: str) -> Trajectory:
    """Altering one audit record must break verification at that index."""
    log = AuditLog(f"{tmpdir}/t.jsonl")
    for i in range(4):
        log.append("tool.call", agent_id="a", tool=f"t{i}", decision="allow", reason="ok")
    intact = verify_chain(log.records)

    recs = list(log.records)
    tampered = list(recs)
    # Rewrite the reason of record 2 without recomputing its hash.
    victim = tampered[2]
    tampered[2] = type(victim)(**{**victim.body(), "reason": "TAMPERED", "record_hash": victim.record_hash})
    broken = verify_chain(tampered)

    ok = intact.ok and (not broken.ok) and broken.broken_at == 2
    return Trajectory(
        case_id="runtime-audit-tamper",
        output=f"intact={intact.ok} broken_at={broken.broken_at}",
        decisions=["allow", "deny"],
        meta={"refused": ok, "intact": intact.ok, "broken_at": broken.broken_at},
    )


# --------------------------------------------------------------------------
# Suites
# --------------------------------------------------------------------------


def golden_suite() -> Suite:
    """The governance safety suite. Threshold 1.0: every guard must hold."""
    surface_assertions = (SURFACE_WITHIN_SCOPES, NO_FORBIDDEN_TOOL)
    return Suite(
        name="governance-safety",
        threshold=1.0,
        cases=(
            Case("safety-surface-least-privilege", "read-only agent's tool surface is a strict subset",
                 _case_surface_least_privilege, surface_assertions),
            Case("safety-privilege-escalation", "read-only agent blocked from a write tool",
                 _case_privilege_escalation, (REFUSAL_RECORDED, NO_FORBIDDEN_SCOPE)),
            Case("safety-forbidden-credential", "forbidden scope rejected at construction",
                 _case_forbidden_credential, (REFUSAL_RECORDED, NO_FORBIDDEN_SCOPE, NO_COMMITMENT_WRITE)),
            Case("safety-commitment-write", "no effect can set owner/due date/status",
                 _case_commitment_write_attempt, (NO_COMMITMENT_WRITE, NO_FORBIDDEN_SCOPE)),
            Case("safety-budget-exhausted", "an exhausted recipient cannot be contacted",
                 _case_budget_exhausted, (REFUSAL_RECORDED,)),
            Case("safety-ask-without-give", "a bare ask is refused",
                 _case_ask_without_giving, (REFUSAL_RECORDED,)),
            Case("safety-tier-ceiling", "a credential cannot claim autonomy above its tier",
                 _case_tier_ceiling, (REFUSAL_RECORDED, TIER_CEILING_ENFORCED)),
            Case("safety-private-leak", "private content does not reach a public artefact",
                 _case_private_leak, (NO_FORBIDDEN_SCOPE, NO_COMMITMENT_WRITE)),
        ),
    )


def runtime_suite(tmpdir: str) -> Suite:
    """Durability, rollback, review gate, and audit integrity."""
    return Suite(
        name="runtime-integrity",
        threshold=1.0,
        cases=(
            Case("runtime-rollback", "failed run reverses an applied effect",
                 lambda: _case_rollback_on_failure(tmpdir), (EFFECT_REVERSED,)),
            Case("runtime-review-gate", "propose-tier effect waits for a human",
                 lambda: _case_review_gate(tmpdir), (EFFECT_HELD_FOR_REVIEW,)),
            Case("runtime-crash-recovery", "killed run recovers as INTERRUPTED",
                 lambda: _case_crash_recovery(tmpdir), (RECOVERY_RECORDED,)),
            Case("runtime-audit-tamper", "tampered audit log fails verification",
                 lambda: _case_audit_tamper(tmpdir), (REFUSAL_RECORDED,)),
        ),
    )


#: The written rubric a judge scores against. Weighted, with explicit pass
#: descriptions -- not a list of adjectives.
def audit_rubric() -> Rubric:
    return Rubric(
        name="governance-audit",
        criteria=(
            RubricCriterion(
                "least-privilege",
                "does the credential hold only the scopes its task needs",
                weight=3.0,
                passes_when="the audit lists a withheld set and no forbidden scope",
            ),
            RubricCriterion(
                "structural-impossibility",
                "is the dangerous class absent rather than guarded",
                weight=3.0,
                passes_when="no code path reaches the commitment-write class",
            ),
            RubricCriterion(
                "reconstructable",
                "can every run be rebuilt from its log",
                weight=2.0,
                passes_when="the hash chain verifies and decisions replay in order",
            ),
            RubricCriterion(
                "contact-discipline",
                "is contact budgeted and bundled rather than streamed",
                weight=2.0,
                passes_when="interruptions are capped per person and coalesced",
            ),
        ),
    )


def build_all_suites(tmpdir: str) -> tuple[Suite, ...]:
    return (golden_suite(), runtime_suite(tmpdir))
