"""Test suite for agent-governance.

The evals module is the *product's* evaluation layer (it ships and is what the
CI gate runs). This file is the ordinary unit suite for the library itself:
construction rules, budget arithmetic, durability, and the MCP protocol
surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_governance.audit import GENESIS_HASH, AuditLog, load_log, verify_chain
from agent_governance.budget import (
    BudgetExhausted,
    BudgetLedger,
    ContactIntent,
    render_bundle,
)
from agent_governance.capabilities import (
    AUTONOMY_IMPLEMENTED,
    AgentCredential,
    AutonomyClass,
    CapabilityError,
    FORBIDDEN_SCOPES,
    ForbiddenCapabilityError,
    RiskTier,
    Scope,
    ToolGrant,
    audit_credential,
)
from agent_governance.evals.mutants import audit_forbidden_paths, score_mutation
from agent_governance.evals.runner import run_all
from agent_governance.runtime import (
    Agent,
    RunStatus,
    RunStore,
    TaskQueue,
)
from agent_governance.server import DEFAULT_AGENTS, GovernedServer, REGISTRY, build_server


# --------------------------------------------------------------------------
# Capabilities
# --------------------------------------------------------------------------


class TestForbiddenScopes:
    @pytest.mark.parametrize("scope", sorted(FORBIDDEN_SCOPES))
    def test_every_forbidden_scope_is_rejected_at_construction(self, scope):
        with pytest.raises(ForbiddenCapabilityError):
            AgentCredential(
                agent_id="x", tier=RiskTier.CONTACT, scopes=frozenset({scope})
            )

    def test_rejection_is_specific_not_generic(self):
        """The forbidden class raises its own type, so it is separable in audits."""
        with pytest.raises(ForbiddenCapabilityError):
            AgentCredential("x", RiskTier.CONTACT, frozenset({"write:issue.owner"}))

    def test_tool_grant_cannot_declare_forbidden_scope(self):
        with pytest.raises(ForbiddenCapabilityError):
            ToolGrant("sneaky", frozenset({"write:issue.status"}))

    def test_tool_grant_cannot_declare_commitment_autonomy(self):
        with pytest.raises(ForbiddenCapabilityError):
            ToolGrant("sneaky", frozenset(), autonomy=AutonomyClass.COMMITMENT_WRITE)

    def test_commitment_class_has_no_implementation(self):
        """The claim 'impossible, not discouraged' rests on this being False."""
        assert AUTONOMY_IMPLEMENTED[AutonomyClass.COMMITMENT_WRITE] is False


class TestCredential:
    def test_requires_identity(self):
        with pytest.raises(CapabilityError):
            AgentCredential("", RiskTier.READ_ONLY)

    def test_tier_ceiling_enforced(self):
        with pytest.raises(CapabilityError):
            AgentCredential(
                "x", RiskTier.READ_ONLY, frozenset({"read:issue"}),
                autonomy=AutonomyClass.CONTACT_BUDGETED,
            )

    def test_within_ceiling_allowed(self):
        c = AgentCredential(
            "x", RiskTier.READ_ONLY, frozenset({"read:issue"}), autonomy=AutonomyClass.READ
        )
        assert c.autonomy is AutonomyClass.READ

    def test_require_fails_closed(self):
        c = AgentCredential("x", RiskTier.READ_ONLY, frozenset({"read:issue"}))
        c.require(Scope.READ_ISSUE)
        with pytest.raises(CapabilityError):
            c.require(Scope.WRITE_ISSUE_NOTE)

    def test_grants_for_filters(self):
        grants = (
            ToolGrant("r", frozenset({"read:issue"})),
            ToolGrant("w", frozenset({"write:issue.note"})),
        )
        c = AgentCredential("x", RiskTier.READ_ONLY, frozenset({"read:issue"}))
        assert [g.name for g in c.grants_for(grants)] == ["r"]

    def test_audit_reports_withheld_and_never_forbidden(self):
        grants = (
            ToolGrant("r", frozenset({"read:issue"})),
            ToolGrant("w", frozenset({"write:issue.note"})),
        )
        c = AgentCredential("x", RiskTier.READ_ONLY, frozenset({"read:issue"}))
        a = audit_credential(c, grants)
        assert a.withheld_tools == ("w",)
        assert a.can_write_commitments is False
        assert a.clean is True


# --------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------


class TestContactIntent:
    def test_ask_without_give_is_rejected(self):
        with pytest.raises(Exception):
            ContactIntent(recipient="p", asks="do this")

    def test_inform_is_allowed(self):
        i = ContactIntent(recipient="p", gives="here is a summary")
        assert i.kind.value == "inform"

    def test_ask_with_give_is_allowed(self):
        i = ContactIntent(recipient="p", gives="summary", asks="approve")
        assert i.kind.value == "ask"


class TestBudgetLedger:
    def test_hard_ceiling(self):
        # window 0 = bundling off: the second intent closes the first window.
        led = BudgetLedger(default_budget=1, window_seconds=0.0)
        assert led.submit(ContactIntent(recipient="p", gives="a")) is None
        msg = led.submit(ContactIntent(recipient="p", gives="b"))
        assert msg is not None and msg.item_count == 1
        led.release(msg)
        assert led.spent("p") == 1
        with pytest.raises(BudgetExhausted):
            led.submit(ContactIntent(recipient="p", gives="c"))

    def test_bundling_coalesces(self):
        led = BudgetLedger(default_budget=5, window_seconds=3600)
        for i in range(4):
            assert led.submit(ContactIntent(recipient="p", gives=f"g{i}")) is None
        msgs = led.flush_and_release()
        assert len(msgs) == 1
        assert msgs[0].item_count == 4
        assert led.spent("p") == 1

    def test_bundle_gives_before_asking(self):
        body = render_bundle(
            [
                ContactIntent(recipient="p", gives="useful summary", asks="approve PR"),
            ]
        )
        assert body.index("What you get") < body.index("What I need")

    def test_savings_reports_compression(self):
        led = BudgetLedger(default_budget=9, window_seconds=3600)
        for i in range(6):
            led.submit(ContactIntent(recipient="p", gives=f"g{i}"))
        led.flush_and_release()
        s = led.savings()
        assert s["compression"] == 6.0
        assert s["avoided"] == 5.0

    def test_report_accounts_per_person(self):
        led = BudgetLedger(default_budget=2, window_seconds=3600)
        led.submit(ContactIntent(recipient="p", gives="x"))
        led.flush_and_release()
        r = led.report()["p"]
        assert r["budget"] == 2 and r["spent"] == 1 and r["remaining"] == 1


# --------------------------------------------------------------------------
# Audit log
# --------------------------------------------------------------------------


class TestAuditLog:
    def test_chain_verifies(self):
        log = AuditLog()
        for i in range(5):
            log.append("e", agent_id="a", tool=f"t{i}", decision="allow")
        assert log.verify().ok
        assert len(log) == 5

    def test_genesis_hash_is_not_empty(self):
        log = AuditLog()
        r = log.append("e", agent_id="a", tool="t", decision="allow")
        assert r.prev_hash == GENESIS_HASH and len(GENESIS_HASH) == 64

    def test_tamper_detected_with_index(self):
        log = AuditLog()
        for i in range(4):
            log.append("e", agent_id="a", tool=f"t{i}", decision="allow")
        recs = list(log.records)
        v = recs[2]
        recs[2] = type(v)(**{**v.body(), "reason": "changed", "record_hash": v.record_hash})
        ver = verify_chain(recs)
        assert not ver.ok and ver.broken_at == 2

    def test_removal_detected(self):
        log = AuditLog()
        for i in range(4):
            log.append("e", agent_id="a", tool=f"t{i}", decision="allow")
        recs = list(log.records)
        del recs[1]
        ver = verify_chain(recs)
        assert not ver.ok

    def test_roundtrip_through_disk(self, tmp_path: Path):
        p = tmp_path / "audit.jsonl"
        log = AuditLog(p)
        for i in range(3):
            log.append("e", agent_id="a", tool=f"t{i}", decision="allow", reason="r")
        assert p.exists()
        reloaded = load_log(p)
        assert len(reloaded) == 3
        assert verify_chain(reloaded).ok

    def test_reconstruct_filters_by_run(self):
        log = AuditLog(run_id="run-1")
        log.append("e", agent_id="a", tool="t", decision="allow")
        assert len(log.reconstruct("run-1")) == 1
        assert log.reconstruct("nope") == []

    def test_denials_are_queryable(self):
        log = AuditLog()
        log.append("e", agent_id="a", tool="t", decision="allow")
        log.append("e", agent_id="a", tool="t", decision="deny", reason="no scope")
        assert len(log.denials()) == 1
        assert log.counts_by_decision() == {"allow": 1, "deny": 1}


# --------------------------------------------------------------------------
# Runtime
# --------------------------------------------------------------------------


def _env(tmp_path: Path):
    store = RunStore(tmp_path / "runs")
    log = AuditLog(tmp_path / "audit.jsonl")
    applied: list[str] = []
    effects = {
        "write_note": lambda p: applied.append(f"write:{p.get('note')}"),
        "remove_note": lambda p: applied.append(f"remove:{p.get('note')}"),
    }
    return store, log, TaskQueue(store, log, effect_handlers=effects), applied


class TestRuntime:
    def test_durable_state_survives_reload(self, tmp_path):
        store, log, queue, _ = _env(tmp_path)
        cred = AgentCredential("a", RiskTier.READ_ONLY, frozenset({"read:issue"}))
        agent = Agent("a", cred)
        st = queue.submit(agent, "task")
        st2 = store.load(st.run_id)
        assert st2.run_id == st.run_id and st2.task == "task"

    def test_recover_marks_interrupted(self, tmp_path):
        store, log, queue, _ = _env(tmp_path)
        cred = AgentCredential("a", RiskTier.READ_ONLY, frozenset({"read:issue"}))
        agent = Agent("a", cred)
        st = queue.submit(agent, "t")
        st.status = RunStatus.RUNNING
        store.save(st)
        rec = store.recover()
        assert any(r.status is RunStatus.INTERRUPTED for r in rec)

    def test_review_gate_holds_then_applies(self, tmp_path):
        store, log, queue, applied = _env(tmp_path)
        cred = AgentCredential("a", RiskTier.PROPOSE, frozenset({"write:issue.note"}))
        agent = Agent("a", cred)

        def body(state, ctx):
            ctx.stage("write_note", payload={"note": "n"}, inverse="remove_note",
                      inverse_payload={"note": "n"})
            return "ok"

        st = queue.submit(agent, "t", tier=RiskTier.PROPOSE)
        queue.run(agent, st, body)
        assert st.status is RunStatus.AWAITING_REVIEW
        assert applied == []
        queue.approve(st, approver="anton")
        assert "write:n" in applied

    def test_reject_rolls_back(self, tmp_path):
        store, log, queue, applied = _env(tmp_path)
        cred = AgentCredential("a", RiskTier.PROPOSE, frozenset({"write:issue.note"}))
        agent = Agent("a", cred)

        def body(state, ctx):
            ctx.stage("write_note", payload={"note": "n"}, inverse="remove_note",
                      inverse_payload={"note": "n"})
            return "ok"

        st = queue.submit(agent, "t", tier=RiskTier.PROPOSE)
        queue.run(agent, st, body)
        queue.reject(st, approver="anton", reason="not now")
        assert st.status is RunStatus.ROLLED_BACK

    def test_approval_needs_a_named_approver(self, tmp_path):
        store, log, queue, _ = _env(tmp_path)
        cred = AgentCredential("a", RiskTier.PROPOSE, frozenset({"write:issue.note"}))
        agent = Agent("a", cred)
        st = queue.submit(agent, "t", tier=RiskTier.PROPOSE)
        queue.run(agent, st, lambda s, c: "ok")
        with pytest.raises(Exception):
            queue.approve(st, approver="  ")

    def test_unregistered_effect_is_a_recorded_failure(self, tmp_path):
        store, log, queue, applied = _env(tmp_path)
        cred = AgentCredential("a", RiskTier.READ_ONLY, frozenset({"read:issue"}))
        agent = Agent("a", cred)

        def body(state, ctx):
            ctx.stage("unregistered_thing", payload={})
            return "ok"

        st = queue.submit(agent, "t", tier=RiskTier.READ_ONLY)
        queue.run(agent, st, body)
        # The effect does not land, the run is recorded as a failure, and the
        # state is durable rather than stuck in RUNNING.
        assert applied == []
        assert st.status is RunStatus.ROLLED_BACK
        assert "fail" in log.decisions()
        assert store.load(st.run_id).status is RunStatus.ROLLED_BACK

    def test_credential_not_transferable(self):
        cred = AgentCredential("a", RiskTier.READ_ONLY, frozenset({"read:issue"}))
        with pytest.raises(Exception):
            Agent("b", cred)

    def test_rollback_actually_reverses(self, tmp_path):
        store, log, queue, applied = _env(tmp_path)
        cred = AgentCredential("a", RiskTier.READ_ONLY, frozenset({"read:issue"}))
        agent = Agent("a", cred)

        def body(state, ctx):
            ctx.stage("write_note", payload={"note": "z"}, inverse="remove_note",
                      inverse_payload={"note": "z"})
            ctx.queue._apply(state, agent)
            raise RuntimeError("boom")

        st = queue.submit(agent, "t", tier=RiskTier.READ_ONLY)
        queue.run(agent, st, body)
        assert "write:z" in applied and "remove:z" in applied
        assert st.status is RunStatus.ROLLED_BACK


# --------------------------------------------------------------------------
# Governed server surface
# --------------------------------------------------------------------------


class TestGovernedServer:
    def test_unknown_agent_gets_nothing(self):
        gov = GovernedServer()
        with pytest.raises(CapabilityError):
            gov.credential("nobody")

    def test_surface_is_a_strict_subset(self):
        gov = GovernedServer()
        assert len(gov.tools_for("analyst")) < len(gov.registry)
        assert gov.withheld_for("analyst")

    def test_denied_call_is_audited(self):
        gov = GovernedServer()
        with pytest.raises(CapabilityError):
            gov.call("analyst", "propose_issue_note", {"note": "x"})
        assert gov.log.counts_by_decision()["deny"] == 1

    def test_unknown_tool_is_audited(self):
        gov = GovernedServer()
        with pytest.raises(CapabilityError):
            gov.call("analyst", "delete_everything", {})
        assert "deny" in gov.log.decisions()

    def test_propose_reports_it_cannot_set_commitments(self):
        gov = GovernedServer()
        r = gov.call("reporter", "propose_issue_note", {"note": "x"})
        assert r["can_set_owner"] is False
        assert r["can_set_due_date"] is False
        assert r["can_set_status"] is False
        assert r["requires_human_consent"] is True

    def test_contact_is_budgeted_and_bundled(self):
        gov = GovernedServer(budget=BudgetLedger(default_budget=3, window_seconds=3600))
        r1 = gov.call("notifier", "contact_person", {"recipient": "d", "gives": "g1"})
        r2 = gov.call("notifier", "contact_person", {"recipient": "d", "gives": "g2"})
        # Two intents inside one window cost no interruption yet.
        assert r1.get("delivered") is None and r2.get("delivered") is None
        assert gov.budget.spent("d") == 0
        # Flushing the window delivers both as ONE message and spends ONE.
        r3 = gov.call("notifier", "contact_person",
                      {"recipient": "d", "gives": "g3", "flush": True})
        assert r3.get("delivered") is True
        assert r3["bundled_items"] == 3
        assert gov.budget.spent("d") == 1

    def test_budget_refusal_is_audited(self):
        gov = GovernedServer(budget=BudgetLedger(default_budget=0, window_seconds=1))
        with pytest.raises(BudgetExhausted):
            gov.call("notifier", "contact_person", {"recipient": "d", "gives": "g"})
        assert "deny" in gov.log.decisions()

    def test_ask_without_give_refused_on_contact(self):
        gov = GovernedServer()
        with pytest.raises(Exception):
            gov.call("notifier", "contact_person", {"recipient": "d", "asks": "do it"})


# --------------------------------------------------------------------------
# Package-level: the eval gate and mutation testing
# --------------------------------------------------------------------------


class TestEvals:
    def test_gate_passes(self, tmp_path):
        rep = run_all(str(tmp_path), include_meta=False)
        assert rep["gate"]["ok"] is True
        assert rep["totals"]["cases"] >= 12

    def test_all_mutants_killed(self):
        ms = score_mutation()
        assert ms.score == 1.0, ms.summary()

    def test_forbidden_path_closed(self):
        a = audit_forbidden_paths(REGISTRY)
        assert a.ok and a.reachable_tools == ()

    def test_every_credential_is_audit_clean(self):
        gov = GovernedServer(DEFAULT_AGENTS)
        for aid in DEFAULT_AGENTS:
            au = gov.audit(aid)
            assert au["can_write_commitments"] is False
            assert not (set(au["held_scopes"]) & set(au["forbidden_scopes"]))


# --------------------------------------------------------------------------
# MCP protocol surface
# --------------------------------------------------------------------------


class TestMCPProtocol:
    """End-to-end over the real SDK transport, not a mock."""

    @pytest.mark.asyncio
    async def test_tools_list_matches_the_credential(self):
        from mcp.client._memory import InMemoryTransport
        from mcp.client.session import ClientSession

        gov = GovernedServer(DEFAULT_AGENTS)
        server, _ = build_server(gov, "analyst")
        async with InMemoryTransport(server) as (r, w):
            async with ClientSession(r, w) as cs:
                await cs.initialize()
                listed = {t.name for t in (await cs.list_tools()).tools}
        expected = {g.name for g in gov.tools_for("analyst")}
        assert listed == expected
        assert "propose_issue_note" not in listed

    @pytest.mark.asyncio
    async def test_withheld_tool_is_not_callable(self):
        from mcp.client._memory import InMemoryTransport
        from mcp.client.session import ClientSession

        gov = GovernedServer(DEFAULT_AGENTS)
        server, _ = build_server(gov, "analyst")
        async with InMemoryTransport(server) as (r, w):
            async with ClientSession(r, w) as cs:
                await cs.initialize()
                res = await cs.call_tool("propose_issue_note", {"note": "x"})
        # The tool is not registered, so the protocol reports an error result.
        assert res.is_error is True

    @pytest.mark.asyncio
    async def test_allowed_tool_round_trips(self):
        from mcp.client._memory import InMemoryTransport
        from mcp.client.session import ClientSession

        gov = GovernedServer(DEFAULT_AGENTS)
        server, _ = build_server(gov, "reporter")
        async with InMemoryTransport(server) as (r, w):
            async with ClientSession(r, w) as cs:
                await cs.initialize()
                res = await cs.call_tool("read_issue", {"issue_id": "ISSUE-9"})
        assert res.is_error is False
        payload = json.loads(res.content[0].text)
        assert payload["issue_id"] == "ISSUE-9"

    @pytest.mark.asyncio
    async def test_protocol_call_is_audited(self):
        from mcp.client._memory import InMemoryTransport
        from mcp.client.session import ClientSession

        gov = GovernedServer(DEFAULT_AGENTS)
        server, _ = build_server(gov, "reporter")
        async with InMemoryTransport(server) as (r, w):
            async with ClientSession(r, w) as cs:
                await cs.initialize()
                await cs.call_tool("read_issue", {"issue_id": "ISSUE-1"})
        assert gov.log.counts_by_decision().get("allow", 0) >= 1
