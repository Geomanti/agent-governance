"""MCP server exposing the governed tool surface over stdio.

This is the artefact the job description's "must have" names directly:
*"have done deep API work... and have authored MCP servers."* It is built on
the official SDK (``mcp>=2.0``), with the governance model wired into the
server surface rather than layered on top of it.

The design point is that **the tool list is per-credential**. Two agents
connecting to the same process see different tools, because
``MCPServer.list_tools`` reflects the credential's scopes. An agent without
``write:issue.note`` does not get a disabled write tool -- it gets no write
tool, and a call to that name is refused and recorded. That is the
"impossible, not discouraged" claim expressed in the protocol surface itself.

Run it:

    python -m agent_governance.server              # stdio, default analyst agent
    python -m agent_governance.server --audit      # print the credential audit
    python -m agent_governance.server --list-tools # print the reachable surface
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .audit import AuditLog, verify_chain
from .budget import BudgetExhausted, BudgetLedger, ContactIntent
from .capabilities import (
    AgentCredential,
    CapabilityError,
    RiskTier,
    Scope,
    ToolGrant,
    audit_credential,
)
from .runtime import Agent, RunStore, RunStatus, TaskQueue

__all__ = [
    "REGISTRY",
    "GovernedServer",
    "build_server",
    "DEFAULT_AGENTS",
    "main",
]


# --------------------------------------------------------------------------
# The tool registry
# --------------------------------------------------------------------------

#: Every tool the platform knows how to expose, with the scopes it needs.
REGISTRY: tuple[ToolGrant, ...] = (
    ToolGrant(
        "read_issue",
        frozenset({Scope.READ_ISSUE.value}),
        description="Read one issue from the tracker.",
    ),
    ToolGrant(
        "read_digest",
        frozenset({Scope.READ_DIGEST.value}),
        description="Read the current team digest.",
    ),
    ToolGrant(
        "read_private_channel",
        frozenset({Scope.READ_PRIVATE.value}),
        description="Read a private channel. Its content may not be quoted publicly.",
    ),
    ToolGrant(
        "draft_update",
        frozenset({Scope.WRITE_DRAFT.value}),
        description="Draft a status update. Produces text; writes nothing to a system of record.",
    ),
    ToolGrant(
        "propose_issue_note",
        frozenset({Scope.WRITE_DRAFT.value, Scope.WRITE_ISSUE_NOTE.value}),
        description=(
            "Propose an issue note. Applies only after a human consents; "
            "cannot set an owner, due date or status."
        ),
    ),
    ToolGrant(
        "contact_person",
        frozenset({Scope.CONTACT_INTERNAL.value}),
        description="Send one bundled internal message, subject to the interruption budget.",
    ),
    ToolGrant(
        "read_ledger",
        frozenset({Scope.LEDGER_READ.value}),
        description="Read the ledger of work absorbed per agent.",
    ),
    ToolGrant(
        "read_audit_log",
        frozenset({Scope.AUDIT_READ.value}),
        description="Read this run's decision audit log.",
    ),
)

_BY_NAME: Mapping[str, ToolGrant] = {g.name: g for g in REGISTRY}


#: Three registered agents, each with a least-privilege credential. The
#: read-only one exists to demonstrate the withheld surface.
DEFAULT_AGENTS: Mapping[str, AgentCredential] = {
    "analyst": AgentCredential(
        agent_id="analyst",
        tier=RiskTier.READ_ONLY,
        scopes=frozenset(
            {
                Scope.READ_ISSUE.value,
                Scope.READ_DIGEST.value,
                Scope.LEDGER_READ.value,
            }
        ),
        owner="platform",
    ),
    "reporter": AgentCredential(
        agent_id="reporter",
        tier=RiskTier.PROPOSE,
        scopes=frozenset(
            {
                Scope.READ_ISSUE.value,
                Scope.READ_DIGEST.value,
                Scope.WRITE_DRAFT.value,
                Scope.WRITE_ISSUE_NOTE.value,
                Scope.LEDGER_READ.value,
                Scope.AUDIT_READ.value,
            }
        ),
        owner="platform",
    ),
    "notifier": AgentCredential(
        agent_id="notifier",
        tier=RiskTier.CONTACT,
        scopes=frozenset(
            {
                Scope.READ_ISSUE.value,
                Scope.READ_DIGEST.value,
                Scope.WRITE_DRAFT.value,
                Scope.CONTACT_INTERNAL.value,
                Scope.AUDIT_READ.value,
            }
        ),
        owner="platform",
    ),
}


# --------------------------------------------------------------------------
# Governed in-process surface
# --------------------------------------------------------------------------


@dataclass
class _Effect:
    """A recorded side effect, so a test can assert what actually landed."""

    tool: str
    payload: Mapping[str, Any]
    agent_id: str


class GovernedServer:
    """The governance core, independent of the MCP transport.

    Splitting this out is deliberate: the guarantees are testable without a
    transport, and the MCP layer is a thin adapter. A reviewer can therefore
    check the governance claim against a plain Python object.
    """

    def __init__(
        self,
        credentials: Mapping[str, AgentCredential] | None = None,
        *,
        registry: Sequence[ToolGrant] = REGISTRY,
        log_path: str | None = None,
        budget: BudgetLedger | None = None,
    ) -> None:
        self.registry = tuple(registry)
        self.credentials = dict(credentials or DEFAULT_AGENTS)
        self.log = AuditLog(log_path)
        self.budget = budget or BudgetLedger(default_budget=2, window_seconds=3600.0)
        self.effects: list[_Effect] = []
        self._sources: dict[str, str] = {}

    # -- surface ------------------------------------------------------------

    def tools_for(self, agent_id: str) -> tuple[ToolGrant, ...]:
        """Only the tools this credential may reach. The withheld set is absent."""
        cred = self.credential(agent_id)
        return cred.grants_for(self.registry)

    def withheld_for(self, agent_id: str) -> tuple[str, ...]:
        """Named for the audit: what this agent cannot reach."""
        reachable = {g.name for g in self.tools_for(agent_id)}
        return tuple(sorted({g.name for g in self.registry} - reachable))

    def credential(self, agent_id: str) -> AgentCredential:
        try:
            return self.credentials[agent_id]
        except KeyError:
            raise CapabilityError(f"unknown agent {agent_id!r}: nothing is granted by default")

    # -- invocation ---------------------------------------------------------

    def call(self, agent_id: str, tool: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Invoke a tool under governance. Every call is audited either way."""
        args = dict(arguments or {})
        cred = self.credential(agent_id)

        grant = _BY_NAME.get(tool)
        if grant is None:
            self._record(cred, tool, "deny", "unknown tool")
            raise CapabilityError(f"unknown tool {tool!r}")

        # Layer 2: the tool must be on this credential's surface.
        if not cred.permits_tool(grant):
            need = sorted(grant.scopes - cred.scopes)
            self._record(cred, tool, "deny", f"missing scope(s) {need}", scopes=need)
            raise CapabilityError(
                f"agent {agent_id!r} cannot call {tool!r}: missing {need}"
            )

        # Layer 3: re-check each required scope at the call site.
        for s in sorted(grant.scopes):
            try:
                cred.require(s)
            except CapabilityError as exc:
                self._record(cred, tool, "deny", str(exc))
                raise

        if tool == "contact_person":
            return self._contact(cred, args)

        handler = getattr(self, f"_do_{tool}", None)
        if handler is None:
            self._record(cred, tool, "deny", "no handler registered")
            raise CapabilityError(f"tool {tool!r} has no registered handler")

        result = handler(cred, args)
        self._record(cred, tool, "allow", "scope check passed")
        return result

    # -- handlers -----------------------------------------------------------

    def _do_read_issue(self, cred: AgentCredential, args: Mapping[str, Any]) -> dict[str, Any]:
        issue_id = str(args.get("issue_id", "ISSUE-1"))
        return {
            "issue_id": issue_id,
            "title": f"Sample issue {issue_id}",
            "status": "open",
            "owner": "someone",
            "due_date": None,
        }

    def _do_read_digest(self, cred: AgentCredential, args: Mapping[str, Any]) -> dict[str, Any]:
        return {"digest": ["3 issues open", "1 PR awaiting review"], "team": args.get("team", "core")}

    def _do_read_private_channel(self, cred: AgentCredential, args: Mapping[str, Any]) -> dict[str, Any]:
        text = str(args.get("text", ""))
        self._sources[cred.agent_id] = text
        self.log.append(
            "read.private",
            agent_id=cred.agent_id,
            tool="read_private_channel",
            decision="allow",
            reason="private content read; may not be quoted publicly",
            tier=cred.tier.name,
            detail={"chars": len(text)},
        )
        return {"private": text, "classification": "private"}

    def _do_draft_update(self, cred: AgentCredential, args: Mapping[str, Any]) -> dict[str, Any]:
        return {"draft": str(args.get("text", "")), "written_to_record": False}

    def _do_propose_issue_note(self, cred: AgentCredential, args: Mapping[str, Any]) -> dict[str, Any]:
        """Stage a proposal. It is recorded, not applied, until human consent."""
        note = str(args.get("note", ""))
        self.effects.append(_Effect("propose_issue_note", {"note": note}, cred.agent_id))
        return {
            "proposed": note,
            "requires_human_consent": True,
            "can_set_owner": False,
            "can_set_due_date": False,
            "can_set_status": False,
        }

    def _do_read_ledger(self, cred: AgentCredential, args: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "entries": [
                {"agent": "reporter", "work_absorbed": "drafted 4 updates"},
                {"agent": "analyst", "work_absorbed": "summarised 12 issues"},
            ],
            "monthly_figure_computed": True,
        }

    def _do_read_audit_log(self, cred: AgentCredential, args: Mapping[str, Any]) -> dict[str, Any]:
        v = verify_chain(self.log.records)
        return {
            "records": len(self.log),
            "chain_ok": v.ok,
            "decisions": self.log.counts_by_decision(),
        }

    def _contact(self, cred: AgentCredential, args: Mapping[str, Any]) -> dict[str, Any]:
        """The budgeted contact path: bundling, ceiling, and value-before-ask.

        Delivery happens on window close, or immediately when the caller asks
        for it. The distinction is the whole point of bundling: an agent that
        pings six times inside one window spends **one** interruption, not six,
        which is only true if the window is allowed to accumulate. Flushing on
        every call would defeat it.
        """
        recipient = str(args.get("recipient", "someone"))
        intent = ContactIntent(
            recipient=recipient,
            summary=str(args.get("summary", "")),
            gives=str(args.get("gives", "")),
            asks=str(args.get("asks", "")),
        )
        try:
            # Returns the previous window's bundle when that window closes.
            closed = self.budget.submit(intent)
        except BudgetExhausted as exc:
            self._record(cred, "contact_person", "deny", str(exc))
            raise

        message = closed
        if message is None and args.get("flush"):
            message = self.budget.window.flush(recipient)

        out: dict[str, Any] = {
            "recipient": recipient,
            "queued": True,
            "remaining_budget": self.budget.remaining(recipient),
        }
        if message is not None:
            self.budget.release(message)
            self.effects.append(
                _Effect("contact_person", {"body": message.body}, cred.agent_id)
            )
            out.update(
                {
                    "delivered": True,
                    "bundled_items": message.item_count,
                    "body": message.body,
                    "remaining_budget": self.budget.remaining(recipient),
                }
            )
        self._record(cred, "contact_person", "allow", "within interruption budget")
        return out

    # -- helpers ------------------------------------------------------------

    def _record(
        self,
        cred: AgentCredential,
        tool: str,
        decision: str,
        reason: str,
        scopes: Sequence[str] = (),
    ) -> None:
        self.log.append(
            "tool.call",
            agent_id=cred.agent_id,
            tool=tool,
            decision=decision,
            reason=reason,
            tier=cred.tier.name,
            scopes=tuple(scopes),
        )

    def audit(self, agent_id: str) -> dict[str, Any]:
        a = audit_credential(self.credential(agent_id), self.registry)
        return {
            "agent_id": a.agent_id,
            "tier": a.tier,
            "autonomy_ceiling": a.autonomy_ceiling,
            "held_scopes": list(a.held_scopes),
            "denied_scopes": list(a.denied_scopes),
            "forbidden_scopes": list(a.forbidden_scopes),
            "registered_tools": list(a.registered_tools),
            "withheld_tools": list(a.withheld_tools),
            "can_write_commitments": a.can_write_commitments,
        }


# --------------------------------------------------------------------------
# MCP adapter
# --------------------------------------------------------------------------


def build_server(governed: GovernedServer | None = None, agent_id: str = "reporter"):
    """Build an MCPServer whose tool list reflects one credential.

    The returned server exposes only the tools ``agent_id`` may reach. Calling
    a withheld tool is refused and audited rather than returning an error the
    caller could retry into.
    """
    from mcp.server.mcpserver import MCPServer

    gov = governed or GovernedServer()
    cred = gov.credential(agent_id)
    reachable = {g.name for g in gov.tools_for(agent_id)}

    server = MCPServer(
        name="agent-governance",
        version="0.1.0",
        instructions=(
            f"Governed tool surface for agent {agent_id!r} "
            f"(tier {cred.tier.name}). Tools outside this credential's scopes are "
            "not registered and cannot be called. The commitment-write class has "
            "no code path."
        ),
    )

    def _register(name: str):
        def deco(fn):
            if name in reachable:
                server.tool(name=name, description=_BY_NAME[name].description)(fn)
            return fn

        return deco

    @_register("read_issue")
    def read_issue(issue_id: str = "ISSUE-1") -> dict[str, Any]:
        """Read one issue from the tracker."""
        return gov.call(agent_id, "read_issue", {"issue_id": issue_id})

    @_register("read_digest")
    def read_digest(team: str = "core") -> dict[str, Any]:
        """Read the current team digest."""
        return gov.call(agent_id, "read_digest", {"team": team})

    @_register("read_private_channel")
    def read_private_channel(text: str = "") -> dict[str, Any]:
        """Read a private channel (content may not be quoted publicly)."""
        return gov.call(agent_id, "read_private_channel", {"text": text})

    @_register("draft_update")
    def draft_update(text: str = "") -> dict[str, Any]:
        """Draft a status update. Writes nothing to a system of record."""
        return gov.call(agent_id, "draft_update", {"text": text})

    @_register("propose_issue_note")
    def propose_issue_note(note: str = "") -> dict[str, Any]:
        """Propose an issue note; applies only after a human consents."""
        return gov.call(agent_id, "propose_issue_note", {"note": note})

    @_register("contact_person")
    def contact_person(
        recipient: str = "someone", gives: str = "", asks: str = "", summary: str = ""
    ) -> dict[str, Any]:
        """Send one bundled internal message within the interruption budget."""
        return gov.call(
            agent_id,
            "contact_person",
            {"recipient": recipient, "gives": gives, "asks": asks, "summary": summary},
        )

    @_register("read_ledger")
    def read_ledger() -> dict[str, Any]:
        """Read the ledger of work absorbed per agent."""
        return gov.call(agent_id, "read_ledger", {})

    @_register("read_audit_log")
    def read_audit_log() -> dict[str, Any]:
        """Read this run's decision audit log."""
        return gov.call(agent_id, "read_audit_log", {})

    return server, gov


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-governance", description=__doc__)
    parser.add_argument("--agent", default="reporter", choices=sorted(DEFAULT_AGENTS))
    parser.add_argument("--audit", action="store_true", help="print the credential audit and exit")
    parser.add_argument("--list-tools", action="store_true", help="print the reachable surface and exit")
    parser.add_argument("--log", default=None, help="path to the audit log")
    args = parser.parse_args(argv)

    gov = GovernedServer(log_path=args.log)

    if args.audit:
        print(json.dumps(gov.audit(args.agent), indent=2))
        return 0

    if args.list_tools:
        for g in gov.tools_for(args.agent):
            print(f"{g.name:24} scopes={sorted(g.scopes)}")
        withheld = gov.withheld_for(args.agent)
        if withheld:
            print(f"\nwithheld ({len(withheld)}): {', '.join(withheld)}")
        return 0

    server, _ = build_server(gov, args.agent)
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
