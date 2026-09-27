"""Capability model: deny-by-default, scope-checked, forbidden-class structural.

The central claim of this module is the one the target job description makes:
a dangerous action should be *impossible* rather than *discouraged*. That is
enforced in three layers, in increasing order of strength:

1. **Construction-time rejection.** A credential that requests a scope in
   ``FORBIDDEN_SCOPES`` cannot be instantiated at all. The forbidden class is
   rejected where the credential is built, so no amount of clever input,
   prompt injection or model error can produce a credential that holds it.

2. **Registration-time filtering.** Tools whose scopes exceed a credential are
   never registered on that agent's MCP surface. The code path is absent from
   the reachable call graph rather than guarded inside it.

3. **Call-time re-check (defence in depth).** Even a direct in-process call to
   a handler re-verifies the scope, so the guarantee does not silently depend
   on the registration layer remaining the only entry point.

Layer 1 makes the class impossible; layers 2 and 3 keep it impossible under
refactoring. ``audit_credential`` reports what a credential can and cannot do,
so the guarantee is inspectable rather than asserted in prose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Iterable, Mapping, Sequence

__all__ = [
    "CapabilityError",
    "ForbiddenCapabilityError",
    "Scope",
    "FORBIDDEN_SCOPES",
    "AutonomyClass",
    "RiskTier",
    "ToolGrant",
    "AgentCredential",
    "CredentialAudit",
    "audit_credential",
]


class CapabilityError(Exception):
    """Raised when an operation exceeds the credential's granted scopes."""


class ForbiddenCapabilityError(CapabilityError):
    """Raised when a credential requests a scope from the forbidden class.

    This is deliberately a subclass of ``CapabilityError`` so callers that only
    handle the general case still fail closed, but it is separable so tests and
    audits can assert that the *construction-time* rejection fired rather than a
    later runtime check.
    """


# --------------------------------------------------------------------------
# Scopes
# --------------------------------------------------------------------------

#: Scopes that are permanently off limits. The rule the job description states
#: is that an agent may never autonomously write a *commitment* -- an owner, a
#: due date, a status -- into a shared work system. That is expressed here as a
#: scope no credential may ever hold.
FORBIDDEN_SCOPES: frozenset[str] = frozenset(
    {
        "write:commitment",
        "write:issue.owner",
        "write:issue.due_date",
        "write:issue.status",
    }
)


class Scope(str, Enum):
    """Known scopes, as ``verb:object`` strings.

    Read scopes are cheap and broadly granted. Write scopes are narrow and
    per-tool. Contact scopes are additionally budgeted at runtime.
    """

    READ_ISSUE = "read:issue"
    READ_DIGEST = "read:digest"
    READ_PRIVATE = "read:private"
    WRITE_DRAFT = "write:draft"
    WRITE_ISSUE_NOTE = "write:issue.note"
    CONTACT_INTERNAL = "contact:internal"
    CONTACT_EXTERNAL = "contact:external"
    LEDGER_READ = "ledger:read"
    AUDIT_READ = "audit:read"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


# --------------------------------------------------------------------------
# Autonomy classes and risk tiers
# --------------------------------------------------------------------------


class AutonomyClass(str, Enum):
    """The classes of action an agent may be trusted with.

    The first three map to the three things the job description says an agent
    *may* do. The fourth is the class it may never do, and it is present here
    only so that the registry can record it as unimplemented and an audit can
    prove it has no code path.
    """

    READ = "read"
    PROPOSE_ON_CONSENT = "propose-on-consent"
    CONTACT_BUDGETED = "contact-budgeted"
    COMMITMENT_WRITE = "commitment-write"


#: Whether each autonomy class has an implementing code path in this server.
#: ``COMMITMENT_WRITE`` is ``False`` by construction, and
#: ``audit_forbidden_paths`` in ``evals.audit`` asserts it stays that way.
AUTONOMY_IMPLEMENTED: Mapping[AutonomyClass, bool] = {
    AutonomyClass.READ: True,
    AutonomyClass.PROPOSE_ON_CONSENT: True,
    AutonomyClass.CONTACT_BUDGETED: True,
    AutonomyClass.COMMITMENT_WRITE: False,
}


class RiskTier(IntEnum):
    """Governance tier. Review depth and autonomy scale with the tier."""

    READ_ONLY = 0
    PROPOSE = 1
    CONTACT = 2


#: Highest autonomy class each tier is permitted to reach.
TIER_AUTONOMY: Mapping[RiskTier, AutonomyClass] = {
    RiskTier.READ_ONLY: AutonomyClass.READ,
    RiskTier.PROPOSE: AutonomyClass.PROPOSE_ON_CONSENT,
    RiskTier.CONTACT: AutonomyClass.CONTACT_BUDGETED,
}


# --------------------------------------------------------------------------
# Grants and credentials
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolGrant:
    """A named tool and the scopes it requires to exist.

    ``autonomy`` records which autonomy class the tool belongs to, so the
    registry can be audited class by class rather than only scope by scope.
    """

    name: str
    scopes: frozenset[str] = field(default_factory=frozenset)
    autonomy: AutonomyClass = AutonomyClass.READ
    description: str = ""

    def __post_init__(self) -> None:
        bad = self.scopes & FORBIDDEN_SCOPES
        if bad:
            raise ForbiddenCapabilityError(
                f"tool {self.name!r} declares forbidden scope(s) {sorted(bad)}"
            )
        if self.autonomy is AutonomyClass.COMMITMENT_WRITE:
            raise ForbiddenCapabilityError(
                f"tool {self.name!r} declares the commitment-write autonomy class, "
                "which has no implementing code path by design"
            )

    @property
    def is_read_only(self) -> bool:
        return all(s.startswith("read:") for s in self.scopes) or not self.scopes


@dataclass(frozen=True)
class AgentCredential:
    """An immutable least-privilege credential for one agent.

    Construction fails closed: any requested scope in ``FORBIDDEN_SCOPES``
    raises ``ForbiddenCapabilityError`` here, at build time, before the agent
    exists. That is layer 1 of the guarantee.
    """

    agent_id: str
    tier: RiskTier
    scopes: frozenset[str] = field(default_factory=frozenset)
    owner: str = "unassigned"
    autonomy: AutonomyClass | None = None

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise CapabilityError("agent_id must be set: every agent needs a registered identity")

        forbidden = self.scopes & FORBIDDEN_SCOPES
        if forbidden:
            raise ForbiddenCapabilityError(
                f"credential for {self.agent_id!r} requests forbidden scope(s) "
                f"{sorted(forbidden)}; the commitment-write class has no code path"
            )

        # A credential may not reach past the autonomy class its tier allows.
        if self.autonomy is not None:
            allowed = TIER_AUTONOMY[self.tier]
            if self.autonomy is AutonomyClass.COMMITMENT_WRITE:
                raise ForbiddenCapabilityError(
                    f"credential for {self.agent_id!r} claims the commitment-write "
                    "autonomy class; that class is unimplemented by design"
                )
            if _AUTONOMY_RANK[self.autonomy] > _AUTONOMY_RANK[allowed]:
                raise CapabilityError(
                    f"credential for {self.agent_id!r} claims autonomy "
                    f"{self.autonomy.value!r} above its tier ceiling "
                    f"{allowed.value!r}"
                )

    # -- capability queries -------------------------------------------------

    def allows(self, scope: str | Scope) -> bool:
        return str(scope) in self.scopes

    def require(self, scope: str | Scope) -> None:
        """Layer 3: fail closed if the scope is not held."""
        if not self.allows(scope):
            raise CapabilityError(
                f"agent {self.agent_id!r} (tier {self.tier.name}) lacks scope {str(scope)!r}"
            )

    def permits_tool(self, grant: ToolGrant) -> bool:
        """True when every scope the tool needs is held. Layer 2's predicate."""
        return grant.scopes <= self.scopes

    def grants_for(self, available: Iterable[ToolGrant]) -> tuple[ToolGrant, ...]:
        """The subset of ``available`` this credential is allowed to see."""
        return tuple(g for g in available if self.permits_tool(g))


_AUTONOMY_RANK: Mapping[AutonomyClass, int] = {
    AutonomyClass.READ: 0,
    AutonomyClass.PROPOSE_ON_CONSENT: 1,
    AutonomyClass.CONTACT_BUDGETED: 2,
    AutonomyClass.COMMITMENT_WRITE: 99,
}


# --------------------------------------------------------------------------
# Auditing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CredentialAudit:
    """An inspectable statement of what a credential can and cannot do."""

    agent_id: str
    tier: str
    autonomy_ceiling: str
    held_scopes: tuple[str, ...]
    denied_scopes: tuple[str, ...]
    forbidden_scopes: tuple[str, ...]
    registered_tools: tuple[str, ...]
    withheld_tools: tuple[str, ...]

    @property
    def can_write_commitments(self) -> bool:
        """Always False for a constructible credential. Asserted in tests."""
        return False

    @property
    def clean(self) -> bool:
        return not (set(self.held_scopes) & FORBIDDEN_SCOPES)


def audit_credential(
    cred: AgentCredential, available: Sequence[ToolGrant]
) -> CredentialAudit:
    """Produce the audit a reviewer would ask for: what is *not* reachable."""
    granted = cred.grants_for(available)
    granted_names = {g.name for g in granted}
    return CredentialAudit(
        agent_id=cred.agent_id,
        tier=cred.tier.name,
        autonomy_ceiling=TIER_AUTONOMY[cred.tier].value,
        held_scopes=tuple(sorted(cred.scopes)),
        denied_scopes=tuple(
            sorted({s for g in available for s in g.scopes} - set(cred.scopes))
        ),
        forbidden_scopes=tuple(sorted(FORBIDDEN_SCOPES)),
        registered_tools=tuple(sorted(granted_names)),
        withheld_tools=tuple(sorted({g.name for g in available} - granted_names)),
    )
