"""agent-governance: a governed agent platform where dangerous actions are impossible.

Four layers, each independently testable:

* ``capabilities`` -- deny-by-default scopes, least-privilege credentials, and
  a forbidden class rejected at construction time.
* ``budget``      -- per-person interruption budgets with message bundling and
  a value-before-ask rule.
* ``runtime``     -- a durable queue, crash-survivable state, atomic rollback,
  and a human review gate that is a state transition rather than a prompt.
* ``evals``       -- golden suites, behavioral assertions, a written rubric,
  a CI gate, and a meta-eval that grades the grader.

``server`` exposes the governed surface over the Model Context Protocol.
"""

from .audit import AuditLog, ChainVerification, verify_chain
from .budget import BudgetExhausted, BudgetLedger, ContactIntent
from .capabilities import (
    AgentCredential,
    AutonomyClass,
    CapabilityError,
    ForbiddenCapabilityError,
    FORBIDDEN_SCOPES,
    RiskTier,
    Scope,
    ToolGrant,
    audit_credential,
)

__version__ = "0.1.0"

__all__ = [
    "AuditLog",
    "ChainVerification",
    "verify_chain",
    "BudgetExhausted",
    "BudgetLedger",
    "ContactIntent",
    "AgentCredential",
    "AutonomyClass",
    "CapabilityError",
    "ForbiddenCapabilityError",
    "FORBIDDEN_SCOPES",
    "RiskTier",
    "Scope",
    "ToolGrant",
    "audit_credential",
    "__version__",
]
