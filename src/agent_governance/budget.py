"""Interruption budget and message bundling.

The job description calls this out as the design choice adoption depends on
more than any other: *"a hard interruption budget per person, message bundling
instead of a stream of pings, and a structure that gives something useful
before it asks for anything."*

So contact is modelled as a **budget that depletes**, not a permission that is
either on or off. Three properties are enforced here:

* **Hard ceiling per person.** A recipient with no budget left cannot be
  contacted at all, whatever the agent's scopes say. The refusal is recorded,
  not silently dropped.
* **Bundling window.** Requests inside the window coalesce into one message
  rather than N. Bundling is what makes a budget survivable: an agent with a
  lot to say spends one interruption, not twenty.
* **Value-before-ask.** A message that requests something must carry something
  useful to the recipient first. A bare ask is rejected at construction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Sequence

__all__ = [
    "BudgetError",
    "BundlingWindow",
    "BudgetLedger",
    "BudgetExhausted",
    "ContactIntent",
    "BundledMessage",
    "ContactPolicy",
]


class BudgetError(Exception):
    """Base class for contact-budget refusals."""


class BudgetExhausted(BudgetError):
    """Raised when a recipient's interruption budget is spent."""


class _Kind(str, Enum):
    ASK = "ask"
    INFORM = "inform"


@dataclass(frozen=True)
class ContactIntent:
    """One thing an agent wants to say to one person.

    ``gives`` is what the recipient gets; ``asks`` is what is requested of
    them. If ``asks`` is non-empty then ``gives`` must be too -- that is the
    value-before-ask rule, enforced in ``__post_init__``.
    """

    recipient: str
    summary: str = ""
    gives: str = ""
    asks: str = ""
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.recipient:
            raise BudgetError("a contact intent needs a recipient")
        if self.asks.strip() and not self.gives.strip():
            raise BudgetError(
                f"intent for {self.recipient!r} asks ({self.asks!r}) without giving; "
                "the structure must give something useful before it asks for anything"
            )

    @property
    def kind(self) -> _Kind:
        return _Kind.ASK if self.asks.strip() else _Kind.INFORM


@dataclass(frozen=True)
class BundledMessage:
    """The single message that reaches a person for a window."""

    recipient: str
    body: str
    items: tuple[ContactIntent, ...]
    interruptions_spent: int = 1

    @property
    def item_count(self) -> int:
        return len(self.items)


class BundlingWindow:
    """Coalesces intents for a recipient inside a time window.

    Intents arriving within ``window_seconds`` of the first intent in the
    current window join that window. Closing a window yields exactly one
    ``BundledMessage``.

    ``window_seconds=0`` is meaningful and permitted: every intent closes the
    previous window immediately, i.e. bundling is switched off. Negative
    windows are rejected, since they describe nothing.
    """

    def __init__(self, window_seconds: float = 3600.0) -> None:
        if window_seconds < 0:
            raise BudgetError("window_seconds cannot be negative")
        self.window_seconds = float(window_seconds)
        self._open: dict[str, list[ContactIntent]] = {}
        self._opened_at: dict[str, float] = {}

    def add(self, intent: ContactIntent) -> BundledMessage | None:
        """Add an intent. Returns a message only when a prior window closes."""
        r = intent.recipient
        closing: BundledMessage | None = None
        if r in self._open:
            opened = self._opened_at[r]
            if intent.created_at - opened >= self.window_seconds:
                closing = self._flush(r)
                self._opened_at[r] = intent.created_at
                self._open[r] = [intent]
                return closing
            self._open[r].append(intent)
            return None
        self._open[r] = [intent]
        self._opened_at[r] = intent.created_at
        return None

    def flush(self, recipient: str) -> BundledMessage | None:
        return self._flush(recipient)

    def flush_all(self) -> list[BundledMessage]:
        return [self._flush(r) for r in list(self._open) if self._open.get(r)]

    def _flush(self, recipient: str) -> BundledMessage | None:
        items = self._open.pop(recipient, [])
        self._opened_at.pop(recipient, None)
        if not items:
            return None
        return BundledMessage(
            recipient=recipient,
            body=render_bundle(items),
            items=tuple(items),
        )


def render_bundle(items: Sequence[ContactIntent]) -> str:
    """Render one bundle. Gives first, then asks -- never the reverse."""
    gives = [i.gives.strip() for i in items if i.gives.strip()]
    asks = [i.asks.strip() for i in items if i.asks.strip()]
    parts: list[str] = []
    if gives:
        parts.append("What you get:")
        parts.extend(f"- {g}" for g in gives)
    if asks:
        parts.append("")
        parts.append("What I need from you:")
        parts.extend(f"- {a}" for a in asks)
    return "\n".join(parts).strip()


@dataclass
class BudgetLedger:
    """Per-person interruption budgets, spent one bundle at a time."""

    default_budget: int = 5
    window_seconds: float = 3600.0
    window: BundlingWindow = field(init=False)
    _budgets: dict[str, int] = field(default_factory=dict, init=False)
    _sent: dict[str, int] = field(default_factory=dict, init=False)
    _bundles: dict[str, int] = field(default_factory=dict, init=False)
    _intents: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.default_budget < 0:
            raise BudgetError("default_budget cannot be negative")
        self.window = BundlingWindow(self.window_seconds)

    # -- budget management --------------------------------------------------

    def set_budget(self, recipient: str, amount: int) -> None:
        if amount < 0:
            raise BudgetError("budget cannot be negative")
        self._budgets[recipient] = amount

    def budget_for(self, recipient: str) -> int:
        return self._budgets.get(recipient, self.default_budget)

    def remaining(self, recipient: str) -> int:
        return max(0, self.budget_for(recipient) - self._sent.get(recipient, 0))

    def spent(self, recipient: str) -> int:
        return self._sent.get(recipient, 0)

    def is_exhausted(self, recipient: str) -> bool:
        return self.remaining(recipient) <= 0

    # -- sending ------------------------------------------------------------

    def submit(self, intent: ContactIntent) -> BundledMessage | None:
        """Feed an intent in. Raises if the recipient has no budget left."""
        r = intent.recipient
        if self.is_exhausted(r):
            raise BudgetExhausted(
                f"interruption budget for {r!r} is exhausted "
                f"({self.spent(r)}/{self.budget_for(r)} spent)"
            )
        self._intents[r] = self._intents.get(r, 0) + 1
        return self.window.add(intent)

    def release(self, message: BundledMessage) -> BundledMessage:
        """Spend one interruption to deliver a bundle. Fails closed if spent."""
        r = message.recipient
        if self.is_exhausted(r):
            raise BudgetExhausted(
                f"cannot deliver to {r!r}: budget exhausted "
                f"({self.spent(r)}/{self.budget_for(r)})"
            )
        self._sent[r] = self._sent.get(r, 0) + 1
        self._bundles[r] = self._bundles.get(r, 0) + 1
        return message

    def flush_and_release(self) -> list[BundledMessage]:
        """Close every open window and deliver, skipping exhausted recipients."""
        out: list[BundledMessage] = []
        for msg in self.window.flush_all():
            if self.is_exhausted(msg.recipient):
                continue
            out.append(self.release(msg))
        return out

    # -- reporting ----------------------------------------------------------

    def report(self) -> dict[str, dict[str, int]]:
        """Per-recipient accounting. The audit view of interruption spend."""
        people = set(self._budgets) | set(self._sent) | set(self._intents)
        return {
            p: {
                "budget": self.budget_for(p),
                "spent": self.spent(p),
                "remaining": self.remaining(p),
                "intents": self._intents.get(p, 0),
                "bundles": self._bundles.get(p, 0),
            }
            for p in sorted(people)
        }

    def savings(self) -> dict[str, float]:
        """How much bundling saved.

        ``compression`` is intents per delivered message: 1.0 means no bundling
        happened, higher is better. ``avoided`` is the interruptions that would
        have been spent without bundling.
        """
        intents = sum(self._intents.values())
        bundles = sum(self._bundles.values())
        return {
            "intents": float(intents),
            "bundles": float(bundles),
            "compression": (intents / bundles) if bundles else 0.0,
            "avoided": float(max(0, intents - bundles)),
        }


class ContactPolicy:
    """Ties a ledger to the scopes a credential holds for contact."""

    def __init__(self, ledger: BudgetLedger) -> None:
        self.ledger = ledger

    def may_contact(self, recipient: str, *, external: bool = False) -> bool:
        if external and self.ledger.is_exhausted(recipient):
            return False
        return not self.ledger.is_exhausted(recipient)
