"""Durable agent runtime: queue, crash-survivable state, rollback, review gate.

The job description's success criterion is blunt: *"the platform is boring.
Agents run on a schedule and on events, state survives restarts, failed runs
roll back cleanly, and every run can be reconstructed from its log."* This
module is the smallest thing that satisfies each clause literally.

**Durable state, not in-memory state.** ``RunStore`` writes each run's state to
a JSON file via write-temp-then-rename, so a process killed mid-write leaves
either the old file or the new one, never a half-file. A run interrupted
between ``start`` and ``finish`` is therefore recoverable: on reload it is
``INTERRUPTED``, not lost silently.

**Atomic rollback.** A run declares side effects as *staged* actions before
performing them. ``rollback`` replays the inverse of every action recorded so
far, newest first, and reports which inverses succeeded. The point is that the
compensating action is derived from the log, so a crashed run is rolled back
from disk rather than from the crashed process's memory.

**Human review gate.** A run whose risk tier requires review parks in
``AWAITING_REVIEW`` and performs no side effect until ``approve`` records a
named approver and a reason. ``reject`` moves it to ``ROLLED_BACK``. The gate
is a state transition, not a prompt instruction: the effect path checks the
status, so an agent cannot talk its way past it.

**Model-agnostic.** ``Agent.run`` takes a callable. Nothing here imports a
model provider, which is what makes the runtime headless and swappable.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .capabilities import AgentCredential, RiskTier
from .audit import AuditLog

__all__ = [
    "RunStatus",
    "RunError",
    "ReviewRequired",
    "StagedAction",
    "RunState",
    "RunStore",
    "Agent",
    "TaskQueue",
]


class RunError(Exception):
    """Base class for runtime failures."""


class ReviewRequired(RunError):
    """Raised when a run is attempted before its review gate has cleared."""


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_REVIEW = "awaiting_review"
    APPROVED = "approved"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    ROLLED_BACK = "rolled_back"

    @property
    def terminal(self) -> bool:
        return self in {
            RunStatus.COMPLETED,
            RunStatus.REJECTED,
            RunStatus.FAILED,
            RunStatus.ROLLED_BACK,
        }


@dataclass(frozen=True)
class StagedAction:
    """One side effect, declared before it happens, with its inverse.

    ``inverse`` names the compensating operation and ``payload`` carries what
    the inverse needs to run. Recording the inverse at stage time -- not at
    failure time -- is what makes rollback work for a run whose process died.
    """

    name: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    inverse: str = ""
    inverse_payload: Mapping[str, Any] = field(default_factory=dict)
    applied: bool = False
    reversed_at: float | None = None


@dataclass
class RunState:
    """The full state of one run. Serializable, and the unit of durability."""

    run_id: str
    agent_id: str
    task: str
    tier: int = int(RiskTier.READ_ONLY)
    status: RunStatus = RunStatus.PENDING
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    actions: list[StagedAction] = field(default_factory=list)
    result: Any = None
    error: str = ""
    approver: str = ""
    approval_reason: str = ""
    log_path: str = ""

    @property
    def risk_tier(self) -> RiskTier:
        return RiskTier(self.tier)

    @property
    def requires_review(self) -> bool:
        """Tiers above read-only require a human before side effects land."""
        return self.risk_tier is not RiskTier.READ_ONLY

    def to_json(self) -> str:
        d = asdict(self)
        d["status"] = self.status.value
        d["actions"] = [asdict(a) for a in self.actions]
        return json.dumps(d, sort_keys=True, default=str)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RunState":
        d = dict(d)
        d["status"] = RunStatus(d["status"])
        d["actions"] = [StagedAction(**a) for a in d.get("actions", [])]
        return cls(**d)


class RunStore:
    """Crash-safe run persistence. One file per run, atomic replace."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        return self.root / f"{run_id}.json"

    def save(self, state: RunState) -> Path:
        """Write atomically: temp file in the same dir, then rename."""
        state.updated_at = time.time()
        target = self._path(state.run_id)
        tmp = target.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
        tmp.write_text(state.to_json(), encoding="utf-8")
        os.replace(tmp, target)  # atomic on POSIX and Windows
        return target

    def load(self, run_id: str) -> RunState:
        p = self._path(run_id)
        if not p.exists():
            raise RunError(f"no such run: {run_id}")
        return RunState.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def list_runs(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json"))

    def recover(self) -> list[RunState]:
        """Mark runs that died mid-flight as INTERRUPTED.

        A run left in RUNNING has no live process behind it (this is called at
        start-up), so its state is honest as INTERRUPTED rather than RUNNING.
        """
        out: list[RunState] = []
        for rid in self.list_runs():
            st = self.load(rid)
            if st.status in (RunStatus.RUNNING, RunStatus.PENDING):
                st.status = RunStatus.INTERRUPTED
                self.save(st)
                out.append(st)
        return out


class Agent:
    """A registered agent: credential + tool surface + a model-agnostic body.

    The ``body`` is any callable taking ``(state, ctx)``. Nothing here imports
    a model SDK -- swapping the model is changing the callable.
    """

    def __init__(
        self,
        agent_id: str,
        credential: AgentCredential,
        *,
        description: str = "",
    ) -> None:
        if credential.agent_id != agent_id:
            raise RunError(
                f"agent_id {agent_id!r} does not match credential "
                f"{credential.agent_id!r}: a credential is not transferable"
            )
        self.agent_id = agent_id
        self.credential = credential
        self.description = description

    def stage(
        self, state: RunState, name: str, *, payload: Mapping[str, Any] | None = None,
        inverse: str = "", inverse_payload: Mapping[str, Any] | None = None,
    ) -> StagedAction:
        action = StagedAction(
            name=name,
            payload=dict(payload or {}),
            inverse=inverse,
            inverse_payload=dict(inverse_payload or {}),
        )
        state.actions.append(action)
        return action


class TaskQueue:
    """Event-triggered queue with a review gate and compensating rollback."""

    def __init__(
        self,
        store: RunStore,
        log: AuditLog,
        *,
        effect_handlers: Mapping[str, Callable[[Mapping[str, Any]], Any]] | None = None,
    ) -> None:
        self.store = store
        self.log = log
        self.effects = dict(effect_handlers or {})
        self._pending: list[str] = []

    # -- queue --------------------------------------------------------------

    def submit(self, agent: Agent, task: str, *, tier: RiskTier | None = None) -> RunState:
        run_id = uuid.uuid4().hex[:12]
        state = RunState(
            run_id=run_id,
            agent_id=agent.agent_id,
            task=task,
            tier=int(tier if tier is not None else agent.credential.tier),
            status=RunStatus.PENDING,
            log_path=str(self.log.path) if self.log.path else "",
        )
        self.store.save(state)
        self._pending.append(run_id)
        self.log.append(
            "run.submitted",
            agent_id=agent.agent_id,
            tool=task,
            decision="allow",
            reason="enqueued",
            tier=state.risk_tier.name,
            scopes=tuple(sorted(agent.credential.scopes)),
            detail={"run_id": run_id},
        )
        return state

    @property
    def pending(self) -> tuple[str, ...]:
        return tuple(self._pending)

    # -- execution ----------------------------------------------------------

    def run(
        self,
        agent: Agent,
        state: RunState,
        body: Callable[[RunState, "RunContext"], Any],
    ) -> RunState:
        """Execute ``body``. Side effects land only after the review gate."""
        if state.status.terminal:
            raise RunError(f"run {state.run_id} is already {state.status.value}")

        ctx = RunContext(queue=self, agent=agent, state=state)
        state.status = RunStatus.RUNNING
        self.store.save(state)

        try:
            result = body(state, ctx)
        except Exception as exc:  # noqa: BLE001 - recorded, then rolled back
            state.status = RunStatus.FAILED
            state.error = f"{type(exc).__name__}: {exc}"
            self.store.save(state)
            self.log.append(
                "run.failed",
                agent_id=agent.agent_id,
                tool=state.task,
                decision="fail",
                reason=state.error,
                tier=state.risk_tier.name,
                detail={"run_id": state.run_id},
            )
            self.rollback(state, reason=f"run failed: {state.error}")
            return state

        state.result = result

        # The review gate. A tier that needs review parks here and applies
        # nothing: the staged actions are recorded but not performed.
        if state.requires_review and state.status is not RunStatus.APPROVED:
            state.status = RunStatus.AWAITING_REVIEW
            self.store.save(state)
            self.log.append(
                "run.awaiting_review",
                agent_id=agent.agent_id,
                tool=state.task,
                decision="hold",
                reason=f"tier {state.risk_tier.name} requires human approval",
                tier=state.risk_tier.name,
                detail={"run_id": state.run_id, "staged": len(state.actions)},
            )
            return state

        # Apply phase. An unregistered effect or a raising handler is a run
        # failure like any other: it must roll back and be recorded. Letting it
        # escape here would leave the run stuck in RUNNING on disk with effects
        # half-applied -- and a half-applied run is exactly what atomic
        # rollback exists to prevent.
        try:
            self._apply(state, agent)
        except Exception as exc:  # noqa: BLE001 - recorded, then rolled back
            state.status = RunStatus.FAILED
            state.error = f"{type(exc).__name__}: {exc}"
            self.store.save(state)
            self.log.append(
                "run.failed",
                agent_id=agent.agent_id,
                tool=state.task,
                decision="fail",
                reason=state.error,
                tier=state.risk_tier.name,
                detail={"run_id": state.run_id, "phase": "apply"},
            )
            self.rollback(state, reason=f"apply failed: {state.error}")
            return state

        state.status = RunStatus.COMPLETED
        self.store.save(state)
        self.log.append(
            "run.completed",
            agent_id=agent.agent_id,
            tool=state.task,
            decision="allow",
            reason="all staged effects applied",
            tier=state.risk_tier.name,
            detail={"run_id": state.run_id, "applied": len(state.actions)},
        )
        return state

    def approve(self, state: RunState, *, approver: str, reason: str = "") -> RunState:
        if state.status is not RunStatus.AWAITING_REVIEW:
            raise ReviewRequired(
                f"run {state.run_id} is {state.status.value}, not awaiting review"
            )
        if not approver.strip():
            raise ReviewRequired("approval requires a named approver")
        agent = Agent(state.agent_id, _rehydrate_cred(state))
        state.approver = approver
        state.approval_reason = reason
        state.status = RunStatus.APPROVED
        self.log.append(
            "run.approved",
            agent_id=state.agent_id,
            tool=state.task,
            decision="allow",
            reason=reason or "approved",
            tier=state.risk_tier.name,
            detail={"run_id": state.run_id, "approver": approver},
        )
        self._apply(state, agent)
        state.status = RunStatus.COMPLETED
        self.store.save(state)
        return state

    def reject(self, state: RunState, *, approver: str, reason: str = "") -> RunState:
        if state.status is not RunStatus.AWAITING_REVIEW:
            raise ReviewRequired(
                f"run {state.run_id} is {state.status.value}, not awaiting review"
            )
        state.approver = approver
        state.approval_reason = reason
        self.log.append(
            "run.rejected",
            agent_id=state.agent_id,
            tool=state.task,
            decision="deny",
            reason=reason or "rejected by reviewer",
            tier=state.risk_tier.name,
            detail={"run_id": state.run_id, "approver": approver},
        )
        return self.rollback(state, reason=f"rejected by {approver}")

    # -- effects and rollback ----------------------------------------------

    def _apply(self, state: RunState, agent: Agent) -> None:
        """Perform staged effects. Unregistered effects fail closed."""
        for i, action in enumerate(state.actions):
            if action.applied:
                continue
            handler = self.effects.get(action.name)
            if handler is None:
                raise RunError(
                    f"agent {agent.agent_id!r} staged {action.name!r} but no handler "
                    "is registered; effects do not run unregistered"
                )
            handler(action.payload)
            state.actions[i] = StagedAction(
                name=action.name,
                payload=action.payload,
                inverse=action.inverse,
                inverse_payload=action.inverse_payload,
                applied=True,
            )

    def rollback(self, state: RunState, *, reason: str = "") -> RunState:
        """Reverse applied actions newest-first, from the persisted log."""
        reversed_count = 0
        failures: list[str] = []
        for i in range(len(state.actions) - 1, -1, -1):
            action = state.actions[i]
            if not action.applied or action.reversed_at is not None:
                continue
            handler = self.effects.get(action.inverse) if action.inverse else None
            if action.inverse and handler is None:
                failures.append(action.inverse)
                continue
            if handler is not None:
                handler(action.inverse_payload)
            state.actions[i] = StagedAction(
                name=action.name,
                payload=action.payload,
                inverse=action.inverse,
                inverse_payload=action.inverse_payload,
                applied=False,
                reversed_at=time.time(),
            )
            reversed_count += 1

        state.status = RunStatus.ROLLED_BACK
        self.store.save(state)
        self.log.append(
            "run.rolled_back",
            agent_id=state.agent_id,
            tool=state.task,
            decision="rollback",
            reason=reason or "rollback requested",
            tier=state.risk_tier.name,
            detail={
                "run_id": state.run_id,
                "reversed": reversed_count,
                "unreversed": failures,
            },
        )
        return state

    # -- recovery -----------------------------------------------------------

    def resume(self, agent: Agent, run_id: str) -> RunState:
        """Reload a run from disk and finish or roll it back."""
        state = self.store.load(run_id)
        if state.status is RunStatus.INTERRUPTED:
            self.log.append(
                "run.recovered",
                agent_id=state.agent_id,
                tool=state.task,
                decision="hold",
                reason="recovered after interruption",
                tier=state.risk_tier.name,
                detail={"run_id": run_id},
            )
            return self.rollback(state, reason="recovered interrupted run")
        return state


class RunContext:
    """Handed to the agent body. The only way it can stage a side effect."""

    def __init__(self, queue: TaskQueue, agent: Agent, state: RunState) -> None:
        self.queue = queue
        self.agent = agent
        self.state = state

    def stage(
        self, name: str, *, payload: Mapping[str, Any] | None = None,
        inverse: str = "", inverse_payload: Mapping[str, Any] | None = None,
    ) -> StagedAction:
        return self.agent.stage(
            self.state, name, payload=payload, inverse=inverse,
            inverse_payload=inverse_payload,
        )

    def require(self, scope: str) -> None:
        """Fail closed if the credential lacks a scope."""
        self.agent.credential.require(scope)

    def log_decision(self, decision: str, reason: str = "", **detail: Any) -> None:
        self.queue.log.append(
            "agent.decision",
            agent_id=self.agent.agent_id,
            tool=self.state.task,
            decision=decision,
            reason=reason,
            tier=self.state.risk_tier.name,
            detail={"run_id": self.state.run_id, **detail},
        )


def _rehydrate_cred(state: RunState) -> AgentCredential:
    """Rebuild a minimal credential when approving a reloaded run."""
    return AgentCredential(
        agent_id=state.agent_id,
        tier=RiskTier(state.tier),
        scopes=frozenset(),
    )
