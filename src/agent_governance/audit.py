"""Hash-chained decision audit log.

Every consequential decision -- an allowed call, a denied call, a rollback --
appends one record. Each record carries the SHA-256 of the previous record, so
the log is tamper-evident: altering or removing any entry breaks the chain at
that point and every verifier after it detects the break.

This is what makes "every run can be reconstructed" a *checkable* statement
rather than a logging convention. ``verify_chain`` returns the index of the
first record whose linkage fails, so a tampered log reports *where* it was
tampered with, not merely *that* it was.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = ["AuditRecord", "AuditLog", "ChainVerification", "GENESIS_HASH"]

#: The ``prev_hash`` of the first record. A fixed, non-empty sentinel so a
#: truncated-to-empty log is distinguishable from a freshly created one.
GENESIS_HASH = "0" * 64


def _canonical(payload: Mapping[str, Any]) -> str:
    """Deterministic serialisation. Keys sorted, no incidental whitespace."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class AuditRecord:
    """One immutable decision record."""

    seq: int
    ts: float
    event: str
    agent_id: str
    tool: str
    decision: str
    reason: str
    run_id: str
    tier: str = ""
    scopes: tuple[str, ...] = ()
    detail: Mapping[str, Any] = field(default_factory=dict)
    prev_hash: str = GENESIS_HASH
    record_hash: str = ""

    def body(self) -> dict[str, Any]:
        """The hashed portion: everything except the hash itself."""
        d = asdict(self)
        d.pop("record_hash", None)
        d["scopes"] = list(self.scopes)
        d["detail"] = dict(self.detail)
        return d

    def compute_hash(self) -> str:
        return hashlib.sha256(_canonical(self.body()).encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        d = self.body()
        d["record_hash"] = self.record_hash
        return json.dumps(d, sort_keys=True, default=str)


@dataclass(frozen=True)
class ChainVerification:
    """Outcome of verifying a log's hash chain."""

    ok: bool
    length: int
    broken_at: int | None = None
    detail: str = ""

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.ok


class AuditLog:
    """Append-only, hash-chained log. Optionally mirrors to a file."""

    def __init__(self, path: str | Path | None = None, run_id: str | None = None) -> None:
        self.path = Path(path) if path else None
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self._records: list[AuditRecord] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch(exist_ok=True)

    # -- writing ------------------------------------------------------------

    def append(
        self,
        event: str,
        *,
        agent_id: str,
        tool: str,
        decision: str,
        reason: str = "",
        tier: str = "",
        scopes: Iterable[str] = (),
        detail: Mapping[str, Any] | None = None,
    ) -> AuditRecord:
        prev = self._records[-1].record_hash if self._records else GENESIS_HASH
        rec = AuditRecord(
            seq=len(self._records),
            ts=time.time(),
            event=event,
            agent_id=agent_id,
            tool=tool,
            decision=decision,
            reason=reason,
            run_id=self.run_id,
            tier=tier,
            scopes=tuple(scopes),
            detail=dict(detail or {}),
            prev_hash=prev,
        )
        rec = AuditRecord(**{**rec.body(), "record_hash": rec.compute_hash()})
        self._records.append(rec)
        if self.path:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(rec.to_json() + "\n")
        return rec

    # -- reading ------------------------------------------------------------

    @property
    def records(self) -> tuple[AuditRecord, ...]:
        return tuple(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def decisions(self) -> list[str]:
        return [r.decision for r in self._records]

    def denials(self) -> list[AuditRecord]:
        return [r for r in self._records if r.decision == "deny"]

    def counts_by_decision(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self._records:
            out[r.decision] = out.get(r.decision, 0) + 1
        return out

    def reconstruct(self, run_id: str | None = None) -> list[AuditRecord]:
        """Every record for a run, in order. The basis of run reconstruction."""
        want = run_id or self.run_id
        return [r for r in self._records if r.run_id == want]

    # -- verification -------------------------------------------------------

    def verify(self) -> ChainVerification:
        return verify_chain(self._records)


def verify_chain(records: Iterable[AuditRecord]) -> ChainVerification:
    """Verify linkage and content hashes. Reports the first break, if any."""
    prev = GENESIS_HASH
    seq = list(records)
    for i, rec in enumerate(seq):
        if rec.seq != i:
            return ChainVerification(
                False, len(seq), i, f"sequence gap: expected seq {i}, found {rec.seq}"
            )
        if rec.prev_hash != prev:
            return ChainVerification(
                False, len(seq), i, f"prev_hash mismatch at seq {i}"
            )
        if rec.compute_hash() != rec.record_hash:
            return ChainVerification(
                False, len(seq), i, f"content hash mismatch at seq {i} (record altered)"
            )
        prev = rec.record_hash
    return ChainVerification(True, len(seq), None, "chain intact")


def load_log(path: str | Path) -> list[AuditRecord]:
    """Read a written log back into records, for verification after the fact."""
    out: list[AuditRecord] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        d["scopes"] = tuple(d.get("scopes", ()))
        out.append(AuditRecord(**d))
    return out
