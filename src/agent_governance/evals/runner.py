"""Evaluation runner: runs every suite, the meta-eval, and the CI gate.

    python -m agent_governance.evals.runner            # gate decision, exit code
    python -m agent_governance.evals.runner --json     # machine-readable

Exit code is 0 when the gate passes and 1 when it blocks, so CI needs no
parsing to decide. The meta-eval runs as part of the gate: a suite that has
started passing its own mutants is reported as a block, because at that point
the suite is no longer measuring what it claims to.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Any, Sequence

from .golden import build_all_suites
from .harness import gate
from .mutants import audit_forbidden_paths, score_mutation
from ..server import REGISTRY, GovernedServer, DEFAULT_AGENTS

__all__ = ["run_all", "main"]


def run_all(tmpdir: str, *, include_meta: bool = True) -> dict[str, Any]:
    """Run suites, meta-eval and the forbidden-path audit. One report.

    ``include_meta`` exists to break a cycle: the meta-eval asks whether the
    *suite* blocks a mutant, so it must run the suite without re-entering the
    meta-eval. The gate decision used for mutation detection is therefore the
    suite gate alone -- which is the correct question, since the meta-eval is
    checking the suites, not checking itself.
    """
    suites = build_all_suites(tmpdir)
    results = [s.run() for s in suites]
    decision = gate(results)
    forbidden = audit_forbidden_paths(REGISTRY)

    gov = GovernedServer(DEFAULT_AGENTS)
    audits = {aid: gov.audit(aid) for aid in sorted(DEFAULT_AGENTS)}

    total_cases = sum(r.total for r in results)
    total_passed = sum(r.passed for r in results)
    assertions = sum(c.assertion_count for r in results for c in r.cases)

    if include_meta:
        mutation = score_mutation()
        meta = {
            "mutation_score": round(mutation.score, 4),
            "killed": mutation.killed,
            "total": mutation.total,
            "survivors": [s.mutant for s in mutation.survivors],
        }
        ok = decision.ok and mutation.score == 1.0 and forbidden.ok
    else:
        meta = {"mutation_score": None, "killed": 0, "total": 0, "survivors": []}
        ok = decision.ok and forbidden.ok

    return {
        "ok": ok,
        "gate": {
            "ok": decision.ok,
            "blocked": decision.blocked,
            "reasons": list(decision.reasons),
            "suites": [
                {
                    "name": r.suite,
                    "passed": r.passed,
                    "total": r.total,
                    "pass_rate": round(r.pass_rate, 4),
                    "threshold": r.threshold,
                    "safety_violations": len(r.safety_violations),
                    "duration_ms": round(r.duration_ms, 2),
                }
                for r in results
            ],
        },
        "totals": {
            "suites": len(results),
            "cases": total_cases,
            "cases_passed": total_passed,
            "assertions": assertions,
        },
        "meta_eval": meta,
        "forbidden_path": {
            "ok": forbidden.ok,
            "verdict": forbidden.verdict,
            "reachable_tools": list(forbidden.reachable_tools),
        },
        "credentials": audits,
    }


def _report(rep: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=" * 68)
    lines.append(f"AGENT-GOVERNANCE EVAL GATE: {'PASS' if rep['ok'] else 'BLOCKED'}")
    lines.append("=" * 68)

    t = rep["totals"]
    lines.append(
        f"suites {t['suites']}  cases {t['cases_passed']}/{t['cases']} passed  "
        f"assertions {t['assertions']}"
    )
    lines.append("")
    for s in rep["gate"]["suites"]:
        state = "PASS" if (s["pass_rate"] >= s["threshold"] and not s["safety_violations"]) else "BLOCK"
        lines.append(
            f"  [{state}] {s['name']:22} {s['passed']}/{s['total']} "
            f"({s['pass_rate']:.0%}) safety={s['safety_violations']} {s['duration_ms']:.1f}ms"
        )

    me = rep["meta_eval"]
    lines.append("")
    lines.append(
        f"meta-eval: mutation score {me['killed']}/{me['total']} "
        f"({me['mutation_score']:.0%})"
        + (f"  SURVIVORS={me['survivors']}" if me["survivors"] else "")
    )
    fp = rep["forbidden_path"]
    lines.append(f"forbidden path: {'CLOSED' if fp['ok'] else 'OPEN'} -- {fp['verdict']}")

    if rep["gate"]["reasons"]:
        lines.append("")
        lines.append("Reasons:")
        for r in rep["gate"]["reasons"]:
            lines.append(f"  - {r}")
    lines.append("=" * 68)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agent-governance-evals", description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit the report as JSON")
    ap.add_argument("--tmpdir", default=None, help="working directory for run state")
    args = ap.parse_args(argv)

    if args.tmpdir:
        rep = run_all(args.tmpdir)
    else:
        with tempfile.TemporaryDirectory() as td:
            rep = run_all(td)

    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        print(_report(rep))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
