# agent-governance

A governed agent platform where the dangerous action is **impossible**, not
discouraged — plus an evaluation harness whose grader is itself graded.

The design rule is inversion-first: start from the failure and work backward.
An agent must never autonomously write a *commitment* (an owner, a due date, a
status) into a shared work system. The weak implementation of that rule is an
instruction in a prompt. The strong one is that the credential physically
cannot hold the scope, so no amount of clever input, prompt injection or model
error produces the forbidden write. This repository is the strong one.

```bash
git clone https://github.com/Geomanti/agent-governance
cd agent-governance
pip install -e '.[test]'

python -m agent_governance.evals.runner   # the eval gate; exit code 1 blocks
pytest tests/ -q                          # the unit suite
```

---

## What's measured

Every figure below is produced by the code in this repository, on the default
registry of **8 tools** and **3 agents**. Reproduce with
`python -m agent_governance.evals.runner`.

| Property | Value |
|---|---|
| Eval gate | **PASS** — 12 cases, 19 assertions, 2 suites |
| Mutation score | **7/7 killed** |
| Forbidden-path audit | **CLOSED** — no reachable tool |
| Unit tests | **53 passed** |
| Forbidden scopes | 4, rejected at credential construction |

Per-agent tool surface — the withheld set is *absent*, not disabled:

| Agent | Tier | Reachable | Withheld | Scopes |
|---|---|---|---|---|
| `analyst` | READ_ONLY | 3 of 8 | 5 | 3 |
| `notifier` | CONTACT | 5 of 8 | 3 | 5 |
| `reporter` | PROPOSE | 6 of 8 | 2 | 6 |

That table is enforced in three places at once (`grant`-time filtering, the
registration surface, and a call-site re-check), which is why the mutation
`surface-unfiltered` is caught rather than merely survived.

---

## The three layers of "impossible"

**1. Construction-time rejection.** A credential requesting any scope in
`FORBIDDEN_SCOPES` cannot be instantiated. The forbidden class is refused where
the credential is built, before the agent exists.

**2. Registration-time filtering.** Tools exceeding a credential are never
registered on that agent's surface. Over MCP, `tools/list` returns a different
set per credential, and a withheld tool answers `Unknown tool:` — there is no
code path to call.

**3. Call-site re-check.** Even a direct in-process call re-verifies the scope,
so the guarantee does not depend on the registration layer remaining the only
entry point.

Verified over the real protocol, not a mock: the server was driven as a
subprocess over stdio, and `tools/call` on `propose_issue_note` as the
`analyst` returned `isError: true` with `Unknown tool`.

```
$ python -m agent_governance.server --agent analyst --list-tools
read_issue               scopes=['read:issue']
read_digest              scopes=['read:digest']
read_ledger              scopes=['ledger:read']

withheld (5): contact_person, draft_update, propose_issue_note, read_audit_log, read_private_channel
```

---

## The evaluation layer

The distinction that matters is falsifiability. A spot check produces an
opinion; an evaluation produces a number a regression can move and CI can block
on. Four things make that true here:

- **Behavioral assertions over trajectories.** Cases assert on the sequence of
  decisions and effects a run produced, not on the text it emitted.
- **Safety assertions are never averaged.** A case passes only if every safety
  assertion holds; quality assertions are scored separately against a
  threshold. Blending them is how a suite starts passing while being unsafe.
- **A written rubric, weighted, with required citations.** A judge that cannot
  cite a criterion is rejected, and one that skips a criterion raises rather
  than scoring it zero.
- **The grader is graded.** `evals/mutants.py` seeds deliberate defects into
  the live system, runs the *entire suite*, and counts a mutant as killed only
  if the gate actually blocks.

### The meta-eval found real holes in this suite

Running the meta-eval against an early version of the golden suite scored
**5/5 killed** with a *predicate-level* harness: it probed one function per
mutant. That number was misleading. Reimplementing it as suite-level mutation —
mutate a live method, run the whole suite, require the gate to block — produced
**6/7**, and the survivor was genuine:

- **`tier-ceiling-ignored` survived.** No case asserted that a credential
  cannot claim an autonomy class above its tier ceiling. The check existed in
  the source and was untested. A `safety-tier-ceiling` case was added, and
  `TIER_CEILING_ENFORCED` now asserts the refusal directly.

A separate falsification pass neutered `grants_for` by hand. The call path
still refused writes, so **every safety case passed while the tool surface
exposed all eight tools** — the guard held and the surface leaked. That is
precisely the class of blind spot a green light hides, and it is why the
`no-forbidden-tool` and `surface-within-scopes` assertions now compute
`allowed_tools` independently by raw set comparison rather than trusting the
filtering helper's own output.

Both gaps are fixed; the score is 7/7. The sequence is reported here because a
mutation score is only meaningful alongside the evidence that it can fall.

---

## Durability, rollback, and the review gate

`RunStore` writes run state via write-temp-then-rename, so a process killed
mid-write leaves either the old file or the new one. A run interrupted between
start and finish is reloaded as `INTERRUPTED`, not lost.

Rollback replays the inverse of every applied action, newest first. The inverse
is recorded at *stage* time, not at failure time, which is what makes rollback
work for a run whose process died. If the apply phase fails — including the
case where an effect has no registered handler — the run is recorded as a
failure *and* rolled back, rather than left stuck in `RUNNING` with effects
half-applied.

The human review gate is a state transition, not a prompt instruction. A
propose-tier run parks in `AWAITING_REVIEW` and applies nothing; `approve`
requires a named approver, `reject` moves the run to `ROLLED_BACK`.

---

## Contact discipline

Contact is a budget that depletes, not a permission that is on or off:

- a hard ceiling per person, with refusals recorded rather than silently dropped;
- a **bundling window**, so six intents inside one window cost one interruption,
  not six;
- a **value-before-ask** rule — a message that requests something must carry
  something useful first, and a bare ask is rejected at construction.

---

## Layout

```
src/agent_governance/
  capabilities.py     scopes, least-privilege credentials, forbidden class
  budget.py           interruption budgets, bundling, value-before-ask
  audit.py            hash-chained, tamper-evident decision log
  runtime.py          durable queue, rollback, review gate
  server.py           governed surface + MCP adapter (mcp>=2.0)
  evals/
    harness.py        assertions, trajectories, rubric, judge, CI gate
    golden.py         the golden cases and reusable assertions
    mutants.py        suite-level mutation testing + forbidden-path audit
    runner.py         the gate: suites + meta-eval, exit code for CI
tests/                53 unit tests, incl. real MCP transport tests
```

---

## Honest scope

- **This is a platform skeleton, not a deployment.** There is no warehouse
  sink, no scheduler, and no model provider wired in — `Agent.run` takes a
  callable precisely so the runtime stays model-agnostic and headless. The
  governance, durability, and evaluation layers are the substance.
- **The mutation score is 7/7 over seven hand-written operators**, not
  exhaustive coverage of the source. It measures the suite against the failure
  classes the design names, which is a smaller claim than "the suite is
  complete".
- **Assertions are as good as their authors.** `_no_commitment_write` inspects
  the effect names a run applied. A handler that ignored its own name and wrote
  a commitment anyway would not be caught by it. The structural guard (layer 1)
  is what prevents that; the assertion is a second net, not the primary one.
- The MCP surface is tested in-memory via `InMemoryTransport` and over real
  stdio as a subprocess; it has not been exercised against a third-party client
  implementation.

## License

MIT — see `LICENSE`.
