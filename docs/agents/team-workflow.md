# Team Workflow: PM + Parallel Specialists + Retro

How to run a piece of work as a small simulated dev team inside a single Claude Code
session, using the built-in `Agent` tool's `isolation: "worktree"` for real parallelism —
no external orchestration app, no extra skill install.

This composes the mattpocock/skills engineering skills (`to-spec`, `to-tickets`,
`implement`, `retro`, `resolving-merge-conflicts`) and this repo's own `code-review` /
`security-review` with a role split. The role list and both dispatch models below are
not invented from scratch — they're pulled from Anthropic's own published engineering
work on this exact problem:

- [Building a C compiler with a team of parallel Claudes](https://www.anthropic.com/engineering/building-c-compiler) —
  16 real Claude instances, one shared repo, genuinely parallel. This is the source for
  the specialist-role list and the decentralized dispatch model below.
- [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) —
  Anthropic's canonical lead-agent + subagents architecture. This is the source for the
  PM/ticket-graph dispatch model, the task-clarity requirement, and the token-cost
  numbers.

Read both if you're about to run a wave larger than 2 tickets — the failure modes
section below only lists the highlights.

## Roles

Specialize by **concern**, not by cloning the same generic "implementer" role for
every ticket — the compiler project's agents were a dedup agent, a performance agent,
a codegen-efficiency agent, an architecture/quality reviewer, and a docs maintainer,
not five identical workers. For this repo, the equivalent split is:

| Role | Who plays it | Skill/tool it uses | Concern |
| --- | --- | --- | --- |
| **Spec Writer** | main session | `/to-spec` | Turn a rough idea into a spec |
| **Planner (PM)** | main session | `/to-tickets` | Decompose into vertical slices, declare blocking edges, pick a dispatch model |
| **Model/Experiment Implementer** | subagent, `Agent` + `isolation: "worktree"` | `/implement` | Feature engineering and model comparison, `stockta/features/`, `stockta/ml/` |
| **API/Data Implementer** | subagent, `Agent` + `isolation: "worktree"` | `/implement` | Serving and data pipeline, `stockta/api/`, `stockta/data/`, `stockta/inference/` |
| **Frontend Implementer** | subagent, `Agent` + `isolation: "worktree"` | `/implement` | Next.js dashboard, K-line and prediction display |
| **Correctness Reviewer** | main session, or a fresh subagent per diff | `/code-review` | Bugs, edge cases, and — for anything touching the train/test split — leakage into the test period (see `docs/leakage_report.md`) |
| **Security Reviewer** | main session or subagent | `/security-review` | Anything touching credentials, or (this repo feeds `quant-trading-platform` as an imported library) anything that could change the exact feature/model/decision pipeline `quant-trading-platform` depends on for no train-serve drift |
| **Docs/Experiment-Log Maintainer** | subagent or main session | plain edits | Keep `docs/model_comparison.md`, `docs/experiment_log.md`, `docs/leakage_report.md`, `docs/cross_sectional_report.md` in sync with what actually shipped |
| **Integrator** | main session | plain git; `/resolving-merge-conflicts` on real conflicts | Merge reviewed branches back |
| **Retro Facilitator** | main session or user-invoked | `/retro` | Close the loop — see below |

Only the Implementer roles run more than one at a time. Reviewers, Integrator, and
Retro are sequential — that's deliberate, not a shortcut (see "Review vs
implementation" below).

## Two dispatch models — pick based on the work's shape

**Model A: PM plans a ticket graph, dispatches waves.** Use when tasks are varied and
have real dependencies (e.g. "add a feature column" blocks "retrain the five models on
it"). This is the multi-agent-research-system pattern: a lead agent turns ambiguous
work into explicit, bounded task specs *before* anything runs in parallel, because
vague delegation is exactly what caused duplicated work and wasted effort in
Anthropic's own early failures with this pattern.

```
idea → SPEC → TICKETS (waves by blocking edges) → parallel IMPLEMENT → REVIEW → INTEGRATE → RETRO
```

**Model B: decentralized, git-lock self-selection.** Use when you have **many similar,
genuinely independent items** — e.g. re-running the five-model comparison across a
batch of newly added tickers, or fixing a batch of near-identical lint findings. Skip
the ticket graph. Each Implementer:

1. Looks in a shared `current_tasks/` file (or a Gitea label query) for unclaimed work.
2. Claims one item by writing/committing a claim marker — git's own conflict handling
   means two agents racing for the same item is a non-event, not a bug to prevent.
3. Does the work in its own worktree, `pull → merge → push → remove claim`.

No central planner needed for this shape of work; it doesn't need one. Don't force
Model A's ticket graph onto a batch of interchangeable items — that's the more likely
of the two mistakes.

## Before dispatching either model: the test suite is the actual feedback loop

*"The task verifier is nearly perfect, otherwise Claude will solve the wrong
problem."* Before dispatching a wave, make sure the tests each Implementer will run
against actually catch the thing you care about — for this repo specifically, that
means the leakage and walk-forward discipline in `docs/experiment_log.md` and
`docs/leakage_report.md`, not just "pytest passes." Test output must be grep-friendly,
since each Implementer starts with an empty context.

## Dispatch (Model A, the common case here)

For each ticket in the wave, call the `Agent` tool with `isolation: "worktree"` in the
**same turn** (multiple tool calls in one message — that's what makes it parallel).
Give each Implementer an explicit, bounded task spec, not a vague pointer — objective,
output shape, and boundary (what it must *not* touch), the same discipline the
research-system lead agent uses for subagents:

- the ticket's full text (title, body, acceptance criteria)
- a pointer to `CONTEXT.md` / `docs/adr/` (see `domain.md`) and this repo's `CLAUDE.md`
  — each Implementer is dropped in with no memory of this conversation, so the
  navigation pointer *is* the context transfer
- an instruction to follow `/implement`: TDD at pre-agreed seams where possible,
  typecheck and run the relevant tests regularly, run the full suite once at the end,
  commit to the worktree's own branch, and stop there — **do not** merge or push

Cap a wave at **3–4 tickets** in flight. Multi-agent parallelism runs roughly **15×**
the token cost of a single chat session (Anthropic's own measured number for their
research system) — that cost has to be worth it for the task, and past 3–4 in flight
the Review step becomes the bottleneck anyway, so the extra parallelism buys nothing.

## Review vs implementation — kept separate on purpose

The Implementer carries the most context pressure: exploration, writing code,
debugging failures. The Reviewer carries the least — it receives a diff, no
exploration needed — which is exactly why coding standards and architecture judgment
belong in Review, not stapled onto an already-loaded Implementer.

For each Implementer's branch: run `/code-review`. Run `/security-review` too for
anything touching credentials or the exact feature/model pipeline `quant-trading-platform`
imports as a library.

Post findings as a Gitea comment on the ticket. If a finding is a real bug, send the
*same* Implementer subagent back into its own worktree for one fix round — don't fold
the fix into the Integrator step, and don't spawn a fresh Implementer with no memory of
why the code looks the way it does.

## Integrate

Once a ticket's branch is clean, merge it into the wave's base branch yourself (plain
`git merge` or `git rebase`, whichever this repo's recent history already uses). If two
branches genuinely conflict despite vertical-slice discipline, run
`/resolving-merge-conflicts` rather than hand-resolving ad hoc — it insists on reading
both sides' original intent first. This shouldn't be routine; if it keeps happening,
that's itself a Retro finding (the slices weren't as independent as planned).

Close each ticket on Gitea once merged, with a comment linking the merge commit.

## Retro

After a wave fully lands, run `/retro` on the wave, not on a single ticket. Beyond
this file's own checklist (wave sizing, blocking-edge accuracy, tool economy, guardrail
gaps), check for the specific failure modes Anthropic documented in both source
articles, since they're the most likely ways this goes wrong here too:

- **Over-spawned for the task's size** — a wave of 3–4 parallel Implementers for work
  that one session could've done directly. Parallelism has a real cost; it should have
  bought something.
- **Vague task specs causing duplicated or overlapping work** — two Implementers
  quietly solved the same sub-problem because the ticket boundary was fuzzy.
  Tighten the ticket, not the agent.
- **An Implementer that didn't know when to stop** — kept iterating past the point of
  a sufficient result. Ticket acceptance criteria should make "done" checkable, not a
  judgment call the Implementer has to make alone.
- **Wrong dispatch model chosen** — Model A's ticket graph forced onto interchangeable
  work (should've been Model B), or Model B's free-for-all claiming used on work with
  real dependencies that needed sequencing.

Feed findings into how the *next* wave is planned. The loop only pays off if a retro
actually changes something about the following wave — otherwise it's theater.

## When not to parallelize

A single ticket, an exploratory/spike task, or anything where the seams aren't clear
yet: just do it directly in the main session. At roughly 15× the token cost of doing it
yourself, parallel dispatch needs the work to actually be parallelizable and valuable
enough to justify that — not every task clears that bar.
