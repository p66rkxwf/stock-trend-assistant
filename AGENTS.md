# stock-trend-assistant

## Agent skills

### Issue tracker

Issues live as Gitea issues on the self-hosted instance at `http://localhost:3030`, repo
`baiwang/stock-trend-assistant`, driven through the REST API with `curl` (no `gh`/`tea` CLI
on this machine). See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical roles, each label string equal to its name. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
