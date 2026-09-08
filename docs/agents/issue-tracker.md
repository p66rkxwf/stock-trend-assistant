# Issue tracker: Gitea (self-hosted)

Issues and specs for this repo live as **Gitea issues** on the self-hosted instance at
`http://localhost:3030`, repo `baiwang/stock-trend-assistant`.

There is no `gh`/`glab`/`tea` CLI on this machine. All operations go through the Gitea
**REST API v1** with `curl`.

## Authentication

The instance is private; every call needs an access token.

```bash
GITEA="http://localhost:3030/api/v1"
OWNER=baiwang
REPO=stock-trend-assistant
TOKEN=$(cat ~/.config/gitea-token)
AUTH=(-H "Authorization: token $TOKEN" -H "Content-Type: application/json")
```

Never echo `$TOKEN`, never paste it into an issue body, and never commit it. The file
lives outside the repo at `~/.config/gitea-token` on purpose.

If a call returns `401`, the token is missing or expired: regenerate it at
`http://localhost:3030/user/settings/applications` with scopes `write:issue` and
`write:repository`, and rewrite the token file.

## Conventions

- **Create an issue**: `POST $GITEA/repos/$OWNER/$REPO/issues` with
  `{"title": "...", "body": "...", "labels": [<label ids>]}`. Build multi-line bodies with a
  heredoc piped through `jq -Rs` so newlines are encoded correctly.
- **Read an issue**: `GET $GITEA/repos/$OWNER/$REPO/issues/<index>` plus
  `GET .../issues/<index>/comments` for the conversation. `<index>` is the per-repo issue
  number shown in the UI, not the global id.
- **List issues**: `GET $GITEA/repos/$OWNER/$REPO/issues?state=open&labels=<name>&limit=50`.
  Gitea filters labels by **name** here, comma-separated.
- **Comment**: `POST $GITEA/repos/$OWNER/$REPO/issues/<index>/comments` with `{"body": "..."}`.
- **Apply labels**: `POST $GITEA/repos/$OWNER/$REPO/issues/<index>/labels` with
  `{"labels": [<label ids>]}`. **Remove**: `DELETE .../issues/<index>/labels/<label id>`.
  Resolve name → id once via `GET $GITEA/repos/$OWNER/$REPO/labels`.
- **Close**: `PATCH $GITEA/repos/$OWNER/$REPO/issues/<index>` with `{"state": "closed"}`
  (comment first if you want a closing note; Gitea has no `--comment` shorthand).

The owner/repo pair is fixed above rather than inferred, because `git remote -v` points at
an HTTP URL that carries no API base path.

## Differences from GitHub worth knowing

- **Labels are ids, not names, on write.** Listing accepts names; attaching requires ids.
- **No sub-issues.** Gitea has no GitHub-style sub-issue endpoint. Parent/child is expressed
  by a task list in the parent body plus a `Part of #<n>` line at the top of the child.
- **Dependencies exist and are the real blocking mechanism**:
  `GET/POST/DELETE $GITEA/repos/$OWNER/$REPO/issues/<index>/dependencies`, POST body
  `{"index": <blocker index>}`. Prefer this over a prose `Blocked by:` line.
- **Issues and PRs share one number space**, as on GitHub. A bare `#42` may be either; an
  issue payload carrying a `pull_request` object is a PR.

## Pull requests as a triage surface

**PRs as a request surface: no.** _(Set to `yes` if this repo treats external PRs as feature
requests; `/triage` reads this flag.)_

This is a solo project with no external contributors, so the flag stays off. Gitea also has
no `authorAssociation` field, so the GitHub recipe for separating external contributors
would not port directly if it were ever turned on.

## When a skill says "publish to the issue tracker"

Create a Gitea issue via the `POST .../issues` call above.

## When a skill says "fetch the relevant ticket"

`GET .../issues/<index>` plus its `/comments`.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a single issue with one **child** issue per ticket.

- **Map**: an issue labelled `wayfinder:map` holding the Notes / Decisions-so-far / Fog body.
- **Child ticket**: an issue labelled `wayfinder:<type>` (`research`/`prototype`/`grilling`/`task`),
  with `Part of #<map>` as the first body line, and listed as a task-list item in the map body.
- **Blocking**: a Gitea issue dependency (`POST .../issues/<child>/dependencies`,
  `{"index": <blocker>}`). A ticket is unblocked when every blocker is closed.
- **Frontier query**: list the map's open children, drop any with an open dependency or an
  assignee; first in map order wins.
- **Claim**: `PATCH .../issues/<index>` with `{"assignees": ["<your username>"]}`, the
  session's first write.
- **Resolve**: comment the answer, `PATCH` the issue to `closed`, then append a context
  pointer to the map's Decisions-so-far.
