# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to the actual label strings used in this repo's issue tracker.

| Label in mattpocock/skills | Label in our tracker | Meaning                                  |
| -------------------------- | -------------------- | ---------------------------------------- |
| `needs-triage`             | `needs-triage`       | Maintainer needs to evaluate this issue  |
| `needs-info`               | `needs-info`         | Waiting on reporter for more information |
| `ready-for-agent`          | `ready-for-agent`    | Fully specified, ready for an AFK agent  |
| `ready-for-human`          | `ready-for-human`    | Requires human implementation            |
| `wontfix`                  | `wontfix`            | Will not be actioned                     |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the corresponding label string from this table.

Edit the right-hand column to match whatever vocabulary you actually use.

## Gitea label ids (this repo)

Gitea filters by label **name** when listing, but attaching or removing a label on an issue
takes the numeric **id**. These are this repo's ids, created by `/setup-matt-pocock-skills`:

| Role | Name | Gitea id |
| ---- | ---- | -------- |
| `needs-triage`    | `needs-triage`    | 1 |
| `needs-info`      | `needs-info`      | 2 |
| `ready-for-agent` | `ready-for-agent` | 3 |
| `ready-for-human` | `ready-for-human` | 4 |
| `wontfix`         | `wontfix`         | 5 |

Ids are per-repo and are **not** the same in the sibling repo, so don't copy them across.
If they ever drift, re-read them with
`GET /api/v1/repos/baiwang/stock-trend-assistant/labels`.
