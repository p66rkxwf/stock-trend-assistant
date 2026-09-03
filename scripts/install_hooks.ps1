# Enable the version-controlled git hooks in scripts/githooks/. Run once.
#
# Uses core.hooksPath instead of copying files into .git/hooks/: the hook itself
# stays in version control, so edits reach everyone on the next pull and nobody
# ends up running a stale copy.
#
# Keep this file ASCII-only: Windows PowerShell 5.1 parses .ps1 with the ANSI
# codepage, not UTF-8, so non-ASCII characters here become a parser error.
# (Same trap as daily_predict.bat documents for cmd.exe.)
$ErrorActionPreference = "Stop"

$root = git rev-parse --show-toplevel
if (-not $root) { throw "Not inside a git repository" }

git config core.hooksPath scripts/githooks
Write-Host "core.hooksPath set to scripts/githooks"
Write-Host "pre-push will run: pytest -m leakage  (data-leakage regression tests)"
Write-Host ""
Write-Host "Run the same checks manually: powershell -File scripts/check_leakage.ps1"
