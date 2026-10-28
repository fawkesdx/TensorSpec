#!/usr/bin/env bash
# Block personal paths, logins, e-mails and credentials from entering git history.
#
#   scripts/privacy_check.sh --staged            # what `git commit` would record (pre-commit hook)
#   scripts/privacy_check.sh --range A..B         # added lines in every commit of a range (pre-push hook)
#   scripts/privacy_check.sh --range X --not --remotes   # commits of X not yet on any remote
#   scripts/privacy_check.sh --tree               # every tracked file at HEAD (audit)
#
# Patterns: .privacy-denylist (what to block) and .privacy-allowlist (placeholders that are fine).
# Emergency bypass for one command: PRIVACY_CHECK_SKIP=1 git commit ...
set -u
[ "${PRIVACY_CHECK_SKIP:-0}" = "1" ] && exit 0
root=$(git rev-parse --show-toplevel) || exit 1
deny="$root/.privacy-denylist"; allow="$root/.privacy-allowlist"
[ -f "$deny" ] || { echo "privacy_check: missing $deny" >&2; exit 1; }
grep -v -E '^\s*(#|$)' "$deny" > "${TMPDIR:-/tmp}/privacy_deny.$$"
if [ -f "$allow" ]; then grep -v -E '^\s*(#|$)' "$allow" > "${TMPDIR:-/tmp}/privacy_allow.$$"; else : > "${TMPDIR:-/tmp}/privacy_allow.$$"; fi
trap 'rm -f "${TMPDIR:-/tmp}"/privacy_deny.$$ "${TMPDIR:-/tmp}"/privacy_allow.$$ "${TMPDIR:-/tmp}"/privacy_raw.$$' EXIT

allowed_out() {  # drop lines matching the allowlist (grep -v exits 1 when nothing is left; that is fine)
  if [ -s "${TMPDIR:-/tmp}/privacy_allow.$$" ]; then grep -a -v -E -f "${TMPDIR:-/tmp}/privacy_allow.$$" || true; else cat; fi
}
scan() {  # stdin: "<location>:<text>" lines
  { grep -a -E -f "${TMPDIR:-/tmp}/privacy_deny.$$" || true; } | allowed_out
}
added_lines() {  # git diff/log -p on stdin -> "<file>:<+line>"; the two pattern files are exempt
  awk '/^\+\+\+ b\//{f=substr($0,7); next} /^\+\+\+ /{f="?"; next}
       /^\+/ && !/^\+\+\+/ && f !~ /(^|\/)\.privacy-(deny|allow)list$/ {print f ":" substr($0,2)}'
}
raw="${TMPDIR:-/tmp}/privacy_raw.$$"
case "${1:-}" in
  --staged) git diff --cached --no-color --diff-filter=AM -U0 > "$raw" || { echo "privacy_check: git diff failed" >&2; exit 1; }
            hits=$(added_lines < "$raw" | scan) ;;
  --range)  shift; git log --no-color -p --format= --diff-filter=AM -U0 "$@" > "$raw" || { echo "privacy_check: git log $* failed" >&2; exit 1; }
            hits=$(added_lines < "$raw" | scan) ;;
  --tree)   git grep -I -n -E -f "${TMPDIR:-/tmp}/privacy_deny.$$" -- . ':!.privacy-denylist' ':!.privacy-allowlist' > "$raw"; rc=$?
            [ $rc -gt 1 ] && { echo "privacy_check: git grep failed" >&2; exit 1; }
            hits=$(allowed_out < "$raw") ;;
  *) echo "usage: $0 --staged | --range <git log revision args> | --tree" >&2; exit 2 ;;
esac
rm -f "$raw"
[ -z "$hits" ] && exit 0
echo "privacy_check: private data would be published. Fix these lines (or add a placeholder to .privacy-allowlist):" >&2
echo "$hits" | head -40 >&2
echo "(bypass once with PRIVACY_CHECK_SKIP=1 if this is a false positive)" >&2
exit 1
