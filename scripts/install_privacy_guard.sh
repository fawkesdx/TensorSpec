#!/usr/bin/env bash
# Install the privacy guard for every clone on this machine, whatever branch each is on.
#
#   scripts/install_privacy_guard.sh [REPO_DIR ...]      # default: this repo only
#
# Copies this repo's scripts/privacy_check.sh, .privacy-denylist and .privacy-allowlist to
# ~/.privacy-guard and writes pre-commit / pre-push hooks there. The hooks run the repo's own
# checker when the checked-out branch ships it and the ~/.privacy-guard copy otherwise, so
# feature branches and worktrees without the guard files are covered too. Every REPO_DIR gets
# core.hooksPath pointed at those hooks (worktrees share the setting). Re-run after editing
# the pattern files. Override the install dir with PRIVACY_GUARD_HOME.
set -eu
src=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
g="${PRIVACY_GUARD_HOME:-$HOME/.privacy-guard}"
mkdir -p "$g/hooks"
cp "$src/scripts/privacy_check.sh" "$g/privacy_check.sh"
cp "$src/.privacy-denylist" "$g/privacy-denylist"
cp "$src/.privacy-allowlist" "$g/privacy-allowlist"
# The fallback copy reads the pattern files from ~/.privacy-guard when the branch has none.
sed -i.bak 's|^deny="\$root/.privacy-denylist"; allow="\$root/.privacy-allowlist"$|deny="${PRIVACY_DENYLIST:-$root/.privacy-denylist}"; allow="${PRIVACY_ALLOWLIST:-$root/.privacy-allowlist}"|' "$g/privacy_check.sh"
rm -f "$g/privacy_check.sh.bak"
grep -q 'PRIVACY_DENYLIST' "$g/privacy_check.sh" || { echo "install_privacy_guard: could not patch privacy_check.sh" >&2; exit 1; }

cat > "$g/run_check.sh" <<'EOF'
#!/usr/bin/env bash
# Run the repo's own privacy checker if the checked-out branch ships it, else the ~/.privacy-guard copy.
root=$(git rev-parse --show-toplevel) || exit 1
g=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -x "$root/scripts/privacy_check.sh" ] && [ -f "$root/.privacy-denylist" ]; then
  exec "$root/scripts/privacy_check.sh" "$@"
fi
[ -f "$root/.privacy-denylist" ] || export PRIVACY_DENYLIST="$g/privacy-denylist"
[ -f "$root/.privacy-allowlist" ] || export PRIVACY_ALLOWLIST="$g/privacy-allowlist"
exec "$g/privacy_check.sh" "$@"
EOF

cat > "$g/hooks/pre-commit" <<'EOF'
#!/usr/bin/env bash
exec "$(dirname "$0")/../run_check.sh" --staged
EOF

cat > "$g/hooks/pre-push" <<'EOF'
#!/usr/bin/env bash
# stdin: <local ref> <local sha> <remote ref> <remote sha> per pushed ref
check="$(dirname "$0")/../run_check.sh"
z=0000000000000000000000000000000000000000
status=0
while read -r lref lsha rref rsha; do
  [ "$lsha" = "$z" ] && continue                      # deleting a remote branch
  if [ "$rsha" = "$z" ]; then range="$lsha --not --remotes"   # new branch: everything not already on a remote
  else range="$rsha..$lsha"; fi
  "$check" --range $range || status=1   # unquoted on purpose: may be several args
done
exit $status
EOF

chmod +x "$g/privacy_check.sh" "$g/run_check.sh" "$g/hooks/pre-commit" "$g/hooks/pre-push"
[ $# -gt 0 ] || set -- "$src"
for r in "$@"; do
  if git -C "$r" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "$r" config core.hooksPath "$g/hooks"
    echo "guarded: $r"
  else
    echo "skipped (not a git repo): $r" >&2
  fi
done
echo "hooks installed in $g/hooks"
