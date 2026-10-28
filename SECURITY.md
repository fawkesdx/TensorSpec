# Security policy

## Supported versions

Only the default branch (`main`) receives security fixes.

## Reporting a vulnerability

Please do not open a public issue for security problems. Use GitHub's private
vulnerability reporting instead: go to the repository's **Security** tab and click
**Report a vulnerability**. You will get an acknowledgement within a week, and a fix
or mitigation plan within 30 days for confirmed issues.

## Scope

This is research software. Reports about code in this repository are welcome.
Issues in third-party dependencies should go to those projects; Dependabot keeps
our pins current.

## What never belongs in this repository

Credentials, tokens, SSH logins, personal home-directory paths and cloud-storage
paths. A pre-commit / pre-push hook (`scripts/privacy_check.sh`) and GitHub push
protection block them; see `.privacy-denylist`.
