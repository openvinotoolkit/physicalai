# GitHub Copilot Instructions

Follow [`AGENTS.md`](../AGENTS.md) for repository guidance, build/test commands, and cross-repo rules.

For coding standards, follow [`docs/development/coding-standards.md`](../docs/development/coding-standards.md) when writing, editing, or reviewing code.

For security rules, follow [`docs/development/security.md`](../docs/development/security.md) when writing, editing, or reviewing runtime or plugin source code (see [`AGENTS.md`](../AGENTS.md) for repository layout). 
For every pull request review, evaluate potential security findings against the security model and accepted assumptions in
[`docs/getting-started/security.md`](../docs/getting-started/security.md). Do not flag behavior that the security
model explicitly treats as trusted or intentionally accepts unless the change violates that boundary or
makes its assumptions inaccurate. When a review comment flags a security issue, append the violated rule numbers from `docs/development/security.md` in the form `(rule 1, rule 2)`.

For agent skills, see [`skills/README.md`](../skills/README.md).
