# Release Process

This repository uses [release-please](https://github.com/googleapis/release-please) for automated releases.

## Pull request title convention

Individual commit messages can be written freely. However, PR titles must follow the [conventional commits](https://www.conventionalcommits.org/) format - `<type>(<optional scope>): <description>` - this is enforced by the `pr-title.yml` workflow.

**Supported types:**

| Type       | Description              | Version bump |
| ---------- | ------------------------ | ------------ |
| `feat`     | New feature              | MINOR        |
| `fix`      | Bug fix                  | PATCH        |
| `perf`     | Performance improvement  | PATCH        |
| `refactor` | Code refactoring         | —            |
| `docs`     | Documentation only       | —            |
| `test`     | Adding/updating tests    | —            |
| `ci`       | CI/CD changes            | —            |
| `chore`    | Maintenance tasks        | —            |
| `revert`   | Revert a previous change | —            |

For **breaking** changes - add `!` after the type to trigger a MAJOR version bump, e.g. `feat!: remove deprecated API`

## Release flow

1. **Merge PRs to `main`** - merge PRs with conventional PR titles.
2. **Draft release PR** - `release-please` runs on every push to `main` and automatically creates or updates a draft PR titled `chore(main): release X.Y.Z` with an updated `CHANGELOG.md`. The next version is determined from merged PR titles.
3. **Review and merge the release PR** - once the team is ready to release, review the release PR and merge it. `release-please` then creates the git tag (e.g. `v0.2.0`) and a GitHub Release with generated release notes.
4. **Automated publish** - after the tag and GitHub Release are created, `publish.yml` builds, smoke-tests and publishes the package to PyPI.

## Versioning

The package version is **not hardcoded** - it is derived from git tags at build time via [`hatch-vcs`](https://github.com/ofek/hatch-vcs). No manual version bumping is needed.

## Testing a Release (TestPyPI)

To validate a build before an official release, trigger the `publish-testpypi.yml` workflow manually from the Actions tab. It runs the same build and smoke-test steps, then publishes to [TestPyPI](https://test.pypi.org/p/physicalai).
