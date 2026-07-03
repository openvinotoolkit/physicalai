# Physical AI Runtime Agent Guide

Physical AI Runtime is the deployment-side repo for the Physical AI workflow: load policies exported from [Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio), run inference, and drive robots with cameras and a control loop.

## Repository Layout

- `src/physicalai/inference/`: `InferenceModel`, manifests, adapters, preprocessors/postprocessors, runners.
- `src/physicalai/capture/`: unified camera API, discovery, transport.
- `src/physicalai/runtime/`: `PolicyRuntime`, execution modes, action queues, callbacks.
- `src/physicalai/robot/`: robot protocol and hardware integrations.
- `src/physicalai/cli/`: `physicalai` / `pai` host CLI (`run` and entry-point subcommands from other packages).
- `src/physicalai/benchmark/`: inference performance tooling.
- `skills/inference/`, `skills/capture/`, `skills/runtime/`: agent skills (canonical). Adapter symlinks under `.claude/skills/` and `.agents/skills/` are committed so clones work out of the box. See `skills/README.md`.
- `docs/`: user and contributor documentation (MkDocs).

## Setup

- Run `uv sync` from the repo root.
- Install optional extras as needed (`physicalai[realsense]`, `[basler]`, `[so101]`, `[trossen]`, `[capture]`, `[transport]`, etc.) — see `pyproject.toml`.

## Build, Test, Lint

- Run tests with `uv run pytest` from the repo root.
- Run repo hooks with `prek run --all-files`.
- Type-check with `pyrefly check` (also run via pre-commit).

## Cross-Repo Rules

- Runtime owns the `physicalai` executable and `pai` alias. Studio contributes training/export subcommands through `physicalai.cli.subcommands` entry points.
- Runtime-owned CLI surface includes `physicalai run` (policy on hardware from config).
- Studio owns export; Runtime consumes exported artifacts via `InferenceModel` / `InferenceModel.from_pretrained(...)`.
- Keep customer-facing instructions stable and avoid exposing internal scaffolding unless the user is contributing to the repo.

## Contribution Notes

- Use Conventional Commits for PR titles and commits.
- Sign commits when committing changes.
- Follow `docs/development/coding-standards.md` for repo-wide coding standards.
- Follow `docs/development/security.md` before changing `src/physicalai/`.
