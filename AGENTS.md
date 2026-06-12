# Physical AI Runtime Agent Guide

Physical AI Runtime is the deployment-side package for loading exported policies, connecting cameras and robots, and running control loops on hardware.

## Repository Layout

- `src/physicalai/capture/`: camera interfaces, concrete camera integrations, discovery, and transport support.
- `src/physicalai/robot/`: robot protocol, concrete robot integrations, connection helpers, and verification.
- `src/physicalai/inference/`: exported policy loading, adapter registry, backend adapters, preprocessing, callbacks, and runners.
- `src/physicalai/runtime/`: `PolicyRuntime`, execution modes, action queues, and control-loop behavior.
- `docs/`: mkdocs documentation.
- `tests/`: unit and integration tests.
- `skills/`: canonical agent skills for repo-specific workflows. Client adapters may expose these through `.claude/skills/`, `.agents/skills/`, or other client paths.

## Setup

- Install dependencies with `uv sync` from the repo root.
- Include hardware extras only when needed, such as `physicalai[realsense]`, `physicalai[basler]`, `physicalai[so101]`, or `physicalai[trossen]`.

## Build, Test, Lint

- Run repo hooks with `prek run --all-files`.
- Run tests with `uv run pytest`.
- Build docs with `uv run mkdocs build` when documentation changes are involved.

## Cross-Repo Rules

- Runtime owns the `physicalai` executable and `pai` alias.
- Training packages, including Studio's `physicalai-train`, contribute CLI subcommands through `physicalai.cli.subcommands`.
- Runtime owns the load side of the export/load contract. Studio produces artifacts; Runtime consumes them with `InferenceModel.load(...)`.
- Runtime core ships ONNX and OpenVINO inference adapters. Additional backends may come from companion distributions.
- Respect Preview markers in docs and APIs. Do not present planned APIs as shipped behavior.

## Contribution Notes

- Use Conventional Commits for PR titles and commits.
- Sign commits when committing changes.
- Follow `.github/copilot-instructions.md` for detailed coding standards when present.
