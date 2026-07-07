# Coding Standards

These standards apply to contributors and agents across the repository.

## Python

- Use `uv` for Python dependency and test commands; do not use `pip` directly in contributor docs or automation.
- Add type hints to functions.
- Prefer `pathlib.Path` over string path manipulation.
- Use `ruff` for linting and formatting, and address all warnings.
- Use Google-style docstrings for public Python APIs.
- Use `loguru` logger in library code; avoid `print()` except in CLI output paths.
- Prefer dataclasses or Pydantic models for structured data.

## Writing Style

This applies to comments, docstrings, commit messages, and PR descriptions.

- State the point first.
- Use active voice.
- Avoid hedging (`may`, `might`, `could potentially`) unless uncertainty is real and relevant.
- Cut filler such as "It is important to note that", "Furthermore", and "Moreover".
- Comments explain why, not what.
- Use Conventional Commits for commit messages and PR titles (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).

## Testing

- Run Python tests with `uv run pytest` from the repo root.
- Keep Python tests under `tests/unit/` (and integration tests under `tests/integration/` when added).
- Mock external services and hardware unless a test is explicitly marked as integration or download-dependent.

## AI and ML (runtime)

- Lazy-load heavy optional dependencies (camera SDKs, robot drivers).
- Account for inference latency and memory when changing adapters, preprocessors, or the control loop.
- Do not add training-only dependencies to the core runtime install path.
