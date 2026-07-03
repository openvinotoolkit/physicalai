# Runtime Security Rules

These rules apply when writing, editing, or reviewing code under `src/physicalai/`.

1. No `# nosec` / `# nosemgrep` without a justification comment explaining why the suppression is safe.

2. No hardcoded secrets. No API keys, tokens, passwords, or credentials in any source file, test, config, or commit message.

3. No path traversal. For user-supplied file paths (export directories, config paths, cache paths): resolve and verify containment using `pathlib.Path`. Never use `assert` for security checks. Correct pattern:

   ```python
   resolved = (base_dir / user_path).resolve()
   if not resolved.is_relative_to(base_dir.resolve()):
       raise ValueError(f"Path escapes base directory: {user_path!r}")
   ```

4. No arbitrary `class_path` import from untrusted manifests or YAML. `instantiate_component` in `inference/component_factory.py` resolves `class_path` via `ComponentRegistry` and `importlib`. Only register trusted short names; treat manifest `class_path` values as untrusted unless the export directory is trusted. Prefer registered `type` names for built-ins.

5. Enforce component nesting limits. `_MAX_COMPONENT_DEPTH` in `component_factory.py` caps recursive manifest/YAML instantiation — do not raise or bypass without a security review.

6. Never use `pickle`, `eval()`, `exec()`, `joblib`, `dill`, or `cloudpickle` on untrusted data. Prefer `json` for structured metadata, `safetensors` for weights, and `numpy.load(..., allow_pickle=False)` for arrays.

7. Avoid `trust_remote_code=True` in Hugging Face loaders unless the repo id is a hardcoded first-party constant, the need is documented, and `revision=` is pinned to a commit SHA.

8. Hugging Face Hub downloads (`inference/utils/_hub.py`): pin `revision=` to a commit SHA for reproducible loads when security matters; never log `HF_TOKEN` or tokens from the environment.

9. Validate `manifest.json` and Hub-sourced JSON fields against expected types before use. Raise on unexpected errors in model loading and manifest parsing — do not silently fall back to insecure defaults.

10. Prefer `.safetensors` over `.ckpt`/`.pt` for processor stats and weights when adding new artifact types.

11. jsonargparse `parser.instantiate` in `cli/run.py` and runtime config loading can construct arbitrary registered classes from YAML — document new `class_path` targets and avoid exposing dangerous constructors through config without validation.
