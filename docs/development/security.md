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

4. No arbitrary `class_path` import from untrusted manifests, YAML, or peer
   payloads. `instantiate_component` in `inference/component_factory.py`
   resolves `class_path` via `ComponentRegistry` and `importlib`. Only register
   trusted short names; treat manifest `class_path` values as untrusted unless
   the export directory is trusted. Prefer registered `type` names for
   built-ins. `physicalai.config.instantiate` is a separate trusted-local /
   parent→child-only construction boundary: never pass robot/camera network
   metadata, Zenoh payloads, shared-memory control requests, or other
   untrusted peer data into it. Camera reconfigure requests may carry only
   explicitly allowlisted scalar settings; the publisher must merge them into
   its trusted startup recipe without accepting a peer-selected `class_path`.

5. Enforce component nesting limits. `_MAX_COMPONENT_DEPTH` in
   `component_factory.py` caps recursive manifest/YAML instantiation;
   `_MAX_CONFIG_DEPTH` in `physicalai.config` caps recursive
   `to_config` / `instantiate` trees — do not raise or bypass either without a
   security review.

6. Never use `pickle`, `eval()`, `exec()`, `joblib`, `dill`, or `cloudpickle` on untrusted data. Prefer `json` for structured metadata, `safetensors` for weights, and `numpy.load(..., allow_pickle=False)` for arrays.

7. Avoid `trust_remote_code=True` in Hugging Face loaders unless the repo id is a hardcoded first-party constant, the need is documented, and `revision=` is pinned to a commit SHA.

8. Hugging Face Hub downloads (`inference/utils/_hub.py`): pin `revision=` to a commit SHA for reproducible loads when security matters; never log `HF_TOKEN` or tokens from the environment.

9. Validate `manifest.json` and Hub-sourced JSON fields against expected types before use. Raise on unexpected errors in model loading and manifest parsing — do not silently fall back to insecure defaults.

10. Prefer `.safetensors` over `.ckpt`/`.pt` for processor stats and weights when adding new artifact types.

11. jsonargparse `parser.instantiate` in `cli/run.py` and runtime config loading can construct arbitrary registered classes from YAML — document new `class_path` targets and avoid exposing dangerous constructors through config without validation.

12. Network transport trust boundary. `physicalai.robot.transport` (Zenoh) applies `/action` payloads to physical hardware **without authentication** — it assumes a trusted, isolated robot-cell network (VLAN/firewall or Zenoh ACL/TLS is the deployer's responsibility). Any new network-capable transport must document its trust boundary the same way, must never deserialize payloads with `pickle` (rule 6 — use msgpack/json), and must not silently widen exposure (e.g. adding remote-code or file-path semantics to wire payloads) without a security review.
