# Runtime Security Rules

These rules apply when writing, editing, or reviewing runtime or plugin source code (see [`AGENTS.md`](../../AGENTS.md) for repository layout).

The [security model](../getting-started/security.md) defines operator-facing trust boundaries and accepted
assumptions. Use the numbered rules below as coding and review requirements. Do not report behavior that the
security model explicitly accepts unless a change crosses that boundary or makes its description inaccurate.

1. Every `# nosec` or `# nosemgrep` suppression must name the suppressed check and explain
   why the code is safe. Do not copy an existing suppression without re-evaluating it in the new context.

2. No hardcoded secrets. Never commit real API keys, tokens, passwords, authenticated URLs, or other
   credentials. Clearly non-secret placeholders and test fixtures are allowed when they cannot grant access
   and are identified as test data.

3. No path traversal across a defined base directory. When an API accepts an untrusted relative name that
   is contractually confined beneath a base directory, resolve it and verify containment with `pathlib.Path`.
   Never use `assert` for security checks. An explicit path selector such as `--config <path>` is allowed to
   name an arbitrary operator-chosen path; automation must apply an allowlist or base-directory policy before
   forwarding an untrusted value. When symlink targets must remain inside the base, use resolved containment:

   ```python
   resolved = (base_dir / user_path).resolve()
   if not resolved.is_relative_to(base_dir.resolve()):
       raise ValueError(f"Path escapes base directory: {user_path!r}")
   ```

   Some artifact stores intentionally use symlinks outside the lexical export directory. Such APIs must use
   documented lexical containment instead and test both traversal rejection and the allowed symlink layout.

4. Preserve dynamic-construction trust boundaries. Name resolution by `ComponentRegistry`,
   `instantiate_component()`, jsonargparse, or `physicalai.config.instantiate()` does not establish trust.
   Operator-reviewed configs and export directories are trusted inputs under the security model; provenance
   and review establish that trust, not directory completeness or successful parsing. Do not pass network
   metadata, transport payloads, shared-memory control requests, or other untrusted peer data into these
   construction paths. Never accept a peer-selected `class_path`. Prefer registered `type` names for built-in
   manifest components, review new registry entries, and use an explicit module-prefix allowlist when a
   feature intentionally supports only a package family.

5. Preserve recursive configuration limits. `_MAX_CONFIG_DEPTH` in `physicalai.config` caps recursive
   normalization and instantiation of `Config` trees, and normalization rejects cycles. Manifest component
   construction relies on those controls for nested typed configuration. Do not raise or bypass the limit or
   cycle checks without a security review.

6. Never use `pickle`, `eval()`, `exec()`, `joblib`, `dill`, or `cloudpickle` on untrusted data. Python
   multiprocessing may pickle objects exchanged within a trusted local parent-child boundary; do not expose
   that channel to attacker-controlled serialized data, and do not treat field validation in `__setstate__`
   as a safe-unpickling control. Prefer `json` for structured metadata, `safetensors` for weights, and
   `numpy.load(..., allow_pickle=False)` for arrays.

7. Do not enable `trust_remote_code=True` in Hugging Face loaders unless the repository and code are reviewed,
   the repository id is a hardcoded first-party constant, the need is documented, and `revision=` is pinned
   to the reviewed commit SHA.

8. Hugging Face Hub loaders must preserve caller-supplied `revision=` values. APIs that promise immutable
   loading must require and validate a commit SHA and test omitted, branch, tag, and commit values. Never log
   `HF_TOKEN`, explicit token arguments, authenticated URLs, or tokens from the environment. The security
   model treats policy revision pinning as an operator safeguard, so an allowed mutable revision is not a
   finding unless an API promises immutability or the change makes the security model inaccurate.

9. Validate `manifest.json` and other Hub-sourced JSON fields against explicit schemas or expected types
   before use. Propagate unexpected model-loading and manifest-parsing errors; do not silently fall back to a
   different backend, artifact, or insecure default.

10. Prefer `.safetensors` over `.ckpt`/`.pt` for processor stats and weights when adding new artifact types.

11. jsonargparse `parser.instantiate()` in runtime config loading can import and construct arbitrary
    `class_path` targets from trusted operator YAML. Document supported targets and validate constructor
    arguments that can affect files, processes, networks, credentials, or hardware. Do not report dynamic
    construction alone as a defect within this trusted boundary; report untrusted data entering the boundary
    or a constructor that violates another numbered rule.

12. Network transports must remain local-only by default. The documented, explicitly enabled remote mode and
    its deployment assumptions are accepted behavior, not a finding by themselves. Flag changes that widen
    default exposure, alter peer-identity assumptions, add peer-selected code or path semantics, or weaken
    payload validation. Preserve fixed payload-size limits before deserialization. If a change alters the
    trust boundary, update the user security model in the same change. Deserialization of untrusted transport
    data must comply with rule 6.
