# Fuzzing

## Intro

Physical AI Runtime is a robot control loop that loads an exported, fine-tuned VLA policy, reads sensor observations, performs inference, and sends action vectors to the hardware. Because the system consumes file-system artifacts (such as manifests, model weights, and statistics files) and NumPy array streams (including camera frames and robot joint readings), it must be able to handle malformed, adversarial, and out-of-range inputs without crashing or generating unsafe actions. Fuzzing is used to identify input-related issues that could cause crashes, unexpected behavior, or unsafe actions.

Fuzzing automation uses [Atheris](https://github.com/google/atheris) - a coverage-guided Python fuzzer backed by libFuzzer.
The GitHub Actions workflow `.github/workflows/fuzz.yml` runs all harnesses in parallel on a schedule and on every PR that touches `tests/fuzz/` or the workflow file itself. Crash artifacts are uploaded for the future analysis.

## Fuzz target inventory

Each row maps a harness file to the component it covers, its input space, and the oracle invariants it checks. Oracle numbers refer to the [Key invariants for fuzzing oracles](#key-invariants-for-fuzzing-oracles) section below.

| Harness                                                                  | Entry Point                                                                  | Input Space                                                                                                                            | Oracles    |
| ------------------------------------------------------------------------ | ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| [`fuzz_manifest.py`](harnesses/fuzz_manifest.py)                         | `Manifest.load(path)`                                                        | Arbitrary JSON bytes (valid and malformed); extreme values in `shape`, `n_obs_steps`, unknown keys                                     | I-3        |
| [`fuzz_component.py`](harnesses/fuzz_component.py)                       | `instantiate_component(spec)`, `resolve_artifact(spec, export_dir)`          | Crafted `artifact` paths (`../`, absolute); bounded nested `init_args`; arbitrary `flat_params` keys for registered safe types         | I-1, I-5   |
| [`fuzz_import_dotted_path.py`](harnesses/fuzz_import_dotted_path.py)     | `import_dotted_path(path)`                                                   | Arbitrary dotted strings (relative imports, no-dot strings, known-safe module roots + fuzz suffix)                                     | I-6        |
| [`fuzz_policy_name.py`](harnesses/fuzz_policy_name.py)                   | `_is_safe_policy_name(name)`, `InferenceModel(export_dir, policy_name=name)` | Arbitrary Unicode strings; strings that pass and fail the safety regex                                                                 | I-2        |
| [`fuzz_detect_backend.py`](harnesses/fuzz_detect_backend.py)             | `InferenceModel._detect_backend()`                                           | Export directories populated with arbitrary filename extensions and combinations                                                       | I-6        |
| [`fuzz_prepare_inputs.py`](harnesses/fuzz_prepare_inputs.py)             | `InferenceModel._prepare_inputs(inputs)`                                     | Flat dicts, nested dicts, mixed flat+nested collisions, arbitrary key names                                                            | I-7        |
| [`fuzz_stats_normalizer.py`](harnesses/fuzz_stats_normalizer.py)         | `StatsNormalizer.__call__(inputs)`                                           | Arbitrary shaped float32 arrays; extreme stat values (inf, nan, zero std, inverted quantiles); all four modes                          | I-8, I-9   |
| [`fuzz_resize_preprocessor.py`](harnesses/fuzz_resize_preprocessor.py)   | `ResizePreprocessor.__call__(inputs)`                                        | Channels-first and channels-last images; uint8 and float32; zero spatial dims; extreme target resolutions; stretch and letterbox modes | I-10, I-11 |
| [`fuzz_resize_smolvla.py`](harnesses/fuzz_resize_smolvla.py)             | `ResizeSmolVLA.__call__(inputs)`                                             | Same image layouts as above; arbitrary target resolution                                                                               | I-10, I-12 |
| [`fuzz_action_normalizer.py`](harnesses/fuzz_action_normalizer.py)       | `ActionNormalizer.__call__(outputs)`                                         | Output dicts with arbitrary key names and array shapes; missing `action` key                                                           | I-13       |
| [`fuzz_action_chunk_trimmer.py`](harnesses/fuzz_action_chunk_trimmer.py) | `ActionChunkTrimmer.__call__(outputs)`                                       | Action arrays with arbitrary first and second dimensions; extreme `n_action_steps`                                                     | I-14       |
| [`fuzz_lerp_smoother.py`](harnesses/fuzz_lerp_smoother.py)               | `LerpSmoother.merge(remaining, incoming)`                                    | 2D float arrays with arbitrary row counts, zero rows, NaN/Inf values, mismatched dims                                                  | I-15, I-16 |
| [`fuzz_action_queue.py`](harnesses/fuzz_action_queue.py)                 | `ChunkedActionQueue.push_chunk(chunk, offset)` + `pop()`                     | Deterministic sequences of arbitrary chunks, extreme offsets, pushes, and pops                                                         | I-17       |
| [`fuzz_transport_codec.py`](harnesses/fuzz_transport_codec.py)           | `decode_action`/`decode_state`/`decode_metadata`, `_unpack_payload`          | Raw bytes into the decode entry points; structure-aware action/state/metadata records with malformed `__np__` markers and missing keys | I-18, I-19 |

## Key invariants for fuzzing oracles

These are the security/safety and correctness properties each harness asserts. A violation is a bug.

**I-1 — No path traversal via artifact**  
`resolve_artifact()` must never return a path lexically outside `export_dir`. Any crafted `artifact` value (including `../`, absolute paths, and null bytes) must either raise `ValueError` or produce a normalized path beneath the resolved `export_dir`. This does not validate a symlink target; reviewed artifact provenance owns that trust boundary.

**I-2 — Policy name safety**  
`_is_safe_policy_name(name)` and `InferenceModel` must agree: if the predicate returns `True`, the model constructor must not raise `ValueError` for the name. Names that pass the regex (`^[a-zA-Z0-9][a-zA-Z0-9_.\-]*$`) are safe and must be accepted.

**I-3 — No crash on malformed JSON**  
`Manifest.load()` must not raise an unhandled exception for any JSON input. Only `ValidationError` (Pydantic) and `FileNotFoundError` are acceptable.

**I-4 — Class resolution requires a type**  
`instantiate_component` on a `class_path` that resolves to a non-type (function, module, instance) must raise `TypeError` and must not call or execute the resolved object.

**I-5 — Depth limit enforced**  
`instantiate_component` relies on `physicalai.config` recursive normalization and `_MAX_CONFIG_DEPTH = 10`. Inputs nested beyond that limit must raise `ConfigError` before dynamic import or construction. They must not recurse indefinitely.

**I-6 — No unexpected exception from import or detection**  
`import_dotted_path()` must raise only `ValueError` for paths that cannot be imported — never `TypeError`, `AttributeError`, or other undocumented exceptions. `_detect_backend()` must raise `ValueError` (not crash) when no model files are found.

**I-7 — Key collision in `_prepare_inputs` raises `ValueError`**  
When both a flat key `"prefix.suffix"` and a nested dict `{"prefix": {"suffix": v}}` are present, `_prepare_inputs` must raise `ValueError`. Silently picking a winner based on insertion order would allow a caller to substitute an arbitrary tensor into the model without any observable error.

**I-8 — Non-listed keys pass through unchanged**  
`StatsDenormalizer` and `StatsNormalizer` must not modify or drop keys that are not in the `features` list. The value at any non-listed key must be byte-for-byte identical in the output.

**I-9 — Non-finite statistics are rejected before arithmetic**
For non-identity modes, statistics containing NaN or Inf must raise `ValueError` before normalization arithmetic. Finite inputs and statistics must not unexpectedly produce non-finite output when the mathematical result is representable in float32.

**I-10 — Preprocessor output is float32 channels-first**  
`ResizePreprocessor` and `ResizeSmolVLA` must produce `float32` output in `(B, C, H, W)` layout when the output is non-empty, regardless of input dtype or channel layout.

**I-11 — No unhandled exception for zero-spatial images**
`ResizePreprocessor` must not raise `ZeroDivisionError` or any unhandled exception when the input image has zero height or width. It must either produce an empty output or raise `ValueError`.

**I-12 — SmolVLA pixel values in [-1, 1]**
All pixel values in the `IMAGES` output of `ResizeSmolVLA` must satisfy `-1.0 - ε ≤ value ≤ 1.0 + ε` (with `ε = 1e-5`). Values outside this range produce out-of-distribution model inputs that can drive the robot with incorrectly scaled actions.

**I-13 — ActionNormalizer always emits `"action"` key**
The output dict of `ActionNormalizer.__call__()` must contain the key `"action"` for any input, even if the input dict does not contain it. Other keys must pass through unchanged.

**I-14 — ActionChunkTrimmer reduces chunk to at most `n_action_steps`**
The second dimension of the output action array must be `≤ n_action_steps`. The trimmer must not increase the action horizon.

**I-15 — LerpSmoother output is float32**
`LerpSmoother.merge()` output dtype must always be `np.float32`, regardless of input dtypes.

**I-16 — LerpSmoother does not introduce NaN**
If neither `remaining` nor `incoming` contains NaN or Inf, the merged output must also not contain NaN or Inf.

**I-17 — ChunkedActionQueue operation sequences preserve queue invariants**
For a deterministic sequence of `push_chunk` and `pop` calls, counters and remaining length must match the reference state. `pop()` returns either `None` or a 1-D array. Thread safety is tested separately with controlled pytest concurrency tests.

**I-18 — Transport codec parser robustness**
`_unpack_payload()`, `decode_action()`, `decode_state()`, and `decode_metadata()` must raise only the documented exceptions (`ValueError`, `TypeError`, `KeyError`, `UnicodeDecodeError`, `OverflowError`, or a `msgpack.exceptions.UnpackException`/`ExtraData`) for arbitrary bytes or a structurally malformed record — never `RecursionError`, `MemoryError`, or any other undocumented exception. `ValueError` covers both the 1 MiB payload-size gate and the `_MAX_PAYLOAD_DEPTH` nesting-depth gate. This is parser robustness only — it does not assert that a successfully decoded action is safe to actuate; hardware-facing dtype/finiteness/bounds validation is a separate, not-yet-implemented contract at this boundary.

**I-19 — Transport codec round-trip preserves value, dtype, and shape**
For a genuinely valid `encode_action()`/`decode_action()`, `encode_state()`/`decode_state()`, or `encode_metadata()`/`decode_metadata()` round trip, the decoded value must equal the original byte-for-byte: dtype, shape (including 0-dimensional/scalar arrays), and values for arrays; exact equality for metadata primitives. `_ensure_contiguous()` is what makes the 0-d case hold — it behaves like `np.ascontiguousarray()` for `ndim >= 1`, but passes a genuine 0-d array through unchanged instead of promoting it to shape `(1,)`.

## Shared Test Utilities

[`harnesses/_helpers.py`](harnesses/_helpers.py) provides fuzz-data-driven constructors used across multiple harnesses:

| Function                                  | Returns                                                                            |
| ----------------------------------------- | ---------------------------------------------------------------------------------- |
| `make_float_array(fdp, ...)`              | `np.ndarray` of arbitrary shape and float32 values (including NaN/Inf)             |
| `make_image_array(fdp, ...)`              | Plausible image array — channels-first or channels-last, uint8 or float32          |
| `make_2d_float_array(fdp, ...)`           | 2-D float32 array with fuzz-derived row and column counts                          |
| `make_2d_same_cols(fdp, cols, ...)`       | 2-D float32 array with a fixed column count (for action-queue tests)               |
| `make_stats_dict(fdp, feature, mode=...)` | Pre-built stats dict (mean/std, min/max, q01/q99) with fuzz-derived float32 values |

All helpers pad short byte sequences with zeros so the harness never throws `IndexError` on exhausted fuzz data.

## Corpus and seeds

- `tests/fuzz/corpus/<harness>/` - working corpus; populated by libFuzzer during fuzzing runs.
- `tests/fuzz/seeds/fuzz_manifest/` - hand-curated valid `manifest.json` examples used to bootstrap the manifest harness.

The fuzzer saves inputs that make it reach behavior it has not seen before. These inputs are useful because a
future run can start from them instead of rediscovering the same parser state, array shape, error path, or
component configuration from random bytes. This can help later runs reach deeper code and find new bugs
sooner.

The working corpus is temporary and can contain many similar or hard-to-understand files. Checked-in seeds are
the small, reviewed subset that is worth keeping permanently. They give the fuzzer useful starting points and
also act as regression cases: if a previously interesting input starts crashing, seed replay reports it before
the timed fuzzing campaign begins. Seeds do not prove that code is fully covered.

Successful daily and `workflow_dispatch` runs upload one `corpus-<harness>-<run-id>` artifact per harness for
seven days.

### Review and promote a saved corpus

Use this process to turn temporary corpus files into permanent regression seeds:

1. Record the successful workflow-run URL, commit SHA, and harness name. Download that harness's artifact with
   the GitHub Actions UI or:

   ```bash
   gh run download <run-id> -n corpus-<harness>-<run-id> -D downloaded-corpus
   ```

2. Treat every downloaded file as untrusted parser input. Do not execute it, use it as configuration, or copy
   the complete artifact into the repository.
3. In a supported Linux environment, install the same Atheris version as the workflow and minimize the corpus
   into a new directory. Keep the downloaded directory unchanged for comparison:

   ```bash
   uv pip install "atheris==3.1.0"
   mkdir minimized-corpus
   uv run python tests/fuzz/harnesses/fuzz_<harness>.py \
     -merge=1 minimized-corpus downloaded-corpus
   ```

4. Replay the minimized corpus without mutation. Unexpected exceptions, hangs, or crashes require triage; do
   not promote those inputs as ordinary seeds:

   ```bash
   uv run python tests/fuzz/harnesses/fuzz_<harness>.py minimized-corpus -runs=0
   ```

5. Determine what each candidate reaches. A successful replay only proves that the input did not crash. For
   each input considered for promotion, identify and record:

   - the harness sub-target selected by `FuzzedDataProvider`;
   - the production function reached, not only a harness helper or re-export;
   - the README invariant (`I-1` through `I-17`) exercised;
   - the meaningful state reached, such as a valid parser record, depth boundary, non-finite statistic, queue
     offset, or validation error;
   - the expected result or documented exception type;
   - why the input is not redundant with an existing seed or focused unit test.

   Repository examples:

   - `fuzz_manifest`: distinguish the raw-JSON and structured-dictionary paths. Record whether the input reaches
     valid manifest fields, `class_path`/`init_args`, processor lists, hardware shapes, or a specific Pydantic
     rejection.
   - `fuzz_component`: distinguish artifact containment, `_MAX_CONFIG_DEPTH`, and registered flat-parameter
     handling. Do not retain a seed merely to reproduce the fixed Hugging Face symlink compatibility case;
     that behavior belongs to its unit test.
   - `fuzz_import_dotted_path`: distinguish arbitrary dotted paths, known-safe module roots, and the no-dot
     rejection path. Confirm that the harness reaches the same `physicalai.config.importing.import_dotted_path`
     implementation used by component construction.
   - `fuzz_stats_normalizer`: record the selected normalization mode, compatible array shape, finite versus
     non-finite statistics, passthrough-key behavior, and whether the expected result is normalization or the
     documented pre-arithmetic `ValueError`.
   - `fuzz_action_queue`: record the smoother, push/pop sequence, offset boundary, and counter transition. The
     deterministic harness does not prove thread safety; concurrent behavior remains a pytest responsibility.
   - `fuzz_policy_name`: record the Unicode or ASCII grammar boundary and whether predicate and constructor
     behavior agree. Numeric-leading names are valid under the current grammar.

   Use temporary instrumentation, a focused spy, or a small test rather than decoding bytes by assumption. For
   a multi-target harness, temporarily wrap `_sub_*` functions and assert which wrapper is called. To confirm
   deeper behavior, wrap the production entry point and inspect its arguments or expected exception. Remove
   debug-only instrumentation before committing; keep a focused test when it documents a durable property. Do
   not infer purpose from a filename or guessed `FuzzedDataProvider` byte layout.

6. Select only 2-8 nonredundant inputs that reach distinct useful behavior. Prefer inputs that unlock a parser
   state, array shape, validation boundary, or error path that is difficult to reach from random bytes.
7. Copy only those selected files into `tests/fuzz/seeds/fuzz_<harness>/`. Give them stable descriptive names
   and document each purpose in `tests/fuzz/seeds/README.md`.
8. If this is the harness's first checked-in seed directory, add it to the `_HARNESS_SEEDS` allowlist in
   `tests/fuzz/test_seed_replay.py` and set its `seed_dir` in `.github/workflows/fuzz.yml`.
9. Run deterministic seed replay and a short harness smoke test:

   ```bash
   uv run pytest tests/fuzz/test_seed_replay.py -v
   uv run python tests/fuzz/harnesses/fuzz_<harness>.py \
     tests/fuzz/seeds/fuzz_<harness> -runs=100
   ```

10. Submit the selected seeds, purpose documentation, allowlist/matrix changes, and source workflow-run URL in
    one PR. Reviewers must be able to explain why each seed is retained.

After merge, seed replay checks these inputs before fuzzing, and the workflow copies them into the harness's
starting corpus. Corpus persistence and automatic reuse across runs remain deferred work.
