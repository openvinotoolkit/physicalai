# Checked-in Fuzz Seeds

Checked-in seeds are deterministic regression inputs, not source-coverage evidence. Every configured seed is
replayed through its harness by `tests/fuzz/test_seed_replay.py`. Seed files must be regular files, must not be
crash artifacts, and must have a distinct documented purpose.

A filename or human-readable payload does not prove which `FuzzedDataProvider` branch it reaches. Before a new
seed is committed, verify its harness dispatch and semantic path with instrumentation or a focused spy, then
minimize it against the rest of that target's corpus.

## `fuzz_manifest`

| Seed                          | Purpose                                                                                         |
| ----------------------------- | ----------------------------------------------------------------------------------------------- |
| `seed_minimal.json`           | Minimal policy-package format and version fields                                                |
| `seed_full.json`              | Full type-based manifest with policy, artifacts, processors, robot, camera, and metadata fields |
| `seed_class_path.json`        | Explicit `class_path` and nested `init_args` component representation                           |
| `seed_nested_components.json` | Nested component configuration below the supported depth limit                                  |

These files bootstrap mutation toward meaningful manifest structures. Their successful replay proves only that
the current harness accepts them as non-crashing inputs; it does not by itself establish which selector path or
source lines they exercise.
