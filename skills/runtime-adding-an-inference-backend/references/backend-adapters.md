# Backend Adapter Patterns

Runtime adapters bridge exported model artifacts to `InferenceModel`.

## Registration

- Use `backend_registry.register("name", extensions=(".ext",))` for adapters safe to import eagerly.
- Use `backend_registry.register_lazy_module(...)` when the adapter has heavy or optional dependencies.
- Keep extensions available without importing heavy dependencies so export directories can be probed cheaply.

## Adapter Behavior

- Load model artifacts from the export directory or explicit model path.
- Convert Runtime-prepared inputs into backend-specific input objects.
- Return outputs in the shape and naming expected by `InferenceModel` postprocessing.
- Release backend resources in cleanup/close paths when the backend needs it.

## Dependency Errors

- Missing dependencies should raise clear messages with the package or extra to install.
- Do not import optional runtime SDKs at module import time unless they are required by the base package.

## Testing

- Test registry name and extension behavior.
- Test missing dependency messages.
- Mock heavy backend runtimes when CI cannot install or execute them.
