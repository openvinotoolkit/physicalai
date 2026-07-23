# Shared construction wire decision

**Status:** Accepted (recorded in the component-config design)

Private robot-owner and camera-publisher startup stdin is a same-package
ephemeral `Popen` handshake, not a persisted peer protocol. Change the
construction payload to `robot: ComponentConfig` / `camera: ComponentConfig`
with a **hard cutover**: rewrite writers, readers, and fixtures in the same
PR as the Shared\* spawn path. Do **not** add `config_format`, dual-read, or
shape-detection fallback.

Do not bump `ROBOT_TRANSPORT_PROTOCOL_VERSION` or camera frame
`PROTOCOL_VERSION` for this change; those version network/frame payloads, not
startup envelopes.

Canonical detail: [component-config.md — Private startup envelopes](component-config.md#private-startup-envelopes-hard-cutover).
