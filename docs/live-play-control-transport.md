# Live Play L1.2a — Editor control endpoint

**Issue:** [#1197](https://github.com/richardthorell/arc/issues/1197).
**Parent:** [Live Play L1 #1194](https://github.com/richardthorell/arc/issues/1194).
**Contract:** [Live Play control protocol](live-play-control-protocol.md).

This slice implements a **trusted editor-main-process-owned TCP control endpoint**.
It deliberately does not start on editor startup, local PIE or loading a project.
[#1140](https://github.com/richardthorell/arc/issues/1140) will request a
session only when launching an external Development game.

## Endpoint and framing

- Bind **127.0.0.1**, ephemeral port by default; no LAN listener or content
  server API is exposed. Android USB can later use ADB reverse to reach this
  loopback listener without opening it to the network.
- Each TCP frame starts with a **4-byte unsigned big-endian byte length**,
  followed by that many bytes of payload. Empty frames, frames exceeding
  **64 KiB**, and excessive combined buffered data are rejected before JSON
  parsing. Untrusted data is never dispatched as a host/ECS operation.
- A newly connected peer's **first frame must be exactly 32 raw bytes** matching
  a random per-server-start secret. Constant-time comparison happens before
  calling the existing L1.1 `receiveLivePlayFrame` validator. There is no
  alternative unauthenticated JSON handshake or arbitrary RPC endpoint.
- The second frame is `hello` in the exact
  `arc.live-play` v1 JSON envelope with matching project GUID, build ID,
  target/profile, session and instance IDs, and schema hash.
- The server replies with `hello.accepted` containing only the intersection
  of the native peer's offered capabilities and the editor session's trusted
  capability grant. An authenticated but incompatible or competing peer is
  disconnected.
- Subsequent messages use L1.1 strict schemas, sender direction, permission
  checks, and monotonically increasing **per-sender** sequences retained for
  the same session across reconnect. The main process can publish accepted
  runtime observations to its *trusted* orchestration callback, which cannot
  execute arbitrary remote operations. Outgoing heartbeat framing uses the
  same contract.
- Inbound buffering is bounded, output backpressure fails closed, idle /
  handshake timeouts close inactive peers, and Stop destroys connections,
  closes the listener, clears state and zeros the in-memory secret.

## Credential handling

Only the trusted main-process orchestrator can construct the
`LivePlayControlServer` and retrieve a **copy** of its one-session secret
through `launchHandoff()`. **Never expose this result to renderer IPC, the
command line, a URL, a build log, saved project settings, cook manifests, or a
temporary discovery file.** The next native-client slice (#1197 follow-up)
must acquire the secret over a protected native inherited pipe/handle or
equivalent OS-protected launcher IPC; the transport is not allowed to silently
fall back to an argv/environment secret. For now, real socket tests act as a
mock Development-runtime peer and pass the secret directly in memory.

The TCP listener is intentionally not opened automatically on app launch.
No Shipping/gameplay target links this TypeScript editor component. LAN
pairing/TLS remains separate, as does the cooked-asset HTTP service in
[#1137](https://github.com/richardthorell/arc/issues/1137).

## This PR / following slices

Included: loopback server, bounded binary framing, authentication, strict L1.1
message validation, handshake/capability intersection, status and telemetry
callbacks, reconnect sequencing, Stop/cleanup, and real socket tests for
bad tokens, malformed frames, wrong identities, oversized messages, permission
checks and reconnection.

Next: a **portable native Development-runtime client** using this exact frame
format, cross-language golden-frame tests and a protected launch-time
credential handoff. Then #1140 wires Windows Play and native process lifecycle;
[#1198](https://github.com/richardthorell/arc/issues/1198), #1199 and
#1200 provide unsaved scene baseline, actual editor mutations and runtime
inspection respectively.
