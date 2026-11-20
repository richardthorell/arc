# ARC Live Play L1.2b — Native Development runtime client

**Tracking:** [#1197](https://github.com/richardthorell/arc/issues/1197),
[Live Play epic #1194](https://github.com/richardthorell/arc/issues/1194).
**Contract:** [L1.1 wire protocol](live-play-control-protocol.md) and
[L1.2a editor control transport](live-play-control-transport.md).

This PR supplies the **native half of the existing editor TCP endpoint**; it does
not change in-editor Play, start a remote session by itself, run game entity
commands, or attach the ordinary packaged Windows/Android Shipping entrypoint.

## Build and linkage

`arc-live-play-dev` is a separate opt-in C++20 static library:

```bash
cmake -S . -B build-dev -DARC_BUILD_LIVE_PLAY_DEV=ON
cmake --build build-dev --target arc-live-play-dev
```

The build target is available as `arc::live-play-dev` **in-tree only**.
Exporting it through the installed SDK is deferred until the private JSON
library dependency can be correctly packaged with it. Default `arc-network`,
`arc-runtime` and packaged client/server builds **do not** link this target.
Engine test builds compile it separately to prevent Windows/Android source
regressions. The library uses WinSock2 on Windows and POSIX sockets on Android/
Linux/macOS. It does not require EditorHost, Electron, Qt or the asset server.

The build-time gate is intentional: even with `ARC_BUILD_LIVE_PLAY_DEV=ON`,
nothing exposes a control listener or starts a client without an explicit
Development launch handoff. Shipping binaries must never request/link the
Development target.

## Protected one-shot launch handoff

The trusted editor process starts its loopback server, receives
`LivePlayControlServer.launchHandoff()`, and writes one record through a
**privileged anonymous pipe / inherited OS handle** to the Development game.
Only a *non-secret* inherited pipe descriptor/handle is identified to the native
entrypoint. The handoff secret is **never** supplied through argv, environment
variables, HTTP query parameters, logs, source files, project settings or cooked
manifests.

- Editor-main-process writer:
  `editor/src/main/livePlayNativeHandoff.ts`.
- Native consumer:
  `arc::network::live_play_dev::read_launch_handoff(native_handle, ...)`.
- An inherited pipe record starts with a 4-byte big-endian length, followed by
  strictly bounded (max 4096 byte) UTF-8 JSON containing the editor server's
  `port`, 64 hexadecimal characters for the 32-byte session `secret`,
  `projectGuid`, `buildId`, `targetProfile`, `sessionId`,
  `instanceId`, `schemaHash` and offered `capabilities`.
- The native consumer validates length/identity/schema/capability shape before
  making any connection. A malformed/incomplete handoff fails closed; no
  file/argv discovery fallback exists. The launcher owns OS inheritance,
  descriptor lifetime, and the secure pipe endpoint.
- **Windows:** pass an explicitly inheritable read HANDLE from the trusted
  platform launcher; native `read_launch_handoff()` consumes that HANDLE with
  `ReadFile`. Node's standard `stdio[3]` abstraction must not be assumed to be
  a native Win32 HANDLE without checking how the platform launcher creates it.
- **Android/POSIX:** pass an inherited read-only file descriptor and read from
  it using `read`; Android Play's actual launch/ADB reverse/pairing is owned
  by #403/#1140. No socket access to external LAN addresses is implemented.

## Native client session flow

1. Consume the handoff from protected IPC.
2. Connect only to `127.0.0.1:port`.
3. Send length-prefixed **raw 32-byte secret**; no JSON is interpreted by the
   editor until the secret matches.
4. Send L1.1 JSON `hello` with schema hash, project/build/target/session/
   instance identity, last sequence and explicit capabilities.
5. Read and verify `hello.accepted` and ensure granted capabilities are no
   broader than the native offer. Reject wrong version, identity, sequence or
   frame shape; never accept commands before the handshake.
6. Report `session.state`, `heartbeat`, `command.ack` or
   `content.generation` through typed, bounded frames. The native `poll()`
   currently accepts editor heartbeat only; later issue #1199 extends the
   explicitly typed, tick-safe command allowlist.
7. Stop/close releases the socket; no native callbacks mutate ECS, VFS or GPU
   handles from the network thread.

The native client is deliberately synchronous with a bounded receive timeout,
so the #1140 runtime host must choose an appropriate lifecycle boundary or
worker thread and never poll the networking client on a latency-critical render
thread. L1.2c will own process lifetime, actual host opt-in, heartbeat scheduling
and reconnection strategy. The current client supports establishing a new
connection but does **not** automatically reconnect or replay commands.

## Cross-language verification

`engine/network/test_support/live_play_wire_v1.json` is a shared golden source
for native `hello` and editor `heartbeat` envelopes. Both:

- `editor/src/common/livePlayProtocol.golden.test.ts` (L1.1 Zod schema) and
- `engine/network/test_support/live_play_dev_tests.cpp` (real native TCP peer)

validate the same data. Native tests also exercise an actual TCP handshake,
runtime state and heartbeat, protected pipe decoding and rejection of a
spoofed editor session.

## Follow-up integration

- **#1140 / #379:** Create a Development game process, safely transfer the
  inherited pipe, start/reuse the editor server, attach lifecycle and publish
  connection status to the Builds/Play UI.
- **#1137:** Serve verified cooked content independently over C9 data plane.
- **#1198:** Unsaved initial scene baseline.
- **#1199:** Editor-to-runtime entity edits, ACK and safe tick application.
- **#1200:** Runtime hierarchy, Inspector and diagnostics.

**Boundary:** Shipping and normal in-editor Play paths remain untouched.
