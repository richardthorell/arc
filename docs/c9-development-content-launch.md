# C9.3 — Development-only remote content source selection

Tracking: [#1136](https://github.com/richardthorell/arc/issues/1136) /
[C9 epic #1133](https://github.com/richardthorell/arc/issues/1133).
Depends on shared packaged bootstrap #1094 and the existing C9.1/C9.2
validated release and HTTP contracts.

## Launch contract

C9.3 adds `arc::project::runtime_content_launch` to the host-owned
`runtime_boot_request`. The selected mode, validated endpoint origin,
project GUID, target ID, cook profile, and private cache root are exposed by
`runtime_boot_context::content` and `packaged_runtime_session::boot()`.
The D0 `arc-distribution.json` schema is unchanged.

There are three explicit modes:

| Mode | Requested behavior |
| --- | --- |
| `packaged` | Use only staged, manifest-verified packaged content. This remains the default. |
| `remote` | Require a compatible verified remote cooked generation; **never** silently substitute a packaged scene. |
| `hybrid` | Prefer verified remote content, with explicit packaged-content fallback. |

Precedence, field by field: **trusted host launch** > **per-project
Development options supplied by the host** > **packaged default**.
Supplying `--content-server` alone selects `remote` even when the project
default mode is packaged. An explicit `--content-mode` overrides that
inference. `packaged` with an explicit launch server is an error.

Windows packaged hosts parse the actual Unicode process argv, supporting:

```text
Game.exe --content-server http://127.0.0.1:8123 --content-mode=hybrid
Game.exe --content-server=https://dev-content.example.org --content-mode=remote
```

These are **non-secret endpoint origins**, never credentials or artifact URLs.
The existing editor-owned `WindowsLivePlaySession` already accepts trusted
main-process-only non-secret argv; C9.4/C9.7 will pass the active editor
server origin once the server exists. Editor Viewport Play is unchanged.

Android uses the same C++ selection and validation functions. Since
NativeActivity has no ordinary argv, the Android host reads **optional**
`getIntent().getStringExtra(...)` values:

- `org.arcengine.CONTENT_SERVER`: HTTPS origin, or loopback HTTP when explicitly
  targeting an ADB-reversed Development session.
- `org.arcengine.CONTENT_MODE`: `packaged`, `remote` or `hybrid`.

For example, after an authorized `adb reverse tcp:8123 tcp:8123`, a
Development APK can be launched with:

```sh
adb shell am start -n org.arcengine.host/android.app.NativeActivity \
  --es org.arcengine.CONTENT_SERVER http://127.0.0.1:8123 \
  --es org.arcengine.CONTENT_MODE hybrid
```

Actual component/application ID is project-specific. Android host metadata
is used only as a **Development request**, not as release authority. The
native platform TLS bridge remains responsible for certificate/hostname
verification; C9.1/C9.2 validate release identity and immutable bytes.
For LAN access, require explicit user opt-in, HTTPS, and an authenticated
C9.4 server. Do not expose an unauthenticated editor server on the LAN.

## Validation and Shipping invariants

- Only a strict URL **origin** is accepted: HTTPS, or explicit Development
  loopback HTTP. Reuse `arc-http` origin validation. Userinfo, resource paths,
  queries, fragments, invalid ports and plaintext LAN are rejected.
- D0 manifest target/configuration and compiled host expectations must agree
  **before** selecting the content source. C9.2 separately validates exact
  project, base-build, target/profile and release-generation compatibility.
- Shipping rejects **any** Development endpoint/mode override, even an
  override requesting `packaged`. No content-server information is stored
  in the signed/staged distribution manifest, and no token is accepted in argv
  or Intent extras.
- Selection errors are structured and deliberately do not echo an unsafe
  endpoint. The effective mode/source/validated origin/cache root are
  available in the launch context for structured diagnostics.
- The cache root is app-private `<writable_storage_root>/RemoteContent`.
  Future C9 acquisition must use this root, not the source tree, CWD, or
  arbitrary paths from the command line.

## Current integration boundary

This PR **selects and validates** a content source; it does **not** start an
editor HTTP server, discover/download a C9 release, or change any ECS world.

Until C9.4 and C9.6 connect verified remote startup:

- `remote` is **rejected before** the runtime can silently start a
  mismatched packaged scene (Windows shared packaged session and Android
  mounted-catalog host).
- `hybrid` currently starts from the packaged fallback; its requested
  remote origin remains in the boot context for later integration.
- An L1 `scene.baseline.status: active` is **never** emitted from the presence
  of a C9.3 endpoint alone.

C9.4 owns the authenticated editor HTTP server; the subsequent cross-track
integration owns C9.2 runtime discovery/verified mount and L1.3 baseline
activation. C9.5/6 own saved-asset publication and hot reload, and C9.7 owns
automatic editor build/cook/server/start/Stop orchestration.

## Tests

`arc_project_tests` covers argv and Android metadata parsing, precedence,
Shipping rejection, URL safety and redacted errors, Windows boot with actual
staged Development distribution, and Android mounted-VFS Development boot.
Native platform integration/physical-device tests remain under #1142.
