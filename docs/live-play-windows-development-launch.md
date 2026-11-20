# Live Play L1.2c — Windows Development process attachment

Tracking: [#1197](https://github.com/richardthorell/arc/issues/1197),
[#379](https://github.com/richardthorell/arc/issues/379),
[#1140](https://github.com/richardthorell/arc/issues/1140),
and [epic #1194](https://github.com/richardthorell/arc/issues/1194).

## Scope

This integrates the merged L1.2a editor endpoint and L1.2b native client
into a **real Windows packaged Development game process**. It does **not**
include the C9 cooked-content server, unsaved scene baseline or entity sync.

The normal in-editor Play path, normal Windows packaged entry, and Shipping
projects are unaffected. Only the explicit Development target links the
Live Play native library.

## Enabling the Development entry

1. Build/install the ARC SDK with `ARC_BUILD_LIVE_PLAY_DEV=ON`. That
   makes **`ARC::LivePlayDev`** and
   **`ARC::PlatformWindowsLivePlayEntry`** available.
2. In a project based on the updated `blank-3d` template, configure
   `ARC_ENABLE_LIVE_PLAY_DEV=ON` (the default is OFF) and build/package
   a **non-Shipping** configuration for a Windows profile.
   Existing project CMake files must opt into this entrypoint explicitly.
3. Use ARC's existing Builds workflow to publish the Development package.
   The editor reads the *project-owned* `current-workflow.json` and
   `arc-distribution.json`; it never accepts a renderer-supplied executable.
4. Select **Windows → This Computer (Standalone)** and press Play.
   The editor starts a localhost control listener, spawns the game,
   passes the session authority and ephemeral secret through an inherited
   anonymous **stdin pipe**, and waits for the authenticated handshake.
5. Stop terminates the attached process, closes the listener and zeros the
   transport secret. Project close/switch and editor shutdown also stop it.

**Current limitation:** This does not yet automatically reconfigure/compile
a Development game at Play time, or convert existing project CMake files.
If a package has not been built with the opt-in entrypoint, the connection
will time out with a clear diagnostic. Automatic configure/build/cook will
be added to the #1140 orchestration pipeline once C9 #1136/#1137 is ready.

## Protocol and lifecycle

- The executable is resolved from a project-scoped published package
  and checked against project GUID, target/profile, non-Shipping
  configuration and safe package-relative entrypoint. No arbitrary
  path, argument or credential comes from the renderer.
- Node's `spawn(..., {stdio:['pipe','pipe','pipe']})` supplies a
  private child **STD_INPUT_HANDLE** pipe. The Dev WinMain entry requires
  `GetFileType(stdin)==FILE_TYPE_PIPE`, consumes one bounded handoff and
  closes the inherited handle. The native client authenticates against
  the editor's 127.0.0.1 listener. No token in argv/env/disk/logs.
- A dedicated native worker exchanges `session.state` and heartbeats,
  independently of the Windows message/render loop. ECS edit commands
  remain disabled. The editor preserves Authoring World state.
- The main process owns status (`starting`, `connected`,
  `disconnected`, `failed`, `stopping`, `stopped`), lifecycle
  and cancellation. Only status and start/stop operations are exposed
  via typed preload IPC; no renderer-level arbitrary path launch.

## Tests / follow-up

- Editor tests validate package/project/target/Shipping authority,
  symlink/escape protection, and a real child process doing the TCP
  handshake and Stop via a platform-neutral fixture.
- Windows native build compiles the opt-in packaged Development entry.
  L1.2 native TCP/pipe contract tests remain unchanged.
- Follow-up #1140: fully automatic Development configure/build/cook,
  verified content-server endpoint and real game-instance CI smoke;
  #1198 unsaved scene handoff; #1199 live edit commands;
  #1200 runtime Hierarchy and Inspector; Android #403 later.
