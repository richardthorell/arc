# Live Play L1.3b2d — Native Windows unsaved-scene activation

Tracking: [L1.3 #1198](https://github.com/richardthorell/arc/issues/1198). Parent: [#1194](https://github.com/richardthorell/arc/issues/1194).
Depends on merged #1211, #1218, #1222, #1225, #1240, #1242 and #1244.

## Startup contract

This integrates the existing **private** L1.3 authoring capture and the
verified C9.1/C9.2 cooked release into the actual Windows Development game
startup. There is no duplicate asset store or raw-scene network format.

1. The editor captures the current Authoring World without Save, cooks it
   ephemerally, verifies its target-cooked manifest/chunks and publishes a
   complete immutable generation to the editor's loopback-only C9 server.
2. Editor-main issues a short-lived 256-bit Bearer token. Only editor-main
   sends the token, server origin, source document GUID, registered scene
   *asset* GUID, authored content SHA-256/revision, release SHA-256, generation
   and target texture family through the existing single-use protected
   inherited stdin pipe. No renderer or CLI learns the credential.
3. The native Development host validates the private pairing. Remote-only
   startup is accepted **only** if it has this pairing, native IO workers,
   a verified non-Shipping distribution and a single matching startup scene.
   Ordinary packaged/Shipping starts remain unchanged.
4. Native C9 discovers the exact release from
   `/releases/<project-guid>/<cook-profile>/current.json`, with Bearer
   limited to `http://127.0.0.1:<port>`. It validates the descriptor's
   SHA-256 against the protected handoff, and C9's existing parser checks
   project GUID, engine version, selected base build, target architecture,
   platform, renderer, texture family and generation.
5. C9 downloads the complete manifest/chunks using its existing SHA-verified
   CAS/transport pipeline; it does **not** mount partial results. The existing
   cooked catalog mounts the verified release with higher priority *before*
   `prepare_packaged_application` reads the startup scene, and only if
   the remote provider actually owns the required scene asset GUID.
6. The normal D1 application materializer reads the newly active scene
   artifact and creates the Play World. Its successful start is the
   authoritative native activation. No remote scene bytes are injected into
   the socket worker or authored project.
7. The editor offers the same scene metadata over the authenticated L1
   control channel. The native worker checks **all** offered identities
   against the protected handoff and the successfully started runtime.
   `scene.baseline.status: active` is emitted only after those checks.
   That status is the gate for L1.4 authoring transforms. Any mismatch
   rejects with `incompatible`, no mounted scene rejects as `unavailable`.
8. `livePlay:start` does not complete simply because the control socket
   connected: it awaits the correlated `active` ACK. Failure, Stop or
   project changes stop the child and HTTP listener before deleting private
   staged files.

## Failure and authority rules

- A malformed/untrusted origin, missing token, wrong project/build/target,
  mismatched release SHA-256, failed HTTP authentication, corrupt chunk,
  incomplete cooked generation, or missing remote startup scene prevents
  successful remote-only startup; never fall back to an old packaged scene.
- The server exposes cooked immutable objects and the mutable read-only
  release pointer, **never** the original `.arcscene`, project database,
  private local paths or credentials.
- The initial cooked generation is immutable for the session. Later
  authored values use L1.4 typed revisioned edits; saved cooked asset
  publish/hot reload belongs to separate C9.5/C9.6 tickets.
- Transient Live Play reconnect revokes authoring edit authority until a
  new verified activation, instead of replaying commands into a new world.
- The implementation targets Development Windows standalone only.
  Android/device orchestration and general Shipping remote updates remain
  owned by C9.7/other tickets.

## Acceptance required before #1198 closes

The verification boundary is not merely an authenticated socket test.
On a real Windows CI/package or developer host:

1. Open a project with a registered scene and a movable cube; build the
   matching Development standalone package.
2. Move the cube in the editor but **do not Save**. Record the original
   on-disk authored scene's SHA-256.
3. Select Windows / This Computer (Standalone), press Play. The editor
   captures and cooks the current dirty scene, verifies and publishes it,
   starts the game and **waits for `scene.baseline.status: active`**.
4. Confirm the cube appears at the unsaved authored transform on the
   external game's *first* materialized scene, not the previously saved
   transform. Verify the source scene's SHA-256 is unchanged.
5. Move it again while attached. L1.4 final-value ACK and on-screen update
   must work without rebuilding or saving; Undo/Redo sends new typed edits.
6. Repeat with invalid token, wrong target/profile/build/release hash,
   modified staged chunk, unavailable server, missing scene, repeated
   start/stop, rapid edits, process crash, project switch and controller
   disconnect. Verify no partial mount, old-scene fallback or source change.
7. Verify in-editor Play still runs unchanged and Shipping rejects
   Development content overrides.

Unit and protocol tests in this PR prove isolated policy, identity and
handoff behavior. **Closing #1198 additionally requires an actual native
Windows runtime/visual scene acceptance check.** Do not infer visual
correctness solely from an HTTP 200 or an authenticated `active` frame.
