# C9.4c2 — Android private device pairing and ADB reverse

Tracks [C9.4 #1137](https://github.com/richardthorell/arc/issues/1137), after
[#1246](https://github.com/richardthorell/arc/pull/1246) and the L1.3 Windows
activation integration [#1247](https://github.com/richardthorell/arc/pull/1247).

## Trusted editor-main device transport

`AndroidLivePlayPairingSession` is an editor-main-only adapter for the
**future** C9.7 Android external Play/deploy handler. It accepts a trusted
app ID, content server identity, token, and the *selected*
`AndroidTargetReference`, then calls `AndroidTargetService.resolveTarget`
to get the current ready serial from the configured Android SDK.

It creates exactly one `adb -s <serial> reverse --no-rebind
tcp:<editor-http-port> tcp:<editor-http-port>` tunnel. The editor's HTTP
listener continues to bind only `127.0.0.1`, never a LAN interface.
The `--no-rebind` mode refuses an existing rule rather than borrowing or
overwriting another session's tunnel. ADB USB, wireless-debug and emulator
devices all use the same explicit-serial flow. The target is resolved again
before and after credential staging; a disappearing, changed or ambiguous
device aborts instead of silently choosing another.

The editor then opens **a raw ADB stdin stream** to
`adb -s <serial> shell -T run-as <applicationId> sh -c <fixed-script>`.
The script uses `umask 077`, writes a temporary file under the
**debuggable application's private files directory**, and atomically renames
to `.arc-live-play-pairing.json`. The UTF-8 JSON record has a strict
2048-byte cap and 90-second expiry. It binds project GUID, packaged base
build, target cook profile, texture family, editor session/instance, C9
generation and descriptor digest. It contains the editor's existing
random bearer token and exact private IPv4 loopback origin.

No bearer is put in an Activity Intent, argv, shell argument, environment,
URL, logcat, renderer IPC or project file. This path necessarily requires
an installed **Debug/debbugable** Android application that permits
`run-as`. Shipping cannot receive or consume this handoff.

## Android native ingestion

The Android NativeActivity host consumes the one-shot record from
`ANativeActivity.internalDataPath`, with `O_NOFOLLOW`, regular-file,
owner, `0600` and single-hardlink checks. It unlinks even malformed records,
bounds/validates JSON and expiry, and rejects wrong project or Shipping
configuration without logging token or source bytes.

If valid, the host performs an authenticated `arc-http` request through the
reversed IPv4-loopback port on an IO worker. The existing C9.1 release
parser checks project/engine/build/Android ARM64 Vulkan target and texture
family, then checks the expected generation and SHA-256 descriptor digest.
It does **not** mount or activate a remote scene. If *any* private pairing
was staged, the host **never starts an old packaged scene instead**. Actual
remote Android world activation is a separate C9.6 requirement.

The existing Debug-only Android network security configuration permits
`127.0.0.1` plaintext for the ADB-reversed editor server, while Shipping
keeps Android's normal HTTPS-only policy.

## Teardown and limits

After failed staging, stale target/session, Stop or project closing,
the session removes its **own** device-scoped pairing file and ADB reverse
mapping. It does not issue `reverse --remove-all`, stop a shared ADB server,
or stop emulators. A disappearing device may prevent immediate cleanup,
but its record expires and is unlinked on the next native ingestion.

C9.4c2 provides the transport and protected Android-side **discovery**,
not an Android Play UI or APK deployer. C9.7 must use this service after
installing the matching Debug APK, launch NativeActivity, await positive
native activation, and own this session's teardown. C9.6 owns remote scene
mount/activation and rollback, while C9.4d owns change-feed replay/reset.
Do not report Android external Play as working solely on the basis of a
validated HTTP response.

## Coverage

`editor/src/main/androidLivePlayPairing.test.ts` covers device serial
binding, no-rebind, stdin-only secret hygiene, stale target/project, invalid
input, staging failure, race with Stop, idempotent cleanup and rule ownership.
Native Android Debug compilation includes the private pairing parser; actual
device `run-as`, USB/Wi-Fi ADB reverse and end-to-end Play acceptance
remain manual/device CI checks for C9.7.
