# Android target selection

The Level Editor toolbar lists connected Android devices and installed Android
Virtual Devices (AVDs) when Android is selected. Configure the SDK in Preferences;
install platform-tools, the Emulator, and an AVD using Android Studio.

Select a stopped AVD, then click **Start Emulator**. Selecting it alone does not
start anything. The emulator opens in its own window; helper consoles stay hidden
on Windows. ARC waits up to three minutes for the matching ADB serial to be online
and `sys.boot_completed` to report `1`. Timeout stops waiting, not the emulator.
Closing the editor or switching platforms does not stop an emulator.

Targets refresh every two seconds while the Android Level Editor toolbar is
active. **Refresh** requests discovery immediately. Unauthorized/offline devices
remain visible but cannot be selected. Authorize USB debugging on the device or
reconnect it, then refresh. Emulator discovery failures do not hide physical
devices. Hover the Android diagnostic for its complete message.

The preferred identity is stored in Electron's local storage, scoped to the
project root and resolved SDK root, never in project files. Initially only a
single ready target is automatically selected. A disconnected preference remains
unavailable until it reconnects or the user explicitly selects another target.

## Main-process contract and deployment handoff

`AndroidTargetService` owns subprocess execution and startup monitoring. The
preload `arcAndroid` bridge only forwards typed IPC:

- `listTargets()` returns devices, installed AVDs, readiness, diagnostics, SDK
  identity, and discovery time. `listDevices()` retains the earlier device snapshot
  shape for existing callers.
- `startEmulator({ sdkRoot, id })` validates the identity against current discovery
  and acknowledges startup scheduling. Poll snapshots for completion or failure.
  Repeated requests are deduplicated; an already-running match is reused.
- `resolveTarget({ sdkRoot, id })` discovers again and returns the exact ready
  `{ sdkRoot, serial }`, or fails. It never selects another device.

Physical devices and unrecognized/multiple running emulator instances use
`device:<serial>`. A uniquely associated installed AVD uses `avd:<encoded-name>`;
its serial may change between boots. Multiple instances of one AVD stay distinct
by serial, and resolving the now-ambiguous AVD identity fails. Serial reuse cannot
be treated as a persistent hardware identity beyond what ADB reports.

The main-process service invokes configured executables without a shell. Renderer
input cannot specify executable paths or extra arguments. Discovery uses bounded
commands and limited concurrent metadata queries. Emulator invocation follows
[Android's command-line documentation](https://developer.android.com/studio/run/emulator-commandline)
(`-list-avds`, `-avd <name>`). No embedded-window flags are supplied.

Future #403 deployment must receive the captured selected identity, call
`resolveTarget` immediately before deployment, use that SDK's ADB with explicit
`-s <serial>`, and fail on disappearance/ambiguity. Never cache a serial across
boots or retry on the next connected device. #411/#410 own orchestration; #403
owns APK configuration, install and application launch. This PR does not connect
target selection to Build or local Play, and adds no packaging, installation,
application launch, Logcat, or runtime controls.

## Validation

Focused tests mock discovery, duplicate instances, unavailable tools, boot
completion, process failure, timeout, SDK changes, persistence, disconnects, and
late results. Existing desktop toolbar tests remain applicable.

Manual acceptance on a configured machine:

1. Connect and authorize a physical device. Select it; verify name, API and serial.
2. Select a stopped installed AVD. Confirm nothing launches until **Start Emulator**.
3. Start it; observe progress until Ready, then resolve the selected identity and
   compare the returned serial with `adb devices -l` / `adb -s <serial> emu avd name`.
4. Disconnect the selected device; confirm no other target is substituted. Switch
   project/platform and back; confirm the preference is retained. Leave an
   emulator running and close the editor; confirm it remains running.
