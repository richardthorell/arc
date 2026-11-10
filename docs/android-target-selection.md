# Android target selection

The Level Editor toolbar lists connected Android devices and installed Android
Virtual Devices (AVDs) when Android is selected. Configure the SDK in Preferences;
install platform-tools, the Emulator, and an AVD using Android Studio.

Blank 3D / Quick Start includes an Android ARM64 Vulkan target and ASTC cook
profile. Existing managed Quick Start projects gain missing Android entries on
open; explicitly disabled targets are preserved. This is project configuration,
not APK build support. The platform menu disables entries until their local
toolchain validations in Settings pass (Android requires Java, SDK and NDK).
Auto-detected valid installations count as configured. Hover a disabled entry
for the missing setup details; availability refreshes after settings changes and
when the editor regains focus. Local Play remains independent of this selection.

Select a stopped AVD, then click **Start Emulator**. Selecting it alone does not
start anything. The emulator opens in its own window; helper consoles stay hidden
on Windows. ARC waits up to three minutes for the matching ADB serial to be online
and `sys.boot_completed` to report `1`. Timeout stops waiting, not the emulator.
Closing the editor or switching platforms does not stop an emulator.

The target menu ends with a separator and **Android Device Manager…**. This opens
Android Studio (or asks you to locate it) and explains how to reach its Device
Manager: **More Actions → Virtual Device Manager** on the welcome screen, or
**View → Tool Windows → Device Manager** in a project. Device Manager is not a
standalone SDK application. This action does not change the selected target or
start an emulator; newly created AVDs appear through automatic discovery.

Targets refresh every two seconds while the Android Level Editor toolbar is
active; no manual refresh is needed. Unauthorized/offline devices
remain visible but cannot be selected. Authorize USB debugging on the device or
reconnect it, then wait for automatic discovery. Emulator discovery failures do not hide physical
devices. Hover the target dropdown for discovery and startup diagnostics.

The closed selector shows only the device name. Each expanded item has three rows:
manufacturer/model with Android release (or API fallback), serial/AVD, and status.
For network debugging only, a resolved IP follows the status in parentheses;
USB/emulator targets and unresolved network addresses have no suffix. IPs come
from ADB endpoints or the matching mDNS connect service, not guessed device interfaces.
The detail rows use muted regular-weight text, with green
ready, yellow pending/unauthorized/stopped, and red unavailable/failed status.
The icon is vertically centered beside the text block. USB metadata and ADB device paths identify the
transport; a missing transport is not assumed to mean USB. Serials disambiguate
devices with identical names without cluttering the closed selector.

The preferred identity is stored in Electron's local storage, scoped to the
project root and resolved SDK root, never in project files. Initially only a
single ready target is automatically selected. If no target is selected, later
polls also select a sole ready device as soon as it connects. A disconnected preference remains
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
