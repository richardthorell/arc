# Project Settings

Open **Project → Project Settings** to edit source-controlled project metadata.
The four categories are General, Startup, Platforms & Targets, and Build & Cook.
Ordinary fields are rendered from the shared versioned catalog; scene, target,
and profile collections use reusable editors. Native project metadata supplies
supported target and texture options.

Changes stay in the dialog until **Apply**. Cancel/close asks before discarding a
draft. Apply validates the complete candidate descriptor using `arc-project`,
checks the original revision again, and atomically replaces `.arcproject`.
Unrelated fields and extension metadata are preserved. Builds and settings Apply
are mutually exclusive. A conflict retains the draft and requires an explicit
reload; it never automatically overwrites external changes.

Applying settings does not save or reopen scenes, material graphs, or Flow
documents. Scene bindings use asset GUIDs; unavailable references stay visible.
Project name changes do not rename folders or generated CMake targets.
Engine-version changes and path relocation are not exposed by this dialog.

Targets represent project intent. Local SDK discovery, credentials, and device
selection remain machine-local in Editor Preferences/the target selector.
Enabling Android does not enable an unfinished Android Build/Play adapter.
Disabling a target keeps its profiles; profiles must reference declared targets
and configurations. Package output must remain inside the project Build root.

## Android handoff

Android identity is stored under `platformSettings.android` in `.arcproject`:
`applicationId`, `versionName`, `versionCode`, `minimumSdk`, and `targetSdk`.
An empty application ID is valid while authoring, but cannot be exported for
packaging. Existing projects need no migration. Defaults retain API 26 minimum
and API 28 target; the current toolchain supports target API up to compile API 35.

Export validated settings as UTF-8 JSON, then pass that generated file to Gradle:

```sh
arc-project settings export-android --project /path/Game.arcproject --json > /path/android-project.json
gradle -p engine/platform/android -ParcProjectSettings=/path/android-project.json :app:assembleDebug
```

The export carries project GUID, application ID, display name, versions, and SDK
levels. Gradle rejects a supplied missing, invalid, or unsupported export; it does
not fall back to the engine-host identity. Without `arcProjectSettings`, existing
engine-host builds retain their original identity and SDK defaults. The namespace
`org.arcengine.host`, NativeActivity, and native library name remain unchanged.
This only configures the existing host; it does not deploy or add runtime control.
Exports are derived build inputs, not a second authoritative settings file.

## Interfaces

`window.arc.settings.project.describe/read/apply/onChanged` is the narrow editor
bridge. Apply accepts catalog version, project GUID, revision, and registered
field changes; it cannot accept arbitrary file paths or whole-file replacement.
The main process owns publication. Native `arc-project settings describe` supplies
supported values, and `settings export-android` supplies the packaging handoff.
