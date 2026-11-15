# Project Settings

Open **Project → Project Settings** to edit source-controlled project metadata.
Navigation has at most two levels: General (Project Information and Startup panels), Platforms
(enabled operating systems only), Build & Release (Cooking & Packaging),
Rendering & Quality, Input, Content & Assets, Plugins & Modules, and Advanced.
Advanced contains read-only paths, descriptor metadata and compiler policy.
Unavailable subsystems and empty Planned pages are omitted.
Ordinary fields are rendered from the shared versioned catalog; scene, target,
and profile collections use reusable editors. Native project metadata supplies
supported target and texture options.

The specification is a product direction, not a claim that all listed options
are implemented. This iteration exposes supported identity/startup/target/cook/
packaging and Android identity fields, plus read-only paths, compiler policy,
renderer/input configuration references and module/plugin declarations. Root
relocation, named build-profile inheritance, window/display options, signing,
and subsystem editors need their own native contracts and integration before
becoming editable. Existing Android keys remain `platformSettings.android`;
the spec's proposed `platforms.android` spelling is not a second storage model.
Apply/Cancel and the existing Android defaults remain unchanged.

Platforms is always available as the target enablement overview. Enabled technical
targets are grouped by operating system: multiple Windows targets produce one
Windows child, not separate architecture/renderer pages. Disabling a platform
hides its child without deleting any settings or profiles. Android Application
and Compatibility are inline sections on one page. Search covers all fields,
including settings for disabled platforms. Cook profiles use expandable summaries.

Distribution variants are intentionally deferred: stores/channels are not targets
or sidebar entries. A future shared native provider registry and variant resolver
must precede variant editing, Build Profile bindings, and packaging integration.
This UI-only navigation slice adds no variant schema, fake store controls, or
changes to native validation and persistence.

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

General contains Project Information and Startup panels. Project Information
includes a three-line description, Studio / Company, a `major.minor.patch`
project version, and a GUID-backed thumbnail selected from project image assets.
Project GUID, engine version and descriptor details are under Advanced.
The project browser previews PNG, JPEG and WebP thumbnails up to 1 MiB; unavailable
or mismatched assets retain the default project icon. Existing custom application
names are preserved even though the separate display-name control is no longer shown.

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
