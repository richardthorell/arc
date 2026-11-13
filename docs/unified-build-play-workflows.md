# Unified Build, Cook, and Play workflows

## Foundation (implemented)

The native `plan_project_workflow` extends the existing package policy with an
explicit target/profile pair and ordered stages. An incompatible pair is rejected;
the editor must not guess a profile from a platform name. The legacy package
planner's default-profile behavior remains for existing callers.

The read-only CLI entry point returns versioned JSON:

```text
arc-project workflow plan --json --project Game.arcproject \
  --profile windows-x64-vulkan --target windows-x64-vulkan --action build
```

Actions are `configure`, `buildCode`, `cook`, `build`, `play`, and `runBuilt`.
Only `play` and `runBuilt` require `--destination local|device`. Device plans
currently accept Android only. Play additionally requires a non-Shipping profile
and an owner-generated `--session-id` (1–64 ASCII letters/digits/hyphens). Play
cooked/package destinations live beneath `Build/PlaySessions/session-<id>` rather
than replacing final output. Paths use the configured project build directory.

| Intent | Stages after validation |
|---|---|
| Configure | Configure |
| Build Code | Configure, compile |
| Cook | Cook |
| Build | Configure, compile, cook, package |
| Play | Snapshot, configure, compile, cook, package, optional deploy, launch, connect |
| Run Built | Optional deploy, launch |

The read-only `arc.build.plan(request)` IPC uses `queryProjectWorkflowPlan` to consume this CLI output directly, validates its version
and captured identities, and never constructs an independent UI stage policy.
Planning does not claim that executable platform support exists.

`ProjectWorkflowRunner` is a main-process orchestration foundation with injected,
trusted stage adapters. It preflights every required capability before executing
any stage, captures project/SDK/device authority, permits only one active run,
and rejects late stage completions after cancellation or authority changes.
Cancellation remains in the draining state until owned work and rollback settle;
it cannot race a new run. Notifications are detached copies.

Adapters prepare private outputs, observe the AbortSignal, terminate only owned
process trees, and supply idempotent rollback. Final publication is synchronous,
authority-checked and must retain the previous output on failure. Missing adapters
fail closed. Saved-content cooking is executable through narrow renderer IPC and
the toolbar's Cook Content Only action. Runnable-output Build and external Play
remain unavailable until their adapters are implemented.

## Executable cooking

`arc.build.executeWorkflow(request)` currently accepts only `action: 'cook'`,
`destination: 'none'`, and an empty session ID. It replans natively from the active
project rather than accepting paths or an executable plan from the renderer.
`workflowSnapshot()`, `onWorkflowState(callback)`, and `cancelWorkflow(requestId)`
expose bounded diagnostics, progress and identity-scoped cancellation. Cancellation
during read-only planning takes effect before any cook process can start.

The service holds the same exclusion boundary as legacy code builds, including
multi-command builds. Read-only projects are rejected. Project replacement,
descriptor edits, SDK changes and cancellation prevent publication. It does not
save scenes/documents or invoke module reload, packaging, deployment or launch.

The cooker is resolved only beside the selected project tool or in its matching
development build/configuration. Missing SDK tools fail explicitly; another
configuration is never substituted. The command uses the saved startup roots and
existing cooker/DDC behavior, including incremental cache reuse and texture-family
variants. A project with no saved cook roots receives the native cooker diagnostic.

Output is written to a new `generation-*` directory under the planned cook root.
Every returned manifest must remain inside that generation and match the requested
profile/platform/architecture/renderer/configuration. Each is verified by the
native cooker before a synchronous rename publishes `current-workflow.json`.
This local build record points to exact manifest paths and native build IDs; it is
not a new asset format. Existing successful generations remain available, and
consumers must use the publication record rather than scan for the newest folder.
Failed private generations are retained for diagnostics (never advertised as
current); explicit generation/cache cleanup remains follow-up work.

Commands have a 30-minute deadline, 1 MiB stdout cap and bounded stderr/diagnostics.
Cancellation targets the owned process tree using Windows taskkill or a POSIX
process group; it never targets the emulator or unrelated applications. No new run
is accepted until the previous process and termination helper have drained.

An opt-in real-tool smoke test is available in `cookWorkflow.smoke.test.ts`:
set `ARC_WORKFLOW_TEST_PROJECT_TOOL` and `ARC_WORKFLOW_TEST_COOKER` to matching
binaries, then run that test from `editor`. It generates an isolated saved-scene
project, cooks and verifies it twice, and checks publication plus source preservation.

## Cook controls

The toolbar now explicitly calls the existing module build **Build Editor Code**.
Its menu includes **Cook Content Only**, which opens a profile/progress dialog.
Profiles come from the active project and enabled authored targets, with explicit
configuration, architecture, renderer, and texture outputs. The current platform's
sole profile is initially selected; multiple matches require a choice. Other
authored platform profiles remain selectable: cooking does not require ADB, a
connected device, or a deployment toolchain. Native planning still validates every
requested profile/target pair.

When documents are dirty, choose **Save All and Cook**, **Cook Saved Content**, or
**Cancel**. Save uses existing registered document services (and the active scene's
normal save dialog); a cancelled/unsupported/failed save prevents cooking. Saves
are explicit sequential authoring operations, not a cross-document transaction;
earlier successful saves are not undone if a later save fails. Cooking uses saved
project startup roots, not the active scene merely because it was saved. Setting
the project's startup scene remains a separate authoring action.

The main process checks the captured project/descriptor/SDK authority again before
starting a request. Closing the dialog during preparation prevents a subsequent
cook from starting. Once cooking starts, closing only hides progress; reopen via
the cooking status or Cook Content Only menu. **Cancel Cook** cancels the exact
request. Progress includes stages, bounded output, errors and verified manifest
locations. Old initial snapshots cannot overwrite newer progress events.

This slice deliberately keeps profile selection in the cook dialog, not a shared
Build profile selector: the existing code build still builds editor modules for
the host and must not imply cross-platform runnable-output support.

## Remaining implementation and delivery

### Runtime build inputs (implemented)

The installed CMake SDK provides `arc_register_runtime_output`. It registers one
runtime executable with explicit dynamically loaded library/plugin targets and
creates the `arc-runtime-output` build target. New blank-3d and blank-headless
projects opt in when configured with `-DARC_WORKFLOW_PROFILE=<authored-profile>`.
Existing projects can register their own executable without renaming their targets:

```cmake
arc_register_runtime_output(TARGET MyGame
  PROJECT_GUID "<project-guid>" ENGINE_VERSION "0.1.0"
  PROFILE "${ARC_WORKFLOW_PROFILE}" RENDERER vulkan ROLE client
  LIBRARIES ARC::RenderVulkan)
```

Build `arc-runtime-output` with the exact profile configuration. This builds the
registered runtime target and its dependencies, not the editor module. It writes
`<binary-directory>/arc-runtime-<configuration>.json` only after those targets
succeed. The record contains project/engine/profile/actual compiled target and
configuration identity, CMake target, exact absolute executable/library/plugin
paths, sizes and SHA-256 hashes. Platform/architecture come from CMake's configured
toolchain, not the cook profile label. Repeated builds refresh the record even if
the linker has no work; failure preserves the previous record. Configure alone
does not emit a successful-build record. This is machine-local derived metadata,
not a portable game asset or distribution format.

The native CLI verifies that metadata against its existing workflow planner:

```text
arc-project package --json --verify-build-only --project Game.arcproject \
  --profile windows-shipping --build-manifest Build/Code/arc-runtime-Shipping.json
```

Verification checks bounded format/version, matching project/engine/profile/
target/configuration/role, absolute existing files, duplicate inputs, sizes and
hashes. It does not launch the executable or prove rendering correctness.
Packaging can replace manual `--executable`, `--library`, `--plugin`, and
`--dll-dir` inputs with `--build-manifest`; mixing the two selection modes is
rejected. `--runtime-config` and `--cook-manifest` remain explicit. Existing native
PE import resolution completes the dependency closure from the selected inputs;
no newest-file or SDK-directory guessing is introduced.

### Windows Shipping execution (implemented)

The existing main-process `workflow:execute` endpoint accepts saved-content
`build` requests as well as `cook`. Build consumes the native plan's validate →
configure → compile → cook → package stages. It never launches a runtime or
reloads an editor module. The toolbar remains unchanged in this slice.

Runtime Build requires a Windows host, a Windows Shipping profile, configured
CMake/Visual Studio, and an installed ARC SDK containing the selected project
tool and cooker. A source-tree CLI is deliberately not treated as an installed
SDK. Visual Studio selection is resolved to its matching generator and instance,
not whichever installation happens to be newest. The native build metadata
verifier rejects incompatible compiled target/profile/configuration identities.

Code builds use a separate profile/environment-keyed `RuntimeCode` directory
under the configured project build root. CMake builds only `arc-runtime-output`;
up-to-date compilation is reused. MSVC runtime DLLs now come from CMake's active
compiler redistributable inventory and enter the hashed metadata. The packager
also follows imports of explicitly loaded libraries such as the Vulkan backend.
The existing installed-SDK Windows distribution CI smoke exercises this metadata
path, including relocated directory/ZIP bootstrap and negative package tests.

Cooking produces a private generation, verifies it, and supplies its exact
manifest to native packaging together with the verified build manifest and the
project's configured renderer settings file. This initial Windows path requires
one cooked texture variant. Packaging produces another private generation; only
successful final publication atomically replaces `current-workflow.json` under
the planned package output. Failure, cancellation, stale authority, and invalid
output leave the previous successful record intact. Failed private generations
remain for diagnostics. Intermediate cooks are not separately published.

Other targets, development packages, code-only UI, cleanup/rebuild UI and external
Play remain follow-ups. Missing deployment tools do not block Cook Only. Android
packaging/boot remains independently owned and is not enabled by this executor.

### Next slices

1. Extend executable adapters beyond Windows Shipping and integrate the parallel
   Android APK work. Add active-session input leases when external Play exists.
2. Wire narrow workflow IPC, capabilities, profile selection and build dashboard.
   Primary Build produces runnable output; retain separately named Build Editor
   Code for module reload. Offer cook-only and code-only, confirmed scoped clean
   and rebuild, plus explicit saved-content handling for dirty documents.
3. Add local standalone Play snapshot providers and development-only authenticated
   runtime control. Keep existing in-editor Play direct. Unsupported referenced
   dirty documents block launch rather than silently using saved content.
4. Integrate Android exact-target deploy/launch and ADB-forwarded session control.
   Re-resolve identity before deployment; never substitute another device.

Pause/resume/step/stop must report runtime acknowledgements. Shipping excludes
the control endpoint. Runtime worlds never mutate authoring data. Run Built uses
the last compatible successful package and its startup scene; it never recooks
or substitutes current unsaved content. Remote desktop, multiplayer and continuous
asset hot reload remain follow-ups.

This foundation does **not** enable standalone/device Play, change the existing
editor-module build semantics, or advertise unfinished packaging as executable.
