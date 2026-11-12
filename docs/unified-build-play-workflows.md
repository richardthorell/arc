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
fail closed. Saved-content cooking is now executable through narrow renderer IPC;
the toolbar remains unchanged until profile selection and the saved-content prompt
are implemented.

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

## Remaining implementation and delivery

1. Complete executable adapters and exact build artifact/dependency manifests.
   Integrate existing native packaging and the parallel Android APK work; do not
   scan directories for likely executables. Extend the process runner and tests to
   compilation/packaging, incremental build reuse and active-session input leases.
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

This foundation does **not** enable standalone/device Play, change the current
toolbar's Build semantics, or advertise unfinished packaging as executable.
