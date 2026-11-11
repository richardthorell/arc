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
fail closed. Execution is not yet exposed through renderer IPC or the toolbar;
only planning is exposed, with project/SDK revalidation after the CLI response.

## Remaining implementation and delivery

1. Complete executable adapters and exact build artifact/dependency manifests.
   Integrate existing native packaging and the parallel Android APK work; do not
   scan directories for likely executables. Add process-tree cancellation,
   adapter integration tests, incremental reuse and active-session input leases.
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
