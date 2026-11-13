# ARC CI workflow naming and checks

GitHub Actions is organized by **intent** at workflow level and by
**platform · architecture · compiler/runtime · configuration — purpose** at job level.

| Workflow | Purpose |
| --- | --- |
| Pull Request | Change-based validation and the stable `PR Gate` required check |
| 01 · Quality — Editor & Native | Independent dependency, C++ format, C++ analysis, editor static, and editor test jobs |
| 02 · Windows x64 — MSVC (VS2022) | Native tests, Electron editor package, and artifact publication orchestrator |
| 03 · Windows x64 — Content, SDK & Distribution | Reusable asset cook, D3 Shipping boot acceptance, SDK matrix and runtime package |
| 02 · Linux x64 — Clang 18 / GCC 14 | Native compiler matrix sharing CMake CI presets; Clang extended validation |
| 03 · Android arm64 — NDK r30 / Clang | Debug APK build |
| 04 · Coverage — Linux x64 / Clang 18 | LLVM test coverage |
| 05 · Performance — CPU Regression | Benchmark regression check |
| 06 · Documentation — Doxygen & Pages | API docs generation/deployment |
| 07 · Maintenance — PR Format Fix / Rebase | Explicit comment-triggered PR maintenance |

## Pull request checks

The `Pull Request` workflow remains the only PR entrypoint. The
`tools/classify_ci_changes.py` classifier continues to choose the required
compiler, Android, coverage, quality, shader and cooker work from changed paths.
The existing `PR Gate` job ID/name is unchanged; it continues to aggregate
required workflow results. Renamed reusable workflow jobs are *not* independent
new branch-protection requirements.

The editor quality workflow is split into five independently diagnosable jobs:

1. **Dependency Policy & Tool Tests** — Python policy checks and tool unit tests.
2. **C++ Formatting** — clang-format without a CMake configure/build.
3. **Native Static Analysis** — clang-tidy with a generated CMake compile database.
4. **Editor Format / ESLint / Typecheck** — Node-only, no CMake build.
5. **Editor Unit & Integration Tests** — Electron/Vitest plus the native
   `arc-project-cli` executable required by the project workflow tests.

For editor-only PRs, C++ formatting/static analysis do not run. For native-only
PRs, editor jobs do not run. Policy always runs when quality is selected.
Full checks continue on pushes to `main` and manual workflow dispatch.

## Scope boundaries

CI.1 changes visibility and quality-job isolation only. It does not change the
existing PR path classifier, packaging, Android runtime tests or branch gate
semantics. Platform/distribution job extraction and new emulator/server tests
belong to CI.2/CI.3.

The obsolete branch-write automation for
`feature/material-editor-foundation` and `feature/ai-assistant-stage-1`
was removed after confirming both target branches no longer exist.

## CI.2: distribution isolation and shared native configurations

`build-msvc.yml` is still the existing entrypoint for Windows PR checks and
`main` builds. It owns the native build/test, editor package, and publication
orchestration. It invokes `windows-distribution.yml` as a nested reusable
workflow, keeping the original four job IDs and their behavior:
`asset-cook`, `d3-shipping-acceptance`, `sdk-validation`, and
`shipping-runtime`. The Shipping runtime still depends on the verified cook
artifact. Artifacts are produced and consumed **within the same GitHub Actions
run**, without cross-workflow-run artifact discovery or permissions. On PRs,
the original `run_cooker` selector still controls both cooker and D3 tests;
SDK and Shipping runtime remain main-only. The editor package still depends
on the Windows native build, and artifact publication waits for native,
distribution, and editor jobs to succeed.

Linux Clang 18 and GCC 14 share `ci-linux-vulkan-base` in
`CMakePresets.json`; each toolchain preset retains its compiler,
binary directory, and original target-specific flags. Cache keys, test runs,
shader checks, and main-only installed-SDK verification are unchanged. The
Clang installed-SDK stages remain in the Clang build job until an artifact
handoff can be added without rebuilding the same SDK twice.

The root README shows the `Pull Request` workflow badge, scoped to
`event=pull_request` rather than to `branch=main` (the PR workflow runs
against PR heads). It is an overview of the latest PR workflow run,
**not** a claim that every PR or every `main` build is green.
