# Build Monitor: BM7.1 native compilation timings

BM7 starts with measured, per-object compilation task intervals for **Ninja** builds launched by the normal editor Build/Rebuild action. BM6's CMake target graph remains independent and does not imply that a target ran or completed.

## Source of truth

- Before the editor invokes `arc-project build`, it remembers the current byte offset of `paths.build/default/.ninja_log`.
- After that CLI process exits, BM7.1 reads only newly appended Ninja v5 log records, keeps object-file outputs (`.o`/`.obj`), and records Ninja's measured start/end times in milliseconds.
- All intervals share the **Ninja invocation's relative clock**. They correctly show overlap and relative order within compilation, but **cannot be aligned to the editor Build stage or CMake configuration time**. They are not timings for the CMake dependency tree.
- The Build Monitor shows up to 256 measured tasks as a scrollable timeline and saves the bounded trace with the existing project-local BM2 build history. Prior builds lacking the field remain compatible.
- The reader refuses symlinked logs, unsupported log versions and replaced/truncated files. It never copies earlier incremental run records into a fresh run. The native build continues if timings cannot be read.
- A configure-only operation, generator without Ninja v5 logs, or no-op incremental build may have no trace. Missing data is never interpreted as zero compile time or successful target completion.

## Limitations and next PRs

This first PR supports post-build timings from the Ninja v5 log. MSBuild/Visual Studio, Makefiles, Android workflow builds, live task events, source-file mapping, target attribution, per-task log filtering, and before/after run comparisons are follow-up tasks. No compiler instrumentation or build behavior is changed.
