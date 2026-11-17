# Build Monitor UI (BM3)

BM3 replaces the session-only command statistics dialog with a **Build Monitor** backed by the merged [BM1 event stream](build-monitor.md) and [BM2 build history](build-history.md). BM4 adds a [measured stage timeline](build-monitor-timeline.md) using runner-reported start/end times. BM5 adds [CMake target dependencies and native artifact observations](build-monitor-target-graph.md) for structured Windows builds, without inventing target completion percentages. BM6 extends this to [the editor's normal command builds](build-monitor-command-graphs.md).

## Layout

- **Left — Build history:** The currently running build is pinned above the project's persisted previous runs. Selecting an earlier run does **not** interrupt or switch the active build.
- **Main summary:** Action, configuration, platform/profile/target, real elapsed or completed duration, error/warning counts and measured stage completion. No fabricated percent-complete estimate.
- **Stages:** Displays the existing planner's stage statuses and measured durations. It is **not** a CMake target dependency graph; graph and target-level timings are scheduled for BM4/BM5. Raw process chunks do not yet carry a target/stage ID, so this UI does not pretend to filter full logs by dependency node.
- **Bottom — Output & diagnostics:** Resizable panel with full stdout/stderr archive reading, search of *loaded* chunks, stream filtering and clickable file diagnostics. Logs are paged from disk in 100-record steps using BM2's byte cursors.
- **Live updates:** BM1 stage events refresh the summary; the selected running build polls additional archived output while keeping UI memory bounded.
- **Errors:** History storage/read failures are displayed in the monitor and do not affect compilation.

## Data behavior

- Initial selection prefers the active run, then the latest persisted run. Both current and previous builds are selectable.
- The history panel is per-project and never mixes runs across projects.
- Changing history selection clears the prior run's diagnostic/output content. The previous run stays selected when another live build event arrives.
- The log view holds at most 1,200 chunks/1,600 events in memory; earlier chunks remain in `Saved/BuildHistory` and can be reread with **Read from start**. Search intentionally covers only loaded chunks, not an invented whole-archive index.
- The compact Build Output panel, configure/build/rebuild/clean buttons, and legacy diagnostics are unchanged. The expand button opens the new monitor; both command builds and active project workflows can be cancelled there, using their existing cancellation APIs.
- No persistent dashboard preferences or build commands are added in BM3.

## Validation

Editor format, lint, typecheck and Vitest, with new tests covering active and archived selections, log paging/filtering, diagnostic navigation, preserving selection during live updates, empty/storage-error states, and opening from the compact Build Output panel.
