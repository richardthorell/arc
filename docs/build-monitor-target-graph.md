# Build Monitor: CMake target dependencies (BM5)

BM5 adds a native CMake File API dependency graph and *evidence-based* target output observations to the existing [BM4 stage timeline](build-monitor-timeline.md).

## Source of truth

- Before the Windows runtime **configure** stage, ARC writes a CMake File API `codemodel-v2` query to the workflow's private build directory.
- After a successful CMake configure, ARC reads the latest native reply for the exact requested configuration and the `arc-runtime-output` root target. The target IDs, names, types, direct dependencies, and artifacts originate from CMake's codemodel target JSON files—not compiler console text.
- ARC includes only up to **128 targets reachable** from the requested root. Unrelated tests/utilities are excluded. The UI labels a truncated graph explicitly. If the reply is missing, invalid, or incompatible, the build still runs; the monitor displays that target metadata is unavailable.
- The workflow runner publishes the graph using its trusted stage context; BM1's shared monitor persists it in the BM2 build history index, including across editor restarts. A `target-graph-changed` event refreshes the live UI.

## Live observations

During the Windows compile stage, the editor watches stdout/stderr for **explicit native artifact output lines** (for example, MSBuild `game.vcxproj -> C:\\...\\game.exe`). A reported artifact counts only if its normalized full path **exactly matches** an artifact already declared by the CMake File API. Each node can be observed at most once per run.

- **Output seen**: CMake-declared artifact explicitly reported by the tool. The monitor records the observed time and its offset on the build timeline.
- **Satisfied**: The explicitly requested `arc-runtime-output` target returned successful `cmake --build`. This does *not* mean it produced a new artifact.
- **Not reported**: No reliable native artifact output was observed. This includes up-to-date incremental targets, skipped targets, custom utility targets, and builds without recognizable output. **This is not failure or pending work.**

The count of observed artifacts is **not a completion percentage**. The graph does not manufacture per-target start times or parallel execution durations. Accurate per-target execution spans would require generator-specific structured build events rather than File API dependency metadata.

## UI

The Build Monitor adds an expandable dependency tree with the requested target at the root, visible graph edges through indentation/expansion, status labels, and an elapsed-time marker for targets whose artifact output was observed. The panel is available for active and archived Windows runtime workflows. Older runs without native metadata show an explicit unavailable state. BM6 extends the same graph view to the normal editor command builds; see [BM6](build-monitor-command-graphs.md).

## Validation

Tests exercise native File API reply traversal, exact configuration and root selection, rejection of traversal paths, chunked output parsing, no invented incremental progress, workflow execution and cancellation behavior, monitor updates, and renderer dependency expansion.

## Scope / follow-up

**BM5** instruments the structured **Windows runtime workflow** first, the one that directly owns the CMake configure and build directory. BM6 adds File API capture to the editor's normal `arc-project` commands. Standalone CLI invocations outside the editor and Android/other generators still need dedicated telemetry integration. There is no universal percent-complete measure and no CMake-target *timing* claim until structured generator-native target events exist.
