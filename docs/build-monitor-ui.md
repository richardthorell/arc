# Build Monitor UI (BM3+)

BM3 introduced the persistent **Build Monitor** backed by the [BM1 event stream](build-monitor.md) and [BM2 build history](build-history.md). Later milestones added [measured workflow stages](build-monitor-timeline.md), [CMake target dependencies](build-monitor-target-graph.md), command-build graphs and [native compiler intervals](build-monitor-compile-tasks.md).

## Timeline-first layout

- **Compact build history:** The running build is pinned above persisted previous runs. Rows contain only status, action/configuration, time and duration. Selecting history never interrupts or retargets the active build.
- **Compact summary:** Action/configuration, state, measured duration, diagnostic counts and selected-run actions share one row. Failed, cancelled and interrupted runs add one concise callout that navigates to the relevant diagnostic or output.
- **Execution timeline:** Planner-provided stages form the authoritative workflow Gantt. Measured stage boundaries share the editor workflow clock; missing timing remains visibly unavailable.
- **SDK preparation:** When the Windows development SDK must be prepared, Configure's authoritative child operations are flattened into **Build SDK** and **Configure project** rows. Build SDK expands to the generated CMake/MSBuild project hierarchy so each engine module is visible while it builds; the redundant Configure parent is hidden in this case.
- **SDK module progress:** Generated project references establish module identity and dependency structure before compilation starts. Explicit native build artifact lines mark observed module-completion milestones. ARC does not invent per-module start times or durations when MSBuild does not provide them.
- **Compile lanes:** Native Ninja/MSBuild compiler intervals are packed into stacked lanes so parallel work remains visible. Their native clock is labelled separately and is never aligned artificially to the workflow clock.
- **Target metadata:** CMake target counts and bounded-data status remain available as context, but dependency relationships no longer define the primary visualization.
- **Parallel task detail:** Selecting Compile shows all measured native compiler intervals. Selecting a CMake target filters only verified Ninja attribution; MSBuild CL tasks remain under Compile because their target cannot be identified safely. Intervals are packed into lanes to expose overlap on the native tool clock, which is explicitly not aligned to workflow time.
- **Output and diagnostics:** The primary surface directly below the graph retains stdout/stderr archive paging, search of loaded chunks, stream filtering and clickable file diagnostics.
- **Secondary details:** Published artifacts and compatible historical comparison remain available in a Details drawer without displacing the graph or output.
- **Live updates:** Stage, target-graph and compile-trace events refresh the selected graph. A running selected build polls additional archived output while keeping memory bounded.

## Data behavior

- Initial selection prefers the active run, then the latest persisted run. Both current and previous builds remain selectable.
- History is project-scoped and never mixes runs across projects.
- Changing selection clears the prior run's diagnostic and output content. Background events never steal selection from a historical run.
- The log view holds at most 1,200 chunks and 1,600 events in memory. Earlier records remain in `Saved/BuildHistory` and can be reread with **Read from start**. Search covers loaded records only.
- Missing, partial, older or truncated graph/timing metadata is labeled explicitly. Missing measurements are never displayed as zero duration.
- The compact Build Output panel and existing configure/build/rebuild/clean commands are unchanged. Active command builds and workflows keep their existing cancellation paths.

## Validation

Editor format, lint, typecheck and Vitest cover deterministic DAG layout, shared dependencies, workflow fallback, interval lane packing, verified target attribution, active and archived selections, paging/filtering, diagnostic navigation, live updates and storage-error states.
