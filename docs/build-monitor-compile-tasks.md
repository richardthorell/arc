# Build Monitor: BM7.1 native compilation timings

BM7 starts with measured, per-object compilation task intervals for **Ninja** builds launched by the normal editor Build/Rebuild action. BM6's CMake target graph remains independent and does not imply that a target ran or completed.

## Source of truth

- Before the editor invokes `arc-project build`, it remembers the current byte offset of `paths.build/default/.ninja_log`.
- After that CLI process exits, BM7.1 reads only newly appended Ninja v5 log records, keeps object-file outputs (`.o`/`.obj`), and records Ninja's measured start/end times in milliseconds.
- All intervals share the **Ninja invocation's relative clock**. They correctly show overlap and relative order within compilation, but **cannot be aligned to the editor Build stage or CMake configuration time**. They are not timings for the CMake dependency tree.
- The Build Monitor shows up to 256 measured tasks (bounded to roughly 24 KiB per run) as a scrollable timeline and saves the bounded trace with the existing project-local BM2 build history. Prior builds lacking the field remain compatible.
- The reader refuses symlinked logs, unsupported log versions and replaced/truncated files. It never copies earlier incremental run records into a fresh run. The native build continues if timings cannot be read.
- A configure-only operation, generator without Ninja v5 logs, or no-op incremental build may have no trace. Missing data is never interpreted as zero compile time or successful target completion.

## BM7.2 — Visual Studio / MSBuild CL task timings

Normal editor Build and Rebuild now opt into MSBuild's **native file logger** when the active CMake generator is Visual Studio. The project CLI appends the following native MSBuild argument after `cmake --build ... --`:

```text
-flp:LogFile=<project>/Build/default/.arc-monitor/arc-msbuild-<runId>.log;Verbosity=diagnostic;ShowTimestamp;NoItemAndPropertyList;Encoding=UTF-8
```

The option is forwarded only for Visual Studio generators and only when `--msbuild-timing-log` specifies an unused, project-local .arc-monitor log. The build itself and its ordinary console output remain authoritative. An unsupported generator ignores the telemetry flag.

- The editor uses a fresh UUID filename for each build and removes the **raw diagnostic log** after reading it. No previous incremental-build timings are reused, and no diagnostic log contents are stored in BM2.
- The parser pairs real timestamped `Task "CL" (TaskId:...)` and `Done executing task "CL". (TaskId:...)` native logging events, keyed by MSBuild node and task ID. A missing or partial task pair is not assigned a synthetic duration.
- MSBuild intervals are relative to the **earliest completed CL task start** in that native invocation, not to the editor Build stage. Cross-task overlap is measured rather than guessed.
- **One CL task can invoke the compiler for multiple source files**, including `/MP` cases. MSBuild's `CL` task interval is not equivalent to a single translation unit or object output. The monitor labels MSBuild entries as CL tasks, never as per-file compile timings.
- The reader has a 16 MiB input limit and a bounded 256-task / ~24 KiB persisted summary. Missing, malformed or symlinked diagnostics are nonfatal. Incomplete traces are marked truncated.

## Limitations and next PRs

Both Ninja object records and MSBuild CL task intervals are now available for **normal editor commands**. They are post-build measurements rather than live task events. Makefiles, Android workflow builds, source-file mapping, target attribution, per-task log filtering, before/after run comparisons, and per-source timing inside MSBuild CL invocations remain follow-ups.


## BM7.3 — source and target attribution, task drill-down

Normal `arc-project configure/build` requests `CMAKE_EXPORT_COMPILE_COMMANDS=ON`. On Ninja builds, CMake generates `Build/default/compile_commands.json` with compiler source files and concrete object outputs. After BM7.1 has measured the object tasks, the editor joins them to the current compilation database by **exact normalized object output**. It only associates project-relative source paths; duplicate or ambiguous output entries are left unmapped. A target name appears only when the object output names a `CMakeFiles/<target>.dir/` path **and** that target is present in the BM6 codemodel graph.

- Clicking a native compile interval expands a measured duration, exact native output/task, and source/target metadata where recorded. Older history remains readable without attribution fields.
- For Ninja entries, **Find output text in build logs** searches the selected run's already persisted raw output for the exact object-output string. This is a text match, not timestamp correlation or a guarantee that compiler output exists for that task.
- MSBuild `CL` tasks remain **unattributed**, since one `CL` task may compile multiple source files. Their measured task intervals and inspection remain available, but per-file source names and task log shortcuts are intentionally not fabricated.
- Enrichment reads only a bounded, regular, non-symlinked compilation database (up to 8 MiB / 25,000 entries), limits the persisted trace to roughly 32 KiB, and never interrupts compilation.

Follow-ups: per-source timing inside MSBuild `CL` invocations, direct task-ID log correlation (if reliable native events become available), makefile-specific instrumentation, and historical compile-task comparisons.

## BM7.4 — live, measured native compile-task updates

While a normal editor **Build/Rebuild** command runs, the main process checks native timing sources at most once every 750 milliseconds:

- **Ninja:** reads only `.ninja_log` records appended after this invocation's starting cursor. A record becomes visible only after a complete native line is written. Old incremental-build timing records never enter this run's trace. Replaced or shrunken logs are ignored.
- **MSBuild:** reads the unique `.arc-monitor` log without deleting it until process exit. Only paired `CL` task start/finish messages count as measured intervals. Unfinished tasks receive no guessed end times. Raw diagnostic logs are still removed after native build termination.
- Unchanged snapshots are suppressed; `compile-trace-changed` events refresh the active run and Build Monitor timeline, including persisted history. BM7.3's CMake database supplies verified source/target metadata where available.
- Polling stops on process exit/error, cancellation, project switch or editor disposal. Read and parsing failures are nonfatal, and history retains only bounded task data.

**Live means flushed native measurements**, not a predicted build percentage. Some generators/logger configurations buffer their timing files and may only expose task records after process exit. No task is treated as complete until the native tool supplies measured start/end records. MSBuild remains task-level, not per-source timing.

Follow-ups: explicit compiler start/in-flight task events from supported native providers, per-translation-unit MSBuild instrumentation, and historical task-duration comparisons.

## BM7.5 — MSBuild source inputs and active CL tasks

Visual Studio/MSBuild diagnostic logger records can include `CL` task parameter `Sources` values. BM7.5 correlates those parameters with **currently active CL tasks on the same MSBuild logging node**, preserving attribution even when separate nodes interleave their output.

- A completed CL task can list its project-relative source inputs. One source may be shown as the task's associated `sourceFile`; multiple inputs are displayed as a list, **not** individual compilation timings. Source paths outside the project and ambiguous concurrent node associations are ignored.
- A live log snapshot now includes `activeTasks`: CL tasks that have started but have no matching finish record. They appear below the measured timeline with **no duration, end time or completion status**. Completed builds drop the active list, while missing final task pairs still mark the final trace truncated.
- Source metadata is bounded to eight inputs per task, and active-task records have a separate count/byte cap. The raw MSBuild log is still removed after finishing, and only the bounded structured trace is persisted.
- If MSBuild does not emit source parameters in its logger mode, ARC keeps the CL timing and shows no source attribution rather than guessing from compiler command lines, incremental `.tlog` files, or a CMake target name.

**Per-translation-unit wall time is still unsupported for MSVC.** The `CL` task's measured elapsed time can include multiple translation units, `/MP` parallelism and task overhead. MSVC's `/Bt` flag reports compiler component times, not independently verified per-file wall-clock intervals, and its historical diagnostic format is undocumented; ARC does not turn it on automatically. A future backend can use a supported per-source event provider where available.
