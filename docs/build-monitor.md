# Build Monitor event stream (BM1)

BM1 introduces a **shared observation contract** for editor build commands and the native-planned project workflows. The existing Build Output UI, builder execution, and workflow stage authorities are unchanged.

- `BuildMonitorService` receives the existing `BuildService` and `CookWorkflowService` snapshots in the Electron main process.
- A legacy CMake/tool command gets a stable UUID in `ArcBuildSnapshot.runId` (also recorded in its completed history entry). Each invocation is one monitor run with a single command stage. Rebuild currently creates separate command runs for clean/configure/compile.
- Project workflows reuse the existing `WorkflowRun.id` and native-plan stage names. Stage status changes are observed, not inferred from console text.
- `BuildMonitorRun` captures source, action, configuration, optional platform/profile/target identity, stage timings, result, and artifact information when supplied.
- `BuildMonitorEvent` emits `run-started`, `stage-changed`, `output`, `diagnostic`, and `run-completed`, each with a run ID, optional active-stage ID, monotonic event sequence, and timestamp. Output chunks for workflows have a separate monotonic cursor, so repeated bounded snapshots do not duplicate them.
- `window.arc.build.monitorSnapshot()` returns the current project-scoped run/event snapshot. `window.arc.build.onMonitorEvent(callback)` subscribes to live events and returns an unsubscribe function. The old `build:state` and `workflow:state` channels are retained.
- A project change clears the in-memory run/event data. A stale workflow for a different project GUID is ignored.

**BM1 limits:** The event ring is bounded to 512 events and 24 runs, without disk persistence. Command output currently comes from the existing diagnostic parser, and workflow output from bounded process chunks. It is not yet a complete raw stdout/stderr log archive. BM2 will add durable per-project history and full log capture. BM4/BM5 will add accurate stage/target timings and dependency graphs; BM1 does not manufacture percentages or target dependencies.
