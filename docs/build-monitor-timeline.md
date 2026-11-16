# Build Monitor: measured stage timeline (BM4)

BM4 adds a **measured time-axis waterfall** to the existing [BM3 Build Monitor](build-monitor-ui.md). It uses runner-produced timestamps rather than inferred percentages, estimated durations or fabricated dependencies.

## Timing ownership and persistence

- `ProjectWorkflowRunner` records `startedAt`, `completedAt`, and monotonic `durationMs` for each executed native workflow stage and for the overall run.
- Stage start is recorded **before** the adapter is invoked; stage completion is recorded after the adapter returns and authority checks succeed. A failed/cancelled stage records its end before rollback. Any not-entered stages remain pending with **no timing**.
- `BuildMonitorService` adopts authoritative runner start/end/duration values and saves them in the existing BM2 per-project run index and BM1 `stage-changed` event stream. A late terminal snapshot can restore missed intermediate stage timing; it does not use monitor observation time when the runner provided data.
- Legacy CMake command builds retain one compile/configure/clean stage per command. Their timing comes from the legacy command snapshot and measured command duration.
- Existing BM1/BM2 records without stage timestamps are still readable. The UI displays **Timing unavailable** rather than pretending to know the stage's position.

## Timeline UI

- Every stage has a single lane with its name/status, measured bar positioned relative to the build start, and elapsed duration.
- Horizontal axis ticks are based on the **actual overall build duration**. Active stages extend to the current clock time while the build runs. Paused or historical stages never extend to today's time.
- Pending stages have **Waiting** instead of an invented bar. A stage with elapsed duration but without an absolute start is listed with its duration and an unplaced **Timing unavailable** label.
- Success, running, cancelled and failed states use distinct visual treatments. Screen readers get the exact state, start offset and elapsed duration for each measured bar.
- The timeline presents real **workflow stage order** and measured elapsed times, not CMake target dependencies. The native workflow is currently sequential; this does not claim parallel execution.

## Scope boundaries

BM4 does not parse compiler output for CMake target progress, create a DAG from log lines, estimate percent-complete, or attribute raw stdout/stderr chunks to a stage when no such attribution was recorded. That work remains for **BM5 (CMake target dependency graph and target-level instrumentation)**.

## Validation

Vitest covers runner success/failure/cancellation timing, missed-event recovery, monitor preservation of recorded durations, timeline offsets, active stage growth, older history without timings, and avoiding historical build growth.
