# Build Monitor BM8.1 — measured history comparisons

BM8.1 provides a focused comparison panel for a **completed successful build**. It uses already persisted BM2 history (run duration and stage measurements) and BM7's bounded native compile traces. No new history schema, backend endpoint, or source instrumentation is required.

## Baseline rules

The default baseline is the most recent **earlier successful run** matching the selected run's project GUID, command/workflow source, action, configuration, platform, target ID and profile. Users can select any other compatible older successful run from the retained history list. A missing duration, non-finite value, failed/cancelled/running status, or a run newer than the selected run disqualifies a baseline.

A baseline with measured duration zero can show absolute changes but never a percentage change, since division by zero would mislead. A run with missing measurements is not silently treated as zero.

## Measurements

- **Build elapsed time:** selected run duration minus baseline run duration, with percentage relative to a nonzero baseline.
- **Stages:** pairs of succeeded stages with unique matching names and recorded nonnegative durations. Missing, ambiguous or unfinished stages are omitted. Stage identities are not inferred from order or dependency graph.
- **Ninja objects:** exact matching object output paths from two Ninja invocation traces. Only recorded task duration is compared; source paths are displayed as labels where available. Duplicate output identities are treated as ambiguous. Up to eight of the largest positive differences are displayed, with the total number of matches reported.
- **MSBuild:** total and stage comparisons work, but task IDs and their source groupings are not stable between invocations. CL task-to-CL task comparisons are deliberately omitted.

The display shows *actual measured differences*, not a statistical or normalized regression judgment. Differences can result from incremental build caches, changed code, machine contention or environment differences.

## Follow-ups

BM8.2 can add longer-window historical trends and filtering while preserving strict comparability. A later track may record explicit build-environment metadata (toolchain version, cold/warm status, machine identity) to support stronger regression assessment. Avoid automatic "performance regression" alerts until that evidence is available.
