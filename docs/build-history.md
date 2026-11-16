# Build Monitor history (BM2)

BM2 extends [BM1](build-monitor.md) with persistent, per-project build history while keeping the existing Build Output UI and build execution semantics.

## Project-local storage

A project stores its build history under its own **`Saved/BuildHistory/`** folder (ignored by Git). This is local diagnostic data, not a source-controlled project setting:

- `index.json` — versioned, atomic index with the last 100 runs, complete stage summaries, outcomes and artifacts
- `logs/<run-id>.jsonl` — append-only raw stdout/stderr chunks, including output beyond the legacy 1 MiB tool buffer
- `events/<run-id>.jsonl` — normalized BM1 events, including diagnostic file/line, stage transitions and timestamps

Index writes are limited to run/stage transitions. Logs and events are streamed directly to disk in bounded per-record chunks rather than retaining all output in RAM. On editor restart, a run left in the `running` state is marked failed/interrupted, with its previously captured logs retained.

The implementation uses a JSON index and JSONL rather than native SQLite modules, avoiding Electron ABI dependencies. History file paths are bound to the active project's validated Saved root, reject unsafe symlinks, and never accept an arbitrary renderer-provided path. Retention keeps 100 most recent runs, deleting old log/event files.

## Preload API

- `window.arc.build.history(offset?, limit?)` returns the project-specific run index and total count; default 50 and maximum 100 entries.
- `window.arc.build.historyDetails(runId, logCursor?, eventCursor?)` returns the selected run and at most 100 JSONL records from each stream. Cursors are opaque byte offsets supplied by the previous response, allowing a UI to page very large logs.
- `window.arc.build.monitorSnapshot()` and `window.arc.build.onMonitorEvent(callback)` still provide the bounded live stream.

Project changes detach the prior history, and only IDs present in the current project's index can be opened. Storage failures are reported in `history().storageError` and **never interrupt a compiler or cook process**.

## Raw output capture

`BuildService` forwards each stdout/stderr chunk independently to the history store. `runWorkflowProcess` now provides an optional, untruncated `onRawOutput(text, stream)` callback through its cook and Windows runtime build adapters. Existing `onOutput` continues providing the bounded display tail. BM2's log storage captures both channels *before* in-memory truncation.

## Scope

BM2 adds history and log retrieval contracts, not the Build Monitor dashboard redesign (BM3), detailed target graph instrumentation (BM4/BM5), or inferred build percentage. No stored logs are uploaded or published with runtime packages.
