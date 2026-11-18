import { Activity } from 'lucide-react';

import type { BuildMonitorRun } from '../../../common/buildMonitorTypes';

const SOURCE_LABELS = {
  'ninja-log': {
    generator: 'Ninja',
    note: 'Compiler object outputs measured by Ninja. Bars show actual task overlap on the Ninja invocation clock, not elapsed time since the editor Build command started. No CMake target completion is inferred.',
    item: 'Object output',
    axis: 'Ninja invocation elapsed →',
    empty: 'No object compilation was recorded in this Ninja invocation.',
  },
  'msbuild-file-log': {
    generator: 'MSBuild',
    note: 'MSBuild CL task start/finish timestamps, relative to the first completed CL task. One CL task may compile multiple source files; these are compiler invocation intervals, not per-file timings. No CMake target completion is inferred.',
    item: 'CL task',
    axis: 'First CL task elapsed →',
    empty: 'No completed CL tasks were recorded by MSBuild.',
  },
} as const;

function duration(ms: number): string {
  return ms < 1000 ? String(ms) + ' ms' : (ms / 1000).toFixed(2) + ' s';
}

/** Native Ninja intervals are relative to Ninja start, NOT the editor command clock. */
export function BuildCompileTasksPanel({ run }: { run: BuildMonitorRun }) {
  const trace = run.compileTrace;
  if (!trace) return null;

  const source = SOURCE_LABELS[trace.source];
  const tasks = trace.tasks.slice(0, 256);
  const endMs = Math.max(1, ...tasks.map((task) => task.endMs));
  return (
    <section className="build-monitor-targets build-monitor-compiles" aria-label="Compilation task timeline">
      <div className="build-monitor-section-title">
        <h4>
          <Activity size={15} aria-hidden="true" /> Native compilation tasks
        </h4>
        <span>{tasks.length} measured · {source.generator}</span>
      </div>
      <p className="build-monitor-target-note">{source.note}</p>
      {tasks.length ? (
        <>
          <div className="build-monitor-compile-heading" aria-hidden="true">
            <span>{source.item}</span>
            <span>{source.axis}</span>
            <span>Duration</span>
          </div>
          <div className="build-monitor-compile-list" role="list" aria-label="Measured compile tasks">
            {tasks.map((task, index) => (
              <div className="build-monitor-compile-row" role="listitem" key={task.output + ':' + index}>
                <span className="build-monitor-compile-name" title={task.output}>
                  {task.output}
                </span>
                <div className="build-monitor-target-track">
                  <span
                    className="build-monitor-compile-bar"
                    style={{
                      left: String((100 * task.startMs) / endMs) + '%',
                      width: String((100 * task.durationMs) / endMs) + '%',
                    }}
                    title={duration(task.startMs) + ' – ' + duration(task.endMs)}
                    aria-label={task.output + ', ' + duration(task.durationMs)}
                  />
                </div>
                <span className="build-monitor-compile-duration">{duration(task.durationMs)}</span>
              </div>
            ))}
          </div>
        </>
      ) : (
        <p className="build-monitor-empty">{source.empty}</p>
      )}
      {trace.truncated && (
        <p className="build-monitor-target-note">
          Timing trace limited to 256 tasks, 24 KiB of history data, or 8–16 MiB of native log data.
        </p>
      )}
    </section>
  );
}
