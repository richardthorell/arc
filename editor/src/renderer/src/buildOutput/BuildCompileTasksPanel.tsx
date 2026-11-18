import { useState } from 'react';
import { Activity, ChevronRight } from 'lucide-react';

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

/** Native intervals are relative to their own tool clock, not the editor command clock. */
export function BuildCompileTasksPanel({
  run,
  onFilterOutput,
}: {
  run: BuildMonitorRun;
  onFilterOutput?: (output: string) => void;
}) {
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const trace = run.compileTrace;
  if (!trace) return null;

  const source = SOURCE_LABELS[trace.source];
  const tasks = trace.tasks.slice(0, 256);
  const selected = selectedIndex === null ? null : (tasks[selectedIndex] ?? null);
  const endMs = Math.max(1, ...tasks.map((task) => task.endMs));
  const measuredLabel = `${tasks.length} measured · ${source.generator}`;
  return (
    <section className="build-monitor-targets build-monitor-compiles" aria-label="Compilation task timeline">
      <div className="build-monitor-section-title">
        <h4>
          <Activity size={15} aria-hidden="true" /> Native compilation tasks
        </h4>
        <span>{measuredLabel}</span>
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
                <button
                  type="button"
                  className="build-monitor-compile-name build-monitor-compile-select"
                  title={task.sourceFile ?? task.output}
                  aria-label={`Inspect compile task: ${task.sourceFile ?? task.output}`}
                  aria-expanded={selectedIndex === index}
                  onClick={() => setSelectedIndex(selectedIndex === index ? null : index)}
                >
                  <ChevronRight size={12} aria-hidden="true" />
                  <span>{task.sourceFile ?? task.output}</span>
                </button>
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
          {selected && (
            <div className="build-monitor-compile-detail" aria-label="Compilation task details">
              <div className="build-monitor-compile-detail-heading">
                <strong>Selected compilation task</strong>
                <span>{duration(selected.durationMs)} measured</span>
              </div>
              <dl>
                <dt>Source</dt>
                <dd>{selected.sourceFile ?? 'Not available from this native compiler task'}</dd>
                <dt>CMake target</dt>
                <dd>{selected.targetName ?? 'Not verified in the tracked target graph'}</dd>
                <dt>Native task</dt>
                <dd>{selected.output}</dd>
              </dl>
              {trace.source === 'ninja-log' && onFilterOutput && (
                <button type="button" className="build-monitor-compile-filter" onClick={() => onFilterOutput(selected.output)}>
                  Find output text in build logs
                </button>
              )}
              <p className="build-monitor-target-note">
                Log search matches recorded text only; it does not establish a per-task timestamp correlation.
              </p>
            </div>
          )}
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
