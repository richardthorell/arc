import { useMemo, useState } from 'react';
import { ChartNoAxesGantt } from 'lucide-react';

import type { BuildCompileTask } from '../../../common/buildCompileTypes';
import type { BuildMonitorRun } from '../../../common/buildMonitorTypes';
import { buildStageTimeline, formatDuration, packCompileTaskLanes } from './buildExecutionTimelineModel';

function taskName(task: BuildCompileTask): string {
  return task.sourceFile ?? task.sourceFiles?.join(', ') ?? task.output;
}

function TimelineAxis({ durationMs }: { durationMs: number }) {
  return (
    <div className="build-timeline-axis" aria-hidden="true">
      {[0, 0.25, 0.5, 0.75, 1].map((part) => (
        <span key={part} style={{ left: `${part * 100}%` }}>
          {formatDuration(durationMs * part)}
        </span>
      ))}
    </div>
  );
}

function WorkflowStageLanes({ run }: { run: BuildMonitorRun }) {
  const timeline = buildStageTimeline(run);
  return (
    <section className="build-stage-gantt" aria-label="Workflow stage timeline">
      <header>
        <strong>Workflow stages</strong>
        <span>Editor workflow clock · {formatDuration(timeline.durationMs)}</span>
      </header>
      <div className="build-stage-gantt-grid">
        <div className="build-stage-gantt-axis-label">Stage</div>
        <TimelineAxis durationMs={timeline.durationMs} />
        {timeline.spans.map((span) => (
          <div className="build-stage-gantt-row" key={span.id}>
            <div className="build-stage-gantt-label">
              <i className={`is-${span.state}`} />
              <strong>{span.label}</strong>
              <small>{span.durationMs === undefined ? span.state : formatDuration(span.durationMs)}</small>
            </div>
            <div className="build-stage-gantt-track">
              {span.startMs !== undefined && span.durationMs !== undefined ? (
                <div
                  className={`build-stage-gantt-bar is-${span.state}`}
                  style={{
                    left: `${(span.startMs / timeline.durationMs) * 100}%`,
                    width: `${Math.max(0.5, (span.durationMs / timeline.durationMs) * 100)}%`,
                  }}
                  title={`${span.label} · ${formatDuration(span.durationMs)}`}
                />
              ) : (
                <span className="build-stage-gantt-unavailable">
                  {span.state === 'pending' ? 'Pending' : 'Timing unavailable'}
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function CompileTaskLanes({ run, onFilterOutput }: { run: BuildMonitorRun; onFilterOutput?: (value: string) => void }) {
  const [selectedTask, setSelectedTask] = useState<BuildCompileTask | null>(null);
  const trace = run.compileTrace;
  const tasks = useMemo(() => trace?.tasks ?? [], [trace?.tasks]);
  const packed = useMemo(() => packCompileTaskLanes(tasks), [tasks]);
  const laneCount = Math.max(1, ...packed.map((entry) => entry.lane + 1));
  const endMs = Math.max(1, ...tasks.map((task) => task.endMs));
  const active = trace?.activeTasks ?? [];

  if (!trace) {
    return <p className="build-execution-empty">Parallel compiler timing is unavailable for this build.</p>;
  }
  if (!tasks.length && !active.length) {
    return <p className="build-execution-empty">No measured compiler tasks were reported for this build.</p>;
  }

  return (
    <section className="build-execution-timing" aria-label="Compiler task timeline">
      <div className="build-execution-timing-heading">
        <div>
          <strong>Compile tasks</strong>
          <span>
            {trace.source === 'ninja-log' ? 'Ninja invocation clock' : 'MSBuild CL event clock'} · {tasks.length}{' '}
            measured
          </span>
        </div>
        {trace.truncated && <small>Bounded trace · some tasks omitted</small>}
      </div>
      <div className="build-compile-gantt">
        <div className="build-compile-gantt-label">Parallel lanes</div>
        <div className="build-execution-lanes" style={{ height: `${Math.max(52, laneCount * 30 + 25)}px` }}>
          <TimelineAxis durationMs={endMs} />
          {packed.map(({ task, lane, index }) => (
            <button
              type="button"
              key={`${task.output}:${index}`}
              className="build-execution-task"
              style={{
                left: `${(task.startMs / endMs) * 100}%`,
                top: `${lane * 30 + 24}px`,
                width: `${Math.max(0.45, (task.durationMs / endMs) * 100)}%`,
              }}
              title={`${taskName(task)} · ${formatDuration(task.durationMs)}`}
              aria-label={`Inspect compile task: ${taskName(task)}`}
              onClick={() => setSelectedTask((current) => (current === task ? null : task))}
            />
          ))}
        </div>
      </div>
      {active.length > 0 && (
        <div className="build-execution-active" aria-label="Active compiler tasks">
          <span>
            {active.length} CL {active.length === 1 ? 'task is' : 'tasks are'} active; no end timestamp or duration has
            been reported.
          </span>
          {active.map((task) => (
            <div key={task.output} title={task.sourceFiles?.join(', ') ?? task.output}>
              <i />
              <code>{task.output}</code>
            </div>
          ))}
        </div>
      )}
      {selectedTask && (
        <div className="build-execution-task-detail">
          <div>
            <strong>{taskName(selectedTask)}</strong>
            <span>{formatDuration(selectedTask.durationMs)}</span>
          </div>
          <code>{selectedTask.output}</code>
          {selectedTask.targetName && (
            <dl>
              <dt>CMake target</dt>
              <dd>{selectedTask.targetName}</dd>
            </dl>
          )}
          {trace.source === 'ninja-log' && onFilterOutput && (
            <button type="button" onClick={() => onFilterOutput(selectedTask.output)}>
              Find output text in build logs
            </button>
          )}
        </div>
      )}
      <p className="build-execution-clock-note">
        Native compiler timing uses its own tool clock and is not aligned to the workflow timeline.
      </p>
    </section>
  );
}

export function BuildExecutionTimeline({
  run,
  onFilterOutput,
}: {
  run: BuildMonitorRun;
  onFilterOutput?: (value: string) => void;
}) {
  return (
    <section className="build-execution" aria-label="Build execution timeline">
      <div className="build-execution-toolbar">
        <div>
          <ChartNoAxesGantt size={15} aria-hidden="true" />
          <strong>Execution timeline</strong>
          <span>Stacked Gantt</span>
          {run.targetGraph && <small>{run.targetGraph.nodes.length} CMake targets</small>}
          {run.targetGraph?.truncated && <small>Target data truncated</small>}
        </div>
      </div>
      <div className="build-execution-scroll">
        <WorkflowStageLanes run={run} />
        <CompileTaskLanes run={run} onFilterOutput={onFilterOutput} />
      </div>
    </section>
  );
}
