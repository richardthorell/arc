import { Fragment, useMemo, useState } from 'react';
import { ChartNoAxesGantt, ChevronDown, ChevronRight } from 'lucide-react';

import type { BuildCompileTask } from '../../../common/buildCompileTypes';
import type { BuildMonitorRun } from '../../../common/buildMonitorTypes';
import {
  buildStageTimeline,
  formatDuration,
  groupCompileTasks,
  packCompileTaskLanes,
} from './buildExecutionTimelineModel';

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

function CompileTaskHierarchy({
  run,
  onFilterOutput,
}: {
  run: BuildMonitorRun;
  onFilterOutput?: (value: string) => void;
}) {
  const [selectedTask, setSelectedTask] = useState<BuildCompileTask | null>(null);
  const trace = run.compileTrace;
  const groups = useMemo(() => groupCompileTasks(trace?.tasks ?? []), [trace?.tasks]);
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(() => new Set());
  const tasks = trace?.tasks ?? [];
  const endMs = Math.max(1, ...tasks.map((task) => task.endMs));
  const active = trace?.activeTasks ?? [];

  if (!trace) return <p className="build-execution-empty is-child">Compiler subtasks are unavailable.</p>;
  if (!tasks.length && !active.length)
    return <p className="build-execution-empty is-child">No measured compiler subtasks were reported.</p>;

  const toggleGroup = (id: string) => {
    setCollapsedGroups((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <section className="build-execution-timing is-child" aria-label="Compiler task hierarchy">
      <div className="build-execution-timing-heading">
        <div>
          <strong>Compile subtasks</strong>
          <span>
            {trace.source === 'ninja-log' ? 'Ninja invocation clock' : 'MSBuild CL event clock'} · {tasks.length}{' '}
            measured
          </span>
        </div>
        {trace.truncated && <small>Bounded trace · some tasks omitted</small>}
      </div>
      <div className="build-hierarchical-gantt">
        <div className="build-stage-gantt-axis-label is-nested">Target / task</div>
        <TimelineAxis durationMs={endMs} />
        {groups.map((group) => {
          const collapsed = collapsedGroups.has(group.id);
          const packed = packCompileTaskLanes(group.tasks);
          const laneCount = Math.max(1, ...packed.map((entry) => entry.lane + 1));
          return (
            <div className="build-hierarchical-group" key={group.id}>
              <button
                type="button"
                className="build-hierarchical-label"
                aria-expanded={!collapsed}
                onClick={() => toggleGroup(group.id)}
              >
                {collapsed ? <ChevronRight size={13} /> : <ChevronDown size={13} />}
                <span>
                  <strong>{group.label}</strong>
                  <small>
                    {group.tasks.length} {group.tasks.length === 1 ? 'task' : 'tasks'}
                    {!group.attributed && trace.source === 'msbuild-file-log' ? ' · unattributed' : ''}
                  </small>
                </span>
              </button>
              <div
                className={`build-hierarchical-lanes${collapsed ? ' is-collapsed' : ''}`}
                style={{ height: collapsed ? '38px' : `${Math.max(38, laneCount * 30 + 8)}px` }}
              >
                {!collapsed &&
                  packed.map(({ task, lane, index }) => (
                    <button
                      type="button"
                      key={`${task.output}:${index}`}
                      className="build-execution-task"
                      style={{
                        left: `${(task.startMs / endMs) * 100}%`,
                        top: `${lane * 30 + 8}px`,
                        width: `${Math.max(0.45, (task.durationMs / endMs) * 100)}%`,
                      }}
                      title={`${taskName(task)} · ${formatDuration(task.durationMs)}`}
                      aria-label={`Inspect compile task: ${taskName(task)}`}
                      onClick={() => setSelectedTask((current) => (current === task ? null : task))}
                    >
                      <span>{taskName(task)}</span>
                    </button>
                  ))}
              </div>
            </div>
          );
        })}
      </div>
      {active.length > 0 && (
        <div className="build-execution-active" aria-label="Active compiler tasks">
          <span>
            {active.length} CL {active.length === 1 ? 'task is' : 'tasks are'} active; duration is still open.
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

type WorkflowTaskSpan = ReturnType<typeof buildStageTimeline>['spans'][number]['tasks'][number];

function WorkflowTaskRows({
  task,
  timelineDuration,
  runStartedAt,
  nested,
  collapsed,
  onToggle,
}: {
  task: WorkflowTaskSpan;
  timelineDuration: number;
  runStartedAt: number;
  nested: boolean;
  collapsed: boolean;
  onToggle: () => void;
}) {
  return (
    <>
      <div className={`build-stage-gantt-row${nested ? ' is-task' : ''}`}>
        <div className={`build-stage-gantt-label${nested ? ' is-task' : ''}`}>
          {task.targetGraph ? (
            <button
              type="button"
              className="build-stage-expand"
              aria-label={`${collapsed ? 'Expand' : 'Collapse'} ${task.label} modules`}
              aria-expanded={!collapsed}
              onClick={onToggle}
            >
              {collapsed ? <ChevronRight size={13} /> : <ChevronDown size={13} />}
            </button>
          ) : nested ? (
            <span className="build-stage-task-branch" aria-hidden="true" />
          ) : (
            <span className="build-stage-expand-placeholder" />
          )}
          <i className={`is-${task.state}`} />
          <strong>{task.label}</strong>
          <small>{task.durationMs === undefined ? task.state : formatDuration(task.durationMs)}</small>
        </div>
        <div className={`build-stage-gantt-track${nested ? ' is-task' : ''}`}>
          {task.startMs !== undefined && task.durationMs !== undefined ? (
            <div
              className={`build-stage-gantt-bar is-${task.state}`}
              style={{
                left: `${(task.startMs / timelineDuration) * 100}%`,
                width: `${Math.max(0.5, (task.durationMs / timelineDuration) * 100)}%`,
              }}
              title={`${task.label} · ${formatDuration(task.durationMs)}`}
            />
          ) : (
            <span className="build-stage-gantt-unavailable">{task.state}</span>
          )}
        </div>
      </div>
      {task.targetGraph &&
        !collapsed &&
        task.targetGraph.nodes
          .filter(
            (target) => target.id !== task.targetGraph?.rootTargetId && target.name.toLowerCase() !== 'zero_check',
          )
          .map((target) => {
            const observedAt = target.artifactObservedAt ? Date.parse(target.artifactObservedAt) : NaN;
            const observedOffset = Number.isFinite(observedAt) ? Math.max(0, observedAt - runStartedAt) : undefined;
            return (
              <div className="build-stage-gantt-row is-module" key={`${task.id}:${target.id}`}>
                <div className="build-stage-gantt-label is-module">
                  <span className="build-stage-task-branch" aria-hidden="true" />
                  <i className={target.artifactObservedAt ? 'is-succeeded' : 'is-pending'} />
                  <strong>{target.name}</strong>
                  <small>{target.artifactObservedAt ? 'built' : 'pending'}</small>
                </div>
                <div className="build-stage-gantt-track is-module">
                  {observedOffset !== undefined ? (
                    <span
                      className="build-stage-target-milestone"
                      style={{ left: `${Math.min(100, (observedOffset / timelineDuration) * 100)}%` }}
                      title={`${target.name} output observed`}
                    />
                  ) : (
                    <span className="build-stage-gantt-unavailable">Pending</span>
                  )}
                </div>
              </div>
            );
          })}
    </>
  );
}

function WorkflowHierarchy({
  run,
  onFilterOutput,
}: {
  run: BuildMonitorRun;
  onFilterOutput?: (value: string) => void;
}) {
  const timeline = buildStageTimeline(run);
  const isCompile = (id: string, label: string) => id === 'compile' || label.toLowerCase() === 'compile';
  const hasCompile = timeline.spans.some((span) => isCompile(span.id, span.label));
  const [collapsedStages, setCollapsedStages] = useState<Set<string>>(() => new Set());
  const [collapsedTasks, setCollapsedTasks] = useState<Set<string>>(() => new Set());
  const toggleStage = (id: string) => {
    setCollapsedStages((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };
  const toggleTask = (id: string) => {
    setCollapsedTasks((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };
  const runStartedAt = Date.parse(run.startedAt);
  return (
    <section className="build-stage-gantt" aria-label="Workflow stage hierarchy">
      <header>
        <strong>Workflow</strong>
        <span>Editor workflow clock · {formatDuration(timeline.durationMs)}</span>
      </header>
      <div className="build-stage-gantt-grid">
        <div className="build-stage-gantt-axis-label">Stage</div>
        <TimelineAxis durationMs={timeline.durationMs} />
        {timeline.spans.map((span) => {
          const compile = isCompile(span.id, span.label);
          const flattenTasks = span.label.toLowerCase() === 'configure' && span.tasks.length > 0;
          const expandable = Boolean(span.tasks.length || (compile && run.compileTrace));
          const expanded = expandable && !collapsedStages.has(span.id);
          return (
            <Fragment key={span.id}>
              {!flattenTasks && (
                <div className="build-stage-gantt-row">
                  <div className="build-stage-gantt-label">
                    {expandable ? (
                      <button
                        type="button"
                        className="build-stage-expand"
                        aria-label={`${expanded ? 'Collapse' : 'Expand'} ${span.label} subtasks`}
                        aria-expanded={expanded}
                        onClick={() => toggleStage(span.id)}
                      >
                        {expanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                      </button>
                    ) : (
                      <span className="build-stage-expand-placeholder" />
                    )}
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
              )}
              {(flattenTasks || expanded) &&
                span.tasks.map((task) => (
                  <WorkflowTaskRows
                    key={`${span.id}:${task.id}`}
                    task={task}
                    timelineDuration={timeline.durationMs}
                    runStartedAt={runStartedAt}
                    nested={!flattenTasks}
                    collapsed={collapsedTasks.has(task.id)}
                    onToggle={() => toggleTask(task.id)}
                  />
                ))}
            </Fragment>
          );
        })}
      </div>
      {hasCompile &&
        run.compileTrace &&
        !collapsedStages.has(timeline.spans.find((span) => isCompile(span.id, span.label))?.id ?? '') && (
          <CompileTaskHierarchy run={run} onFilterOutput={onFilterOutput} />
        )}
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
          <span>Hierarchical Gantt</span>
          {run.targetGraph && <small>{run.targetGraph.nodes.length} CMake targets</small>}
          {run.targetGraph?.truncated && <small>Target data truncated</small>}
        </div>
      </div>
      <div className="build-execution-scroll">
        <WorkflowHierarchy run={run} onFilterOutput={onFilterOutput} />
      </div>
    </section>
  );
}
