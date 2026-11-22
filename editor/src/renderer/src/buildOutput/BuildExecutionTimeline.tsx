import { Fragment, useMemo, useState } from 'react';
import { ChartNoAxesGantt, ChevronDown, ChevronRight } from 'lucide-react';

import type { BuildCompileTask } from '../../../common/buildCompileTypes';
import type { BuildMonitorRun } from '../../../common/buildMonitorTypes';
import type { BuildTargetNode } from '../../../common/buildTargetTypes';
import {
  buildStageTimeline,
  formatDuration,
  groupCompileTasks,
  packCompileTaskLanes,
} from './buildExecutionTimelineModel';

function taskName(task: BuildCompileTask): string {
  return task.sourceFile ?? task.sourceFiles?.join(', ') ?? task.output;
}

const WORKFLOW_WINDOW_MS = 120_000;
const WORKFLOW_GRID_MS = 30_000;

type TimelineViewport = { startMs: number; durationMs: number };

function timelineSpanStyle(startMs: number, durationMs: number, viewport: TimelineViewport) {
  const endMs = startMs + durationMs;
  const viewportEnd = viewport.startMs + viewport.durationMs;
  const visibleStart = Math.max(startMs, viewport.startMs);
  const visibleEnd = Math.min(endMs, viewportEnd);
  if (visibleEnd < visibleStart) return undefined;
  return {
    left: `${((visibleStart - viewport.startMs) / viewport.durationMs) * 100}%`,
    width: `${Math.max(0.5, ((visibleEnd - visibleStart) / viewport.durationMs) * 100)}%`,
  };
}

function timelinePointStyle(pointMs: number, viewport: TimelineViewport) {
  if (pointMs < viewport.startMs || pointMs > viewport.startMs + viewport.durationMs) return undefined;
  return { left: `${((pointMs - viewport.startMs) / viewport.durationMs) * 100}%` };
}

function TimelineAxis({
  durationMs,
  startMs = 0,
  fixedStepMs,
}: {
  durationMs: number;
  startMs?: number;
  fixedStepMs?: number;
}) {
  const parts = fixedStepMs
    ? Array.from(
        { length: Math.floor((startMs + durationMs) / fixedStepMs) - Math.ceil(startMs / fixedStepMs) + 1 },
        (_, index) => (Math.ceil(startMs / fixedStepMs) + index) * fixedStepMs,
      )
    : [0, 0.25, 0.5, 0.75, 1].map((part) => startMs + durationMs * part);
  return (
    <div className="build-timeline-axis" aria-hidden="true">
      {parts.map((value) => (
        <span key={value} style={{ left: `${((value - startMs) / durationMs) * 100}%` }}>
          {formatDuration(value)}
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

function isExternalTarget(target: BuildTargetNode): boolean {
  const identity = `${target.id}/${target.name}/${target.sourceDirectory ?? ''}/${target.buildDirectory ?? ''}`
    .replace(/\\/gu, '/')
    .toLowerCase();
  return (
    identity.includes('third_party') ||
    identity.includes('third-party') ||
    identity.includes('/_deps/') ||
    identity.includes('/external/')
  );
}

function moduleRank(target: BuildTargetNode, task: WorkflowTaskSpan): number {
  if (target.artifactObservedAt) return 0;
  if (task.state === 'running' && target.activePhase) return 1;
  return 2;
}

function moduleStart(target: BuildTargetNode): number {
  const value = target.buildStartedAt ? Date.parse(target.buildStartedAt) : NaN;
  return Number.isFinite(value) ? value : Number.POSITIVE_INFINITY;
}

function sortedModules(targets: BuildTargetNode[], task: WorkflowTaskSpan): BuildTargetNode[] {
  return [...targets].sort((left, right) => {
    const leftRank = moduleRank(left, task);
    const rightRank = moduleRank(right, task);
    if (leftRank !== rightRank) return leftRank - rightRank;
    if (leftRank !== 2 && moduleStart(left) !== moduleStart(right)) return moduleStart(left) - moduleStart(right);
    return left.name.localeCompare(right.name);
  });
}

function ModuleRow({
  target,
  task,
  viewport,
  runStartedAt,
  timelineNowMs,
  external,
}: {
  target: BuildTargetNode;
  task: WorkflowTaskSpan;
  viewport: TimelineViewport;
  runStartedAt: number;
  timelineNowMs: number;
  external?: boolean;
}) {
  const observedAt = target.artifactObservedAt ? Date.parse(target.artifactObservedAt) : NaN;
  const observedOffset = Number.isFinite(observedAt) ? Math.max(0, observedAt - runStartedAt) : undefined;
  const startedAt = target.buildStartedAt ? Date.parse(target.buildStartedAt) : NaN;
  const startedOffset = Number.isFinite(startedAt) ? Math.max(0, startedAt - runStartedAt) : undefined;
  const active = task.state === 'running' && !target.artifactObservedAt ? target.activePhase : undefined;
  const status = target.artifactObservedAt
    ? 'Built'
    : active === 'compiling'
      ? 'Compiling…'
      : active === 'linking'
        ? 'Linking…'
        : 'Pending';
  const pointStyle = observedOffset === undefined ? undefined : timelinePointStyle(observedOffset, viewport);
  const barEnd = observedOffset ?? (active ? timelineNowMs : undefined);
  const barStyle =
    startedOffset !== undefined && barEnd !== undefined
      ? timelineSpanStyle(startedOffset, Math.max(0, barEnd - startedOffset), viewport)
      : undefined;
  return (
    <div className={`build-stage-gantt-row is-module${external ? ' is-external' : ''}`}>
      <div className={`build-stage-gantt-label is-module${external ? ' is-external' : ''}`}>
        <span className="build-stage-task-branch" aria-hidden="true" />
        <i
          className={target.artifactObservedAt ? 'is-succeeded' : active ? 'is-building' : 'is-pending'}
          aria-hidden="true"
        />
        <strong>{target.name}</strong>
        <small>{status}</small>
      </div>
      <div className="build-stage-gantt-track is-module">
        {barStyle ? (
          <span
            className={`build-stage-gantt-bar is-module ${target.artifactObservedAt ? 'is-succeeded' : 'is-running'}`}
            style={barStyle}
            title={`${target.name} · ${formatDuration(Math.max(0, barEnd! - startedOffset!))}`}
          />
        ) : pointStyle ? (
          <span className="build-stage-target-milestone" style={pointStyle} title={`${target.name} output observed`} />
        ) : !target.artifactObservedAt ? (
          <span className="build-stage-gantt-unavailable">Pending</span>
        ) : null}
      </div>
    </div>
  );
}

function WorkflowTaskRows({
  task,
  viewport,
  runStartedAt,
  timelineNowMs,
  nested,
  collapsed,
  onToggle,
}: {
  task: WorkflowTaskSpan;
  viewport: TimelineViewport;
  runStartedAt: number;
  timelineNowMs: number;
  nested: boolean;
  collapsed: boolean;
  onToggle: () => void;
}) {
  const [externalCollapsed, setExternalCollapsed] = useState(true);
  const targets = (task.targetGraph?.nodes ?? []).filter(
    (target) => target.id !== task.targetGraph?.rootTargetId && target.name.toLowerCase() !== 'zero_check',
  );
  const projectTargets = sortedModules(
    targets.filter((target) => !isExternalTarget(target)),
    task,
  );
  const externalTargets = sortedModules(targets.filter(isExternalTarget), task);
  const externalActive = externalTargets.find(
    (target) => task.state === 'running' && !target.artifactObservedAt && target.activePhase,
  );
  const externalBuilt = externalTargets.length > 0 && externalTargets.every((target) => target.artifactObservedAt);
  const externalProgress = externalTargets.some((target) => target.artifactObservedAt);
  const externalBuilding = Boolean(externalActive || (externalProgress && task.state === 'running'));
  const moduleEntries: Array<
    | { kind: 'target'; target: BuildTargetNode; rank: number; name: string; startedAt: number }
    | { kind: 'external'; rank: number; name: string; startedAt: number }
  > = [
    ...projectTargets.map((target) => ({
      kind: 'target' as const,
      target,
      rank: moduleRank(target, task),
      name: target.name,
      startedAt: moduleStart(target),
    })),
    ...(externalTargets.length
      ? [
          {
            kind: 'external' as const,
            rank: externalBuilt ? 0 : externalBuilding ? 1 : 2,
            name: 'Third-party',
            startedAt: Math.min(...externalTargets.map(moduleStart)),
          },
        ]
      : []),
  ].sort((left, right) => {
    if (left.rank !== right.rank) return left.rank - right.rank;
    if (left.rank !== 2 && left.startedAt !== right.startedAt) return left.startedAt - right.startedAt;
    return left.name.localeCompare(right.name);
  });
  const taskMeasured = task.startMs !== undefined && task.durationMs !== undefined;
  const taskStyle = taskMeasured ? timelineSpanStyle(task.startMs!, task.durationMs!, viewport) : undefined;
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
          {taskStyle ? (
            <div
              className={`build-stage-gantt-bar is-${task.state}`}
              style={taskStyle}
              title={`${task.label} · ${formatDuration(task.durationMs!)}`}
            />
          ) : !taskMeasured ? (
            <span className="build-stage-gantt-unavailable">{task.state}</span>
          ) : null}
        </div>
      </div>
      {task.targetGraph &&
        !collapsed &&
        moduleEntries.map((entry) =>
          entry.kind === 'target' ? (
            <ModuleRow
              key={`${task.id}:${entry.target.id}`}
              target={entry.target}
              task={task}
              viewport={viewport}
              runStartedAt={runStartedAt}
              timelineNowMs={timelineNowMs}
            />
          ) : (
            <Fragment key={`${task.id}:third-party`}>
              <div className="build-stage-gantt-row is-module-group">
                <div className="build-stage-gantt-label is-module is-group">
                  <button
                    type="button"
                    className="build-stage-expand"
                    aria-label={`${externalCollapsed ? 'Expand' : 'Collapse'} Third-party modules`}
                    aria-expanded={!externalCollapsed}
                    onClick={() => setExternalCollapsed((value) => !value)}
                  >
                    {externalCollapsed ? <ChevronRight size={13} /> : <ChevronDown size={13} />}
                  </button>
                  <i className={externalBuilt ? 'is-succeeded' : externalBuilding ? 'is-building' : 'is-pending'} />
                  <strong>Third-party</strong>
                  <small>
                    {externalActive?.activePhase === 'compiling'
                      ? 'Compiling…'
                      : externalActive?.activePhase === 'linking'
                        ? 'Linking…'
                        : externalBuilt
                          ? 'Built'
                          : externalBuilding
                            ? 'Building…'
                            : `${externalTargets.length} modules`}
                  </small>
                </div>
                <div className="build-stage-gantt-track is-module">
                  <span className="build-stage-gantt-unavailable">{externalTargets.length} modules</span>
                </div>
              </div>
              {!externalCollapsed &&
                externalTargets.map((target) => (
                  <ModuleRow
                    key={`${task.id}:${target.id}`}
                    target={target}
                    task={task}
                    viewport={viewport}
                    runStartedAt={runStartedAt}
                    timelineNowMs={timelineNowMs}
                    external
                  />
                ))}
            </Fragment>
          ),
        )}
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
  const [followRight, setFollowRight] = useState(true);
  const [scrubStart, setScrubStart] = useState(0);
  const maxScrubStart = Math.max(0, timeline.durationMs - WORKFLOW_WINDOW_MS);
  const viewportStart = followRight ? maxScrubStart : Math.min(scrubStart, maxScrubStart);
  const viewport = useMemo<TimelineViewport>(
    () => ({ startMs: viewportStart, durationMs: WORKFLOW_WINDOW_MS }),
    [viewportStart],
  );
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
      <div className="build-timeline-scrubber">
        <input
          type="range"
          aria-label="Scrub workflow timeline"
          min={0}
          max={Math.max(1, maxScrubStart)}
          step={1000}
          value={maxScrubStart === 0 ? 0 : viewportStart}
          disabled={maxScrubStart === 0}
          onChange={(event) => {
            const value = Number(event.currentTarget.value);
            setScrubStart(value);
            setFollowRight(value >= maxScrubStart - 500);
          }}
        />
        <button
          type="button"
          className={followRight ? 'is-following' : ''}
          aria-pressed={followRight}
          onClick={() => setFollowRight(true)}
        >
          {followRight ? 'Following live' : 'Follow live'}
        </button>
      </div>
      <div className="build-stage-gantt-grid">
        <div className="build-stage-gantt-axis-label">Stage</div>
        <TimelineAxis durationMs={viewport.durationMs} startMs={viewport.startMs} fixedStepMs={WORKFLOW_GRID_MS} />
        {timeline.spans.map((span) => {
          const compile = isCompile(span.id, span.label);
          const flattenTasks = span.label.toLowerCase() === 'configure' && span.tasks.length > 0;
          const expandable = Boolean(span.tasks.length || (compile && run.compileTrace));
          const expanded = expandable && !collapsedStages.has(span.id);
          const spanMeasured = span.startMs !== undefined && span.durationMs !== undefined;
          const spanStyle = spanMeasured ? timelineSpanStyle(span.startMs!, span.durationMs!, viewport) : undefined;
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
                    {spanStyle ? (
                      <div
                        className={`build-stage-gantt-bar is-${span.state}`}
                        style={spanStyle}
                        title={`${span.label} · ${formatDuration(span.durationMs!)}`}
                      />
                    ) : !spanMeasured ? (
                      <span className="build-stage-gantt-unavailable">
                        {span.state === 'pending' ? 'Pending' : 'Timing unavailable'}
                      </span>
                    ) : null}
                  </div>
                </div>
              )}
              {(flattenTasks || expanded) &&
                span.tasks.map((task) => (
                  <WorkflowTaskRows
                    key={`${span.id}:${task.id}`}
                    task={task}
                    viewport={viewport}
                    runStartedAt={runStartedAt}
                    timelineNowMs={timeline.durationMs}
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
        <WorkflowHierarchy key={run.id} run={run} onFilterOutput={onFilterOutput} />
      </div>
    </section>
  );
}
