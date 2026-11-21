import type { BuildCompileTask } from '../../../common/buildCompileTypes';
import type { BuildMonitorRun, BuildMonitorStageState } from '../../../common/buildMonitorTypes';

export type BuildStageTimelineSpan = {
  id: string;
  label: string;
  state: BuildMonitorStageState;
  startMs?: number;
  durationMs?: number;
  tasks: Array<{
    id: string;
    label: string;
    state: BuildMonitorStageState;
    startMs?: number;
    durationMs?: number;
  }>;
};

export type BuildStageTimeline = {
  spans: BuildStageTimelineSpan[];
  durationMs: number;
};

const timestamp = (value?: string): number | undefined => {
  if (!value) return undefined;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : undefined;
};

/**
 * Workflow timestamps share one wall clock. Older records may retain only
 * measured durations; their planner order is sequential, so those durations
 * can safely be packed after the preceding measured stage. Missing durations
 * remain unavailable and never become zero-length successes.
 */
export const buildStageTimeline = (run: BuildMonitorRun, now = Date.now()): BuildStageTimeline => {
  const runStart = timestamp(run.startedAt);
  let sequentialCursor = 0;
  const spans = run.stages.map((stage) => {
    const started = timestamp(stage.startedAt);
    const completed = timestamp(stage.completedAt);
    const explicitStart = runStart !== undefined && started !== undefined ? Math.max(0, started - runStart) : undefined;
    const measuredDuration =
      stage.durationMs ??
      (started !== undefined && completed !== undefined
        ? Math.max(0, completed - started)
        : stage.state === 'running' && started !== undefined
          ? Math.max(0, now - started)
          : undefined);
    const startMs = explicitStart ?? (measuredDuration !== undefined ? sequentialCursor : undefined);
    if (startMs !== undefined && measuredDuration !== undefined)
      sequentialCursor = Math.max(sequentialCursor, startMs + measuredDuration);
    const tasks = (stage.tasks ?? []).map((task) => {
      const taskStarted = timestamp(task.startedAt);
      const taskCompleted = timestamp(task.completedAt);
      return {
        id: task.id,
        label: task.name,
        state: task.state,
        startMs: runStart !== undefined && taskStarted !== undefined ? Math.max(0, taskStarted - runStart) : undefined,
        durationMs:
          task.durationMs ??
          (taskStarted !== undefined && taskCompleted !== undefined
            ? Math.max(0, taskCompleted - taskStarted)
            : task.state === 'running' && taskStarted !== undefined
              ? Math.max(0, now - taskStarted)
              : undefined),
      };
    });
    return { id: stage.id, label: stage.name, state: stage.state, startMs, durationMs: measuredDuration, tasks };
  });
  const runDuration =
    run.durationMs ??
    (runStart !== undefined
      ? Math.max(0, (timestamp(run.completedAt) ?? (run.state === 'running' ? now : runStart)) - runStart)
      : 0);
  return { spans, durationMs: Math.max(1, runDuration, sequentialCursor) };
};

export type PackedCompileTask = { task: BuildCompileTask; lane: number; index: number };

export type CompileTaskGroup = {
  id: string;
  label: string;
  attributed: boolean;
  tasks: BuildCompileTask[];
};

/** Stable, presentation-only hierarchy built from verified compiler attribution. */
export const groupCompileTasks = (tasks: readonly BuildCompileTask[]): CompileTaskGroup[] => {
  const groups = new Map<string, CompileTaskGroup>();
  for (const task of tasks) {
    const id = task.targetName ? `target:${task.targetName}` : 'unattributed';
    let group = groups.get(id);
    if (!group) {
      group = {
        id,
        label: task.targetName ?? 'Compiler tasks',
        attributed: Boolean(task.targetName),
        tasks: [],
      };
      groups.set(id, group);
    }
    group.tasks.push(task);
  }
  return [...groups.values()].sort((left, right) => {
    if (left.attributed !== right.attributed) return left.attributed ? -1 : 1;
    return left.label.localeCompare(right.label);
  });
};

export const packCompileTaskLanes = (tasks: readonly BuildCompileTask[]): PackedCompileTask[] => {
  const sorted = tasks
    .map((task, index) => ({ task, index }))
    .sort(
      (left, right) =>
        left.task.startMs - right.task.startMs || left.task.endMs - right.task.endMs || left.index - right.index,
    );
  const laneEnds: number[] = [];
  return sorted.map(({ task, index }) => {
    let lane = laneEnds.findIndex((end) => end <= task.startMs);
    if (lane < 0) lane = laneEnds.length;
    laneEnds[lane] = Math.max(task.startMs, task.endMs);
    return { task, lane, index };
  });
};

export const formatDuration = (milliseconds: number): string =>
  milliseconds < 1000 ? `${Math.round(milliseconds)} ms` : `${(milliseconds / 1000).toFixed(2)} s`;
