import type { BuildCompileTask } from '../../../common/buildCompileTypes';
import type { BuildMonitorRun } from '../../../common/buildMonitorTypes';

export type BuildTimingDelta = {
  beforeMs: number;
  afterMs: number;
  changeMs: number;
  /** Undefined for a zero baseline; never infer a percentage. */
  changePercent?: number;
};

export type BuildStageDelta = BuildTimingDelta & { name: string };
export type BuildObjectDelta = BuildTimingDelta & { output: string; sourceFile?: string };
export type BuildHistoryComparison = {
  total: BuildTimingDelta;
  stages: BuildStageDelta[];
  /** Only stable, exact Ninja object-output matches may be compared. */
  objects: BuildObjectDelta[];
  matchedObjects: number;
};

function duration(value: number | undefined): value is number {
  return value !== undefined && Number.isFinite(value) && value >= 0;
}

function delta(beforeMs: number, afterMs: number): BuildTimingDelta {
  return {
    beforeMs,
    afterMs,
    changeMs: afterMs - beforeMs,
    ...(beforeMs > 0 ? { changePercent: ((afterMs - beforeMs) / beforeMs) * 100 } : {}),
  };
}

/** Use one unique measured completion per stage name. Ignore ambiguous/partial stages. */
function stages(run: BuildMonitorRun): Map<string, number> {
  const values = new Map<string, number>();
  const ambiguous = new Set<string>();
  for (const stage of run.stages) {
    if (stage.state !== 'succeeded' || !duration(stage.durationMs)) continue;
    if (values.has(stage.name)) ambiguous.add(stage.name);
    else values.set(stage.name, stage.durationMs);
  }
  for (const name of ambiguous) values.delete(name);
  return values;
}

function objects(tasks: BuildCompileTask[]): Map<string, BuildCompileTask> {
  const values = new Map<string, BuildCompileTask>();
  const ambiguous = new Set<string>();
  for (const task of tasks) {
    if (!task.output || !duration(task.durationMs)) continue;
    if (values.has(task.output)) ambiguous.add(task.output);
    else values.set(task.output, task);
  }
  for (const output of ambiguous) values.delete(output);
  return values;
}

function measuredSuccessful(run: BuildMonitorRun): boolean {
  return run.state === 'succeeded' && duration(run.durationMs) && Number.isFinite(Date.parse(run.startedAt));
}

/**
 * BM8.1: strict comparisons only. Do not mix platform, profile, config, target,
 * source, action, project or incomplete/failed runs. Older runs must precede
 * the selected run, even when the history list is not sorted.
 */
export function comparisonBaselines(selected: BuildMonitorRun, history: BuildMonitorRun[]): BuildMonitorRun[] {
  if (!measuredSuccessful(selected)) return [];
  const started = Date.parse(selected.startedAt);
  return history
    .filter(
      (candidate) =>
        candidate.id !== selected.id &&
        measuredSuccessful(candidate) &&
        candidate.projectGuid === selected.projectGuid &&
        candidate.source === selected.source &&
        candidate.action === selected.action &&
        candidate.configuration === selected.configuration &&
        (candidate.platform ?? null) === (selected.platform ?? null) &&
        (candidate.profile ?? null) === (selected.profile ?? null) &&
        (candidate.targetId ?? null) === (selected.targetId ?? null) &&
        Date.parse(candidate.startedAt) < started,
    )
    .sort((a, b) => Date.parse(b.startedAt) - Date.parse(a.startedAt) || a.id.localeCompare(b.id));
}

export function compareBuildHistory(
  selected: BuildMonitorRun,
  baseline: BuildMonitorRun,
): BuildHistoryComparison | null {
  if (!comparisonBaselines(selected, [baseline]).length) return null;
  const previousStages = stages(baseline);
  const nextStages = stages(selected);
  const stageDeltas: BuildStageDelta[] = [];
  for (const [name, after] of nextStages) {
    const before = previousStages.get(name);
    if (before !== undefined) stageDeltas.push({ name, ...delta(before, after) });
  }
  stageDeltas.sort((a, b) => b.changeMs - a.changeMs || a.name.localeCompare(b.name));

  const objectDeltas: BuildObjectDelta[] = [];
  const comparableNinja =
    selected.compileTrace?.source === 'ninja-log' &&
    baseline.compileTrace?.source === 'ninja-log' &&
    selected.compileTrace.clock === baseline.compileTrace.clock;
  if (comparableNinja) {
    const earlier = objects(baseline.compileTrace!.tasks);
    const latest = objects(selected.compileTrace!.tasks);
    for (const [output, task] of latest) {
      const before = earlier.get(output);
      if (before) {
        objectDeltas.push({
          output,
          ...(task.sourceFile ? { sourceFile: task.sourceFile } : {}),
          ...delta(before.durationMs, task.durationMs),
        });
      }
    }
    objectDeltas.sort((a, b) => b.changeMs - a.changeMs || a.output.localeCompare(b.output));
  }
  return {
    total: delta(baseline.durationMs!, selected.durationMs!),
    stages: stageDeltas,
    objects: objectDeltas.slice(0, 8),
    matchedObjects: objectDeltas.length,
  };
}
