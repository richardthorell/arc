import type { BuildCompileTask } from '../../../common/buildCompileTypes';
import type { BuildMonitorRun, BuildMonitorStageState } from '../../../common/buildMonitorTypes';

export type BuildExecutionNode = {
  id: string;
  kind: 'stage' | 'target';
  label: string;
  detail: string;
  state: BuildMonitorStageState | 'observed' | 'satisfied' | 'unreported';
  x: number;
  y: number;
  width: number;
  height: number;
  targetId?: string;
  root?: boolean;
};

export type BuildExecutionEdge = {
  id: string;
  from: string;
  to: string;
  kind: 'stage' | 'dependency' | 'contains';
};

export type BuildExecutionGraphModel = {
  nodes: BuildExecutionNode[];
  edges: BuildExecutionEdge[];
  bounds: { x: number; y: number; width: number; height: number };
};

const STAGE_WIDTH = 168;
const STAGE_HEIGHT = 68;
const TARGET_WIDTH = 190;
const TARGET_HEIGHT = 72;

const targetDepths = (run: BuildMonitorRun): Map<string, number> => {
  const graph = run.targetGraph;
  if (!graph) return new Map();
  const nodes = new Map(graph.nodes.map((node) => [node.id, node]));
  const memo = new Map<string, number>();
  const visiting = new Set<string>();
  const depth = (id: string): number => {
    const cached = memo.get(id);
    if (cached !== undefined) return cached;
    if (visiting.has(id)) return 0;
    visiting.add(id);
    const node = nodes.get(id);
    const value = node?.dependencies.length
      ? 1 + Math.max(...node.dependencies.filter((dependency) => nodes.has(dependency)).map(depth), 0)
      : 0;
    visiting.delete(id);
    memo.set(id, value);
    return value;
  };
  for (const node of graph.nodes) depth(node.id);
  return memo;
};

export const buildExecutionGraphModel = (run: BuildMonitorRun, includeTargets: boolean): BuildExecutionGraphModel => {
  const nodes: BuildExecutionNode[] = run.stages.map((stage, index) => ({
    id: `stage:${stage.id}`,
    kind: 'stage',
    label: stage.name,
    detail: stage.durationMs === undefined ? 'Timing unavailable' : formatDuration(stage.durationMs),
    state: stage.state,
    x: 48 + index * 218,
    y: 42,
    width: STAGE_WIDTH,
    height: STAGE_HEIGHT,
  }));
  const edges: BuildExecutionEdge[] = run.stages.slice(1).map((stage, index) => ({
    id: `stage-edge:${run.stages[index].id}:${stage.id}`,
    from: `stage:${run.stages[index].id}`,
    to: `stage:${stage.id}`,
    kind: 'stage',
  }));

  if (includeTargets && run.targetGraph) {
    const graph = run.targetGraph;
    const depths = targetDepths(run);
    const layers = new Map<number, typeof graph.nodes>();
    for (const node of graph.nodes) {
      const depth = depths.get(node.id) ?? 0;
      const layer = layers.get(depth) ?? [];
      layer.push(node);
      layers.set(depth, layer);
    }
    for (const layer of layers.values())
      layer.sort((left, right) => left.name.localeCompare(right.name) || left.id.localeCompare(right.id));
    const maximumDepth = Math.max(0, ...depths.values());
    const stageWidth = Math.max(760, run.stages.length * 218 + 24);
    const graphWidth = Math.max(760, (maximumDepth + 1) * 244 + 72);
    const originX = 48 + Math.max(0, (stageWidth - graphWidth) / 2);
    const originY = 176;
    for (const [depth, layer] of [...layers].sort(([left], [right]) => left - right)) {
      layer.forEach((target, index) => {
        const root = target.id === graph.rootTargetId;
        nodes.push({
          id: `target:${target.id}`,
          kind: 'target',
          label: target.name,
          detail: target.type.replaceAll('_', ' ').toLowerCase(),
          state: target.satisfiedAt ? 'satisfied' : target.artifactObservedAt ? 'observed' : 'unreported',
          x: originX + depth * 244,
          y: originY + index * 102,
          width: TARGET_WIDTH,
          height: TARGET_HEIGHT,
          targetId: target.id,
          root,
        });
        for (const dependency of target.dependencies) {
          if (!depths.has(dependency)) continue;
          edges.push({
            id: `target-edge:${dependency}:${target.id}`,
            from: `target:${dependency}`,
            to: `target:${target.id}`,
            kind: 'dependency',
          });
        }
      });
    }
    const compile = run.stages.find((stage) => stage.name.toLowerCase() === 'compile');
    if (compile) {
      edges.push({
        id: `contains:${compile.id}:${graph.rootTargetId}`,
        from: `stage:${compile.id}`,
        to: `target:${graph.rootTargetId}`,
        kind: 'contains',
      });
    }
  }

  if (!nodes.length) {
    nodes.push({
      id: 'stage:unavailable',
      kind: 'stage',
      label: run.action,
      detail: 'Stage metadata unavailable',
      state: run.state,
      x: 48,
      y: 42,
      width: STAGE_WIDTH,
      height: STAGE_HEIGHT,
    });
  }
  const right = Math.max(...nodes.map((node) => node.x + node.width));
  const bottom = Math.max(...nodes.map((node) => node.y + node.height));
  return { nodes, edges, bounds: { x: 20, y: 18, width: right, height: bottom + 30 } };
};

export type PackedCompileTask = { task: BuildCompileTask; lane: number; index: number };

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

export const tasksForSelection = (run: BuildMonitorRun, node: BuildExecutionNode | undefined): BuildCompileTask[] => {
  const tasks = run.compileTrace?.tasks ?? [];
  if (!node || node.kind === 'stage') return node?.label.toLowerCase() === 'compile' ? tasks : [];
  if (run.compileTrace?.source !== 'ninja-log') return [];
  const targetName = run.targetGraph?.nodes.find((target) => target.id === node.targetId)?.name;
  return targetName ? tasks.filter((task) => task.targetName === targetName) : [];
};

export const formatDuration = (milliseconds: number): string =>
  milliseconds < 1000 ? `${Math.round(milliseconds)} ms` : `${(milliseconds / 1000).toFixed(2)} s`;
