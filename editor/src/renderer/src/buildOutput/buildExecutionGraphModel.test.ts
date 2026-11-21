import { describe, expect, it } from 'vitest';

import type { BuildMonitorRun } from '../../../common/buildMonitorTypes';
import { buildExecutionGraphModel, packCompileTaskLanes, tasksForSelection } from './buildExecutionGraphModel';

const run = (): BuildMonitorRun => ({
  id: 'run',
  source: 'command',
  projectGuid: 'project',
  action: 'build',
  configuration: 'Debug',
  startedAt: '2026-11-20T12:00:00Z',
  state: 'succeeded',
  warningCount: 0,
  errorCount: 0,
  stages: [
    { id: 'configure', name: 'Configure', state: 'succeeded', durationMs: 10 },
    { id: 'compile', name: 'Compile', state: 'succeeded', durationMs: 100 },
    { id: 'package', name: 'Package', state: 'succeeded', durationMs: 20 },
  ],
  targetGraph: {
    revision: 1,
    generator: 'Ninja',
    configuration: 'Debug',
    rootTargetId: 'game',
    discoveredAt: '2026-11-20T12:00:00Z',
    truncated: false,
    nodes: [
      { id: 'game', name: 'game', type: 'EXECUTABLE', dependencies: ['render', 'scene'], artifacts: [] },
      { id: 'render', name: 'render', type: 'STATIC_LIBRARY', dependencies: ['core'], artifacts: [] },
      { id: 'scene', name: 'scene', type: 'STATIC_LIBRARY', dependencies: ['core'], artifacts: [] },
      { id: 'core', name: 'core', type: 'STATIC_LIBRARY', dependencies: [], artifacts: [] },
    ],
  },
  compileTrace: {
    source: 'ninja-log',
    clock: 'ninja-invocation',
    truncated: false,
    tasks: [
      { output: 'render-a.obj', targetName: 'render', startMs: 0, endMs: 50, durationMs: 50 },
      { output: 'render-b.obj', targetName: 'render', startMs: 10, endMs: 30, durationMs: 20 },
      { output: 'scene.obj', targetName: 'scene', startMs: 50, endMs: 80, durationMs: 30 },
    ],
  },
});

describe('Build execution graph model', () => {
  it('lays out workflow stages and shared target dependencies deterministically', () => {
    const first = buildExecutionGraphModel(run(), true);
    const second = buildExecutionGraphModel(run(), true);
    expect(second).toEqual(first);
    expect(first.nodes.filter((node) => node.kind === 'stage').map((node) => node.label)).toEqual([
      'Configure',
      'Compile',
      'Package',
    ]);
    expect(first.edges).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ from: 'target:core', to: 'target:render', kind: 'dependency' }),
        expect.objectContaining({ from: 'target:core', to: 'target:scene', kind: 'dependency' }),
        expect.objectContaining({ from: 'target:render', to: 'target:game', kind: 'dependency' }),
      ]),
    );
    expect(first.nodes.find((node) => node.id === 'target:core')!.x).toBeLessThan(
      first.nodes.find((node) => node.id === 'target:game')!.x,
    );
  });

  it('falls back to authoritative workflow stages when target metadata is unavailable', () => {
    const value = run();
    value.targetGraph = undefined;
    const model = buildExecutionGraphModel(value, true);
    expect(model.nodes.every((node) => node.kind === 'stage')).toBe(true);
    expect(model.edges).toHaveLength(2);
  });

  it('packs overlapping tasks into separate lanes and reuses free lanes', () => {
    const packed = packCompileTaskLanes([
      { output: 'a', startMs: 0, endMs: 50, durationMs: 50 },
      { output: 'b', startMs: 10, endMs: 30, durationMs: 20 },
      { output: 'c', startMs: 50, endMs: 80, durationMs: 30 },
      { output: 'd', startMs: 50, endMs: 50, durationMs: 0 },
    ]);
    expect(packed.map(({ task, lane }) => [task.output, lane])).toEqual([
      ['a', 0],
      ['b', 1],
      ['d', 0],
      ['c', 0],
    ]);
  });

  it('filters only verified Ninja target attribution and never guesses MSBuild targets', () => {
    const value = run();
    const model = buildExecutionGraphModel(value, true);
    const render = model.nodes.find((node) => node.id === 'target:render');
    expect(tasksForSelection(value, render).map((task) => task.output)).toEqual(['render-a.obj', 'render-b.obj']);
    value.compileTrace = { ...value.compileTrace!, source: 'msbuild-file-log', clock: 'msbuild-cl-events' };
    expect(tasksForSelection(value, render)).toEqual([]);
    expect(
      tasksForSelection(
        value,
        model.nodes.find((node) => node.label === 'Compile'),
      ),
    ).toHaveLength(3);
  });
});
