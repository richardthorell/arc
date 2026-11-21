import { describe, expect, it } from 'vitest';

import type { BuildMonitorRun } from '../../../common/buildMonitorTypes';
import { buildStageTimeline, groupCompileTasks, packCompileTaskLanes } from './buildExecutionTimelineModel';

const run = (): BuildMonitorRun => ({
  id: 'run',
  source: 'workflow',
  projectGuid: 'project',
  action: 'build',
  configuration: 'Shipping',
  startedAt: '2026-11-20T12:00:00.000Z',
  completedAt: '2026-11-20T12:00:00.200Z',
  durationMs: 200,
  state: 'succeeded',
  warningCount: 0,
  errorCount: 0,
  stages: [
    {
      id: 'validate',
      name: 'validate',
      state: 'succeeded',
      startedAt: '2026-11-20T12:00:00.000Z',
      completedAt: '2026-11-20T12:00:00.020Z',
      durationMs: 20,
    },
    {
      id: 'compile',
      name: 'compile',
      state: 'succeeded',
      startedAt: '2026-11-20T12:00:00.030Z',
      completedAt: '2026-11-20T12:00:00.180Z',
      durationMs: 150,
    },
    { id: 'package', name: 'package', state: 'pending' },
  ],
});

describe('Build execution timeline model', () => {
  it('places workflow stages on their authoritative shared clock and preserves missing timing', () => {
    expect(buildStageTimeline(run(), Date.parse('2026-11-20T12:00:00.200Z'))).toEqual({
      durationMs: 200,
      spans: [
        { id: 'validate', label: 'validate', state: 'succeeded', startMs: 0, durationMs: 20 },
        { id: 'compile', label: 'compile', state: 'succeeded', startMs: 30, durationMs: 150 },
        { id: 'package', label: 'package', state: 'pending', startMs: undefined, durationMs: undefined },
      ],
    });
  });

  it('packs overlapping compiler tasks into stacked lanes and reuses free lanes', () => {
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

  it('groups verified targets while keeping unattributed compiler work explicit', () => {
    const groups = groupCompileTasks([
      { output: 'b.obj', startMs: 0, endMs: 10, durationMs: 10, targetName: 'game' },
      { output: 'a.obj', startMs: 0, endMs: 10, durationMs: 10, targetName: 'engine' },
      { output: 'unknown.obj', startMs: 0, endMs: 10, durationMs: 10 },
    ]);
    expect(groups.map(({ id, label, attributed, tasks }) => [id, label, attributed, tasks.length])).toEqual([
      ['target:engine', 'engine', true, 1],
      ['target:game', 'game', true, 1],
      ['unattributed', 'Compiler tasks', false, 1],
    ]);
  });
});
