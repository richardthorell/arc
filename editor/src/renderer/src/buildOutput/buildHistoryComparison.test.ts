import { describe, expect, it } from 'vitest';

import type { BuildMonitorRun } from '../../../common/buildMonitorTypes';
import { compareBuildHistory, comparisonBaselines } from './buildHistoryComparison';

function run(overrides: Partial<BuildMonitorRun> = {}): BuildMonitorRun {
  return {
    id: 'current',
    source: 'command',
    projectGuid: 'project-a',
    action: 'build',
    configuration: 'Debug',
    platform: 'windows',
    profile: 'desktop',
    targetId: 'windows-x64',
    startedAt: '2026-11-20T10:00:00.000Z',
    state: 'succeeded',
    durationMs: 1000,
    stages: [
      { id: 'current:configure', name: 'Configure', state: 'succeeded', durationMs: 200 },
      { id: 'current:compile', name: 'Compile', state: 'succeeded', durationMs: 800 },
    ],
    warningCount: 0,
    errorCount: 0,
    ...overrides,
  };
}

const earlier = '2026-11-19T10:00:00.000Z';

describe('BM8.1 compatible build history selection', () => {
  it('selects only earlier successful comparable runs and orders newest first', () => {
    const current = run();
    const base = run({ id: 'baseline', startedAt: earlier, durationMs: 1200 });
    const older = run({ id: 'older', startedAt: '2026-11-18T09:00:00.000Z' });
    const invalid: BuildMonitorRun[] = [
      run({ id: 'failed', startedAt: earlier, state: 'failed' }),
      run({ id: 'running', startedAt: earlier, state: 'running' }),
      run({ id: 'newer', startedAt: '2026-11-21T10:00:00.000Z' }),
      run({ id: 'same-time' }),
      run({ id: 'other-project', startedAt: earlier, projectGuid: 'project-b' }),
      run({ id: 'other-source', startedAt: earlier, source: 'workflow' }),
      run({ id: 'other-action', startedAt: earlier, action: 'configure' }),
      run({ id: 'other-platform', startedAt: earlier, platform: 'android' }),
      run({ id: 'other-target', startedAt: earlier, targetId: 'other' }),
      run({ id: 'other-profile', startedAt: earlier, profile: 'steam' }),
      run({ id: 'other-config', startedAt: earlier, configuration: 'Shipping' }),
      run({ id: 'unmeasured', startedAt: earlier, durationMs: undefined }),
    ];
    const options = comparisonBaselines(current, [older, ...invalid, current, base]);
    expect(options.map((item) => item.id)).toEqual(['baseline', 'older']);
    expect(comparisonBaselines(run({ state: 'failed' }), [base])).toEqual([]);
    expect(comparisonBaselines(run({ durationMs: undefined }), [base])).toEqual([]);
  });

  it('compares only positive measured times and never divides by a zero baseline', () => {
    const current = run({ durationMs: 1500 });
    const base = run({ id: 'baseline', startedAt: earlier, durationMs: 1000 });
    expect(compareBuildHistory(current, base)?.total).toEqual({
      beforeMs: 1000,
      afterMs: 1500,
      changeMs: 500,
      changePercent: 50,
    });
    expect(compareBuildHistory(current, { ...base, durationMs: 0 })?.total).toEqual({
      beforeMs: 0,
      afterMs: 1500,
      changeMs: 1500,
    });
    expect(compareBuildHistory(current, { ...base, durationMs: Number.NaN })).toBeNull();
  });

  it('uses only uniquely named succeeded measured stages without guessing missing stages', () => {
    const current = run({
      stages: [
        { id: 'a', name: 'Compile', state: 'succeeded', durationMs: 900 },
        { id: 'b', name: 'Configure', state: 'succeeded', durationMs: 100 },
        { id: 'c', name: 'Link', state: 'running', durationMs: 100 },
        { id: 'd', name: 'Duplicate', state: 'succeeded', durationMs: 20 },
        { id: 'e', name: 'Duplicate', state: 'succeeded', durationMs: 25 },
      ],
    });
    const baseline = run({
      id: 'baseline',
      startedAt: earlier,
      stages: [
        { id: 'a', name: 'Compile', state: 'succeeded', durationMs: 500 },
        { id: 'b', name: 'Configure', state: 'succeeded', durationMs: 200 },
        { id: 'c', name: 'Link', state: 'succeeded', durationMs: 100 },
        { id: 'd', name: 'Duplicate', state: 'succeeded', durationMs: 12 },
      ],
    });
    expect(compareBuildHistory(current, baseline)?.stages).toEqual([
      { name: 'Compile', beforeMs: 500, afterMs: 900, changeMs: 400, changePercent: 80 },
      { name: 'Configure', beforeMs: 200, afterMs: 100, changeMs: -100, changePercent: -50 },
    ]);
  });

  it('matches Ninja object outputs exactly and suppresses ambiguous or unmatched entries', () => {
    const current = run({
      compileTrace: {
        source: 'ninja-log',
        clock: 'ninja-invocation',
        truncated: false,
        tasks: [
          { output: 'CMakeFiles/arc.dir/main.cpp.o', sourceFile: 'Source/main.cpp', startMs: 0, endMs: 130, durationMs: 130 },
          { output: 'CMakeFiles/arc.dir/other.cpp.o', startMs: 0, endMs: 75, durationMs: 75 },
          { output: 'same-object.o', startMs: 0, endMs: 20, durationMs: 20 },
          { output: 'same-object.o', startMs: 0, endMs: 30, durationMs: 30 },
        ],
      },
    });
    const baseline = run({
      id: 'baseline',
      startedAt: earlier,
      compileTrace: {
        source: 'ninja-log',
        clock: 'ninja-invocation',
        truncated: false,
        tasks: [
          { output: 'CMakeFiles/arc.dir/main.cpp.o', startMs: 0, endMs: 100, durationMs: 100 },
          { output: 'CMakeFiles/arc.dir/other.cpp.o', startMs: 0, endMs: 120, durationMs: 120 },
          { output: 'same-object.o', startMs: 0, endMs: 25, durationMs: 25 },
        ],
      },
    });
    const comparison = compareBuildHistory(current, baseline);
    expect(comparison?.matchedObjects).toBe(2);
    expect(comparison?.objects).toEqual([
      {
        output: 'CMakeFiles/arc.dir/main.cpp.o',
        sourceFile: 'Source/main.cpp',
        beforeMs: 100,
        afterMs: 130,
        changeMs: 30,
        changePercent: 30,
      },
      {
        output: 'CMakeFiles/arc.dir/other.cpp.o',
        beforeMs: 120,
        afterMs: 75,
        changeMs: -45,
        changePercent: -37.5,
      },
    ]);
  });

  it('never compares unstable MSBuild CL IDs as if they were the same source compilation', () => {
    const trace = {
      source: 'msbuild-file-log' as const,
      clock: 'msbuild-cl-events' as const,
      truncated: false,
      tasks: [{ output: 'CL task #3', startMs: 0, endMs: 90, durationMs: 90 }],
    };
    const current = run({ compileTrace: trace });
    const baseline = run({ id: 'baseline', startedAt: earlier, compileTrace: trace });
    expect(compareBuildHistory(current, baseline)?.objects).toEqual([]);
    expect(compareBuildHistory(current, baseline)?.matchedObjects).toBe(0);
  });

  it('refuses incompatible direct comparisons, even if a caller bypasses baseline selection', () => {
    expect(compareBuildHistory(run(), run({ id: 'bad', startedAt: earlier, configuration: 'Shipping' }))).toBeNull();
  });
});
