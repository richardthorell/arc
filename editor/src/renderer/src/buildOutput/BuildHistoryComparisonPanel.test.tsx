// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import type { BuildMonitorRun } from '../../../common/buildMonitorTypes';
import { BuildHistoryComparisonPanel } from './BuildHistoryComparisonPanel';

function run(id: string, startedAt: string, durationMs: number): BuildMonitorRun {
  return {
    id,
    source: 'command',
    projectGuid: 'project-a',
    action: 'build',
    configuration: 'Debug',
    startedAt,
    state: 'succeeded',
    durationMs,
    stages: [{ id: id + ':Compile', name: 'Compile', state: 'succeeded', durationMs: durationMs / 2 }],
    warningCount: 0,
    errorCount: 0,
  };
}

afterEach(cleanup);

describe('BM8.1 historical build comparison', () => {
  const current = run('new', '2026-11-20T10:00:00Z', 1000);
  const recent = run('recent', '2026-11-19T10:00:00Z', 800);
  const older = run('older', '2026-11-18T10:00:00Z', 1400);

  it('defaults to the latest compatible history and lets the user choose an older run', () => {
    render(<BuildHistoryComparisonPanel current={current} historyRuns={[older, recent, current]} />);
    expect(screen.getByRole('region', { name: 'Build performance comparison' })).toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: 'Comparison baseline' })).toHaveValue('latest');
    expect(screen.getByText('+200 ms (+25.0%)')).toBeInTheDocument();
    expect(screen.getByText('Measured stage changes')).toBeInTheDocument();
    fireEvent.change(screen.getByRole('combobox', { name: 'Comparison baseline' }), {
      target: { value: 'older' },
    });
    expect(screen.getByText('−400 ms (-28.6%)')).toBeInTheDocument();
    expect(screen.getByText(/incremental caching or machine load/)).toBeInTheDocument();
  });

  it('shows a clear empty state rather than comparing a differently configured build', () => {
    render(<BuildHistoryComparisonPanel current={current} historyRuns={[{ ...recent, configuration: 'Shipping' }]} />);
    expect(screen.getByText(/No earlier successful build matches/)).toBeInTheDocument();
    expect(screen.queryByRole('combobox', { name: 'Comparison baseline' })).not.toBeInTheDocument();
  });

  it('reports zero-baseline percentage as unavailable and omits MSBuild task comparisons', () => {
    const trace = {
      source: 'msbuild-file-log' as const,
      clock: 'msbuild-cl-events' as const,
      truncated: false,
      tasks: [{ output: 'CL task #1', startMs: 0, endMs: 50, durationMs: 50 }],
    };
    render(
      <BuildHistoryComparisonPanel
        current={{ ...current, compileTrace: trace }}
        historyRuns={[{ ...recent, durationMs: 0, compileTrace: trace }]}
      />,
    );
    expect(screen.getByText(/percentage unavailable \(zero baseline\)/)).toBeInTheDocument();
    expect(screen.getByText(/CL task IDs and source grouping are not stable/)).toBeInTheDocument();
  });
});
