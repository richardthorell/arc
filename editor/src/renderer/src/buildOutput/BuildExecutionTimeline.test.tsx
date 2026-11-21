// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import type { BuildMonitorRun } from '../../../common/buildMonitorTypes';
import { BuildExecutionTimeline } from './BuildExecutionTimeline';

const workflow = (): BuildMonitorRun => ({
  id: 'run',
  source: 'workflow',
  projectGuid: 'project',
  action: 'build',
  configuration: 'Shipping',
  startedAt: '2026-11-20T12:00:00.000Z',
  completedAt: '2026-11-20T12:00:01.000Z',
  durationMs: 1000,
  state: 'succeeded',
  warningCount: 0,
  errorCount: 0,
  stages: [
    { id: 'validate', name: 'validate', state: 'succeeded', durationMs: 100 },
    { id: 'configure', name: 'configure', state: 'succeeded', durationMs: 200 },
    { id: 'compile', name: 'compile', state: 'succeeded', durationMs: 500 },
    { id: 'cook', name: 'cook', state: 'succeeded', durationMs: 100 },
    { id: 'package', name: 'package', state: 'succeeded', durationMs: 100 },
  ],
  compileTrace: {
    source: 'ninja-log',
    clock: 'ninja-invocation',
    truncated: false,
    tasks: [
      { output: 'first.obj', startMs: 0, endMs: 80, durationMs: 80 },
      { output: 'second.obj', startMs: 10, endMs: 70, durationMs: 60 },
    ],
  },
});

afterEach(cleanup);

describe('BuildExecutionTimeline', () => {
  it('renders workflow stages and parallel compiler work as Gantt timelines rather than a relationship graph', () => {
    render(<BuildExecutionTimeline run={workflow()} />);
    expect(screen.getByRole('region', { name: 'Build execution timeline' })).toHaveTextContent('Stacked Gantt');
    expect(screen.getByRole('region', { name: 'Workflow stage timeline' })).toHaveTextContent('validate');
    expect(screen.getByRole('region', { name: 'Workflow stage timeline' })).toHaveTextContent('package');
    expect(screen.getByRole('region', { name: 'Compiler task timeline' })).toHaveTextContent('2 measured');
    expect(screen.getByRole('button', { name: 'Inspect compile task: first.obj' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Inspect compile task: second.obj' })).toBeInTheDocument();
    expect(screen.queryByText('Execution graph')).not.toBeInTheDocument();
  });
});
