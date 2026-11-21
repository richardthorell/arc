// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
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
    {
      id: 'configure',
      name: 'configure',
      state: 'succeeded',
      durationMs: 200,
      tasks: [
        {
          id: 'development-sdk',
          name: 'Build SDK',
          state: 'succeeded',
          startedAt: '2026-11-20T12:00:00.100Z',
          completedAt: '2026-11-20T12:00:00.250Z',
          durationMs: 150,
          targetGraph: {
            revision: 2,
            generator: 'Visual Studio 17 2022',
            configuration: 'Shipping',
            rootTargetId: 'sdk',
            discoveredAt: '2026-11-20T12:00:00.100Z',
            truncated: false,
            nodes: [
              { id: 'sdk', name: 'arc-development-sdk', type: 'UTILITY', dependencies: ['assets'], artifacts: [] },
              {
                id: 'assets',
                name: 'arc-assets',
                type: 'STATIC_LIBRARY',
                dependencies: [],
                artifacts: ['engine/assets/Shipping/arc-assets.lib'],
                artifactObservedAt: '2026-11-20T12:00:00.220Z',
              },
              {
                id: 'zero-check',
                name: 'ZERO_CHECK',
                type: 'UTILITY',
                dependencies: [],
                artifacts: [],
              },
            ],
          },
        },
      ],
    },
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
  it('renders expandable workflow, target and compiler task hierarchy', () => {
    render(<BuildExecutionTimeline run={workflow()} />);
    expect(screen.getByRole('region', { name: 'Build execution timeline' })).toHaveTextContent('Hierarchical Gantt');
    expect(screen.getByRole('region', { name: 'Workflow stage hierarchy' })).toHaveTextContent('validate');
    expect(screen.getByRole('region', { name: 'Workflow stage hierarchy' })).toHaveTextContent('package');
    expect(screen.getByRole('region', { name: 'Workflow stage hierarchy' })).toHaveTextContent('Build SDK');
    expect(screen.queryByText('configure', { exact: true })).not.toBeInTheDocument();
    expect(screen.getByText('arc-assets')).toBeInTheDocument();
    expect(screen.queryByText('ZERO_CHECK')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Collapse Build SDK modules' })).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByRole('region', { name: 'Compiler task hierarchy' })).toHaveTextContent('2 measured');
    expect(screen.getByRole('button', { name: 'Collapse compile subtasks' })).toHaveAttribute('aria-expanded', 'true');
    const taskGroup = screen.getByRole('button', { name: /Compiler tasks/ });
    expect(taskGroup).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByRole('button', { name: 'Inspect compile task: first.obj' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Inspect compile task: second.obj' })).toBeInTheDocument();

    fireEvent.click(taskGroup);
    expect(taskGroup).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByRole('button', { name: 'Inspect compile task: first.obj' })).not.toBeInTheDocument();
    fireEvent.click(taskGroup);

    fireEvent.click(screen.getByRole('button', { name: 'Collapse compile subtasks' }));
    expect(screen.queryByRole('region', { name: 'Compiler task hierarchy' })).not.toBeInTheDocument();
  });
});
