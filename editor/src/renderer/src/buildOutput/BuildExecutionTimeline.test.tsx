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
              {
                id: 'render',
                name: 'arc-render',
                type: 'STATIC_LIBRARY',
                dependencies: [],
                artifacts: [],
                activePhase: 'compiling',
                activePhaseStartedAt: '2026-11-20T12:00:00.230Z',
              },
              {
                id: 'third_party/meshoptimizer/meshoptimizer.vcxproj',
                name: 'meshoptimizer',
                type: 'STATIC_LIBRARY',
                dependencies: [],
                artifacts: [],
                artifactObservedAt: '2026-11-20T12:00:00.210Z',
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
    expect(screen.getByText('arc-render')).toBeInTheDocument();
    expect(screen.getByText('Third-party')).toBeInTheDocument();
    expect(screen.queryByText('meshoptimizer')).not.toBeInTheDocument();
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

    fireEvent.click(screen.getByRole('button', { name: 'Expand Third-party modules' }));
    expect(screen.getByText('meshoptimizer')).toBeInTheDocument();
  });

  it('shows an active SDK module with a blue phase spinner', () => {
    const run = workflow();
    run.state = 'running';
    delete run.completedAt;
    delete run.durationMs;
    const configure = run.stages[1];
    configure.state = 'running';
    const sdk = configure.tasks![0];
    sdk.state = 'running';
    delete sdk.completedAt;
    delete sdk.durationMs;

    render(<BuildExecutionTimeline run={run} />);

    const status = screen.getByText('Compiling…');
    expect(status.closest('.build-stage-gantt-label')).toHaveTextContent('arc-render');
    expect(status.closest('.build-stage-gantt-label')?.querySelector('.is-building')).toBeInTheDocument();
  });

  it('uses a fixed 30-second scale and follows the right edge until scrubbed', () => {
    const run = workflow();
    run.durationMs = 189_000;
    run.completedAt = '2026-11-20T12:03:09.000Z';

    render(<BuildExecutionTimeline run={run} />);

    const scrubber = screen.getByRole('slider', { name: 'Scrub workflow timeline' });
    expect(scrubber).toHaveAttribute('max', '69000');
    expect(scrubber).toHaveValue('69000');
    expect(screen.getByRole('button', { name: 'Following live' })).toHaveAttribute('aria-pressed', 'true');

    fireEvent.change(scrubber, { target: { value: '30000' } });
    expect(screen.getByRole('button', { name: 'Follow live' })).toHaveAttribute('aria-pressed', 'false');
    fireEvent.click(screen.getByRole('button', { name: 'Follow live' }));
    expect(screen.getByRole('button', { name: 'Following live' })).toHaveAttribute('aria-pressed', 'true');
  });
});
