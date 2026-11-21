// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { BuildMonitorRun } from '../../../common/buildMonitorTypes';
import { BuildExecutionGraph } from './BuildExecutionGraph';

vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
  callback(0);
  return 1;
});
vi.stubGlobal('cancelAnimationFrame', () => undefined);

const workflow = (): BuildMonitorRun => ({
  id: 'run',
  source: 'workflow',
  projectGuid: 'project',
  action: 'build',
  configuration: 'Shipping',
  startedAt: '2026-11-20T12:00:00.000Z',
  state: 'running',
  warningCount: 0,
  errorCount: 0,
  stages: [
    { id: 'validate', name: 'validate', state: 'succeeded' },
    { id: 'configure', name: 'configure', state: 'succeeded' },
    { id: 'compile', name: 'compile', state: 'running' },
    { id: 'cook', name: 'cook', state: 'pending' },
    { id: 'package', name: 'package', state: 'pending' },
  ],
});

afterEach(cleanup);

describe('BuildExecutionGraph live telemetry', () => {
  it('expands a CMake target graph that arrives after the workflow graph mounted', () => {
    const initial = workflow();
    const view = render(<BuildExecutionGraph run={initial} />);
    expect(screen.getByText('Workflow stages')).toBeInTheDocument();
    expect(screen.queryByText('arc-runtime-output')).not.toBeInTheDocument();

    view.rerender(
      <BuildExecutionGraph
        run={{
          ...initial,
          targetGraph: {
            revision: 1,
            generator: 'Ninja',
            configuration: 'Shipping',
            rootTargetId: 'runtime',
            discoveredAt: '2026-11-20T12:00:01.000Z',
            truncated: false,
            nodes: [
              {
                id: 'runtime',
                name: 'arc-runtime-output',
                type: 'EXECUTABLE',
                dependencies: ['engine'],
                artifacts: [],
              },
              { id: 'engine', name: 'arc-engine', type: 'STATIC_LIBRARY', dependencies: [], artifacts: [] },
            ],
          },
        }}
      />,
    );

    expect(screen.getByText('2 native targets')).toBeInTheDocument();
    expect(screen.getByText('arc-runtime-output')).toBeInTheDocument();
    expect(screen.getByText('arc-engine')).toBeInTheDocument();
  });
});
