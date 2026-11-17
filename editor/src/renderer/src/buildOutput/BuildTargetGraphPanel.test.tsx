// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import type { BuildMonitorRun } from '../../../common/buildMonitorTypes';
import { BuildTargetGraphPanel } from './BuildTargetGraphPanel';

const start = '2026-11-16T12:00:00.000Z';
const run = (): BuildMonitorRun => ({
  id: 'workflow-run',
  projectGuid: 'test',
  source: 'workflow',
  action: 'build',
  configuration: 'Shipping',
  startedAt: start,
  durationMs: 10000,
  state: 'succeeded',
  stages: [],
  errorCount: 0,
  warningCount: 0,
  targetGraph: {
    revision: 3,
    generator: 'Visual Studio 17 2022',
    configuration: 'Shipping',
    discoveredAt: start,
    rootTargetId: 'root',
    truncated: false,
    nodes: [
      {
        id: 'root',
        name: 'arc-runtime-output',
        type: 'UTILITY',
        dependencies: ['game'],
        artifacts: [],
        satisfiedAt: '2026-11-16T12:00:10.000Z',
      },
      {
        id: 'game',
        name: 'my_game',
        type: 'EXECUTABLE',
        dependencies: ['core'],
        artifacts: ['Shipping/my_game.exe'],
        artifactObservedAt: '2026-11-16T12:00:08.000Z',
      },
      { id: 'core', name: 'arc_core', type: 'STATIC_LIBRARY', dependencies: [], artifacts: ['Shipping/arc_core.lib'] },
    ],
  },
});

afterEach(cleanup);

describe('BM5 target dependency graph', () => {
  it('shows native dependencies on demand and marks only explicit observed outputs', () => {
    render(<BuildTargetGraphPanel run={run()} nowMs={Date.parse(start) + 10000} />);
    expect(screen.getByRole('tree', { name: 'CMake target dependencies' })).toBeInTheDocument();
    expect(screen.getByText('arc-runtime-output')).toBeInTheDocument();
    expect(screen.getByText('my_game')).toBeInTheDocument();
    expect(screen.queryByText('arc_core')).not.toBeInTheDocument();
    expect(screen.getByText('1 artifact output observed · 3 targets')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Expand my_game' }));
    expect(screen.getByText('arc_core')).toBeInTheDocument();
    expect(screen.getByText('Output seen')).toBeInTheDocument();
    expect(screen.getByText('Not reported')).toBeInTheDocument();
    expect(screen.getByTitle('Artifact output observed 8.0 s into build')).toHaveStyle({ left: '80%' });
    expect(screen.getByText('Satisfied')).toBeInTheDocument();
  });

  it('automatically expands the requested target when graph metadata arrives live', () => {
    const pending = run();
    pending.targetGraph = undefined;
    pending.state = 'running';
    const { rerender } = render(<BuildTargetGraphPanel run={pending} nowMs={Date.parse(start)} />);
    expect(screen.getByText('Waiting for CMake configure to report target dependencies…')).toBeInTheDocument();
    rerender(<BuildTargetGraphPanel run={run()} nowMs={Date.parse(start) + 10000} />);
    expect(screen.getByText('my_game')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Collapse arc-runtime-output' }));
    expect(screen.queryByText('my_game')).not.toBeInTheDocument();
  });

  it('does not mistake target satisfaction for native artifact compilation', () => {
    const value = run();
    value.targetGraph!.nodes[1].artifactObservedAt = undefined;
    render(<BuildTargetGraphPanel run={value} nowMs={Date.parse(start) + 10000} />);
    expect(screen.getByText('0 artifact outputs observed · 3 targets')).toBeInTheDocument();
    expect(screen.getByText('Satisfied')).toBeInTheDocument();
    expect(screen.queryByText('Output seen')).not.toBeInTheDocument();
  });

  it('labels the editor module subtree for an ordinary CLI build without implying build-all completion', () => {
    const value = run();
    value.source = 'command';
    value.targetGraph = {
      ...value.targetGraph!,
      revision: 2,
      rootTargetId: 'game',
      nodes: [
        {
          id: 'game',
          name: 'MyEditor',
          type: 'MODULE_LIBRARY',
          dependencies: [],
          artifacts: ['MyEditor.dll'],
          artifactObservedAt: '2026-11-16T12:00:08.000Z',
        },
      ],
    };
    render(<BuildTargetGraphPanel run={value} nowMs={Date.parse(start) + 10000} />);
    expect(screen.getByText('MyEditor')).toBeInTheDocument();
    expect(screen.getByText(/Editor module dependencies come from CMake File API/)).toBeInTheDocument();
    expect(screen.getByText('Output seen')).toBeInTheDocument();
    expect(screen.queryByText('Satisfied')).not.toBeInTheDocument();
  });

  it('states when legacy builds cannot provide a target graph', () => {
    const value = run();
    value.source = 'command';
    value.targetGraph = undefined;
    render(<BuildTargetGraphPanel run={value} nowMs={Date.parse(start)} />);
    expect(
      screen.getByText(/Normal Build and Configure can show the editor module dependency subtree/),
    ).toBeInTheDocument();
  });
});
