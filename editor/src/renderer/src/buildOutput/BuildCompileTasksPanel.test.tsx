import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { BuildMonitorRun } from '../../../common/buildMonitorTypes';
import { BuildCompileTasksPanel } from './BuildCompileTasksPanel';

function run(): BuildMonitorRun {
  return {
    id: 'run-1',
    source: 'command',
    projectGuid: 'proj',
    action: 'build',
    configuration: 'Debug',
    state: 'succeeded',
    startedAt: '2026-11-17T12:00:00Z',
    stages: [],
    warningCount: 0,
    errorCount: 0,
    compileTrace: {
      source: 'ninja-log',
      clock: 'ninja-invocation',
      truncated: false,
      tasks: [
        { output: 'CMakeFiles/a.dir/a.cpp.obj', startMs: 10, endMs: 90, durationMs: 80 },
        { output: 'CMakeFiles/b.dir/b.cpp.obj', startMs: 15, endMs: 50, durationMs: 35 },
      ],
    },
  };
}

describe('BM7.1 compilation timeline', () => {
  it('shows overlapping measured tasks on the Ninja clock without claiming command timing', () => {
    render(<BuildCompileTasksPanel run={run()} />);
    expect(screen.getByRole('region', { name: 'Compilation task timeline' })).toBeInTheDocument();
    expect(screen.getByText('CMakeFiles/a.dir/a.cpp.obj')).toBeInTheDocument();
    expect(screen.getByText('CMakeFiles/b.dir/b.cpp.obj')).toBeInTheDocument();
    expect(screen.getByText(/Ninja invocation clock/)).toBeInTheDocument();
    expect(screen.getByText('80 ms')).toBeInTheDocument();
  });

  it('does not render for generators without measured native data', () => {
    const value = run();
    value.compileTrace = undefined;
    const { container } = render(<BuildCompileTasksPanel run={value} />);
    expect(container).toBeEmptyDOMElement();
  });
});
