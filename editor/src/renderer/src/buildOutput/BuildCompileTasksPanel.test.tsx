// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

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

afterEach(cleanup);

describe('BM7.1 compilation timeline', () => {
  it('shows overlapping measured tasks on the Ninja clock without claiming command timing', () => {
    render(<BuildCompileTasksPanel run={run()} />);
    expect(screen.getByRole('region', { name: 'Compilation task timeline' })).toBeInTheDocument();
    expect(screen.getByText('CMakeFiles/a.dir/a.cpp.obj')).toBeInTheDocument();
    expect(screen.getByText('CMakeFiles/b.dir/b.cpp.obj')).toBeInTheDocument();
    expect(screen.getByText(/Ninja invocation clock/)).toBeInTheDocument();
    expect(screen.getByText('80 ms')).toBeInTheDocument();
  });

  it('labels MSBuild CL task intervals without claiming per-object or per-file timings', () => {
    const value = run();
    value.compileTrace = {
      source: 'msbuild-file-log',
      clock: 'msbuild-cl-events',
      truncated: false,
      tasks: [{ output: 'CL task #19 (node 2)', startMs: 0, endMs: 600, durationMs: 600 }],
    };
    render(<BuildCompileTasksPanel run={value} />);
    expect(screen.getByText('1 measured · MSBuild')).toBeInTheDocument();
    expect(screen.getByText('CL task #19 (node 2)')).toBeInTheDocument();
    expect(screen.getByText(/One CL task may compile multiple source files/)).toBeInTheDocument();
    expect(screen.getByText('600 ms')).toBeInTheDocument();
  });

  it('does not render for generators without measured native data', () => {
    const value = run();
    value.compileTrace = undefined;
    const { container } = render(<BuildCompileTasksPanel run={value} />);
    expect(container).toBeEmptyDOMElement();
  });
});
