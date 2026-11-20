// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

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

  it('opens verified source details and only filters Ninja logs on explicit request', () => {
    const value = run();
    value.compileTrace!.tasks[0].sourceFile = 'Source/a.cpp';
    value.compileTrace!.tasks[0].targetName = 'a';
    const onFilterOutput = vi.fn();
    render(<BuildCompileTasksPanel run={value} onFilterOutput={onFilterOutput} />);
    const toggle = screen.getByRole('button', { name: 'Inspect compile task: Source/a.cpp' });
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByText('Selected compilation task')).toBeInTheDocument();
    expect(screen.getByText('Source/a.cpp', { selector: 'dd' })).toBeInTheDocument();
    expect(screen.getByText('a', { selector: 'dd' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Find output text in build logs' }));
    expect(onFilterOutput).toHaveBeenCalledWith('CMakeFiles/a.dir/a.cpp.obj');
    fireEvent.click(toggle);
    expect(screen.queryByText('Selected compilation task')).not.toBeInTheDocument();
  });

  it('does not fabricate per-file MSBuild metadata or correlate CL task IDs to output', () => {
    const value = run();
    value.compileTrace = {
      source: 'msbuild-file-log',
      clock: 'msbuild-cl-events',
      truncated: false,
      tasks: [{ output: 'CL task #7 (node 2)', startMs: 0, endMs: 80, durationMs: 80 }],
    };
    const onFilterOutput = vi.fn();
    render(<BuildCompileTasksPanel run={value} onFilterOutput={onFilterOutput} />);
    fireEvent.click(screen.getByRole('button', { name: 'Inspect compile task: CL task #7 (node 2)' }));
    expect(screen.getByText('Not available from this native compiler task')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Find output text in build logs' })).not.toBeInTheDocument();
  });

  it('labels recorded intervals as live only while their build is running', () => {
    const value = run();
    value.state = 'running';
    const { rerender } = render(<BuildCompileTasksPanel run={value} />);
    expect(screen.getByText('2 measured · Ninja · live')).toBeInTheDocument();
    expect(screen.getByText(/Completed tasks appear as the native tool flushes/)).toBeInTheDocument();
    rerender(<BuildCompileTasksPanel run={{ ...value, state: 'succeeded' }} />);
    expect(screen.getByText('2 measured · Ninja')).toBeInTheDocument();
    expect(screen.queryByText(/Completed tasks appear as the native tool flushes/)).not.toBeInTheDocument();
  });

  it('shows CL task input files without assigning an individual source duration', () => {
    const value = run();
    value.compileTrace = {
      source: 'msbuild-file-log',
      clock: 'msbuild-cl-events',
      truncated: false,
      tasks: [
        {
          output: 'CL task #10 (node 1)',
          startMs: 0,
          endMs: 400,
          durationMs: 400,
          sourceFiles: ['Source/main.cpp', 'Source/math.cpp'],
        },
      ],
    };
    render(<BuildCompileTasksPanel run={value} />);
    fireEvent.click(screen.getByRole('button', { name: 'Inspect compile task: CL task #10 (node 1)' }));
    expect(screen.getByText('CL task inputs')).toBeInTheDocument();
    expect(screen.getByText('Source/main.cpp')).toBeInTheDocument();
    expect(screen.getByText('Source/math.cpp')).toBeInTheDocument();
    expect(screen.getByText(/Durations apply to whole CL tasks, not individual files/)).toBeInTheDocument();
    expect(screen.getByText('400 ms')).toBeInTheDocument();
  });

  it('shows active native CL tasks without durations and hides them when the run finishes', () => {
    const value = run();
    value.state = 'running';
    value.compileTrace = {
      source: 'msbuild-file-log',
      clock: 'msbuild-cl-events',
      truncated: false,
      tasks: [],
      activeTasks: [{ output: 'CL task #15 (node 2)', sourceFiles: ['Source/ui.cpp', 'Source/input.cpp'] }],
    };
    const { rerender } = render(<BuildCompileTasksPanel run={value} />);
    const active = screen.getByLabelText('Active compiler tasks');
    expect(active).toHaveTextContent('1 MSBuild CL task in progress');
    expect(active).toHaveTextContent('CL task #15 (node 2)');
    expect(active).toHaveTextContent('Source/ui.cpp');
    expect(active).toHaveTextContent('no end timestamp or duration recorded yet');
    rerender(<BuildCompileTasksPanel run={{ ...value, state: 'succeeded' }} />);
    expect(screen.queryByLabelText('Active compiler tasks')).not.toBeInTheDocument();
    expect(screen.getByText('0 measured · MSBuild')).toBeInTheDocument();
  });

  it('does not render for generators without measured native data', () => {
    const value = run();
    value.compileTrace = undefined;
    const { container } = render(<BuildCompileTasksPanel run={value} />);
    expect(container).toBeEmptyDOMElement();
  });
});
