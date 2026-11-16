import { describe, expect, it, vi } from 'vitest';

import type { ArcBuildSnapshot } from '../common/buildTypes';
import type { WorkflowExecutionSnapshot, WorkflowRun } from '../common/projectWorkflowTypes';
import { BuildMonitorService } from './buildMonitorService';

const command = (overrides: Partial<ArcBuildSnapshot> = {}): ArcBuildSnapshot => ({
  revision: 1,
  state: 'building',
  configuration: 'RelWithDebInfo',
  buildRequired: false,
  reloadRequired: false,
  restartRequired: false,
  runId: 'command-001',
  startedAt: '2026-11-16T12:00:00.000Z',
  diagnostics: [],
  history: [],
  ...overrides,
});

const workflow = (
  state: WorkflowRun['state'],
  stages: WorkflowRun['stages'],
  latestOutput?: WorkflowExecutionSnapshot['latestOutput'],
): WorkflowExecutionSnapshot =>
  ({
    requestId: 'request-001',
    planning: false,
    error: null,
    output: latestOutput ? [latestOutput.text] : [],
    latestOutput,
    run: {
      id: 'workflow-001',
      state,
      plan: {
        projectGuid: 'project-a',
        action: 'build',
        configuration: 'Shipping',
        profile: 'windows-shipping',
        targetId: 'windows-x64-vulkan',
        platform: 'windows',
      },
      stages,
      artifacts: [],
    },
  }) as WorkflowExecutionSnapshot;

describe('BuildMonitorService BM1 event stream', () => {
  it('correlates command start, output, diagnostics and completion with one stable ID', () => {
    const observed = vi.fn();
    const monitor = new BuildMonitorService(observed, () => '2026-11-16T12:00:03.000Z');
    monitor.observeCommand(command(), 'project-a');
    monitor.observeCommand(command(), 'project-a');
    monitor.observeCommand(
      command({
        diagnostics: [
          { sequence: 1, severity: 'info', message: 'Compiling engine module' },
          { sequence: 2, severity: 'error', message: 'symbol missing', file: 'main.cpp', line: 10 },
        ],
      }),
      'project-a',
    );

    const completed = command({
      state: 'failed',
      completedAt: '2026-11-16T12:00:05.000Z',
      diagnostics: [
        { sequence: 1, severity: 'info', message: 'Compiling engine module' },
        { sequence: 2, severity: 'error', message: 'symbol missing', file: 'main.cpp', line: 10 },
      ],
      history: [
        {
          id: 1,
          runId: 'command-001',
          kind: 'build',
          startedAt: '2026-11-16T12:00:00.000Z',
          durationMs: 5000,
          configuration: 'RelWithDebInfo',
          result: 'failed',
          errorCount: 1,
          warningCount: 0,
        },
      ],
    });
    monitor.observeCommand(completed, 'project-a');
    monitor.observeCommand(completed, 'project-a');

    const snapshot = monitor.snapshot();
    expect(snapshot.runs).toHaveLength(1);
    expect(snapshot.runs[0]).toMatchObject({
      id: 'command-001',
      source: 'command',
      state: 'failed',
      durationMs: 5000,
      errorCount: 1,
      stages: [{ id: 'command-001:command', name: 'Compile', state: 'failed' }],
    });
    expect(snapshot.activeRunId).toBeNull();
    expect(snapshot.events.map((event) => event.kind)).toEqual([
      'run-started',
      'output',
      'diagnostic',
      'stage-changed',
      'run-completed',
    ]);
    expect(snapshot.events.map((event) => event.sequence)).toEqual([1, 2, 3, 4, 5]);
    expect(observed).toHaveBeenCalledTimes(5);
  });

  it('waits for command child exit before recording cancellation as completed', () => {
    const monitor = new BuildMonitorService();
    monitor.observeCommand(command(), 'project-a');
    monitor.observeCommand(command({ state: 'cancelled' }), 'project-a');
    expect(monitor.snapshot().activeRunId).toBe('command-001');
    monitor.observeCommand(
      command({
        state: 'cancelled',
        history: [
          {
            id: 1,
            runId: 'command-001',
            kind: 'build',
            startedAt: '2026-11-16T12:00:00.000Z',
            durationMs: 100,
            configuration: 'RelWithDebInfo',
            result: 'cancelled',
            errorCount: 0,
            warningCount: 0,
          },
        ],
      }),
      'project-a',
    );
    expect(monitor.snapshot().runs[0].state).toBe('cancelled');
    expect(monitor.snapshot().events.filter((event) => event.kind === 'run-completed')).toHaveLength(1);
  });

  it('normalizes native-planned workflow stages and consumes output chunks exactly once', () => {
    const monitor = new BuildMonitorService();
    monitor.observeWorkflow(
      workflow('running', [
        { stage: 'validate', state: 'running' },
        { stage: 'configure', state: 'pending' },
      ]),
      'project-a',
    );
    monitor.observeWorkflow(
      workflow(
        'running',
        [
          { stage: 'validate', state: 'succeeded' },
          { stage: 'configure', state: 'running' },
        ],
        { sequence: 1, text: 'Configured target' },
      ),
      'project-a',
    );
    monitor.observeWorkflow(
      workflow(
        'running',
        [
          { stage: 'validate', state: 'succeeded' },
          { stage: 'configure', state: 'running' },
        ],
        { sequence: 1, text: 'Configured target' },
      ),
      'project-a',
    );
    monitor.observeWorkflow(
      workflow(
        'succeeded',
        [
          { stage: 'validate', state: 'succeeded' },
          { stage: 'configure', state: 'succeeded' },
        ],
        { sequence: 2, text: 'Build complete' },
      ),
      'project-a',
    );

    const snapshot = monitor.snapshot();
    expect(snapshot.runs).toHaveLength(1);
    expect(snapshot.runs[0]).toMatchObject({
      id: 'workflow-001',
      source: 'workflow',
      action: 'build',
      targetId: 'windows-x64-vulkan',
      profile: 'windows-shipping',
      state: 'succeeded',
    });
    expect(snapshot.runs[0].stages.map((item) => [item.id, item.state])).toEqual([
      ['workflow-001:validate', 'succeeded'],
      ['workflow-001:configure', 'succeeded'],
    ]);
    expect(
      snapshot.events
        .filter((event) => event.kind === 'output')
        .map((event) => (event.kind === 'output' ? event.text : '')),
    ).toEqual(['Configured target', 'Build complete']);
    expect(snapshot.events.filter((event) => event.kind === 'run-completed')).toHaveLength(1);
  });

  it('isolates projects and ignores stale workflow notifications', () => {
    const monitor = new BuildMonitorService();
    monitor.observeCommand(command(), 'project-a');
    monitor.observeCommand(command({ runId: undefined, state: 'idle' }), 'project-b');
    expect(monitor.snapshot()).toMatchObject({
      projectGuid: 'project-b',
      activeRunId: null,
      runs: [],
      events: [],
    });
    monitor.observeWorkflow(workflow('succeeded', [{ stage: 'validate', state: 'succeeded' }]), 'project-b');
    expect(monitor.snapshot().runs).toHaveLength(0);
    monitor.observeCommand(command({ runId: 'command-b' }), 'project-b');
    expect(monitor.snapshot().runs.map((run) => run.id)).toEqual(['command-b']);
  });

  it('does not break builds if an observer throws', () => {
    const monitor = new BuildMonitorService(() => {
      throw new Error('renderer closed');
    });
    expect(() => monitor.observeCommand(command(), 'project-a')).not.toThrow();
    expect(monitor.snapshot().runs).toHaveLength(1);
  });
});
