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
): WorkflowExecutionSnapshot => ({
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
      destination: 'none',
      sessionId: '',
      succeeded: true,
      version: 1,
      engineVersion: '0.1.0',
      architecture: 'x86_64',
      renderer: 'vulkan',
      finalizer: 'windows',
      cookOutput: 'Build/Cooked',
      packageOutput: 'Build/Packages',
      stages: ['validate', 'configure'],
    },
    authority: {
      projectGuid: 'project-a',
      projectRoot: '/project',
      descriptorPath: '/project/Test.arcproject',
      projectRevision: 1,
      sdkIdentity: 'sdk-test',
    },
    stages,
    artifacts: [],
  },
});

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
      action: 'build',
      state: 'failed',
      durationMs: 5000,
      errorCount: 1,
      stages: [
        {
          id: 'command-001:command',
          name: 'Compile',
          state: 'failed',
          durationMs: 5000,
          completedAt: '2026-11-16T12:00:05.000Z',
        },
      ],
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
    expect(snapshot.events[2]).toMatchObject({ runId: 'command-001', stageId: 'command-001:command' });
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

  it('preserves runner-measured times and durations rather than main-process observation time', () => {
    const mainTime = '2026-11-16T15:00:00.000Z';
    const monitor = new BuildMonitorService(
      () => {},
      () => mainTime,
    );
    const startedAt = '2026-11-16T12:00:00.000Z';
    const stageStart = '2026-11-16T12:00:01.000Z';
    const stageEnd = '2026-11-16T12:00:02.000Z';
    const snapshot = workflow('succeeded', [
      {
        stage: 'validate',
        state: 'succeeded',
        startedAt: stageStart,
        completedAt: stageEnd,
        durationMs: 901.25,
      },
    ]);
    snapshot.run!.startedAt = startedAt;
    snapshot.run!.completedAt = '2026-11-16T12:00:03.000Z';
    snapshot.run!.durationMs = 2801.5;
    monitor.observeWorkflow(snapshot, 'project-a');
    const result = monitor.snapshot().runs[0];
    expect(result.startedAt).toBe(startedAt);
    expect(result.completedAt).toBe(snapshot.run!.completedAt);
    expect(result.durationMs).toBe(2801.5);
    expect(result.stages[0]).toMatchObject({
      startedAt: stageStart,
      completedAt: stageEnd,
      durationMs: 901.25,
      state: 'succeeded',
    });
    expect(result.stages[0].startedAt).not.toBe(mainTime);
    expect(monitor.snapshot().events.filter((event) => event.kind === 'stage-changed')).toHaveLength(1);
  });

  it('recovers a missed stage-start notification from the completed runner snapshot', () => {
    const monitor = new BuildMonitorService();
    const value = workflow('failed', [
      {
        stage: 'validate',
        state: 'succeeded',
        startedAt: '2026-11-16T12:00:00.000Z',
        completedAt: '2026-11-16T12:00:01.000Z',
        durationMs: 1000,
      },
      {
        stage: 'configure',
        state: 'failed',
        startedAt: '2026-11-16T12:00:01.000Z',
        completedAt: '2026-11-16T12:00:04.000Z',
        durationMs: 3000,
      },
    ]);
    value.run!.startedAt = '2026-11-16T12:00:00.000Z';
    value.run!.completedAt = '2026-11-16T12:00:04.000Z';
    monitor.observeWorkflow(value, 'project-a');
    expect(monitor.snapshot().runs[0].stages).toMatchObject([
      { name: 'validate', startedAt: '2026-11-16T12:00:00.000Z', durationMs: 1000, state: 'succeeded' },
      { name: 'configure', startedAt: '2026-11-16T12:00:01.000Z', durationMs: 3000, state: 'failed' },
    ]);
  });

  it('preserves CMake dependency graphs and emits updates only for native target observations', () => {
    const monitor = new BuildMonitorService();
    const run = workflow('running', [
      { stage: 'validate', state: 'succeeded' },
      { stage: 'configure', state: 'succeeded' },
    ]);
    run.run!.targetGraph = {
      revision: 1,
      generator: 'Visual Studio 17 2022',
      configuration: 'Shipping',
      rootTargetId: 'root',
      discoveredAt: '2026-11-16T12:00:00.000Z',
      truncated: false,
      nodes: [{ id: 'root', name: 'arc-runtime-output', type: 'UTILITY', dependencies: [], artifacts: [] }],
    };
    monitor.observeWorkflow(run, 'project-a');
    monitor.observeWorkflow(run, 'project-a');
    expect(monitor.snapshot().events.filter((event) => event.kind === 'target-graph-changed')).toHaveLength(1);
    run.run!.targetGraph.revision = 2;
    run.run!.targetGraph.nodes[0].satisfiedAt = '2026-11-16T12:00:04.000Z';
    monitor.observeWorkflow(run, 'project-a');
    const snapshot = monitor.snapshot();
    expect(snapshot.runs[0].targetGraph?.nodes[0].satisfiedAt).toBe('2026-11-16T12:00:04.000Z');
    expect(snapshot.events.filter((event) => event.kind === 'target-graph-changed')).toHaveLength(2);
  });

  it('persists command CMake target graphs and deduplicates the native artifact updates', () => {
    const monitor = new BuildMonitorService();
    const graph = {
      revision: 1,
      configuration: 'Debug',
      generator: 'Ninja',
      rootTargetId: 'editor',
      discoveredAt: '2026-11-17T10:00:00.000Z',
      truncated: false,
      nodes: [{ id: 'editor', name: 'MyEditor', type: 'MODULE_LIBRARY', dependencies: [], artifacts: ['MyEditor.so'] }],
    };
    const snapshot = command({ configuration: 'Debug', targetGraph: graph });
    monitor.observeCommand(snapshot, 'project-a');
    monitor.observeCommand(snapshot, 'project-a');
    expect(monitor.snapshot().events.filter((event) => event.kind === 'target-graph-changed')).toHaveLength(1);
    monitor.observeCommand(
      command({
        ...snapshot,
        targetGraph: {
          ...graph,
          revision: 2,
          nodes: [
            {
              ...graph.nodes[0],
              artifactObservedAt: '2026-11-17T10:00:05.000Z',
            },
          ],
        },
      }),
      'project-a',
    );
    const run = monitor.snapshot().runs[0];
    expect(run.source).toBe('command');
    expect(run.targetGraph?.nodes[0].artifactObservedAt).toBe('2026-11-17T10:00:05.000Z');
    expect(monitor.snapshot().events.filter((event) => event.kind === 'target-graph-changed')).toHaveLength(2);
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
