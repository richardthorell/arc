import { describe, expect, it, vi } from 'vitest';
import type { WorkflowAuthority, WorkflowPlan, WorkflowStage } from '../common/projectWorkflowTypes';
import { ProjectWorkflowRunner, type WorkflowStageAdapter } from './projectWorkflowRunner';

const plan = (): WorkflowPlan => ({
  succeeded: true,
  version: 1,
  action: 'build',
  profile: 'windows-dev',
  targetId: 'windows-x64-vulkan',
  destination: 'none',
  sessionId: '',
  projectGuid: 'project',
  engineVersion: '0.1',
  configuration: 'Debug',
  platform: 'windows',
  architecture: 'x86_64',
  renderer: 'vulkan',
  finalizer: 'windows-directory',
  cookOutput: '/project/Build/Cooked/windows-dev',
  packageOutput: '/project/Build/Packages/windows-dev',
  stages: ['validate', 'configure', 'compile', 'cook', 'package'],
});
const authority = (): WorkflowAuthority => ({
  projectGuid: 'project',
  projectRoot: '/project',
  descriptorPath: '/project/a.arcproject',
  projectRevision: 1,
  sdkIdentity: 'sdk',
});

function setup() {
  const order: string[] = [];
  const adapters: Partial<Record<WorkflowStage, WorkflowStageAdapter>> = {};
  for (const stage of plan().stages)
    adapters[stage] = {
      capability: vi.fn(() => ({ available: true })),
      execute: vi.fn(async () => {
        order.push(stage);
      }),
      rollback: vi.fn(async () => {
        order.push(`undo:${stage}`);
      }),
    };
  const commit = vi.fn(() => []);
  const current = vi.fn(() => true);
  const notify = vi.fn();
  const runner = new ProjectWorkflowRunner(adapters, current, { commit }, notify);
  return { runner, adapters, order, commit, current, notify };
}

describe('project workflow orchestration', () => {
  it('does not enter a stage cancelled by a progress observer', async () => {
    const { runner, notify, order, commit } = setup();
    notify.mockImplementation((run) => {
      if (run.stages.some((stage: { state: string }) => stage.state === 'running')) runner.cancel(run.id);
    });
    expect((await runner.execute(plan(), authority())).state).toBe('cancelled');
    expect(order).toEqual([]);
    expect(commit).not.toHaveBeenCalled();
  });
  it('rolls back prepared work when publication fails', async () => {
    const { runner, commit, order } = setup();
    commit.mockImplementation(() => {
      throw new Error('Atomic publish failed');
    });
    const result = await runner.execute(plan(), authority());
    expect(result.state).toBe('failed');
    expect(result.artifacts).toEqual([]);
    expect(order.filter((value) => value.startsWith('undo:'))).toEqual(
      [...plan().stages].reverse().map((stage) => `undo:${stage}`),
    );
  });
  it('observer failures cannot turn successful publication into rollback', async () => {
    const { runner, notify, commit } = setup();
    notify.mockImplementation(() => {
      throw new Error('Renderer disconnected');
    });
    expect((await runner.execute(plan(), authority())).state).toBe('succeeded');
    expect(commit).toHaveBeenCalledOnce();
  });
  it.each(plan().stages)('cancels at %s without executing subsequent work', async (stage) => {
    const { runner, adapters, commit, order } = setup();
    adapters[stage]!.execute = async (context) => {
      runner.cancel(context.runId);
      expect(context.signal.aborted).toBe(true);
    };
    const result = await runner.execute(plan(), authority());
    expect(result.state).toBe('cancelled');
    expect(commit).not.toHaveBeenCalled();
    expect(order.filter((value) => !value.startsWith('undo:'))).toEqual(
      plan().stages.slice(0, plan().stages.indexOf(stage)),
    );
  });
  it('executes native stages in order and publishes once', async () => {
    const { runner, order, commit } = setup();
    const result = await runner.execute(plan(), authority());
    expect(result.state).toBe('succeeded');
    expect(order).toEqual(plan().stages);
    expect(commit).toHaveBeenCalledOnce();
    expect(result.stages.every((stage) => stage.state === 'succeeded')).toBe(true);
  });
  it('records actual stage boundaries and monotonic durations from the runner', async () => {
    const { runner, notify } = setup();
    const result = await runner.execute(plan(), authority());
    expect(result.startedAt).toEqual(expect.any(String));
    expect(result.completedAt).toEqual(expect.any(String));
    expect(result.durationMs).toBeGreaterThanOrEqual(0);
    for (const stage of result.stages) {
      expect(stage.startedAt).toEqual(expect.any(String));
      expect(stage.completedAt).toEqual(expect.any(String));
      expect(stage.durationMs).toBeGreaterThanOrEqual(0);
      expect(Date.parse(stage.completedAt!)).toBeGreaterThanOrEqual(Date.parse(stage.startedAt!));
    }
    // Progress notifications carry the same real start time throughout each stage.
    const running = notify.mock.calls
      .map(([snapshot]) => snapshot)
      .filter((snapshot) => snapshot.stages.some((stage: { state: string }) => stage.state === 'running'));
    expect(running.length).toBe(plan().stages.length);
    expect(running[0].stages[0]).toMatchObject({ state: 'running', startedAt: expect.any(String) });
    expect(running[0].stages[0].completedAt).toBeUndefined();
  });

  it('closes a failing stage at failure time and leaves unstarted stages unmeasured', async () => {
    const { runner, adapters } = setup();
    adapters.compile!.execute = async () => {
      throw new Error('compile failed');
    };
    const result = await runner.execute(plan(), authority());
    expect(result.state).toBe('failed');
    expect(result.stages[2]).toMatchObject({
      stage: 'compile',
      state: 'failed',
      startedAt: expect.any(String),
      completedAt: expect.any(String),
      durationMs: expect.any(Number),
    });
    expect(result.stages[3]).toMatchObject({ stage: 'cook', state: 'pending' });
    expect(result.stages[3].startedAt).toBeUndefined();
    expect(result.stages[3].completedAt).toBeUndefined();
    expect(result.completedAt).toEqual(expect.any(String));
  });

  it('records only native-reported target artifacts and satisfies the requested root after compile', async () => {
    const { runner, adapters, notify } = setup();
    const graph = {
      revision: 1,
      generator: 'Visual Studio 17 2022',
      configuration: 'Debug',
      rootTargetId: 'output',
      discoveredAt: '2026-11-16T10:00:00.000Z',
      truncated: false,
      nodes: [
        { id: 'output', name: 'arc-runtime-output', type: 'UTILITY', dependencies: ['game'], artifacts: [] },
        { id: 'game', name: 'game', type: 'EXECUTABLE', dependencies: [], artifacts: ['Debug/game.exe'] },
      ],
    };
    adapters.configure!.execute = async (context) => {
      context.publishTargetGraph(graph);
    };
    adapters.compile!.execute = async (context) => {
      context.reportTargetArtifact('game');
      context.reportTargetArtifact('game'); // duplicate must not produce another notification
      context.reportTargetArtifact('not-a-target');
      context.reportRequestedTargetSatisfied();
    };
    const result = await runner.execute(plan(), authority());
    expect(result.state).toBe('succeeded');
    expect(result.targetGraph?.nodes).toMatchObject([
      { id: 'output', satisfiedAt: expect.any(String) },
      { id: 'game', artifactObservedAt: expect.any(String) },
    ]);
    expect(result.targetGraph?.nodes[0]).not.toHaveProperty('artifactObservedAt');
    expect(result.targetGraph?.nodes[1]).not.toHaveProperty('satisfiedAt');
    expect(result.targetGraph?.revision).toBe(3);
    const revisions = notify.mock.calls.map(([entry]) => entry.targetGraph?.revision).filter(Boolean);
    expect([...new Set(revisions)]).toEqual([1, 2, 3]);
    expect(revisions.filter((revision) => revision === 2)).toHaveLength(1);
  });

  it('publishes measured compiler intervals without changing the workflow clock', async () => {
    const { runner, adapters, notify } = setup();
    adapters.compile!.execute = async (context) => {
      context.publishCompileTrace({
        source: 'msbuild-file-log',
        clock: 'msbuild-cl-events',
        truncated: false,
        tasks: [{ output: 'CL task #1', startMs: 0, endMs: 25, durationMs: 25 }],
      });
    };
    const result = await runner.execute(plan(), authority());
    expect(result.compileTrace).toMatchObject({ clock: 'msbuild-cl-events', tasks: [{ durationMs: 25 }] });
    expect(notify.mock.calls.some(([entry]) => entry.compileTrace?.tasks.length === 1)).toBe(true);
  });

  it('cannot publish target progress after cancelling its owned workflow', async () => {
    const { runner, adapters, commit } = setup();
    adapters.compile!.execute = async (context) => {
      runner.cancel(context.runId);
      context.reportRequestedTargetSatisfied();
      context.reportTargetArtifact('game');
    };
    const result = await runner.execute(plan(), authority());
    expect(result.state).toBe('cancelled');
    expect(result.targetGraph).toBeUndefined();
    expect(commit).not.toHaveBeenCalled();
  });

  it('preflights every capability before any work', async () => {
    const { runner, adapters, order, commit } = setup();
    adapters.package!.capability = () => ({ available: false, reason: 'Package adapter unavailable' });
    await expect(runner.execute(plan(), authority())).rejects.toThrow('package: Package adapter unavailable');
    expect(order).toEqual([]);
    expect(commit).not.toHaveBeenCalled();
  });
  it('cook-only does not require compile, package or deployment adapters', async () => {
    const { runner, adapters, order } = setup();
    delete adapters.compile;
    delete adapters.package;
    const result = await runner.execute({ ...plan(), action: 'cook', stages: ['validate', 'cook'] }, authority());
    expect(result.state).toBe('succeeded');
    expect(order).toEqual(['validate', 'cook']);
  });
  it.each(plan().stages)('rolls back failed %s and leaves the previous publication alone', async (failed) => {
    const { runner, adapters, commit, order } = setup();
    adapters[failed]!.execute = async () => {
      throw new Error('stage failed');
    };
    const result = await runner.execute(plan(), authority());
    expect(result.state).toBe('failed');
    expect(commit).not.toHaveBeenCalled();
    const entered = plan().stages.slice(0, plan().stages.indexOf(failed) + 1);
    expect(order.filter((value) => value.startsWith('undo:'))).toEqual(
      entered.reverse().map((value) => `undo:${value}`),
    );
  });
  it('drains cancellation before accepting new work and rejects late publication', async () => {
    const { runner, adapters, commit } = setup();
    let finish!: () => void;
    adapters.validate!.execute = () =>
      new Promise<void>((resolve) => {
        finish = resolve;
      });
    const pending = runner.execute(plan(), authority());
    const id = runner.snapshot()!.id;
    runner.cancel('another-run');
    expect(runner.snapshot()?.state).toBe('running');
    runner.cancel(id);
    expect(runner.snapshot()?.state).toBe('cancelling');
    await expect(runner.execute(plan(), authority())).rejects.toThrow('draining');
    finish();
    expect((await pending).state).toBe('cancelled');
    expect(commit).not.toHaveBeenCalled();
  });
  it('rejects changed authority after asynchronous work', async () => {
    const { runner, adapters, current, commit } = setup();
    adapters.cook!.execute = async () => {
      current.mockReturnValue(false);
    };
    const result = await runner.execute(plan(), authority());
    expect(result.state).toBe('failed');
    expect(result.error).toContain('authority changed');
    expect(commit).not.toHaveBeenCalled();
  });
  it('captures input identities and isolates notifications', async () => {
    const { runner, adapters, notify } = setup();
    const input = authority();
    adapters.validate!.execute = async () => {
      input.sdkIdentity = 'different-sdk';
    };
    notify.mockImplementation((run) => {
      run.authority.sdkIdentity = 'mutated';
    });
    const result = await runner.execute(plan(), input);
    expect(result.authority.sdkIdentity).toBe('sdk');
    expect(runner.snapshot()?.authority.sdkIdentity).toBe('sdk');
  });
  it('requires explicit device identity', async () => {
    const { runner } = setup();
    await expect(runner.execute({ ...plan(), destination: 'device' }, authority())).rejects.toThrow('explicit target');
  });
});
