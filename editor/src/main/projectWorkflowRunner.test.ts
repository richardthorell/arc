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
