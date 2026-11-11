import { describe, expect, it, vi } from 'vitest';
import type { ArcProjectCandidate } from '../common/projectTypes';
import type { WorkflowPlan, WorkflowRequest } from '../common/projectWorkflowTypes';
import { BuildService } from './buildService';
import { queryProjectWorkflowPlan } from './projectWorkflowPlanner';

vi.mock('./projectWorkflowPlanner', () => ({ queryProjectWorkflowPlan: vi.fn() }));
const request: WorkflowRequest = {
  action: 'cook',
  profile: 'android-dev',
  targetId: 'android-arm64-vulkan',
  destination: 'none',
  sessionId: '',
};

describe('BuildService native planning', () => {
  it('uses the active descriptor and never invokes the host or builds', async () => {
    const project = { descriptorPath: '/Game.arcproject', descriptor: { guid: 'project' } } as ArcProjectCandidate;
    const result = { ...request, projectGuid: 'project' } as WorkflowPlan;
    vi.mocked(queryProjectWorkflowPlan).mockResolvedValue(result);
    const host = { command: vi.fn() };
    const notify = vi.fn();
    const service = new BuildService(
      () => project,
      () => '/sdk/arc-project',
      host,
      notify,
    );
    expect(await service.plan(request)).toEqual(result);
    expect(queryProjectWorkflowPlan).toHaveBeenCalledWith('/sdk/arc-project', '/Game.arcproject', request);
    expect(host.command).not.toHaveBeenCalled();
    expect(notify).not.toHaveBeenCalled();
  });
  it.each(['project', 'sdk'])('rejects a late response after %s replacement', async (change) => {
    let project = { descriptorPath: '/Game.arcproject', descriptor: { guid: 'project' } } as ArcProjectCandidate;
    let tool = '/sdk/arc-project';
    let complete!: (value: WorkflowPlan) => void;
    vi.mocked(queryProjectWorkflowPlan).mockImplementation(
      () =>
        new Promise((resolve) => {
          complete = resolve;
        }),
    );
    const service = new BuildService(
      () => project,
      () => tool,
      { command: vi.fn() },
      vi.fn(),
    );
    const pending = service.plan(request);
    if (change === 'project') project = { ...project, descriptorPath: '/Other.arcproject' };
    else tool = '/other-sdk/arc-project';
    complete({ ...request, projectGuid: 'project' } as WorkflowPlan);
    await expect(pending).rejects.toThrow('changed while planning');
  });
});
