import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { ArcProjectCandidate } from '../common/projectTypes';
import type { WorkflowPlan, WorkflowRequest } from '../common/projectWorkflowTypes';
import { queryProjectWorkflowPlan } from './projectWorkflowPlanner';
import { CookWorkflowService, resolveWorkflowCooker } from './cookWorkflowService';

vi.mock('./projectWorkflowPlanner', () => ({ queryProjectWorkflowPlan: vi.fn() }));
const roots: string[] = [];
afterEach(() => {
  vi.resetAllMocks();
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});
const request: WorkflowRequest = {
  action: 'cook',
  profile: 'dev',
  targetId: 'windows-x64-vulkan',
  destination: 'none',
  sessionId: '',
};
function setup() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-cook-service-'));
  roots.push(root);
  const descriptorPath = path.join(root, 'Game.arcproject');
  fs.writeFileSync(descriptorPath, '{}');
  const project = {
    descriptorPath,
    projectRoot: root,
    writable: true,
    descriptor: { guid: 'project', cookProfiles: [], targetPlatforms: [] },
  } as unknown as ArcProjectCandidate;
  const legacy = vi.fn(() => false);
  const service = new CookWorkflowService(
    () => project,
    () => process.execPath,
    legacy,
    vi.fn(),
    () => process.execPath,
  );
  let finish!: (value: WorkflowPlan) => void;
  vi.mocked(queryProjectWorkflowPlan).mockImplementation(
    () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
  );
  return {
    root,
    project,
    legacy,
    service,
    finish: () => finish({ ...request, projectGuid: 'project', stages: ['validate', 'cook'] } as WorkflowPlan),
  };
}
describe('cook workflow service authority', () => {
  it('exposes authored profiles only for enabled matching targets without requiring a device', () => {
    const { project, service } = setup();
    project.descriptor.targetPlatforms = [{ id: 'android-arm64-vulkan', enabled: true }];
    project.descriptor.cookProfiles = [
      {
        id: 'custom-mobile',
        platform: 'android',
        architecture: 'arm64',
        renderer: 'vulkan',
        api: '1.1',
        configuration: 'development',
        textures: { outputs: ['astc'], quality: 'high' },
      },
    ];
    expect(service.options()).toMatchObject({ profiles: [{ id: 'custom-mobile', targetId: 'android-arm64-vulkan' }] });
    expect(service.options().unavailableReason).toBeUndefined();
    project.descriptor.targetPlatforms[0].enabled = false;
    expect(service.options().profiles).toEqual([]);
    expect(queryProjectWorkflowPlan).not.toHaveBeenCalled();
  });
  it.each(['descriptor', 'project', 'reopen'])(
    'rejects a dialog captured before %s changed, before any planning',
    async (change) => {
      const { project, service } = setup();
      const authority = service.options().authority;
      if (change === 'descriptor') fs.writeFileSync(project.descriptorPath, '{"updated":true}');
      else if (change === 'project') project.descriptor.guid = 'replacement';
      else service.invalidate();
      await expect(service.execute(request, authority)).rejects.toThrow('reopen Cook Content Only');
      expect(queryProjectWorkflowPlan).not.toHaveBeenCalled();
    },
  );
  it('blocks unsupported actions and read-only projects before planning', async () => {
    const { project, service } = setup();
    await expect(service.execute({ ...request, action: 'build' })).rejects.toThrow('not ready');
    project.writable = false;
    await expect(service.execute(request)).rejects.toThrow('writable');
    expect(queryProjectWorkflowPlan).not.toHaveBeenCalled();
  });
  it('shares the operation lock with legacy builds', async () => {
    const { service, legacy } = setup();
    legacy.mockReturnValue(true);
    await expect(service.execute(request)).rejects.toThrow('already running');
  });
  it.each(['cancel', 'descriptor', 'project'])('rejects late planning after %s', async (change) => {
    const { service, project, finish } = setup();
    const pending = service.execute(request);
    expect(service.isBusy()).toBe(true);
    await expect(service.execute(request)).rejects.toThrow('already running');
    if (change === 'cancel') service.cancel(service.snapshot().requestId!);
    else if (change === 'descriptor') fs.writeFileSync(project.descriptorPath, '{"changed":true}');
    else project.projectRoot = '/different';
    finish();
    await expect(pending).rejects.toThrow('changed while planning');
    expect(service.isBusy()).toBe(false);
    expect(service.snapshot().run).toBeNull();
  });
  it('resolves only the selected tool installation and never another build configuration', () => {
    const { root } = setup();
    const extension = process.platform === 'win32' ? '.exe' : '';
    const tool = path.join(root, 'tools', 'project_cli', 'Debug', `arc-project${extension}`);
    const cooker = path.join(root, 'tools', 'asset_cooker', 'Debug', `arc-cook${extension}`);
    fs.mkdirSync(path.dirname(cooker), { recursive: true });
    fs.writeFileSync(cooker, 'tool');
    expect(path.resolve(resolveWorkflowCooker(tool))).toBe(cooker);
    expect(resolveWorkflowCooker(tool.replace('Debug', 'Release'))).toBe('');
  });
});
