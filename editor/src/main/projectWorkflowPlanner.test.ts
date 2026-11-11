import { execFile } from 'node:child_process';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { queryProjectWorkflowPlan } from './projectWorkflowPlanner';

vi.mock('node:child_process', () => ({ execFile: vi.fn() }));
const request = {
  sessionId: '',
  action: 'cook',
  profile: 'android-dev',
  targetId: 'android-arm64-vulkan',
  destination: 'none',
} as const;
const output = () => ({
  ...request,
  succeeded: true,
  version: 1,
  projectGuid: 'project',
  engineVersion: '0.1',
  configuration: 'Debug',
  platform: 'android',
  architecture: 'arm64',
  renderer: 'vulkan',
  finalizer: 'android-gradle',
  cookOutput: '/out/cook',
  packageOutput: '/out/package',
  stages: ['validate', 'cook'],
});
beforeEach(() => vi.resetAllMocks());
function response(value: unknown, error: Error | null = null) {
  vi.mocked(execFile).mockImplementation((...args: unknown[]) => {
    const callback = args[3] as (error: Error | null, stdout: string) => void;
    callback(error, JSON.stringify(value));
    return {} as ReturnType<typeof execFile>;
  });
}
describe('native workflow planning bridge', () => {
  it('passes explicit identities and uses native stages without shell execution', async () => {
    response(output());
    expect(await queryProjectWorkflowPlan('/tool', '/project/Game.arcproject', request)).toEqual(output());
    expect(execFile).toHaveBeenCalledWith(
      '/tool',
      [
        'workflow',
        'plan',
        '--json',
        '--project',
        '/project/Game.arcproject',
        '--profile',
        'android-dev',
        '--target',
        'android-arm64-vulkan',
        '--action',
        'cook',
        '--destination',
        'none',
      ],
      expect.objectContaining({ shell: false, windowsHide: true, timeout: 30_000 }),
      expect.any(Function),
    );
  });
  it('rejects wrong schema versions and mismatched request identities', async () => {
    response({ ...output(), version: 2 });
    await expect(queryProjectWorkflowPlan('/tool', '/project', request)).rejects.toThrow();
    response({ ...output(), profile: 'other' });
    await expect(queryProjectWorkflowPlan('/tool', '/project', request)).rejects.toThrow('captured request');
  });
  it('preserves native diagnostics and validates input before launching', async () => {
    response({ succeeded: false, error: 'Target disabled' }, new Error('exit 1'));
    await expect(queryProjectWorkflowPlan('/tool', '/project', request)).rejects.toThrow('Target disabled');
    vi.mocked(execFile).mockClear();
    await expect(queryProjectWorkflowPlan('/tool', '/project', { ...request, profile: '' })).rejects.toThrow();
    expect(execFile).not.toHaveBeenCalled();
  });
});
