import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { WorkflowPlan } from '../common/projectWorkflowTypes';
import { ProjectWorkflowRunner } from './projectWorkflowRunner';
import { createCookWorkflowAdapter } from './cookWorkflowAdapter';
import type { WorkflowProcessRequest } from './workflowProcess';

const roots: string[] = [];
afterEach(() => {
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});
function fixture(options: { fail?: boolean; wrongTarget?: boolean; escape?: boolean; variants?: boolean } = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-cook-adapter-'));
  roots.push(root);
  const plan: WorkflowPlan = {
    version: 1,
    succeeded: true,
    action: 'cook',
    destination: 'none',
    sessionId: '',
    projectGuid: 'project',
    engineVersion: '0.1',
    profile: 'android-dev',
    targetId: 'android-arm64-vulkan',
    configuration: 'Debug',
    platform: 'android',
    architecture: 'arm64',
    renderer: 'vulkan',
    finalizer: 'android-gradle',
    cookOutput: path.join(root, 'Build', 'Cooked', 'android-dev'),
    packageOutput: path.join(root, 'Build', 'Packages'),
    stages: ['validate', 'cook'],
  };
  const process = vi.fn(async (request: WorkflowProcessRequest) => {
    if (request.args[0] === 'verify') {
      if (options.fail) throw new Error('Corrupt blob');
      return 'verified';
    }
    const output = request.args[request.args.indexOf('--output') + 1];
    const manifests = (options.variants ? ['astc', 'etc2'] : ['astc']).map((family) => {
      const manifest = path.join(options.escape ? root : output, `${family}.arccookmanifest`);
      fs.writeFileSync(
        manifest,
        JSON.stringify({
          buildId: `native-${family}`,
          target: {
            name: plan.profile,
            platform: options.wrongTarget ? 'windows' : plan.platform,
            architecture: plan.architecture,
            renderer: plan.renderer,
            configuration: 'development',
          },
        }),
      );
      return { manifest };
    });
    return JSON.stringify({ event: 'cook.complete', ...(options.variants ? { variants: manifests } : manifests[0]) });
  });
  const cook = createCookWorkflowAdapter(processExecPath(), process);
  const runner = new ProjectWorkflowRunner(
    {
      validate: { capability: () => ({ available: true }), execute: async () => {}, rollback: async () => {} },
      cook: cook.adapter,
    },
    () => true,
    cook.publication,
    () => {},
  );
  const run = () =>
    runner.execute(plan, {
      projectGuid: 'project',
      projectRoot: root,
      descriptorPath: path.join(root, 'Game.arcproject'),
      projectRevision: 1,
      sdkIdentity: 'sdk',
    });
  return { root, plan, runner, process, run };
}
const processExecPath = () => process.execPath;

describe('executable cook adapter', () => {
  it.each([false, true])(
    'verifies all manifests and publishes an immutable generation (variants=%s)',
    async (variants) => {
      const fixture_ = fixture({ variants });
      const result = await fixture_.run();
      expect(result.state).toBe('succeeded');
      expect(result.artifacts).toHaveLength(variants ? 2 : 1);
      expect(fixture_.process.mock.calls[0][0].args).toContain('android-dev');
      expect(fixture_.process.mock.calls.every(([call]) => ['cook', 'verify'].includes(call.args[0]))).toBe(true);
      const record = JSON.parse(fs.readFileSync(path.join(fixture_.plan.cookOutput, 'current-workflow.json'), 'utf8'));
      expect(record.artifacts).toEqual(result.artifacts);
      expect(result.artifacts[0].buildId).toBe('native-astc');
      expect(fs.existsSync(result.artifacts[0].manifestPath)).toBe(true);
    },
  );
  it.each([{ fail: true }, { wrongTarget: true }, { escape: true }])(
    'preserves previous output on failure %j',
    async (options) => {
      const value = fixture(options);
      fs.mkdirSync(value.plan.cookOutput, { recursive: true });
      const record = path.join(value.plan.cookOutput, 'current-workflow.json');
      fs.writeFileSync(record, 'previous');
      const result = await value.run();
      expect(result.state).toBe('failed');
      expect(result.artifacts).toEqual([]);
      expect(fs.readFileSync(record, 'utf8')).toBe('previous');
    },
  );
  it('does not publish when cancelled during verification', async () => {
    const value = fixture();
    const implementation = value.process.getMockImplementation()!;
    value.process.mockImplementation(async (request) => {
      if (request.args[0] === 'verify') value.runner.cancel(value.runner.snapshot()!.id);
      return implementation(request);
    });
    expect((await value.run()).state).toBe('cancelled');
    expect(fs.existsSync(path.join(value.plan.cookOutput, 'current-workflow.json'))).toBe(false);
  });
  it('rejects output outside the project before spawning', async () => {
    const value = fixture();
    value.plan.cookOutput = path.dirname(value.root);
    expect((await value.run()).state).toBe('failed');
    expect(value.process).not.toHaveBeenCalled();
  });
});
