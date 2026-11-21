import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { WorkflowPlan } from '../common/projectWorkflowTypes';
import { ProjectWorkflowRunner } from './projectWorkflowRunner';
import { createWindowsBuildWorkflow, windowsBuildCapability } from './windowsBuildWorkflow';
import type { WorkflowProcessRequest } from './workflowProcess';

const roots: string[] = [];
afterEach(() => {
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});

function fixture(
  options: {
    fail?: number;
    stale?: number;
    cancel?: number;
    variants?: boolean;
    wrongIdentity?: boolean;
    escape?: boolean;
    cmakeGraph?: boolean;
    emitNativeArtifact?: boolean;
    development?: boolean;
  } = {},
) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc shipping workflow '));
  roots.push(root);
  fs.writeFileSync(path.join(root, 'CMakeLists.txt'), 'fixture');
  const config = path.join(root, 'Renderer.json');
  fs.writeFileSync(config, '{}');
  const plan: WorkflowPlan = {
    version: 1,
    succeeded: true,
    action: 'build',
    destination: 'none',
    sessionId: '',
    projectGuid: 'project',
    engineVersion: '0.1.0',
    profile: 'custom-release',
    targetId: 'windows-x64-vulkan',
    configuration: 'Shipping',
    platform: 'windows',
    architecture: 'x86_64',
    renderer: 'vulkan',
    finalizer: 'windows-directory',
    cookOutput: path.join(root, 'Build', 'Cook'),
    packageOutput: path.join(root, 'Build', 'Packages'),
    stages: ['validate', 'configure', 'compile', 'cook', 'package'],
  };
  fs.mkdirSync(plan.packageOutput, { recursive: true });
  const record = path.join(plan.packageOutput, 'current-workflow.json');
  fs.writeFileSync(record, 'previous successful package');
  let current = true;
  let count = 0;
  const runProcess = vi.fn(async (request: WorkflowProcessRequest) => {
    const index = ++count;
    if (index === options.fail) throw new Error('Injected process failure');
    if (index === options.stale) current = false;
    if (index === options.cancel) runner.cancel(runner.snapshot()!.id);
    const arg = (key: string) => request.args[request.args.indexOf(key) + 1];
    if (options.cmakeGraph && request.args[0] === '-S') {
      const reply = path.join(arg('-B'), '.cmake', 'api', 'v1', 'reply');
      expect(fs.existsSync(path.join(arg('-B'), '.cmake', 'api', 'v1', 'query', 'codemodel-v2'))).toBe(true);
      fs.mkdirSync(reply, { recursive: true });
      fs.writeFileSync(
        path.join(reply, 'index-20261116-100000.json'),
        JSON.stringify({
          cmake: { generator: { name: 'Visual Studio 17 2022' } },
          reply: { 'codemodel-v2': { jsonFile: 'codemodel-v2.json' } },
        }),
      );
      fs.writeFileSync(
        path.join(reply, 'codemodel-v2.json'),
        JSON.stringify({
          configurations: [
            {
              name: 'Shipping',
              targets: [
                { id: 'root', name: 'arc-runtime-output', jsonFile: 'target-root.json' },
                { id: 'game', name: 'game', jsonFile: 'target-game.json' },
              ],
            },
          ],
        }),
      );
      fs.writeFileSync(
        path.join(reply, 'target-root.json'),
        JSON.stringify({
          id: 'root',
          name: 'arc-runtime-output',
          type: 'UTILITY',
          dependencies: [{ id: 'game' }],
        }),
      );
      fs.writeFileSync(
        path.join(reply, 'target-game.json'),
        JSON.stringify({
          id: 'game',
          name: 'game',
          type: 'EXECUTABLE',
          artifacts: [{ path: 'Shipping/game.exe' }],
        }),
      );
    }
    if (options.emitNativeArtifact && request.args[0] === '--build') {
      const artifact = path.join(arg('--build'), 'Shipping', 'game.exe');
      request.onRawOutput?.(`game.vcxproj -> ${artifact}\n`, 'stdout');
    }
    if (request.args[0] === 'cook') {
      const manifests = (options.variants ? ['bc', 'other'] : ['bc']).map((name) => {
        const manifest = path.join(arg('--output'), `${name}.arccookmanifest`);
        fs.writeFileSync(
          manifest,
          JSON.stringify({
            buildId: name,
            target: {
              name: plan.profile,
              platform: plan.platform,
              architecture: plan.architecture,
              renderer: plan.renderer,
              configuration: 'shipping',
            },
          }),
        );
        return { manifest };
      });
      return JSON.stringify({ event: 'cook.complete', variants: manifests });
    }
    if (request.args.includes('--verify-build-only'))
      return JSON.stringify({ succeeded: true, executable: path.join(root, 'exact-runtime.exe') });
    if (request.args[0] === 'package' && request.args.includes('--manifest')) {
      const output = arg('--output');
      fs.mkdirSync(output);
      const manifest = path.join(output, 'packed.arccookmanifest');
      fs.copyFileSync(arg('--manifest'), manifest);
      return JSON.stringify({ event: 'package.complete', manifest });
    }
    if (request.args[0] === 'package') {
      const output = arg('--output');
      fs.mkdirSync(output);
      const manifest = path.join(output, 'arc-distribution.json');
      fs.writeFileSync(
        manifest,
        JSON.stringify({
          projectGuid: options.wrongIdentity ? 'other' : plan.projectGuid,
          target: { id: plan.targetId, cookProfile: plan.profile, configuration: plan.configuration },
        }),
      );
      return JSON.stringify({ succeeded: true, output: options.escape ? root : output, manifest });
    }
    return '';
  });
  const environment = {
    cmake: process.execPath,
    sdkRoot: path.join(root, 'SDK'),
    visualStudio: path.join(root, 'VS'),
    generator: 'Visual Studio 17 2022',
    ...(options.development
      ? {
          development: {
            buildDirectory: path.join(root, 'EngineBuild'),
            engineVersion: '0.1.0',
            architecture: 'x86_64',
          },
        }
      : {}),
  };
  const workflow = createWindowsBuildWorkflow(
    process.execPath,
    process.execPath,
    environment,
    path.join(root, 'Build'),
    config,
    runProcess,
    undefined,
    'win32',
  );
  const runner = new ProjectWorkflowRunner(
    workflow.adapters,
    () => current,
    workflow.publication,
    () => {},
  );
  return {
    plan,
    record,
    runProcess,
    root,
    environment,
    run: () =>
      runner.execute(plan, {
        projectGuid: plan.projectGuid,
        projectRoot: root,
        descriptorPath: path.join(root, 'Game.arcproject'),
        projectRevision: 1,
        sdkIdentity: 'sdk',
      }),
  };
}

describe('Windows Shipping workflow', () => {
  it('rejects a development SDK with a different engine or architecture', () => {
    const value = fixture({ development: true });
    expect(
      windowsBuildCapability({ ...value.plan, engineVersion: '2.0.0' }, value.environment, 'win32').available,
    ).toBe(false);
    expect(windowsBuildCapability({ ...value.plan, architecture: 'arm64' }, value.environment, 'win32').available).toBe(
      false,
    );
  });
  it('prepares the exact development SDK configuration before configuring the project', async () => {
    const value = fixture({ development: true });
    const result = await value.run();
    expect(result.state).toBe('succeeded');
    expect(result.stages.find((stage) => stage.stage === 'configure')?.tasks).toMatchObject([
      { id: 'development-sdk', name: 'Build ARC development SDK', state: 'succeeded' },
      { id: 'cmake-configure', name: 'Configure project', state: 'succeeded' },
    ]);
    const calls = value.runProcess.mock.calls.map(([request]) => request.args);
    expect(calls[0]).toEqual([
      '--build',
      path.join(value.root, 'EngineBuild'),
      '--config',
      'Shipping',
      '--target',
      'arc-development-sdk',
    ]);
    expect(calls[1]).toContain(`-DARC_DIR=${path.join(value.environment.sdkRoot, 'lib', 'cmake', 'ARC')}`);
  });
  it('uses explicit runtime metadata and profile, and publishes only packaged output without launching or reloading', async () => {
    const value = fixture();
    const result = await value.run();
    expect(result.state).toBe('succeeded');
    expect(result.artifacts).toHaveLength(1);
    expect(result.artifacts[0].kind).toBe('package');
    const calls = value.runProcess.mock.calls.map(([request]) => request.args);
    expect(calls.map((args) => args[0])).toEqual([
      '-S',
      '--build',
      'package',
      'cook',
      'verify',
      'package',
      'verify',
      'package',
    ]);
    expect(calls[0]).toContain('-DARC_WORKFLOW_PROFILE=custom-release');
    expect(calls[0]).toContain(`-DCMAKE_PREFIX_PATH=${value.environment.sdkRoot}`);
    expect(calls[1]).toContain('arc-runtime-output');
    expect(calls[1]).toContain('Shipping');
    expect(calls[2]).toContain('--verify-build-only');
    expect(calls[5]).toContain('--manifest');
    expect(calls[7]).toContain('--build-manifest');
    expect(calls[7]).not.toContain('--executable');
    expect(calls[7]).not.toContain('--dll-dir');
    expect(JSON.parse(fs.readFileSync(value.record, 'utf8')).artifacts).toEqual(result.artifacts);
    expect(fs.existsSync(path.join(value.plan.cookOutput, 'current-workflow.json'))).toBe(false);
    // Incremental compilation retains the same captured-profile code directory.
    expect((await value.run()).state).toBe('succeeded');
    expect(value.runProcess.mock.calls[8][0].args).toEqual(calls[0]);
  });

  it('discovers CMake dependencies and reports native artifacts during the Windows build', async () => {
    const value = fixture({ cmakeGraph: true, emitNativeArtifact: true });
    const result = await value.run();
    expect(result.state).toBe('succeeded');
    expect(result.targetGraph).toMatchObject({
      configuration: 'Shipping',
      rootTargetId: 'root',
      revision: 3,
      nodes: [
        { id: 'root', dependencies: ['game'], satisfiedAt: expect.any(String) },
        { id: 'game', artifactObservedAt: expect.any(String) },
      ],
    });
  });

  it('never claims unreported dependencies were rebuilt on an incremental success', async () => {
    const value = fixture({ cmakeGraph: true });
    const result = await value.run();
    expect(result.state).toBe('succeeded');
    expect(result.targetGraph?.nodes.find((node) => node.id === 'root')?.satisfiedAt).toEqual(expect.any(String));
    expect(result.targetGraph?.nodes.find((node) => node.id === 'game')?.artifactObservedAt).toBeUndefined();
  });

  it.each([1, 2, 3, 4, 5, 6, 7, 8])('preserves previous output when command %s fails', async (fail) => {
    const value = fixture({ fail });
    expect((await value.run()).state).toBe('failed');
    expect(value.runProcess).toHaveBeenCalledTimes(fail);
    expect(fs.readFileSync(value.record, 'utf8')).toBe('previous successful package');
  });
  it.each([1, 2, 3, 4, 5, 6, 7, 8])(
    'rejects late completion after authority changes during command %s',
    async (stale) => {
      const value = fixture({ stale });
      expect((await value.run()).state).toBe('failed');
      expect(value.runProcess).toHaveBeenCalledTimes(stale);
      expect(fs.readFileSync(value.record, 'utf8')).toBe('previous successful package');
    },
  );
  it.each([1, 2, 3, 4, 5, 6, 7, 8])(
    'cancels without publishing or starting another stage at command %s',
    async (cancel) => {
      const value = fixture({ cancel });
      expect((await value.run()).state).toBe('cancelled');
      expect(value.runProcess).toHaveBeenCalledTimes(cancel);
      expect(fs.readFileSync(value.record, 'utf8')).toBe('previous successful package');
    },
  );
  it.each([{ variants: true }, { wrongIdentity: true }, { escape: true }])(
    'rejects ambiguous or invalid output %j',
    async (options) => {
      const value = fixture(options);
      expect((await value.run()).state).toBe('failed');
      expect(fs.readFileSync(value.record, 'utf8')).toBe('previous successful package');
    },
  );
  it('rejects unsupported platforms/configurations and missing SDK before work', async () => {
    const value = fixture();
    expect(windowsBuildCapability(value.plan, undefined, 'win32').available).toBe(false);
    expect(windowsBuildCapability(value.plan, value.environment, 'linux').available).toBe(false);
    value.plan.configuration = 'Debug';
    await expect(value.run()).rejects.toThrow('Windows Shipping');
    expect(value.runProcess).not.toHaveBeenCalled();
  });
  it('rejects output escape without running a command', async () => {
    const value = fixture();
    value.plan.packageOutput = path.dirname(value.root);
    expect((await value.run()).error).toContain('inside the project');
    expect(fs.readFileSync(value.record, 'utf8')).toBe('previous successful package');
  });
});
