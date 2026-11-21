import { createHash } from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { prepareLivePlayCookedStaging, resolveLivePlaySceneAsset, type LivePlayCookInput } from './livePlayEphemeralCook';
import type { LivePlaySceneBaseline } from './livePlaySceneBaseline';
import type { WorkflowProcessRequest } from './workflowProcess';

const roots: string[] = [];
const sceneGuid = '22222222-2222-2222-2222-222222222222';
const assetGuid = '33333333-3333-3333-3333-333333333333';
afterEach(() => {
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});

function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-live-play-cook-test-'));
  roots.push(root);
  const projectRoot = path.join(root, 'project');
  const content = path.join(projectRoot, 'Content', 'Scenes');
  const privateRoot = path.join(root, 'private-session');
  fs.mkdirSync(content, { recursive: true });
  fs.mkdirSync(privateRoot);
  const original = path.join(content, 'Level.arcscene');
  fs.writeFileSync(original, JSON.stringify({ format: 'arc.scene', scene: { id: sceneGuid }, entities: [] }));
  fs.writeFileSync(`${original}.arcmeta`, JSON.stringify({
    format: 'arc.asset-meta',
    formatVersion: 1,
    guid: assetGuid,
    type: 'a7ca55e7-0000-0001-0000-000000000001',
  }));
  const scenePath = path.join(privateRoot, 'initial-scene.arcscene');
  const authored = JSON.stringify({
    format: 'arc.scene',
    scene: { id: sceneGuid },
    entities: [{ id: '44444444-4444-4444-4444-444444444444', components: { Transform: { position: [1, 2, 3] } } }],
  });
  fs.writeFileSync(scenePath, authored);
  const descriptorPath = path.join(projectRoot, 'Game.arcproject');
  fs.writeFileSync(descriptorPath, '{}');
  const tool = path.join(root, 'arc-cook');
  fs.writeFileSync(tool, '');
  const baseline: LivePlaySceneBaseline = {
    projectGuid: '11111111-1111-1111-1111-111111111111',
    buildId: 'runtime-build',
    targetProfile: 'windows-x64-vulkan',
    sessionId: 'session-7',
    instanceId: 'instance-8',
    sceneGuid,
    sceneName: 'Level',
    sourceScenePath: original,
    sceneRevision: 12,
    historyRevision: 10,
    entityCount: 1,
    sha256: createHash('sha256').update(authored).digest('hex'),
    byteLength: Buffer.byteLength(authored),
    scenePath,
    directory: privateRoot,
  };
  const controller = new AbortController();
  const input: LivePlayCookInput = {
    baseline, projectGuid: baseline.projectGuid, projectRoot, descriptorPath,
    assetRoots: ['Content'], cooker: tool, profile: baseline.targetProfile,
    target: { platform: 'windows', architecture: 'x86_64', renderer: 'vulkan' },
    signal: controller.signal,
    isCurrent: () => true,
  };
  const calls: WorkflowProcessRequest[] = [];
  const run = vi.fn(async (request: WorkflowProcessRequest) => {
    calls.push(request);
    const argument = (name: string) => request.args[request.args.indexOf(name) + 1];
    if (request.args[0] === 'cook') {
      const output = argument('--output');
      const variant = path.join(output, 'windows-x64-vulkan.arccookmanifest');
      fs.writeFileSync(variant, JSON.stringify({
        buildId: 'cook-build', roots: [assetGuid],
        target: { name: input.profile, ...input.target, configuration: 'development' },
      }));
      return JSON.stringify({ event: 'cook.complete', variants: [{ family: 'bc', manifest: variant }] });
    }
    if (request.args[0] === 'package') {
      const manifest = path.join(argument('--output'), 'windows-x64-vulkan.arccookmanifest');
      fs.copyFileSync(argument('--manifest'), manifest);
      fs.writeFileSync(path.join(argument('--output'), 'startup.arcpak'), 'package-data');
      return JSON.stringify({ event: 'package.complete', manifest });
    }
    if (request.args[0] === 'verify') return 'verified';
    throw new Error('Unexpected tool call');
  });
  return { root, input, baseline, run, calls, controller, original, content, privateRoot };
}

describe('Live Play private cooked scene staging', () => {
  it('binds source scene to its separate registered asset GUID', () => {
    const t = fixture();
    expect(resolveLivePlaySceneAsset(t.input)).toBe(assetGuid);
    const metadata = JSON.parse(fs.readFileSync(`${t.original}.arcmeta`, 'utf8'));
    metadata.guid = sceneGuid;
    fs.writeFileSync(`${t.original}.arcmeta`, JSON.stringify(metadata));
    // GUID values are separate domains, but either syntactically valid GUID
    // can represent an asset. The original scene identity is still checked.
    expect(resolveLivePlaySceneAsset(t.input)).toBe(sceneGuid);
    fs.writeFileSync(t.original, JSON.stringify({ format: 'arc.scene', scene: { id: assetGuid } }));
    expect(() => resolveLivePlaySceneAsset(t.input)).toThrow(/identity differs/);
  });

  it('cooks and verifies then packages a private generation without publishing current', async () => {
    const t = fixture();
    const result = await prepareLivePlayCookedStaging(t.input, t.run);
    expect(result).toMatchObject({
      projectGuid: t.baseline.projectGuid,
      runtimeBuildId: 'runtime-build',
      sessionId: t.baseline.sessionId,
      sceneGuid,
      sceneAssetGuid: assetGuid,
      authoringSha256: t.baseline.sha256,
    });
    expect(result.variants).toHaveLength(1);
    expect(result.variants[0].cookManifestBuildId).toBe('cook-build');
    expect(fs.existsSync(result.variants[0].manifestPath)).toBe(true);
    expect(t.calls.map((call) => call.args[0])).toEqual(['cook', 'verify', 'package', 'verify']);
    expect(t.calls[0].args).toContain('--ephemeral-asset-guid');
    expect(t.calls[0].args).toContain(t.baseline.scenePath);
    expect(fs.existsSync(path.join(t.input.projectRoot, 'Build'))).toBe(false);
    expect(fs.readFileSync(t.original, 'utf8')).toContain(sceneGuid);
    expect(result.directory.startsWith(t.privateRoot)).toBe(true);
    expect(fs.existsSync(path.join(t.privateRoot, 'current.json'))).toBe(false);
  });

  it('rejects a forged or mutated capture before running the cooker', async () => {
    const t = fixture();
    fs.appendFileSync(t.baseline.scenePath, '\n');
    await expect(prepareLivePlayCookedStaging(t.input, t.run)).rejects.toThrow(/changed after/);
    expect(t.run).not.toHaveBeenCalled();
  });

  it('rejects unknown, unsaved, or out-of-root active scene paths', async () => {
    const t = fixture();
    await expect(prepareLivePlayCookedStaging({
      ...t.input, baseline: { ...t.baseline, sourceScenePath: '' },
    }, t.run)).rejects.toThrow(/Save the scene once/);
    const outsider = path.join(t.root, 'outside.arcscene');
    fs.writeFileSync(outsider, fs.readFileSync(t.original));
    expect(() => resolveLivePlaySceneAsset({
      ...t.input, baseline: { ...t.baseline, sourceScenePath: outsider },
    })).toThrow(/does not belong/);
    expect(t.run).not.toHaveBeenCalled();
  });

  it('rejects stale sessions and cleans partially prepared cooked generations', async () => {
    const t = fixture();
    const run = vi.fn(async (request: WorkflowProcessRequest) => {
      if (request.args[0] === 'cook') throw new Error('broken cook');
      return '';
    });
    await expect(prepareLivePlayCookedStaging(t.input, run)).rejects.toThrow(/broken cook/);
    expect(fs.readdirSync(t.privateRoot)).toEqual(['initial-scene.arcscene']);
    await expect(prepareLivePlayCookedStaging({
      ...t.input, isCurrent: () => false,
    }, run)).rejects.toThrow(/session changed/);
    t.controller.abort();
    await expect(prepareLivePlayCookedStaging(t.input, run)).rejects.toThrow();
  });

  it('rejects incorrectly targeted packaged manifests and discards their output', async () => {
    const t = fixture();
    const run = async (request: WorkflowProcessRequest) => {
      const response = await t.run(request);
      if (request.args[0] === 'package') {
        const manifest = JSON.parse(fs.readFileSync(JSON.parse(response).manifest, 'utf8'));
        manifest.target.platform = 'android';
        fs.writeFileSync(JSON.parse(response).manifest, JSON.stringify(manifest));
      }
      return response;
    };
    await expect(prepareLivePlayCookedStaging(t.input, run)).rejects.toThrow(/does not match the target/);
    expect(fs.readdirSync(t.privateRoot)).toEqual(['initial-scene.arcscene']);
  });
});
