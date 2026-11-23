import { createHash } from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { EditorCookedContentServer } from './editorCookedContentServer';
import {
  LivePlayContentPublication,
  selectLivePlayTextureFamily,
  type LivePlayContentPublicationInput,
} from './livePlayContentPublication';
import type { LivePlayRemoteRelease } from './livePlayRemoteRelease';
import type { LivePlaySceneBaseline } from './livePlaySceneBaseline';
import type { WorkflowProcessRequest } from './workflowProcess';

const projectGuid = '11111111-2222-4333-8444-555555555555';
const sceneGuid = '22222222-2222-2222-2222-222222222222';
const sceneAssetGuid = '33333333-3333-3333-3333-333333333333';
const sha256 = (value: Buffer | string) => createHash('sha256').update(value).digest('hex');

const roots: string[] = [];
const servers: EditorCookedContentServer[] = [];
afterEach(async () => {
  await Promise.all(servers.splice(0).map((server) => server.stop()));
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});

function fixture(families: readonly string[] = ['bc']) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-c9-4b-publish-'));
  roots.push(root);
  const projectRoot = path.join(root, 'project');
  const contentRoot = path.join(projectRoot, 'Content');
  const sessionRoot = path.join(root, 'private');
  fs.mkdirSync(contentRoot, { recursive: true });
  fs.mkdirSync(sessionRoot);
  const scene = path.join(contentRoot, 'Scene.arcscene');
  fs.writeFileSync(scene, JSON.stringify({ format: 'arc.scene', scene: { id: sceneGuid }, entities: [] }));
  fs.writeFileSync(
    scene + '.arcmeta',
    JSON.stringify({
      format: 'arc.asset-meta',
      formatVersion: 1,
      guid: sceneAssetGuid,
      type: 'a7ca55e7-0000-0001-0000-000000000001',
    }),
  );
  const authoring = JSON.stringify({
    format: 'arc.scene',
    scene: { id: sceneGuid },
    entities: [{ guid: '44444444-4444-4444-4444-444444444444', transform: { position: [1, 2, 3] } }],
  });
  const scenePath = path.join(sessionRoot, 'initial-scene.arcscene');
  fs.writeFileSync(scenePath, authoring);
  const descriptorPath = path.join(projectRoot, 'Game.arcproject');
  fs.writeFileSync(descriptorPath, '{}');
  const cooker = path.join(root, 'arc-cook');
  fs.writeFileSync(cooker, '');
  const baseline: LivePlaySceneBaseline = {
    projectGuid,
    buildId: 'runtime-build-1',
    targetProfile: 'windows-x64-vulkan',
    sessionId: 'session-test-1',
    instanceId: 'instance-test-1',
    sceneGuid,
    sceneName: 'Scene',
    sourceScenePath: scene,
    sceneRevision: 7,
    historyRevision: 5,
    entityCount: 1,
    sha256: sha256(authoring),
    byteLength: Buffer.byteLength(authoring),
    scenePath,
    directory: sessionRoot,
  };
  const server = new EditorCookedContentServer({
    projectRoot,
    sessionRoot,
    projectGuid,
    runtimeBuildId: baseline.buildId,
    sessionId: baseline.sessionId,
    instanceId: baseline.instanceId,
  });
  servers.push(server);
  let current = true;
  const input: LivePlayContentPublicationInput = {
    baseline,
    projectGuid,
    projectRoot,
    descriptorPath,
    assetRoots: ['Content'],
    cooker,
    profile: baseline.targetProfile,
    target: { platform: 'windows', architecture: 'x86_64', renderer: 'vulkan' },
    textureFamilies: families,
    engineVersion: '0.1.0',
    server,
    isCurrent: () => current,
  };
  const cookedPayload = Buffer.from('binary cooked scene payload');
  const packaged = Buffer.alloc(4096 + cookedPayload.length);
  Buffer.from('ARCPAK1\0').copy(packaged);
  cookedPayload.copy(packaged, 4096);
  const chunkName = 'startup-' + sha256(packaged) + '.arcpak';
  const calls: WorkflowProcessRequest[] = [];
  const run = vi.fn(async (request: WorkflowProcessRequest) => {
    calls.push(request);
    const arg = (name: string) => request.args[request.args.indexOf(name) + 1];
    if (request.args[0] === 'cook') {
      const variants = families.map((family) => {
        const manifest = path.join(arg('--output'), family + '.arccookmanifest');
        fs.writeFileSync(
          manifest,
          JSON.stringify({
            format: 'arc.cook-manifest',
            version: 1,
            buildId: 'cook-build-1',
            roots: [sceneAssetGuid],
            target: {
              name: baseline.targetProfile,
              platform: 'windows',
              architecture: 'x86_64',
              renderer: 'vulkan',
              configuration: 'development',
              textures: family,
            },
            artifacts: [],
          }),
        );
        return { family, manifest };
      });
      return JSON.stringify({ event: 'cook.complete', variants });
    }
    if (request.args[0] === 'verify') return 'verified';
    if (request.args[0] === 'package') {
      const manifest = arg('--manifest');
      const output = arg('--output');
      const data = JSON.parse(fs.readFileSync(manifest, 'utf8')) as Record<string, unknown>;
      data.artifacts = [
        {
          asset: sceneAssetGuid,
          type: 'a7ca55e7-0000-0001-0000-000000000001',
          name: 'scene',
          schemaVersion: 1,
          schema: 'a7ca55e7-0000-0003-0000-000000000001',
          hash: sha256(cookedPayload),
          chunk: chunkName,
          offset: 4096,
          size: cookedPayload.length,
          storedSize: cookedPayload.length,
          compressed: false,
        },
      ];
      const packedManifest = path.join(output, 'windows-x64-vulkan.arccookmanifest');
      fs.writeFileSync(packedManifest, JSON.stringify(data));
      fs.writeFileSync(path.join(output, chunkName), packaged);
      return JSON.stringify({ event: 'package.complete', manifest: packedManifest });
    }
    throw new Error('Unexpected cooker operation');
  });
  return {
    root,
    input,
    baseline,
    server,
    run,
    calls,
    chunkName,
    packaged,
    getCurrent: () => current,
    changeProject: () => {
      current = false;
    },
  };
}

describe('C9.4b editor live-play publication orchestration', () => {
  it('cooks an unsaved scene, seals C9.1 release and exposes only verified HTTP objects', async () => {
    const f = fixture();
    await f.server.start();
    const token = f.server.issueClientToken();
    const publication = new LivePlayContentPublication(f.input, f.run);
    const result = await publication.start();
    expect(result).toMatchObject({
      projectGuid,
      profile: 'windows-x64-vulkan',
      textureFamily: 'bc',
      sceneGuid,
      sceneAssetGuid,
      generation: 1,
      objectCount: 2,
    });
    expect(result.descriptorSha256).toMatch(/^[a-f0-9]{64}$/u);
    expect(f.calls.map((call) => call.args[0])).toEqual(['cook', 'verify', 'package', 'verify']);
    expect(f.calls[0].args).toContain(f.baseline.scenePath);
    expect(f.calls[0].args).toContain('--ephemeral-asset-guid');
    const origin = f.server.snapshot().origin!;
    const response = await fetch(origin + '/releases/' + projectGuid + '/windows-x64-vulkan/current.json', {
      headers: { Authorization: 'Bearer ' + token },
    });
    expect(response.status).toBe(200);
    const metadata = (await response.json()) as {
      target: { cookProfile: string; textureFamily: string };
      chunks: Array<{ sha256: string }>;
      cookManifest: { sha256: string };
      baseBuildId: string;
      cookBuildId: string;
    };
    expect(metadata.target).toEqual({
      platform: 'windows',
      architecture: 'x86_64',
      renderer: 'vulkan',
      cookProfile: 'windows-x64-vulkan',
      textureFamily: 'bc',
    });
    expect(metadata.baseBuildId).toBe('runtime-build-1');
    expect(metadata.cookBuildId).toBe('cook-build-1');
    expect(metadata.chunks).toHaveLength(1);
    const chunk = await fetch(origin + '/objects/sha256/' + metadata.chunks[0].sha256, {
      headers: { Authorization: 'Bearer ' + token },
    });
    expect(chunk.status).toBe(200);
    expect(Buffer.from(await chunk.arrayBuffer())).toEqual(f.packaged);
    expect(fs.readFileSync(f.baseline.sourceScenePath, 'utf8')).not.toContain('[1,2,3]');
    expect(fs.readFileSync(f.baseline.scenePath, 'utf8')).toContain('[1,2,3]');
    expect(() => publication.start()).toThrow(/already started/);
    await publication.cancel();
  });

  it('rejects a damaged package without exposing half-cooked releases', async () => {
    const f = fixture();
    await f.server.start();
    const broken = async (request: WorkflowProcessRequest) => {
      const result = await f.run(request);
      if (request.args[0] === 'package') {
        const file = path.join(request.args[request.args.indexOf('--output') + 1], f.chunkName);
        const damaged = Buffer.from(fs.readFileSync(file));
        damaged[4096] ^= 0xff;
        fs.writeFileSync(file, damaged);
      }
      return result;
    };
    const publication = new LivePlayContentPublication(f.input, broken);
    await expect(publication.start()).rejects.toThrow(/artifact hash mismatch/);
    expect(f.server.snapshot().profiles).toEqual([]);
  });

  it('cancels in-progress cook on Stop without publishing or leaving a pending task', async () => {
    const f = fixture();
    await f.server.start();
    let started!: () => void;
    const didStart = new Promise<void>((resolve) => {
      started = resolve;
    });
    const runner = vi.fn((request: WorkflowProcessRequest): Promise<string> => {
      started();
      return new Promise<string>((_resolve, reject) => {
        if (request.signal.aborted) reject(new Error('Cook aborted'));
        else request.signal.addEventListener('abort', () => reject(new Error('Cook aborted')), { once: true });
      });
    });
    const publication = new LivePlayContentPublication(f.input, runner);
    const pending = publication.start();
    const rejected = expect(pending).rejects.toThrow(/Cook aborted/);
    await didStart;
    await publication.cancel();
    await rejected;
    expect(f.server.snapshot().profiles).toHaveLength(0);
    expect(fs.readdirSync(f.baseline.directory).sort()).toEqual(['initial-scene.arcscene']);
    expect(() => publication.start()).toThrow(/cancelled/);
  });

  it('rejects stale projects while cooking and never promotes their private release', async () => {
    const f = fixture();
    await f.server.start();
    const runner = async (request: WorkflowProcessRequest) => {
      const result = await f.run(request);
      if (request.args[0] === 'cook') f.changeProject();
      return result;
    };
    await expect(new LivePlayContentPublication(f.input, runner).start()).rejects.toThrow(/session changed/);
    expect(f.server.snapshot().profiles).toHaveLength(0);
  });

  it('fails closed on ambiguous and mismatched texture-family selection', async () => {
    const release = {
      variants: [
        { family: 'bc', descriptorPath: '', descriptorSha256: '', cookedBuildId: '', chunkCount: 1 },
        { family: 'portable', descriptorPath: '', descriptorSha256: '', cookedBuildId: '', chunkCount: 1 },
      ],
    } satisfies Pick<LivePlayRemoteRelease, 'variants'>;
    expect(selectLivePlayTextureFamily(release, ['bc'])).toBe('bc');
    expect(() => selectLivePlayTextureFamily(release, ['astc'])).toThrow(/No cooked texture family/);
    expect(() => selectLivePlayTextureFamily(release, ['bc', 'portable'])).toThrow(/Multiple compatible/);
    expect(() => selectLivePlayTextureFamily(release, ['BC'])).toThrow(/invalid texture-family/);
    expect(() => selectLivePlayTextureFamily(release, [])).toThrow(/invalid texture-family/);
    const f = fixture(['bc', 'portable']);
    await f.server.start();
    const publication = new LivePlayContentPublication(f.input, f.run);
    await expect(publication.start()).rejects.toThrow(/explicit target texture-family/);
    expect(f.server.snapshot().profiles).toHaveLength(0);
  });

  it('rejects identity mismatches and never invokes the cooker', async () => {
    const f = fixture();
    await f.server.start();
    const publication = new LivePlayContentPublication({ ...f.input, projectGuid: sceneGuid }, f.run);
    await expect(publication.start()).rejects.toThrow(/matching trusted session/);
    expect(f.run).not.toHaveBeenCalled();
    expect(f.server.snapshot().profiles).toHaveLength(0);
  });
});
