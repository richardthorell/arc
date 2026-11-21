import { createHash } from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { prepareLivePlayRemoteRelease } from './livePlayRemoteRelease';
import type { LivePlayCookedStaging } from './livePlayEphemeralCook';

const roots: string[] = [];
const assetGuid = '33333333-3333-3333-3333-333333333333';
const sceneGuid = '22222222-2222-2222-2222-222222222222';
const digest = (value: Buffer | string) => createHash('sha256').update(value).digest('hex');

afterEach(() => {
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});

function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-c9-live-play-release-'));
  roots.push(root);
  const packageDirectory = path.join(root, 'packages', 'bc');
  fs.mkdirSync(packageDirectory, { recursive: true });
  const payload = Buffer.from('binary cooked scene');
  const bytes = Buffer.alloc(4096 + payload.length);
  Buffer.from('ARCPAK1\0').copy(bytes, 0);
  payload.copy(bytes, 4096);
  const chunkName = `startup-${digest(bytes)}.arcpak`;
  fs.writeFileSync(path.join(packageDirectory, chunkName), bytes);
  const manifestPath = path.join(packageDirectory, 'windows-x64-vulkan.arccookmanifest');
  const manifest = {
    format: 'arc.cook-manifest',
    version: 1,
    buildId: 'cook-build-abc',
    roots: [assetGuid],
    target: {
      name: 'windows-x64-vulkan',
      platform: 'windows',
      architecture: 'x86_64',
      renderer: 'vulkan',
      configuration: 'development',
      textures: 'bc',
    },
    artifacts: [{
      asset: assetGuid,
      type: 'a7ca55e7-0000-0001-0000-000000000001',
      name: 'scene',
      schemaVersion: 1,
      schema: 'a7ca55e7-0000-0003-0000-000000000001',
      hash: digest(payload),
      chunk: chunkName,
      offset: 4096,
      size: payload.length,
      storedSize: payload.length,
      compressed: false,
    }],
  };
  fs.writeFileSync(manifestPath, JSON.stringify(manifest));
  const staged: LivePlayCookedStaging = {
    directory: root,
    projectGuid: '11111111-2222-4333-8444-555555555555',
    runtimeBuildId: 'published-runtime-123',
    profile: 'windows-x64-vulkan',
    sessionId: 'session-7',
    instanceId: 'instance-8',
    sceneGuid,
    sceneAssetGuid: assetGuid,
    authoringRevision: 12,
    authoringSha256: 'a'.repeat(64),
    variants: [{ family: 'bc', manifestPath, packageDirectory, cookManifestBuildId: 'cook-build-abc' }],
  };
  const controller = new AbortController();
  const input = {
    staged,
    engineVersion: '0.1.0',
    generation: 1,
    signal: controller.signal,
    isCurrent: () => true,
  };
  return { root, staged, controller, input, manifestPath, packageDirectory, chunkName, bytes, manifest };
}

describe('C9 Live Play immutable release assembly', () => {
  it('assembles complete, verified C9 v1 remote metadata and immutable objects', async () => {
    const f = fixture();
    const published = await prepareLivePlayRemoteRelease(f.input);
    expect(published).toMatchObject({
      runtimeBuildId: 'published-runtime-123',
      projectGuid: f.staged.projectGuid,
      sceneGuid,
      sceneAssetGuid: assetGuid,
      generation: 1,
    });
    expect(published.variants).toHaveLength(1);
    const variant = published.variants[0];
    const releaseBytes = fs.readFileSync(variant.descriptorPath);
    const release = JSON.parse(releaseBytes.toString('utf8'));
    expect(release).toMatchObject({
      format: 'arc.remote-content',
      formatVersion: 1,
      baseBuildId: 'published-runtime-123',
      cookBuildId: 'cook-build-abc',
      complete: true,
      generation: 1,
      target: { cookProfile: 'windows-x64-vulkan', textureFamily: 'bc' },
    });
    expect(variant.descriptorSha256).toBe(digest(releaseBytes));
    expect(release.cookManifest.path).toBe(`objects/sha256/${release.cookManifest.sha256}`);
    expect(release.chunks).toHaveLength(1);
    for (const object of [release.cookManifest, ...release.chunks]) {
      const stored = fs.readFileSync(path.join(published.root, object.path));
      expect(stored.byteLength).toBe(object.size);
      expect(digest(stored)).toBe(object.sha256);
    }
    expect(fs.readFileSync(f.manifestPath, 'utf8')).toContain('cook-build-abc');
    await expect(prepareLivePlayRemoteRelease(f.input)).rejects.toThrow(/already been published/);
  });

  it('rejects a corrupted cooked artifact even when the package object itself has a valid SHA-256', async () => {
    const f = fixture();
    const wrong = Buffer.from(f.bytes);
    wrong[4096] ^= 0xff;
    fs.writeFileSync(path.join(f.packageDirectory, f.chunkName), wrong);
    await expect(prepareLivePlayRemoteRelease(f.input)).rejects.toThrow(/artifact hash mismatch/);
    expect(fs.existsSync(path.join(f.root, 'release'))).toBe(false);
    expect(fs.readdirSync(f.root).some((name) => name.startsWith('release-candidate-'))).toBe(false);
  });

  it('rejects missing and extra cooked chunks before publishing a generation', async () => {
    const f = fixture();
    fs.unlinkSync(path.join(f.packageDirectory, f.chunkName));
    await expect(prepareLivePlayRemoteRelease(f.input)).rejects.toThrow(/file set differs/);
    fs.writeFileSync(path.join(f.packageDirectory, f.chunkName), f.bytes);
    fs.writeFileSync(path.join(f.packageDirectory, 'unlisted.arcpak'), f.bytes);
    await expect(prepareLivePlayRemoteRelease(f.input)).rejects.toThrow(/file set differs/);
    expect(fs.existsSync(path.join(f.root, 'release'))).toBe(false);
  });

  it('rejects wrong build or target identity while keeping the previous artifacts private', async () => {
    const f = fixture();
    const manifest = { ...f.manifest, buildId: 'forged-cook-build' };
    fs.writeFileSync(f.manifestPath, JSON.stringify(manifest));
    await expect(prepareLivePlayRemoteRelease(f.input)).rejects.toThrow(/build, profile/);
    manifest.buildId = 'cook-build-abc';
    manifest.target = { ...manifest.target, configuration: 'shipping' };
    fs.writeFileSync(f.manifestPath, JSON.stringify(manifest));
    await expect(prepareLivePlayRemoteRelease(f.input)).rejects.toThrow(/does not meet ARC/);
    expect(fs.existsSync(path.join(f.root, 'release'))).toBe(false);
  });

  it('fails closed on stale sessions, cancellation, unsafe family and target path escapes', async () => {
    const f = fixture();
    await expect(prepareLivePlayRemoteRelease({ ...f.input, isCurrent: () => false })).rejects.toThrow(
      /session or project changed/,
    );
    await expect(prepareLivePlayRemoteRelease({
      ...f.input,
      staged: { ...f.staged, variants: [{ ...f.staged.variants[0], family: '../escape' }] },
    })).rejects.toThrow(/unsafe target texture family/);
    f.controller.abort();
    await expect(prepareLivePlayRemoteRelease(f.input)).rejects.toThrow();
    expect(fs.existsSync(path.join(f.root, 'release'))).toBe(false);
  });
});
