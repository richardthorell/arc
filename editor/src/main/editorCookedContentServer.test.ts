import { createHash } from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import type { LivePlayRemoteRelease } from './livePlayRemoteRelease';
import { EditorCookedContentServer } from './editorCookedContentServer';

const projectGuid = '11111111-2222-4333-8444-555555555555';
const runtimeBuildId = 'runtime-123';
const sessionId = 'session-abc';
const instanceId = 'instance-def';
const sceneGuid = '22222222-2222-2222-2222-222222222222';
const digest = (bytes: Buffer | string) => createHash('sha256').update(bytes).digest('hex');

const roots: string[] = [];
const servers: EditorCookedContentServer[] = [];

afterEach(async () => {
  await Promise.all(servers.splice(0).map((server) => server.stop()));
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});

function fixture() {
  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-editor-c9-http-'));
  const projectRoot = path.join(temporaryRoot, 'project');
  const sessionRoot = path.join(temporaryRoot, 'private-session');
  fs.mkdirSync(projectRoot);
  fs.mkdirSync(sessionRoot);
  roots.push(temporaryRoot);
  const options = { projectRoot, sessionRoot, projectGuid, runtimeBuildId, sessionId, instanceId };
  const server = new EditorCookedContentServer(options);
  servers.push(server);
  return { temporaryRoot, projectRoot, sessionRoot, options, server };
}

function releaseFixture(
  sessionRoot: string,
  options: { profile?: string; family?: string; generation?: number; name?: string } = {},
) {
  const target = options.profile ?? 'windows-x64-vulkan';
  const family = options.family ?? 'bc';
  const generation = options.generation ?? 1;
  const root = path.join(sessionRoot, options.name ?? 'build-a', 'release');
  const objectRoot = path.join(root, 'objects', 'sha256');
  const descriptorFolder = path.join(root, 'targets', family);
  fs.mkdirSync(objectRoot, { recursive: true });
  fs.mkdirSync(descriptorFolder, { recursive: true });

  const manifestBytes = Buffer.from('a verified C9 cooked manifest');
  const chunkBytes = Buffer.from('ARCPAK1\0binary cooked scene bytes');
  const objects = [manifestBytes, chunkBytes].map((bytes) => ({
    bytes,
    sha256: digest(bytes),
    path: 'objects/sha256/' + digest(bytes),
    size: bytes.length,
  }));
  for (const item of objects) fs.writeFileSync(path.join(objectRoot, item.sha256), item.bytes);

  const payload = {
    format: 'arc.remote-content',
    formatVersion: 1,
    projectGuid,
    engineVersion: '0.1.0',
    baseBuildId: runtimeBuildId,
    cookBuildId: 'cooked-build-' + generation,
    target: {
      platform: target.startsWith('android') ? 'android' : 'windows',
      architecture: target.startsWith('android') ? 'arm64' : 'x86_64',
      renderer: 'vulkan',
      cookProfile: target,
      textureFamily: family,
    },
    generation,
    complete: true,
    cookManifestName: 'startup.arccookmanifest',
    cookManifest: { path: objects[0].path, sha256: objects[0].sha256, size: objects[0].size },
    chunks: [
      { name: 'scene.arcpak', path: objects[1].path, sha256: objects[1].sha256, size: objects[1].size },
    ],
  };
  const descriptor = Buffer.from(JSON.stringify(payload) + '\n');
  const descriptorPath = path.join(descriptorFolder, 'current.json');
  fs.writeFileSync(descriptorPath, descriptor);
  const release: LivePlayRemoteRelease = {
    root,
    projectGuid,
    runtimeBuildId,
    sessionId,
    instanceId,
    sceneGuid,
    sceneAssetGuid: '33333333-3333-3333-3333-333333333333',
    authoringSha256: 'a'.repeat(64),
    generation,
    variants: [{
      family,
      descriptorPath,
      descriptorSha256: digest(descriptor),
      cookedBuildId: payload.cookBuildId,
      chunkCount: 1,
    }],
  };
  return { release, target, family, payload, descriptor, objects, objectRoot };
}

async function request(server: EditorCookedContentServer, url: string, token?: string, init: RequestInit = {}) {
  const origin = server.snapshot().origin;
  if (!origin) throw new Error('C9 test server is not listening');
  return fetch(origin + url, {
    ...init,
    headers: {
      ...(token ? { Authorization: 'Bearer ' + token } : {}),
      ...(init.headers ?? {}),
    },
  });
}

describe('C9.4a editor-owned cooked HTTP server', () => {
  it('binds only to loopback and serves no content until a verified private publication is promoted', async () => {
    const f = fixture();
    await f.server.start();
    const token = f.server.issueClientToken();
    expect(f.server.snapshot()).toMatchObject({
      state: 'listening', projectGuid, sessionId, profiles: [],
    });
    expect(f.server.snapshot().origin).toMatch(/^http:\/\/127\.0\.0\.1:\d+$/u);
    expect((await request(f.server, '/releases/' + projectGuid + '/windows-x64-vulkan/current.json', token)).status)
      .toBe(404);
    const published = releaseFixture(f.sessionRoot);
    await f.server.publish(published.release, published.target);
    const response = await request(f.server, '/releases/' + projectGuid + '/' + published.target + '/current.json', token);
    expect(response.status).toBe(200);
    expect(response.headers.get('cache-control')).toContain('no-store');
    expect(response.headers.get('etag')).toBe('"' + digest(published.descriptor) + '"');
    expect(await response.text()).toBe(published.descriptor.toString('utf8'));
    expect(f.server.snapshot().profiles).toEqual([{
      profile: published.target, family: published.family, generation: 1,
    }]);
    expect(f.server.snapshot().bytesSent).toBeGreaterThan(0);
  });

  it('rejects unauthenticated access, unknown projects and non-cooked route or traversal', async () => {
    const f = fixture();
    await f.server.start();
    const published = releaseFixture(f.sessionRoot);
    await f.server.publish(published.release, published.target);
    const releaseUrl = '/releases/' + projectGuid + '/' + published.target + '/current.json';
    expect((await request(f.server, releaseUrl)).status).toBe(401);
    expect((await request(f.server, releaseUrl, 'a'.repeat(64))).status).toBe(401);
    const token = f.server.issueClientToken();
    expect((await request(f.server, releaseUrl, 'b'.repeat(64))).status).toBe(401);
    expect((await request(f.server, releaseUrl, token)).status).toBe(200);
    expect((await request(f.server, '/releases/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/' +
      published.target + '/current.json', token)).status).toBe(404);
    expect((await request(f.server, '/Content/main.arcscene', token)).status).toBe(404);
    expect((await request(f.server, '/objects/sha256/000000', token)).status).toBe(404);
    expect((await request(f.server, '/objects%2fsha256%2f' + published.objects[0].sha256, token)).status).toBe(400);
    expect((await request(f.server, '/objects/sha256/' + published.objects[0].sha256 + '?path=../', token)).status)
      .toBe(400);
    expect((await request(f.server, releaseUrl, token, { method: 'POST' })).status).toBe(405);
    expect(f.server.snapshot().rejectedRequests).toBeGreaterThan(3);
  });

  it('returns exact object bytes with strict HTTP 206 range, ETag and conditional guards', async () => {
    const f = fixture();
    await f.server.start();
    const published = releaseFixture(f.sessionRoot);
    await f.server.publish(published.release, published.target);
    const token = f.server.issueClientToken();
    const item = published.objects[1];
    const url = '/objects/sha256/' + item.sha256;
    const full = await request(f.server, url, token);
    expect(full.status).toBe(200);
    expect(Buffer.from(await full.arrayBuffer())).toEqual(item.bytes);
    expect(full.headers.get('etag')).toBe('"' + item.sha256 + '"');

    const partial = await request(f.server, url, token, {
      headers: { Range: 'bytes=2-9', 'If-Match': '"' + item.sha256 + '"' },
    });
    expect(partial.status).toBe(206);
    expect(partial.headers.get('content-range')).toBe('bytes 2-9/' + item.size);
    expect(partial.headers.get('content-length')).toBe('8');
    expect(Buffer.from(await partial.arrayBuffer())).toEqual(item.bytes.subarray(2, 10));
    expect((await request(f.server, url, token, { headers: { Range: 'bytes=999-1000' } })).status).toBe(416);
    expect((await request(f.server, url, token, { headers: { Range: 'bytes=-3' } })).status).toBe(400);
    expect((await request(f.server, url, token, { headers: { Range: 'bytes=0-3,5-7' } })).status).toBe(400);
    expect((await request(f.server, url, token, { headers: { 'If-Match': '"other"' } })).status).toBe(412);
  });

  it('atomically rejects malformed, missing, modified or cross-project release without losing last-good', async () => {
    const f = fixture();
    await f.server.start();
    const token = f.server.issueClientToken();
    const good = releaseFixture(f.sessionRoot);
    await f.server.publish(good.release, good.target);
    const address = '/releases/' + projectGuid + '/' + good.target + '/current.json';
    const newer = releaseFixture(f.sessionRoot, { name: 'build-b', generation: 2 });
    const changed = { ...newer.release, projectGuid: 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa' };
    await expect(f.server.publish(changed, newer.target)).rejects.toThrow(/different project/);
    fs.writeFileSync(path.join(newer.objectRoot, newer.objects[1].sha256), 'bad bytes');
    await expect(f.server.publish(newer.release, newer.target)).rejects.toThrow(/size\/hash/);
    expect((await request(f.server, address, token)).status).toBe(200);
    expect((await request(f.server, address, token)).headers.get('etag'))
      .toBe('"' + digest(good.descriptor) + '"');
    await expect(f.server.publish(good.release, good.target)).rejects.toThrow(/monotonically/);
  });

  it('rejects symlink escape and modified published object before serving', async () => {
    const f = fixture();
    await f.server.start();
    const token = f.server.issueClientToken();
    const published = releaseFixture(f.sessionRoot);
    await f.server.publish(published.release, published.target);
    const target = published.objects[1];
    const file = path.join(published.objectRoot, target.sha256);
    fs.writeFileSync(file, Buffer.alloc(target.size));
    expect((await request(f.server, '/objects/sha256/' + target.sha256, token)).status).toBe(503);
    if (process.platform !== 'win32') {
      const later = releaseFixture(f.sessionRoot, { name: 'build-later', generation: 2 });
      const outside = path.join(f.projectRoot, 'source.bin');
      fs.writeFileSync(outside, later.objects[1].bytes);
      fs.unlinkSync(path.join(later.objectRoot, later.objects[1].sha256));
      fs.symlinkSync(outside, path.join(later.objectRoot, later.objects[1].sha256));
      await expect(f.server.publish(later.release, later.target)).rejects.toThrow(/symlink/);
    }
  });

  it('serves separate target catalogs to concurrent clients and releases the port on stop', async () => {
    const f = fixture();
    await f.server.start();
    const token = f.server.issueClientToken();
    const windows = releaseFixture(f.sessionRoot, { name: 'windows', profile: 'windows-x64-vulkan' });
    const android = releaseFixture(f.sessionRoot, {
      name: 'android', profile: 'android-arm64-vulkan', family: 'astc',
    });
    await f.server.publish(windows.release, windows.target);
    await f.server.publish(android.release, android.target);
    const paths = [windows, android].map((item) =>
      '/releases/' + projectGuid + '/' + item.target + '/current.json');
    const results = await Promise.all(paths.map((url) => request(f.server, url, token)));
    expect(results.map((item) => item.status)).toEqual([200, 200]);
    expect((await Promise.all(results.map((item) => item.json()))) as Array<{ target: { cookProfile: string } }>)
      .toMatchObject([{ target: { cookProfile: windows.target } }, { target: { cookProfile: android.target } }]);
    const origin = f.server.snapshot().origin!;
    const port = Number(new URL(origin).port);
    await f.server.stop();
    expect(f.server.snapshot()).toMatchObject({ state: 'stopped', origin: null, profiles: [] });
    expect(() => f.server.issueClientToken()).toThrow(/not active/);
    const second = new EditorCookedContentServer({ ...f.options, port });
    servers.push(second);
    await second.start();
    expect(second.snapshot().origin).toBe(origin);
    expect((await request(second, paths[0], second.issueClientToken())).status).toBe(404);
  });
});
