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
    chunks: [{ name: 'scene.arcpak', path: objects[1].path, sha256: objects[1].sha256, size: objects[1].size }],
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
    variants: [
      {
        family,
        descriptorPath,
        descriptorSha256: digest(descriptor),
        cookedBuildId: payload.cookBuildId,
        chunkCount: 1,
      },
    ],
  };
  return { release, target, family, payload, descriptor, objects, objectRoot };
}

async function request(server: EditorCookedContentServer, url: string, token?: string, init: RequestInit = {}) {
  const origin = server.snapshot().origin;
  if (!origin) throw new Error('C9 test server is not listening');
  const headers = new Headers(init.headers);
  if (token) headers.set('Authorization', 'Bearer ' + token);
  return fetch(origin + url, { ...init, headers });
}

describe('C9.4a editor-owned cooked HTTP server', () => {
  it('binds only to loopback and serves no content until a verified private publication is promoted', async () => {
    const f = fixture();
    await f.server.start();
    const token = f.server.issueClientToken();
    expect(f.server.snapshot()).toMatchObject({
      state: 'listening',
      projectGuid,
      sessionId,
      profiles: [],
    });
    expect(f.server.snapshot().origin).toMatch(/^http:\/\/127\.0\.0\.1:\d+$/u);
    expect(
      (await request(f.server, '/releases/' + projectGuid + '/windows-x64-vulkan/current.json', token)).status,
    ).toBe(404);
    const published = releaseFixture(f.sessionRoot);
    await f.server.publish(published.release, published.target);
    const response = await request(
      f.server,
      '/releases/' + projectGuid + '/' + published.target + '/current.json',
      token,
    );
    expect(response.status).toBe(200);
    expect(response.headers.get('cache-control')).toContain('no-store');
    expect(response.headers.get('etag')).toBe('"' + digest(published.descriptor) + '"');
    expect(await response.text()).toBe(published.descriptor.toString('utf8'));
    expect(f.server.snapshot().profiles).toEqual([
      {
        profile: published.target,
        family: published.family,
        generation: 1,
      },
    ]);
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
    expect(
      (
        await request(
          f.server,
          '/releases/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/' + published.target + '/current.json',
          token,
        )
      ).status,
    ).toBe(404);
    expect((await request(f.server, '/Content/main.arcscene', token)).status).toBe(404);
    expect((await request(f.server, '/objects/sha256/000000', token)).status).toBe(404);
    expect((await request(f.server, '/objects%2fsha256%2f' + published.objects[0].sha256, token)).status).toBe(400);
    expect(
      (await request(f.server, '/objects/sha256/' + published.objects[0].sha256 + '?path=../', token)).status,
    ).toBe(400);
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
    expect((await request(f.server, address, token)).headers.get('etag')).toBe('"' + digest(good.descriptor) + '"');
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
      name: 'android',
      profile: 'android-arm64-vulkan',
      family: 'astc',
    });
    await f.server.publish(windows.release, windows.target);
    await f.server.publish(android.release, android.target);
    const paths = [windows, android].map((item) => '/releases/' + projectGuid + '/' + item.target + '/current.json');
    const results = await Promise.all(paths.map((url) => request(f.server, url, token)));
    expect(results.map((item) => item.status)).toEqual([200, 200]);
    expect(
      (await Promise.all(results.map((item) => item.json()))) as Array<{ target: { cookProfile: string } }>,
    ).toMatchObject([{ target: { cookProfile: windows.target } }, { target: { cookProfile: android.target } }]);
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

  it('serves an authenticated C9.1c replay of the initial verified SHA-addressed release', async () => {
    const f = fixture();
    await f.server.start();
    const token = f.server.issueClientToken();
    const url = '/sessions/' + sessionId + '/changes?after=0';
    expect((await request(f.server, url, token)).status).toBe(503);
    const first = releaseFixture(f.sessionRoot);
    await f.server.publish(first.release, first.target);
    expect((await request(f.server, url)).status).toBe(401);
    const response = await request(f.server, url, token);
    expect(response.status).toBe(200);
    expect(response.headers.get('cache-control')).toContain('no-store');
    const feed = (await response.json()) as {
      format: string;
      formatVersion: number;
      projectGuid: string;
      engineVersion: string;
      baseBuildId: string;
      target: { platform: string; architecture: string; renderer: string; cookProfile: string; textureFamily: string };
      afterSequence: number;
      oldestSequence: number;
      newestSequence: number;
      currentGeneration: number;
      currentRelease: { path: string; sha256: string; size: number };
      mode: string;
      events: Array<{
        sequence: number;
        generation: number;
        release: { path: string; sha256: string; size: number };
        changed: string[];
        removed: string[];
      }>;
    };
    expect(feed).toMatchObject({
      format: 'arc.remote-content-feed',
      formatVersion: 1,
      projectGuid,
      engineVersion: '0.1.0',
      baseBuildId: runtimeBuildId,
      target: first.payload.target,
      afterSequence: 0,
      oldestSequence: 1,
      newestSequence: 1,
      currentGeneration: 1,
      mode: 'replay',
    });
    expect(feed.events).toEqual([
      {
        sequence: 1,
        generation: 1,
        release: feed.currentRelease,
        changed: [first.release.sceneAssetGuid],
        removed: [],
      },
    ]);
    expect(feed.currentRelease).toEqual({
      path: 'objects/sha256/' + digest(first.descriptor),
      sha256: digest(first.descriptor),
      size: first.descriptor.length,
    });
    const immutable = await request(f.server, '/' + feed.currentRelease.path, token);
    expect(immutable.status).toBe(200);
    expect(Buffer.from(await immutable.arrayBuffer())).toEqual(first.descriptor);
    const head = await request(f.server, '/sessions/' + sessionId + '/changes?after=1', token);
    expect((await head.json()) as { mode: string; events: unknown[] }).toMatchObject({
      mode: 'up-to-date',
      events: [],
    });
    expect(f.server.snapshot().changeFeeds).toEqual([
      { profile: first.target, oldestSequence: 1, newestSequence: 1 },
    ]);
  });

  it('increments the per-target journal only after verified generation commit and replays changed/removed GUIDs', async () => {
    const f = fixture();
    await f.server.start();
    const token = f.server.issueClientToken();
    const first = releaseFixture(f.sessionRoot);
    const second = releaseFixture(f.sessionRoot, { name: 'second', generation: 2 });
    const removedGuid = 'aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee';
    await f.server.publish(first.release, first.target);
    await expect(f.server.publish(second.release, second.target)).rejects.toThrow(/explicit changed\/removed/);
    await expect(
      f.server.publish(second.release, second.target, second.family, {
        changed: [first.release.sceneAssetGuid],
        removed: [first.release.sceneAssetGuid],
      }),
    ).rejects.toThrow(/duplicate or overlapping/);
    await f.server.publish(second.release, second.target, second.family, {
      changed: [first.release.sceneAssetGuid],
      removed: [removedGuid],
    });
    const url = '/sessions/' + sessionId + '/changes?after=1';
    const feed = (await (await request(f.server, url, token)).json()) as {
      mode: string;
      currentGeneration: number;
      newestSequence: number;
      events: Array<{ sequence: number; changed: string[]; removed: string[]; release: { sha256: string } }>;
    };
    expect(feed.mode).toBe('replay');
    expect(feed.currentGeneration).toBe(2);
    expect(feed.newestSequence).toBe(2);
    expect(feed.events).toEqual([
      {
        sequence: 2,
        generation: 2,
        release: {
          path: 'objects/sha256/' + digest(second.descriptor),
          sha256: digest(second.descriptor),
          size: second.descriptor.length,
        },
        changed: [first.release.sceneAssetGuid],
        removed: [removedGuid],
      },
    ]);
    // The prior descriptor must remain independently fetchable, even though
    // the mutable current pointer has advanced to the second generation.
    const old = await request(f.server, '/objects/sha256/' + digest(first.descriptor), token);
    expect(Buffer.from(await old.arrayBuffer())).toEqual(first.descriptor);
    expect(
      (await (await request(f.server, '/sessions/' + sessionId + '/changes?after=2', token)).json()) as {
        mode: string;
        events: unknown[];
      },
    ).toMatchObject({ mode: 'up-to-date', events: [] });
    expect((await request(f.server, '/sessions/' + sessionId + '/changes?after=3', token)).status).toBe(409);
  });

  it('trims replay history and requires a complete immutable snapshot for disconnected clients', async () => {
    const f = fixture();
    servers.pop();
    await f.server.stop();
    const server = new EditorCookedContentServer({ ...f.options, journalLimit: 2 });
    servers.push(server);
    await server.start();
    const token = server.issueClientToken();
    const released = [];
    for (let generation = 1; generation <= 4; ++generation) {
      const next = releaseFixture(f.sessionRoot, { name: 'gen-' + generation, generation });
      await server.publish(
        next.release,
        next.target,
        next.family,
        { changed: [next.release.sceneAssetGuid], removed: [] },
      );
      released.push(next);
    }
    const prefix = '/sessions/' + sessionId + '/changes?after=';
    const reset = (await (await request(server, prefix + '0', token)).json()) as {
      mode: string;
      currentRelease: { sha256: string };
      oldestSequence: number;
      newestSequence: number;
      events: unknown[];
    };
    expect(reset).toMatchObject({
      mode: 'full-snapshot',
      oldestSequence: 3,
      newestSequence: 4,
      events: [],
    });
    expect(reset.currentRelease.sha256).toBe(digest(released[3].descriptor));
    const snapshot = await request(server, '/objects/sha256/' + reset.currentRelease.sha256, token);
    expect(Buffer.from(await snapshot.arrayBuffer())).toEqual(released[3].descriptor);
    const replay = (await (await request(server, prefix + '2', token)).json()) as {
      mode: string;
      events: Array<{ sequence: number; generation: number }>;
    };
    expect(replay.mode).toBe('replay');
    expect(replay.events.map((event) => [event.sequence, event.generation])).toEqual([
      [3, 3],
      [4, 4],
    ]);
    const upToDate = (await (await request(server, prefix + '4', token)).json()) as { mode: string };
    expect(upToDate.mode).toBe('up-to-date');
  });

  it('isolates separate Windows/Android feed cursors and requires target selection when ambiguous', async () => {
    const f = fixture();
    await f.server.start();
    const token = f.server.issueClientToken();
    const windows = releaseFixture(f.sessionRoot, { name: 'windows-feed' });
    const android = releaseFixture(f.sessionRoot, {
      name: 'android-feed',
      profile: 'android-arm64-vulkan',
      family: 'astc',
    });
    await f.server.publish(windows.release, windows.target);
    await f.server.publish(android.release, android.target);
    const feed = '/sessions/' + sessionId + '/changes?after=0';
    expect((await request(f.server, feed, token)).status).toBe(400);
    for (const item of [windows, android]) {
      const result = await request(f.server, feed + '&profile=' + item.target, token);
      expect(result.status).toBe(200);
      expect((await result.json()) as unknown).toMatchObject({
        target: item.payload.target,
        newestSequence: 1,
        mode: 'replay',
      });
    }
    expect((await request(f.server, feed + '&profile=nonexistent', token)).status).toBe(404);
    expect((await request(f.server, '/sessions/other/changes?after=0', token)).status).toBe(404);
    expect((await request(f.server, feed + '&profile=android-arm64-vulkan')).status).toBe(401);
  });

  it('rejects malformed cursors and unsafe queries, preserving previous journal on failed publish', async () => {
    const f = fixture();
    await f.server.start();
    const token = f.server.issueClientToken();
    const first = releaseFixture(f.sessionRoot);
    await f.server.publish(first.release, first.target);
    for (const invalid of [
      'after=',
      'after=-1',
      'after=01',
      'after=1.0',
      'after=9007199254740992',
      'after=1&extra=2',
      'profile=windows-x64-vulkan&after=0',
      'after=0&profile=../windows',
      'after=0&profile=windows-x64-vulkan&profile=android-arm64-vulkan',
    ]) {
      expect(
        (await request(f.server, '/sessions/' + sessionId + '/changes?' + invalid, token)).status,
      ).toBe(400);
    }
    expect(
      (
        await request(
          f.server,
          '/releases/' + projectGuid + '/' + first.target + '/current.json?after=0',
          token,
        )
      ).status,
    ).toBe(400);
    const newer = releaseFixture(f.sessionRoot, { name: 'broken-feed', generation: 2 });
    fs.writeFileSync(path.join(newer.objectRoot, newer.objects[1].sha256), 'bad');
    await expect(
      f.server.publish(newer.release, newer.target, newer.family, {
        changed: [first.release.sceneAssetGuid],
        removed: [],
      }),
    ).rejects.toThrow(/size\/hash/);
    const unchanged = (await (await request(f.server, '/sessions/' + sessionId + '/changes?after=1', token)).json()) as {
      newestSequence: number;
      currentGeneration: number;
      mode: string;
    };
    expect(unchanged).toMatchObject({ newestSequence: 1, currentGeneration: 1, mode: 'up-to-date' });
    await f.server.stop();
    expect(f.server.snapshot().changeFeeds).toEqual([]);
  });

});
