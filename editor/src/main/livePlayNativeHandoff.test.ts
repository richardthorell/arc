import { PassThrough } from 'node:stream';
import { afterEach, describe, expect, it } from 'vitest';
import { LivePlayControlServer } from './livePlayControlServer';
import {
  encodeLivePlayNativeHandoff,
  writeLivePlayNativeHandoff,
  type NativeLivePlayLaunchContext,
} from './livePlayNativeHandoff';

const context: NativeLivePlayLaunchContext = {
  authority: {
    projectGuid: '11111111-1111-1111-1111-111111111111',
    buildId: 'dev-build-42',
    targetProfile: 'windows-x64-vulkan',
    sessionId: 'session-1',
    instanceId: 'runtime-1',
  },
  schemaHash: 'a'.repeat(64),
  capabilities: ['session.observe', 'content.observe'],
};

const servers: LivePlayControlServer[] = [];
afterEach(async () => {
  for (const server of servers.splice(0)) await server.stop();
});

async function start(): Promise<LivePlayControlServer> {
  const server = new LivePlayControlServer({
    authority: context.authority,
    schemaHash: context.schemaHash,
    grantedCapabilities: context.capabilities,
  });
  servers.push(server);
  await server.start();
  return server;
}

describe('Native Development runtime protected inherited-pipe handoff', () => {
  it('serializes the native contract as bounded length-prefixed JSON with a unique launch secret', async () => {
    const server = await start();
    const before = server.launchHandoff();
    const frame = encodeLivePlayNativeHandoff(server, context);
    const length = frame.readUInt32BE(0);
    expect(length).toBe(frame.byteLength - 4);
    expect(length).toBeLessThanOrEqual(4096);
    const decoded = JSON.parse(frame.subarray(4).toString('utf8')) as Record<string, unknown>;
    expect(decoded).toMatchObject({
      port: before.port,
      projectGuid: context.authority.projectGuid,
      sessionId: context.authority.sessionId,
      instanceId: context.authority.instanceId,
      schemaHash: context.schemaHash,
      capabilities: ['session.observe', 'content.observe'],
    });
    expect(decoded.secret).toBe(before.secret.toString('hex'));
    expect(before.secret).toHaveLength(32);
    frame.fill(0);
    before.secret.fill(0);
  });

  it('writes through the protected pipe and erases the temporary outgoing frame', async () => {
    const server = await start();
    const writer = new PassThrough();
    const pieces: Buffer[] = [];
    writer.on('data', (data: Buffer) => pieces.push(Buffer.from(data)));
    await writeLivePlayNativeHandoff(writer, server, context);
    const data = Buffer.concat(pieces);
    expect(data.readUInt32BE(0)).toBe(data.length - 4);
    expect(JSON.parse(data.subarray(4).toString('utf8'))).toMatchObject({
      sessionId: context.authority.sessionId,
      buildId: context.authority.buildId,
    });
    writer.destroy();
  });

  it('binds the exact cooked release and short-lived pairing token only to the protected pipe', async () => {
    const server = await start();
    const content = {
      origin: 'http://127.0.0.1:31415',
      token: 'f'.repeat(64),
      sceneGuid: '22222222-2222-2222-2222-222222222222',
      sceneAssetGuid: '33333333-3333-3333-3333-333333333333',
      authoringSha256: 'd'.repeat(64),
      authoringRevision: 12,
      descriptorSha256: 'e'.repeat(64),
      generation: 1,
      textureFamily: 'bc',
    };
    const frame = encodeLivePlayNativeHandoff(server, { ...context, content });
    const data = JSON.parse(frame.subarray(4).toString('utf8')) as Record<string, unknown>;
    expect(data.content).toEqual(content);
    expect(frame.length).toBeLessThanOrEqual(4100);
    frame.fill(0);
    const bad = (override: Partial<typeof content>) =>
      encodeLivePlayNativeHandoff(server, { ...context, content: { ...content, ...override } });
    expect(() => bad({ origin: 'https://example.com' })).toThrow(/C9 content pairing/);
    expect(() => bad({ origin: 'http://127.0.0.1:31415/path' })).toThrow(/C9 content pairing/);
    expect(() => bad({ token: 'missing' })).toThrow(/C9 content pairing/);
    expect(() => bad({ sceneAssetGuid: '../../invalid' })).toThrow(/C9 content pairing/);
    expect(() => bad({ generation: 0 })).toThrow(/C9 content pairing/);
    expect(() => bad({ descriptorSha256: 'x'.repeat(64) })).toThrow(/C9 content pairing/);
  });

  it('rejects invalid launch identity before writing any credential bytes', async () => {
    const server = await start();
    const bad = {
      ...context,
      authority: { ...context.authority, projectGuid: '../../invalid' },
    };
    expect(() => encodeLivePlayNativeHandoff(server, bad)).toThrow(/Invalid Live Play native launch context/);
  });
});
