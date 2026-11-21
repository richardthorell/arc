import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { NativeLivePlayLaunchContext } from './livePlayNativeHandoff';
import { WindowsLivePlaySession, resolvePublishedWindowsLivePlay } from './windowsLivePlaySession';

const projectGuid = '11111111-1111-1111-1111-111111111111';
const targetId = 'windows-x64-vulkan';
const profile = 'windows-default';
const folders: string[] = [];
const sessions: WindowsLivePlaySession[] = [];

afterEach(async () => {
  for (const session of sessions.splice(0)) await session.stop();
  for (const folder of folders.splice(0)) fs.rmSync(folder, { recursive: true, force: true });
});

function projectFixture(): { projectRoot: string; packageOutput: string; executable: string } {
  const projectRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-windows-live-play-'));
  folders.push(projectRoot);
  const packageOutput = path.join(projectRoot, 'Saved', 'Packages', 'Windows');
  const output = path.join(packageOutput, 'generation-1', 'output');
  fs.mkdirSync(output, { recursive: true });
  const executable = path.join(output, 'ExampleGame.exe');
  fs.writeFileSync(executable, 'fixture binary');
  fs.writeFileSync(
    path.join(output, 'arc-distribution.json'),
    JSON.stringify({
      projectGuid,
      target: { id: targetId, cookProfile: profile, configuration: 'Debug' },
      entrypoint: { kind: 'executable', value: 'ExampleGame.exe' },
    }),
  );
  fs.writeFileSync(
    path.join(packageOutput, 'current-workflow.json'),
    JSON.stringify({
      projectGuid,
      artifacts: [
        {
          kind: 'package',
          path: output,
          buildId: 'fixture-build-42',
          profile,
          targetId,
        },
      ],
    }),
  );
  return { projectRoot, packageOutput, executable };
}

function sessionContext(): NativeLivePlayLaunchContext {
  return {
    authority: {
      projectGuid,
      buildId: 'fixture-build-42',
      targetProfile: profile,
      sessionId: 'session-123',
      instanceId: 'runtime-456',
    },
    schemaHash: 'a'.repeat(64),
    capabilities: ['session.observe', 'content.observe'],
  };
}

describe('Windows Live Play published artifact authority', () => {
  it('resolves only the exact project-published Windows profile and entrypoint', () => {
    const fixture = projectFixture();
    expect(
      resolvePublishedWindowsLivePlay(fixture.projectRoot, fixture.packageOutput, projectGuid, profile, targetId),
    ).toEqual({
      executable: fs.realpathSync(fixture.executable),
      cwd: path.dirname(fs.realpathSync(fixture.executable)),
      buildId: 'fixture-build-42',
    });
    expect(() =>
      resolvePublishedWindowsLivePlay(
        fixture.projectRoot,
        fixture.packageOutput,
        projectGuid,
        'other-profile',
        targetId,
      ),
    ).toThrow(/Build the selected Windows profile/);
    expect(() =>
      resolvePublishedWindowsLivePlay(
        fixture.projectRoot,
        fixture.packageOutput,
        '33333333-3333-3333-3333-333333333333',
        profile,
        targetId,
      ),
    ).toThrow(/different project/);
  });

  it('rejects Shipping and entrypoints that escape the published package', () => {
    const fixture = projectFixture();
    const output = path.dirname(fixture.executable);
    const manifestPath = path.join(output, 'arc-distribution.json');
    const base = JSON.parse(fs.readFileSync(manifestPath, 'utf8')) as {
      target: { configuration: string };
      entrypoint: { kind: string; value: string };
    };
    base.target.configuration = 'Shipping';
    fs.writeFileSync(manifestPath, JSON.stringify(base));
    expect(() =>
      resolvePublishedWindowsLivePlay(fixture.projectRoot, fixture.packageOutput, projectGuid, profile, targetId),
    ).toThrow(/incompatible/);

    base.target.configuration = 'Debug';
    base.entrypoint.value = '../../../../outside.exe';
    fs.writeFileSync(manifestPath, JSON.stringify(base));
    expect(() =>
      resolvePublishedWindowsLivePlay(fixture.projectRoot, fixture.packageOutput, projectGuid, profile, targetId),
    ).toThrow();
  });
});

// A real child process exercises WindowsLivePlaySession's protected inherited
// stdio channel, actual localhost TCP handshake, state notification and Stop.
// The native C++ client has its own cross-language TCP acceptance in CTest.
it.skipIf(process.platform === 'win32')(
  'attaches to a child process without secrets in argv and cleans up Stop',
  async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-live-play-child-'));
    folders.push(root);
    const executable = path.join(root, 'ExampleGame.exe');
    fs.symlinkSync(process.execPath, executable);
    const script = path.join(root, 'runtime.cjs');
    fs.writeFileSync(
      script,
      `
const net = require('node:net');
let chunks = [];
process.stdin.on('data', (part) => chunks.push(part));
process.stdin.on('end', () => {
  const buffer = Buffer.concat(chunks);
  const size = buffer.readUInt32BE(0);
  const launch = JSON.parse(buffer.subarray(4, 4 + size).toString('utf8'));
  const socket = net.createConnection(launch.port, '127.0.0.1');
  const encode = (payload) => {
    const header = Buffer.alloc(4);
    header.writeUInt32BE(payload.length, 0);
    socket.write(Buffer.concat([header, payload]));
  };
  const message = (type, payload, sequence) => ({
    protocol: 'arc.live-play', protocolVersion: 1, sender: 'runtime',
    projectGuid: launch.projectGuid, buildId: launch.buildId, targetProfile: launch.targetProfile,
    sessionId: launch.sessionId, instanceId: launch.instanceId,
    messageId: 'runtime-' + sequence, sequence, type, payload
  });
  socket.on('connect', () => {
    encode(Buffer.from(launch.secret, 'hex'));
    encode(Buffer.from(JSON.stringify(message('hello', {
      schemaHash: launch.schemaHash, lastReceivedSequence: 0, capabilities: launch.capabilities
    }, 1))));
  });
  socket.on('data', () => {
    encode(Buffer.from(JSON.stringify(message('session.state', {
      state: 'running', tick: 1, revision: 1
    }, 2))));
  });
  socket.on('error', () => process.exit(4));
  setInterval(() => {}, 1000);
});
`,
    );
    const states: string[] = [];
    const seen: string[] = [];
    const session = new WindowsLivePlaySession(
      (snapshot) => states.push(snapshot.state),
      (message) => seen.push(message.type),
      'win32',
    );
    sessions.push(session);
    const snapshot = await session.start({ executable, cwd: root, context: sessionContext(), args: [script] });
    expect(snapshot.state).toBe('connected');
    await vi.waitFor(() => expect(seen).toContain('session.state'), { timeout: 3000 });
    expect(states).toContain('connected');
    expect(snapshot.pid).toBeTypeOf('number');
    await session.stop();
    expect(session.snapshot()).toMatchObject({ state: 'stopped', pid: null, sessionId: null });
    expect(states.at(-1)).toBe('stopped');
  },
);

it('fails closed on missing or non-Windows host executable', async () => {
  const session = new WindowsLivePlaySession();
  sessions.push(session);
  await expect(
    session.start({ executable: '/tmp/nonexistent.exe', cwd: '/tmp', context: sessionContext() }),
  ).rejects.toThrow(/Windows standalone Play is only supported on Windows/);
});
