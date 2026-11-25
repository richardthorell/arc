import { describe, expect, it, vi } from 'vitest';

import type { AndroidTargetReference } from '../common/androidDeviceTypes';
import {
  AndroidLivePlayPairingSession,
  type AndroidAdbSecretWriter,
  type AndroidLivePlayPairingInput,
} from './androidLivePlayPairing';
import type { AdbCommandExecutor } from './adbClient';

const target: AndroidTargetReference = { sdkRoot: '/android-sdk', id: 'device:phone' };
const input: AndroidLivePlayPairingInput = {
  target,
  applicationId: 'org.arcengine.sample',
  origin: 'http://127.0.0.1:43210',
  projectGuid: '11111111-2222-4333-8444-555555555555',
  buildId: 'android-build-1',
  targetProfile: 'android-arm64-vulkan',
  textureFamily: 'astc',
  sessionId: 'session-1',
  instanceId: 'runtime-1',
  descriptorSha256: 'a'.repeat(64),
  generation: 1,
  token: 'b'.repeat(64),
  isCurrent: () => true,
};

function fixture() {
  const state = { serial: 'phone', ready: true, current: true, writerFails: false, reverseFails: false };
  const resolveTarget = vi.fn(async (_reference: AndroidTargetReference) => {
    if (!state.ready) throw new Error('Selected Android target unavailable');
    return { sdkRoot: target.sdkRoot, serial: state.serial };
  });
  const commands: string[][] = [];
  const execute = vi.fn<AdbCommandExecutor>(async (_file, args) => {
    commands.push(args);
    if (state.reverseFails && args.includes('--no-rebind')) throw new Error('existing reverse rule');
    return { stdout: '', stderr: '' };
  });
  const payloads: string[] = [];
  const write = vi.fn<AndroidAdbSecretWriter>(async (_file, args, bytes) => {
    expect(args.join(' ')).not.toContain(input.token);
    expect(args.join(' ')).not.toContain(input.descriptorSha256);
    payloads.push(Buffer.from(bytes).toString('utf8'));
    if (state.writerFails) throw new Error('private stage refused');
  });
  const session = new AndroidLivePlayPairingSession({ resolveTarget }, execute, write, () => 1_750_000_000_000);
  return { state, session, resolveTarget, execute, write, commands, payloads };
}

describe('C9.4c2 Android debug-device pairing transport', () => {
  it('pins ADB reverse to the selected serial and stages a bounded secret by private stdin only', async () => {
    const f = fixture();
    const info = await f.session.start(input);
    expect(info).toEqual({
      serial: 'phone',
      applicationId: 'org.arcengine.sample',
      origin: input.origin,
      expiresAtEpochMs: 1_750_000_090_000,
    });
    expect(f.resolveTarget).toHaveBeenCalledTimes(3);
    expect(f.commands[0]).toEqual(['-s', 'phone', 'reverse', '--no-rebind', 'tcp:43210', 'tcp:43210']);
    const staged = JSON.parse(f.payloads[0]) as Record<string, unknown>;
    expect(staged).toMatchObject({
      format: 'arc.android-live-play-pairing',
      version: 1,
      projectGuid: input.projectGuid,
      buildId: input.buildId,
      targetProfile: input.targetProfile,
      textureFamily: 'astc',
      token: input.token,
      generation: 1,
    });
    expect(staged.expiresAtEpochMs).toBe(info.expiresAtEpochMs);
    expect(f.write).toHaveBeenCalledTimes(1);
    const [, args, buffer] = f.write.mock.calls[0];
    expect(args).toContain('run-as');
    expect(args).toContain(input.applicationId);
    expect(args.join(' ')).not.toContain(input.token);
    expect(args.join(' ')).not.toContain(input.origin);
    expect((buffer as Buffer).every((byte: number) => byte === 0)).toBe(true);
    await f.session.stop();
    expect(f.commands).toContainEqual(['-s', 'phone', 'reverse', '--remove', 'tcp:43210']);
    expect(f.commands).not.toContainEqual(['reverse', '--remove-all']);
    expect(f.commands.find((args) => args.includes('rm'))).toContain('files/.arc-live-play-pairing.json');
    await f.session.stop();
    expect(f.commands.filter((args) => args.includes('--remove'))).toHaveLength(1);
  });

  it('never places a credential in ADB argv, URL query, environment or stderr', async () => {
    const f = fixture();
    await f.session.start(input);
    const commandText = JSON.stringify([...f.commands, ...f.write.mock.calls.map((call) => call[1])]);
    expect(commandText).not.toContain(input.token);
    expect(commandText).not.toContain(input.descriptorSha256);
    expect(f.write.mock.calls[0][1]).toEqual(
      expect.arrayContaining(['shell', '-T', 'run-as', 'org.arcengine.sample', 'sh', '-c']),
    );
    await f.session.stop();
  });

  it('refuses existing reverse rules without removing another session mapping', async () => {
    const f = fixture();
    f.state.reverseFails = true;
    await expect(f.session.start(input)).rejects.toThrow('existing reverse rule');
    expect(f.write).not.toHaveBeenCalled();
    expect(f.commands.filter((args) => args.includes('--remove'))).toHaveLength(0);
    await f.session.stop();
  });

  it('removes only the owned reverse rule when private staging fails', async () => {
    const f = fixture();
    f.state.writerFails = true;
    await expect(f.session.start(input)).rejects.toThrow('private stage refused');
    expect(f.commands.at(-1)).toEqual(['-s', 'phone', 'reverse', '--remove', 'tcp:43210']);
    expect(f.commands.some((args) => args.includes('rm'))).toBe(true);
    await f.session.stop();
    expect(f.commands.filter((args) => args.includes('--remove'))).toHaveLength(1);
  });

  it('does not silently rebind when a selected AVD changes serial during launch', async () => {
    const f = fixture();
    f.resolveTarget.mockImplementationOnce(async () => ({ sdkRoot: target.sdkRoot, serial: 'emulator-5554' }));
    f.resolveTarget.mockImplementationOnce(async () => ({ sdkRoot: target.sdkRoot, serial: 'emulator-5556' }));
    await expect(f.session.start(input)).rejects.toThrow('changed');
    expect(f.write).not.toHaveBeenCalled();
    expect(f.commands.at(-1)).toEqual(['-s', 'emulator-5554', 'reverse', '--remove', 'tcp:43210']);
  });

  it('fails closed when project/session changes after reverse and before writing the secret', async () => {
    const f = fixture();
    let current = true;
    f.resolveTarget.mockImplementationOnce(async () => {
      current = false;
      return { sdkRoot: target.sdkRoot, serial: 'phone' };
    });
    await expect(f.session.start({ ...input, isCurrent: () => current })).rejects.toThrow('stale');
    expect(f.write).not.toHaveBeenCalled();
    // No reverse was installed because project changed during discovery.
    expect(f.commands).toHaveLength(0);
  });

  it('rejects untrusted ports, profiles, packages, tokens and stale projects without touching ADB', async () => {
    const invalid: Array<Partial<AndroidLivePlayPairingInput>> = [
      { origin: 'http://192.168.1.10:43210' },
      { origin: 'https://127.0.0.1:43210' },
      { origin: 'http://127.0.0.1:0' },
      { origin: 'http://127.0.0.1:65536' },
      { origin: 'http://127.0.0.1:43210/secret' },
      { applicationId: 'org.arcengine.test;rm -rf /' },
      { applicationId: 'invalid' },
      { projectGuid: 'not-guid' },
      { textureFamily: '../astc' },
      { token: 'b'.repeat(63) + '\n' },
      { token: 'B'.repeat(64) },
      { descriptorSha256: 'broken' },
      { sessionId: 'bad\nsession' },
      { generation: 0 },
      { isCurrent: () => false },
    ];
    for (const changes of invalid) {
      const f = fixture();
      await expect(f.session.start({ ...input, ...changes })).rejects.toThrow('Untrusted Android');
      expect(f.execute).not.toHaveBeenCalled();
      expect(f.write).not.toHaveBeenCalled();
    }
  });

  it('rejects repeated starts; Stop while a writer is in flight waits for cleanup', async () => {
    const f = fixture();
    let finish!: () => void;
    const gate = new Promise<void>((resolve) => {
      finish = resolve;
    });
    f.write.mockImplementation(async () => gate);
    const pending = f.session.start(input);
    expect(() => f.session.start(input)).toThrow('already started');
    await vi.waitFor(() => expect(f.write).toHaveBeenCalledTimes(1));
    const stopping = f.session.stop();
    finish();
    await expect(pending).rejects.toThrow('stale');
    await stopping;
    expect(f.commands.at(-1)).toEqual(['-s', 'phone', 'reverse', '--remove', 'tcp:43210']);
    expect(() => f.session.start(input)).toThrow('stopped');
  });
});
