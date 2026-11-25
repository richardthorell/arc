import { spawn } from 'node:child_process';
import path from 'node:path';

import type { AndroidTargetReference } from '../common/androidDeviceTypes';
import { AdbClient, executeAdbCommand, type AdbCommandExecutor } from './adbClient';
import type { AndroidTargetService } from './androidTargetService';

const applicationIdPattern = /^[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+$/u;
const guidPattern = /^[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}$/u;
const tokenPattern = /^[a-f0-9]{64}$/u;
const profilePattern = /^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/u;
const familyPattern = /^[a-z0-9_-]{1,32}$/u;
const sessionPattern = /^[A-Za-z0-9_-]{1,128}$/u;
const pairingFile = '.arc-live-play-pairing.json';
const pairingLifetimeMs = 90_000;

export type AndroidLivePlayPairingInput = Readonly<{
  target: AndroidTargetReference;
  applicationId: string;
  origin: string;
  projectGuid: string;
  buildId: string;
  targetProfile: string;
  textureFamily: string;
  sessionId: string;
  instanceId: string;
  descriptorSha256: string;
  generation: number;
  token: string;
  /** Trusted main-process authority, not renderer-supplied. */
  isCurrent: () => boolean;
}>;

export type AndroidLivePlayPairingSnapshot = Readonly<{
  serial: string;
  applicationId: string;
  origin: string;
  expiresAtEpochMs: number;
}>;

/** ADB stdin (not shell args/env) is the only place where session bytes flow. */
export type AndroidAdbSecretWriter = (
  executable: string,
  args: readonly string[],
  payload: Buffer,
  timeoutMs: number,
) => Promise<void>;

export const writePrivateAndroidPairing: AndroidAdbSecretWriter = (executable, args, payload, timeoutMs) =>
  new Promise((resolve, reject) => {
    const child = spawn(executable, [...args], {
      shell: false,
      windowsHide: true,
      stdio: ['pipe', 'ignore', 'ignore'],
    });
    let failed = false;
    const timer = setTimeout(() => {
      failed = true;
      child.kill();
    }, timeoutMs);
    timer.unref?.();
    child.once('error', () => {
      failed = true;
    });
    // Always await actual child exit before the caller deletes private device
    // files or reverses. A timed-out ADB command must not race the cleanup.
    child.once('close', (code) => {
      clearTimeout(timer);
      if (failed || code !== 0) reject(new Error('ADB could not stage the private Android Live Play credential'));
      else resolve();
    });
    child.stdin.once('error', () => {
      failed = true;
      child.kill();
    });
    child.stdin.end(payload);
  });

/**
 * C9.4c2: main-process only, one selected ready ADB device and one bounded
 * private credential. C9.7 owns APK install/launch; never accept a serial,
 * binary path or secret from renderer IPC.
 */
export class AndroidLivePlayPairingSession {
  private pending: Promise<AndroidLivePlayPairingSnapshot> | null = null;
  private attached: { adb: AdbClient; serial: string; port: number; applicationId: string } | null = null;
  private stopped = false;

  constructor(
    private readonly targets: Pick<AndroidTargetService, 'resolveTarget'>,
    private readonly execute: AdbCommandExecutor = executeAdbCommand,
    private readonly write: AndroidAdbSecretWriter = writePrivateAndroidPairing,
    private readonly now: () => number = Date.now,
  ) {}

  start(input: AndroidLivePlayPairingInput): Promise<AndroidLivePlayPairingSnapshot> {
    if (this.stopped || this.pending) throw new Error('Android content pairing session already started or stopped');
    this.pending = this.prepare(input);
    return this.pending;
  }

  private checkCurrent(input: AndroidLivePlayPairingInput): void {
    if (this.stopped || !input.isCurrent())
      throw new Error('Android Live Play pairing belongs to a stopped or stale session');
  }

  private async prepare(input: AndroidLivePlayPairingInput): Promise<AndroidLivePlayPairingSnapshot> {
    const match = /^http:\/\/127\.0\.0\.1:([1-9][0-9]{0,4})$/u.exec(input.origin);
    const port = Number(match?.[1]);
    if (
      !match ||
      !Number.isInteger(port) ||
      port > 65535 ||
      !applicationIdPattern.test(input.applicationId) ||
      input.applicationId.length > 200 ||
      !guidPattern.test(input.projectGuid) ||
      !sessionPattern.test(input.buildId) ||
      !sessionPattern.test(input.sessionId) ||
      !sessionPattern.test(input.instanceId) ||
      !profilePattern.test(input.targetProfile) ||
      !familyPattern.test(input.textureFamily) ||
      !tokenPattern.test(input.token) ||
      !tokenPattern.test(input.descriptorSha256) ||
      !Number.isSafeInteger(input.generation) ||
      input.generation < 1 ||
      !input.isCurrent()
    )
      throw new Error('Untrusted Android Live Play pairing authority');
    this.checkCurrent(input);

    const selected = await this.targets.resolveTarget(input.target);
    this.checkCurrent(input);
    const adbPath = path.join(selected.sdkRoot, 'platform-tools', process.platform === 'win32' ? 'adb.exe' : 'adb');
    if (
      selected.sdkRoot !== input.target.sdkRoot ||
      !selected.serial ||
      selected.serial.length > 512 ||
      /\s/u.test(selected.serial)
    )
      throw new Error('Android Live Play requires one fresh selected ADB target');
    const adb = new AdbClient(adbPath, this.execute);
    const peer = { adb, serial: selected.serial, port, applicationId: input.applicationId };
    const address = 'tcp:' + port;

    // Never replace an existing reverse rule: it may belong to another editor
    // session. The device's 127.0.0.1 port remains device-local, not LAN.
    await adb.executeForDevice(peer.serial, ['reverse', '--no-rebind', address, address], 8000);
    this.attached = peer;
    try {
      this.checkCurrent(input);
      const refreshed = await this.targets.resolveTarget(input.target);
      if (refreshed.serial !== selected.serial || refreshed.sdkRoot !== selected.sdkRoot)
        throw new Error('Selected Android device changed during content pairing');
      this.checkCurrent(input);
      const expiresAtEpochMs = this.now() + pairingLifetimeMs;
      const payload = Buffer.from(
        JSON.stringify({
          format: 'arc.android-live-play-pairing',
          version: 1,
          projectGuid: input.projectGuid,
          buildId: input.buildId,
          targetProfile: input.targetProfile,
          textureFamily: input.textureFamily,
          sessionId: input.sessionId,
          instanceId: input.instanceId,
          descriptorSha256: input.descriptorSha256,
          generation: input.generation,
          origin: input.origin,
          token: input.token,
          expiresAtEpochMs,
        }),
        'utf8',
      );
      if (payload.length > 2048) {
        payload.fill(0);
        throw new Error('Android pairing record exceeds the bounded native handoff');
      }
      // Fixed literals and a validated package are the ONLY ADB argv. ADB
      // transports secret bytes over stdin into the debuggable app's private
      // files directory; no Intent extras, logcat, URL or shell argv secrets.
      const script =
        'umask 077; mkdir -p files && cat > files/' +
        pairingFile +
        '.tmp && mv -f files/' +
        pairingFile +
        '.tmp files/' +
        pairingFile;
      try {
        await this.write(
          adbPath,
          ['-s', peer.serial, 'shell', '-T', 'run-as', peer.applicationId, 'sh', '-c', script],
          payload,
          10000,
        );
      } finally {
        payload.fill(0);
      }
      this.checkCurrent(input);
      const finalTarget = await this.targets.resolveTarget(input.target);
      if (finalTarget.serial !== selected.serial || finalTarget.sdkRoot !== selected.sdkRoot)
        throw new Error('Selected Android device changed after private handoff');
      this.checkCurrent(input);
      return Object.freeze({
        serial: peer.serial,
        applicationId: peer.applicationId,
        origin: input.origin,
        expiresAtEpochMs,
      });
    } catch (error) {
      await this.cleanup();
      throw error;
    }
  }

  private async cleanup(): Promise<void> {
    const peer = this.attached;
    this.attached = null;
    if (!peer) return;
    // Cleanup is scoped to the serial and rule owned by this instance; never
    // execute global "adb reverse --remove-all" or affect other devices.
    await peer.adb
      .executeForDevice(
        peer.serial,
        [
          'shell',
          '-T',
          'run-as',
          peer.applicationId,
          'rm',
          '-f',
          'files/' + pairingFile,
          'files/' + pairingFile + '.tmp',
        ],
        5000,
      )
      .catch(() => undefined);
    await peer.adb
      .executeForDevice(peer.serial, ['reverse', '--remove', 'tcp:' + peer.port], 5000)
      .catch(() => undefined);
  }

  async stop(): Promise<void> {
    this.stopped = true;
    try {
      await this.pending;
    } catch {
      // Failure cleanup is handled in prepare; Stop is idempotent.
    }
    await this.cleanup();
  }
}
