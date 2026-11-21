import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import type { LivePlayMessage } from '../common/livePlayProtocol';
import type { WindowsLivePlaySnapshot } from '../common/livePlaySessionTypes';
import type { NativeLivePlayLaunchContext } from './livePlayNativeHandoff';
import { writeLivePlayNativeHandoff } from './livePlayNativeHandoff';
import { LivePlayControlServer, type LivePlayControlStatus } from './livePlayControlServer';
import { LivePlayBaselineOutcomeGate } from './livePlayBaselineOutcomeGate';

/** Trusted main-process input, never a renderer-supplied executable or argv. */
export type WindowsLivePlayLaunch = {
  executable: string;
  cwd: string;
  context: NativeLivePlayLaunchContext;
  /** Main-process-only non-secret arguments (empty for normal game launches). */
  args?: readonly string[];
};

type Session = {
  epoch: number;
  server: LivePlayControlServer;
  process: ChildProcessWithoutNullStreams | null;
  resolveConnected: (() => void) | null;
  rejectConnected: ((error: Error) => void) | null;
  connectedTimer: ReturnType<typeof setTimeout> | null;
  stopRequested: boolean;
  baselineOutcome: LivePlayBaselineOutcomeGate;
};

const connectionTimeoutMs = 15_000;

export class WindowsLivePlaySession {
  private current: Session | null = null;
  private epoch = 0;
  private snapshotValue: WindowsLivePlaySnapshot = {
    state: 'stopped',
    sessionId: null,
    instanceId: null,
    pid: null,
    lastReceivedSequence: 0,
    error: null,
  };

  constructor(
    private readonly publish: (snapshot: WindowsLivePlaySnapshot) => void = () => {},
    private readonly onMessage: (message: LivePlayMessage) => void = () => {},
    private readonly platform: NodeJS.Platform = process.platform,
  ) {}

  snapshot(): WindowsLivePlaySnapshot {
    return { ...this.snapshotValue };
  }

  /**
   * L1.3b2: await a native activation decision for an already-verified cooked
   * C9 release. Do not call with the L1.3a authored .arcscene snapshot.
   * The C9 producer is deliberately not wired until #1137/#995 are available.
   */
  async awaitBaselineStatus(
    payload: Extract<LivePlayMessage, { type: 'scene.baseline.offer' }>['payload'],
    timeoutMs?: number,
  ): Promise<Extract<LivePlayMessage, { type: 'scene.baseline.status' }>['payload']> {
    const session = this.current;
    if (!session || this.snapshotValue.state !== 'connected' || session.stopRequested)
      throw new Error('Live Play is not connected to a Development runtime');
    return session.baselineOutcome.waitForStatus(() => session.server.sendBaselineOffer(payload), timeoutMs);
  }

  private update(change: Partial<WindowsLivePlaySnapshot>): void {
    this.snapshotValue = { ...this.snapshotValue, ...change };
    try {
      this.publish(this.snapshot());
    } catch {
      // A closed or reloaded renderer cannot interrupt the native process owner.
    }
  }

  async start(input: WindowsLivePlayLaunch): Promise<WindowsLivePlaySnapshot> {
    if (this.platform !== 'win32') throw new Error('Windows standalone Play is only supported on Windows');
    if (this.current) throw new Error('A Windows Live Play session is already active');
    const executable = fs.realpathSync(input.executable);
    const cwd = fs.realpathSync(input.cwd);
    if (!fs.statSync(executable).isFile() || path.extname(input.executable).toLowerCase() !== '.exe')
      throw new Error('The Development game executable is invalid');
    if (!fs.statSync(cwd).isDirectory()) throw new Error('The Development game directory is invalid');

    const epoch = ++this.epoch;
    const context = input.context;
    const server = new LivePlayControlServer({
      authority: context.authority,
      schemaHash: context.schemaHash,
      grantedCapabilities: context.capabilities,
      onMessage: (message) => {
        if (this.current?.epoch !== epoch) return;
        if (message.type === 'scene.baseline.status') this.current.baselineOutcome.receive(message.payload);
        this.onMessage(message);
      },
      onStatus: (status) => this.onServerStatus(epoch, status),
    });
    const session: Session = {
      epoch,
      server,
      process: null,
      resolveConnected: null,
      rejectConnected: null,
      connectedTimer: null,
      stopRequested: false,
      baselineOutcome: new LivePlayBaselineOutcomeGate(),
    };
    this.current = session;
    this.update({
      state: 'starting',
      sessionId: context.authority.sessionId,
      instanceId: context.authority.instanceId,
      pid: null,
      lastReceivedSequence: 0,
      error: null,
    });
    try {
      await server.start();
      if (this.current !== session) throw new Error('Live Play was cancelled before launch');

      // stdin is a private inherited anonymous pipe on Windows
      // (STARTF_USESTDHANDLES). No password-bearing argv, env, path, or log.
      const child = spawn(executable, [...(input.args ?? [])], {
        cwd,
        shell: false,
        windowsHide: false,
        detached: false,
        stdio: ['pipe', 'pipe', 'pipe'],
      });
      session.process = child;
      this.update({ pid: child.pid ?? null });
      // The child may exit before the secret handoff; never leave an
      // unhandled error from an early closed stdin pipe.
      child.stdin.on('error', () => {});
      child.stdout.on('data', () => {});
      child.stderr.on('data', () => {});
      child.once('error', (error) => this.onProcessExit(session, error.message));
      child.once('close', (code, signal) =>
        this.onProcessExit(
          session,
          `Game exited before/during Live Play (code ${String(code)}, signal ${String(signal)})`,
        ),
      );

      const attached = new Promise<void>((resolve, reject) => {
        session.resolveConnected = resolve;
        session.rejectConnected = reject;
        session.connectedTimer = setTimeout(
          () => reject(new Error('Development game did not connect to the editor')),
          connectionTimeoutMs,
        );
      });
      // Attach the rejection handler before the one-shot pipe write: a child
      // that dies mid-handoff may reject its connection promise immediately.
      void attached.catch(() => {});
      await writeLivePlayNativeHandoff(child.stdin, server, context);
      child.stdin.end();
      await attached;
      if (session.stopRequested || this.current !== session) throw new Error('Live Play was stopped during connection');
      return this.snapshot();
    } catch (error) {
      await this.stop();
      throw error;
    } finally {
      if (session.connectedTimer) clearTimeout(session.connectedTimer);
      session.connectedTimer = null;
      session.resolveConnected = null;
      session.rejectConnected = null;
    }
  }

  private onServerStatus(epoch: number, status: LivePlayControlStatus): void {
    const session = this.current;
    if (!session || session.epoch !== epoch || session.stopRequested) return;
    if (status.state === 'connected') {
      this.update({ state: 'connected', lastReceivedSequence: status.lastReceivedSequence, error: null });
      session.resolveConnected?.();
    } else if (status.state === 'disconnected' || status.state === 'error') {
      session.baselineOutcome.cancel('Live Play runtime disconnected before scene activation');
      this.update({
        state: status.state === 'error' ? 'failed' : 'disconnected',
        error: status.lastError ?? 'Development runtime disconnected',
        lastReceivedSequence: status.lastReceivedSequence,
      });
      session.rejectConnected?.(new Error(status.lastError ?? 'Development runtime disconnected'));
    } else if (status.state === 'waiting' && this.snapshotValue.state !== 'starting') {
      this.update({ state: 'starting' });
    } else if (this.snapshotValue.state === 'connected') {
      this.update({ lastReceivedSequence: status.lastReceivedSequence });
    }
  }

  private onProcessExit(session: Session, message: string): void {
    if (this.current !== session || session.stopRequested) return;
    session.baselineOutcome.cancel('Live Play runtime exited before scene activation');
    this.update({ state: 'failed', pid: null, error: message });
    session.rejectConnected?.(new Error(message));
    void session.server.stop();
    this.current = null;
  }

  async stop(): Promise<WindowsLivePlaySnapshot> {
    const session = this.current;
    if (!session) {
      this.update({ state: 'stopped', pid: null, sessionId: null, instanceId: null, error: null });
      return this.snapshot();
    }
    this.current = null;
    session.stopRequested = true;
    session.baselineOutcome.cancel('Live Play stopped before scene activation');
    if (session.connectedTimer) clearTimeout(session.connectedTimer);
    session.rejectConnected?.(new Error('Live Play stopped'));
    this.update({ state: 'stopping' });
    const child = session.process;
    if (child && child.exitCode === null && child.signalCode === null) {
      child.kill();
      await new Promise<void>((resolve) => {
        const timeout = setTimeout(resolve, 2500);
        child.once('close', () => {
          clearTimeout(timeout);
          resolve();
        });
      });
      if (child.exitCode === null && child.signalCode === null) child.kill('SIGKILL');
    }
    await session.server.stop();
    this.update({
      state: 'stopped',
      pid: null,
      sessionId: null,
      instanceId: null,
      lastReceivedSequence: 0,
      error: null,
    });
    return this.snapshot();
  }
}

/** Make only the project-selected published package an eligible launch source. */
export function resolvePublishedWindowsLivePlay(
  projectRoot: string,
  packageOutput: string,
  projectGuid: string,
  profile: string,
  targetId: string,
): { executable: string; cwd: string; buildId: string } {
  const root = fs.realpathSync(projectRoot);
  const relative = path.relative(root, path.resolve(packageOutput));
  if (!relative || relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative))
    throw new Error('The selected package output is outside the project');
  const recordRoot = fs.realpathSync(packageOutput);
  const publication = JSON.parse(fs.readFileSync(path.join(recordRoot, 'current-workflow.json'), 'utf8')) as {
    projectGuid?: string;
    artifacts?: Array<{ path?: string; buildId?: string; profile?: string; targetId?: string; kind?: string }>;
  };
  if (publication.projectGuid !== projectGuid) throw new Error('Published build belongs to a different project');
  const artifact = publication.artifacts?.find(
    (item) => item.kind === 'package' && item.profile === profile && item.targetId === targetId,
  );
  if (!artifact?.path || !artifact.buildId)
    throw new Error('Build the selected Windows profile before standalone Play');
  const output = fs.realpathSync(artifact.path);
  const local = path.relative(recordRoot, output);
  if (!local || local === '..' || local.startsWith(`..${path.sep}`) || path.isAbsolute(local))
    throw new Error('Published package escaped its project output');
  const manifest = JSON.parse(fs.readFileSync(path.join(output, 'arc-distribution.json'), 'utf8')) as {
    projectGuid?: string;
    target?: { id?: string; cookProfile?: string; configuration?: string };
    entrypoint?: { kind?: string; value?: string };
  };
  if (
    manifest.projectGuid !== projectGuid ||
    manifest.target?.id !== targetId ||
    manifest.target?.cookProfile !== profile ||
    manifest.target?.configuration === 'Shipping' ||
    manifest.entrypoint?.kind !== 'executable' ||
    !manifest.entrypoint.value
  )
    throw new Error('Published package is incompatible with Development standalone Play');
  const executable = fs.realpathSync(path.join(output, manifest.entrypoint.value));
  const entry = path.relative(output, executable);
  if (!entry || entry === '..' || entry.startsWith(`..${path.sep}`) || path.isAbsolute(entry))
    throw new Error('Published entrypoint escaped the package');
  return { executable, cwd: path.dirname(executable), buildId: artifact.buildId };
}
