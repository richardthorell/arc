import { createHash } from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

/**
 * A native scene.autosave is a copy of the live Authoring World serialized with
 * the regular ARC scene format. It does not mark the document saved, change its
 * path or advance authoring history. L1.3b will consume this staged baseline;
 * it is not a cooked C9 release or a runtime launch argument.
 */
export type LivePlaySceneBaselineIdentity = {
  projectRoot: string;
  projectGuid: string;
  buildId: string;
  targetProfile: string;
  sessionId: string;
  instanceId: string;
};

export type LivePlaySceneBaseline = Readonly<{
  projectGuid: string;
  buildId: string;
  targetProfile: string;
  sessionId: string;
  instanceId: string;
  sceneGuid: string;
  sceneName: string;
  sceneRevision: number;
  historyRevision: number;
  entityCount: number;
  sha256: string;
  byteLength: number;
  /** Editor-main-process-only path; never send to the renderer or runtime argv. */
  scenePath: string;
  directory: string;
}>;

type CaptureResponse = {
  succeeded: boolean;
  error?: string;
  payload?: unknown;
  sceneRevision?: number;
};

type CaptureHost = {
  command(type: string, payload: Record<string, unknown>): Promise<CaptureResponse>;
};

type PendingCapture = { sessionId: string; directory: string; cancelled: boolean };

export const MAX_LIVE_PLAY_SCENE_BYTES = 16 * 1024 * 1024;

/** Windows read-only attributes must be cleared before recursive deletion. */
function discardStagedDirectory(directory: string): void {
  for (const filename of ['initial-scene.arcscene', 'baseline.json']) {
    try {
      fs.chmodSync(path.join(directory, filename), 0o600);
    } catch {
      // A partial/cancelled capture may not have produced this file.
    }
  }
  fs.rmSync(directory, { recursive: true, force: true });
}

export class LivePlaySceneBaselineStore {
  private active: LivePlaySceneBaseline | null = null;
  private pending: PendingCapture | null = null;

  constructor(
    private readonly host: CaptureHost,
    private readonly temporaryRoot: string = os.tmpdir(),
  ) {}

  current(): LivePlaySceneBaseline | null {
    return this.active;
  }

  async prepare(identity: LivePlaySceneBaselineIdentity): Promise<LivePlaySceneBaseline> {
    if (this.active || this.pending) throw new Error('A Live Play scene baseline is already active or being captured');
    if (Object.values(identity).some((value) => typeof value !== 'string' || !value.trim()))
      throw new Error('Live Play baseline requires a complete trusted session identity');

    const projectRoot = fs.realpathSync(identity.projectRoot);
    const directory = fs.mkdtempSync(path.join(this.temporaryRoot, 'arc-live-play-baseline-'));
    fs.chmodSync(directory, 0o700);
    const relative = path.relative(projectRoot, directory);
    if (!relative || (relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative))) {
      discardStagedDirectory(directory);
      throw new Error('Live Play baseline storage must be outside the authoring project');
    }

    const pending: PendingCapture = { sessionId: identity.sessionId, directory, cancelled: false };
    this.pending = pending;
    const scenePath = path.join(directory, 'initial-scene.arcscene');
    try {
      // The native host synchronously copies the Authoring World at command
      // execution time, then writes a sealed .arcscene using the normal serializer.
      const response = await this.host.command('scene.autosave', { path: scenePath });
      if (pending.cancelled || this.pending !== pending) throw new Error('Live Play baseline capture was cancelled');
      if (!response.succeeded) throw new Error(response.error || 'Native host rejected Live Play scene capture');

      const info = fs.lstatSync(scenePath);
      if (!info.isFile() || info.isSymbolicLink() || info.size === 0 || info.size > MAX_LIVE_PLAY_SCENE_BYTES)
        throw new Error('Live Play scene capture is missing, invalid, or exceeds the 16 MiB session limit');
      const bytes = fs.readFileSync(scenePath);
      const document = JSON.parse(bytes.toString('utf8')) as {
        format?: unknown;
        formatVersion?: unknown;
        scene?: { id?: unknown; name?: unknown };
        entities?: unknown;
        dependencies?: unknown;
      };
      const capture = response.payload as { sceneGuid?: unknown; historyRevision?: unknown } | null;
      if (
        document.format !== 'arc.scene' ||
        !Number.isSafeInteger(document.formatVersion) ||
        (document.formatVersion as number) < 1 ||
        !Array.isArray(document.entities) ||
        !Array.isArray(document.dependencies) ||
        typeof document.scene?.id !== 'string' ||
        !/^(?:[a-f\d]{32}|[a-f\d]{8}(?:-[a-f\d]{4}){3}-[a-f\d]{12})$/iu.test(document.scene.id) ||
        document.scene.id !== capture?.sceneGuid
      )
        throw new Error('Native host returned an incompatible or mismatched ARC scene baseline');

      const sceneRevision = response.sceneRevision;
      const historyRevision = capture?.historyRevision;
      if (
        !Number.isSafeInteger(sceneRevision) ||
        (sceneRevision ?? -1) < 0 ||
        !Number.isSafeInteger(historyRevision) ||
        (typeof historyRevision === 'number' && historyRevision < 0)
      )
        throw new Error('Native scene baseline has no valid authoring revision');

      const result: LivePlaySceneBaseline = Object.freeze({
        projectGuid: identity.projectGuid,
        buildId: identity.buildId,
        targetProfile: identity.targetProfile,
        sessionId: identity.sessionId,
        instanceId: identity.instanceId,
        sceneGuid: document.scene.id,
        sceneName: typeof document.scene.name === 'string' ? document.scene.name : '',
        sceneRevision: sceneRevision as number,
        historyRevision: historyRevision as number,
        entityCount: document.entities.length,
        sha256: createHash('sha256').update(bytes).digest('hex'),
        byteLength: bytes.byteLength,
        scenePath,
        directory,
      });

      // Immutable, session-private authoring input. This is not a C9 publish.
      fs.chmodSync(scenePath, 0o400);
      fs.writeFileSync(
        path.join(directory, 'baseline.json'),
        JSON.stringify({
          format: 'arc.live-play.scene-baseline',
          formatVersion: 1,
          ...Object.fromEntries(
            Object.entries(result).filter(([key]) => key !== 'scenePath' && key !== 'directory'),
          ),
        }),
        { encoding: 'utf8', flag: 'wx', mode: 0o400 },
      );
      if (pending.cancelled || this.pending !== pending) throw new Error('Live Play baseline capture was cancelled');
      this.active = result;
      return result;
    } finally {
      if (this.pending === pending) this.pending = null;
      if (this.active?.directory !== directory) discardStagedDirectory(directory);
    }
  }

  /** Cancellation is safe during native capture; its finalizer reaps late output. */
  clear(sessionId?: string): void {
    if (this.pending && (!sessionId || this.pending.sessionId === sessionId)) this.pending.cancelled = true;
    if (this.active && (!sessionId || this.active.sessionId === sessionId)) {
      const directory = this.active.directory;
      this.active = null;
      discardStagedDirectory(directory);
    }
  }
}
