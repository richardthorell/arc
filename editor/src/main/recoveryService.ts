import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

import type { ArcProjectCandidate } from '../common/projectTypes';
import type { RecoveryGeneration, RecoverySnapshot } from '../common/editorWorkflowTypes';

type RecoveryHostResponse = {
  succeeded: boolean;
  error?: string;
  payload?: unknown;
  sceneRevision?: number;
};

type RecoveryHost = {
  query(type: string, payload?: Record<string, unknown>): Promise<RecoveryHostResponse>;
  command(type: string, payload?: Record<string, unknown>): Promise<RecoveryHostResponse>;
};

type SceneSnapshot = {
  dirty?: boolean;
  sceneGuid?: string;
  sceneName?: string;
  activeScenePath?: string;
  historyRevision?: number;
};

const idleDelayMs = 5_000;
const minimumAutosaveIntervalMs = 120_000;
const heartbeatIntervalMs = 5_000;
const maximumGenerations = 20;
const maximumProjectBytes = 2 * 1024 * 1024 * 1024;

const readJson = <T>(filePath: string, fallback: T): T => {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8')) as T;
  } catch {
    return fallback;
  }
};

const writeJsonAtomic = (filePath: string, value: unknown): void => {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const temporary = `${filePath}.tmp-${process.pid}-${Date.now()}`;
  fs.writeFileSync(temporary, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
  fs.renameSync(temporary, filePath);
};

export class RecoveryService {
  private readonly root: string;
  private readonly host: RecoveryHost;
  private project: ArcProjectCandidate | null = null;
  private timer: NodeJS.Timeout | null = null;
  private lastMutationAt = 0;
  private lastAutosaveAt = 0;
  private saving = false;
  private error = '';
  private lifecycleGeneration = 0;
  private readonly knownProjectRoots = new Map<string, string>();

  constructor(
    root: string,
    host: RecoveryHost,
    private readonly settings: () => Record<string, unknown> = () => ({}),
  ) {
    this.root = root;
    this.host = host;
  }

  start(project: ArcProjectCandidate): void {
    this.stop(true);
    this.project = project;
    ++this.lifecycleGeneration;
    this.lastMutationAt = Date.now();
    this.lastAutosaveAt = 0;
    const projectRoot = this.projectRoot();
    fs.mkdirSync(projectRoot, { recursive: true });
    fs.writeFileSync(this.heartbeatPath(), new Date().toISOString(), 'utf8');
    this.timer = setInterval(() => void this.tick(), heartbeatIntervalMs);
  }

  noteMutation(): void {
    this.lastMutationAt = Date.now();
  }

  stop(clean: boolean): void {
    ++this.lifecycleGeneration;
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    if (this.project) {
      if (clean) {
        fs.writeFileSync(this.cleanMarkerPath(), new Date().toISOString(), 'utf8');
      }
      this.project = null;
    }
  }

  snapshot(projectGuid?: string, projectPath?: string): RecoverySnapshot {
    const guid = projectGuid || this.project?.descriptor.guid || '';
    if (!guid)
      return {
        projectGuid: '',
        uncleanShutdown: false,
        heartbeatAt: '',
        generations: [],
        totalBytes: 0,
        error: this.error,
      };
    const projectRoot =
      this.project?.descriptor.guid === guid
        ? this.projectRoot()
        : projectPath
          ? path.join(projectPath, 'Saved', 'Recovery', guid)
          : this.knownProjectRoots.get(guid) ?? path.join(this.root, guid);
    this.knownProjectRoots.set(guid, projectRoot);
    const index = readJson<RecoveryGeneration[]>(path.join(projectRoot, 'index.json'), []);
    const generations = index.filter((entry) => fs.existsSync(entry.recoveryPath));
    const heartbeatAt = readJsonTimestamp(path.join(projectRoot, 'heartbeat'));
    const cleanAt = readJsonTimestamp(path.join(projectRoot, 'clean'));
    return {
      projectGuid: guid,
      uncleanShutdown: Boolean(heartbeatAt && (!cleanAt || Date.parse(heartbeatAt) > Date.parse(cleanAt))),
      heartbeatAt,
      generations,
      totalBytes: generations.reduce((sum, entry) => sum + entry.size, 0),
      error: this.error,
    };
  }

  async restore(id: string): Promise<RecoveryHostResponse> {
    const generation = this.findGeneration(id);
    if (!generation) return { succeeded: false, error: 'Recovery generation no longer exists' };
    return this.host.command('scene.openRecovery', {
      path: generation.recoveryPath,
      originalPath: generation.originalPath,
    });
  }

  discard(id: string): boolean {
    const generation = this.findGeneration(id);
    if (!generation) return false;
    try {
      fs.rmSync(generation.recoveryPath, { force: true });
      const snapshot = this.snapshot(generation.projectGuid);
      this.writeIndex(
        snapshot.generations.filter((entry) => entry.id !== id),
        generation.projectGuid,
      );
      return true;
    } catch (error) {
      this.error = error instanceof Error ? error.message : String(error);
      return false;
    }
  }

  private async tick(): Promise<void> {
    const project = this.project;
    const lifecycleGeneration = this.lifecycleGeneration;
    if (!project) return;
    const projectRoot = this.projectRoot();
    fs.writeFileSync(path.join(projectRoot, 'heartbeat'), new Date().toISOString(), 'utf8');
    const now = Date.now();
    const settings = this.settings();
    if (settings['editor.autosave.enabled'] === false) return;
    const idleDelay = this.finiteSetting(settings, 'editor.autosave.idleSeconds', idleDelayMs / 1000, 1, 300) * 1000;
    const minimumInterval =
      this.finiteSetting(
        settings,
        'editor.autosave.minimumIntervalSeconds',
        minimumAutosaveIntervalMs / 1000,
        10,
        3600,
      ) * 1000;
    if (
      this.saving ||
      now - this.lastMutationAt < idleDelay ||
      (this.lastAutosaveAt && now - this.lastAutosaveAt < minimumInterval)
    )
      return;
    this.saving = true;
    try {
      const response = await this.host.query('scene.hierarchy');
      const scene = response.payload as SceneSnapshot | undefined;
      if (
        lifecycleGeneration !== this.lifecycleGeneration ||
        this.project !== project ||
        !response.succeeded ||
        !scene?.dirty ||
        !scene.sceneGuid
      )
        return;
      const documentRoot = path.join(projectRoot, scene.sceneGuid);
      fs.mkdirSync(documentRoot, { recursive: true });
      const nowIso = new Date().toISOString();
      const id = `${Date.now()}-${crypto.randomBytes(4).toString('hex')}`;
      const recoveryPath = path.join(documentRoot, `${id}.arcscene`);
      const saved = await this.host.command('scene.autosave', { path: recoveryPath });
      if (lifecycleGeneration !== this.lifecycleGeneration || this.project !== project) {
        fs.rmSync(recoveryPath, { force: true });
        return;
      }
      if (!saved.succeeded) throw new Error(saved.error || 'Native host rejected the recovery save');
      const size = fs.statSync(recoveryPath).size;
      const generation: RecoveryGeneration = {
        id,
        projectGuid: project.descriptor.guid,
        documentGuid: scene.sceneGuid,
        documentName: scene.sceneName || 'Untitled',
        originalPath: scene.activeScenePath || '',
        recoveryPath,
        createdAt: nowIso,
        historyRevision: scene.historyRevision || 0,
        sceneRevision: saved.sceneRevision || 0,
        size,
      };
      this.lastAutosaveAt = Date.now();
      this.prune([generation, ...this.snapshot().generations]);
      this.error = '';
    } catch (error) {
      this.error = error instanceof Error ? error.message : String(error);
    } finally {
      this.saving = false;
    }
  }

  private prune(entries: RecoveryGeneration[]): void {
    const settings = this.settings();
    const generationLimit = Math.trunc(
      this.finiteSetting(settings, 'editor.recovery.generations', maximumGenerations, 1, 100),
    );
    const projectBudget = this.finiteSetting(
      settings,
      'editor.recovery.projectBudgetBytes',
      maximumProjectBytes,
      64 * 1024 * 1024,
      64 * 1024 * 1024 * 1024,
    );
    const retained: RecoveryGeneration[] = [];
    let bytes = 0;
    for (const entry of entries.sort((left, right) => right.createdAt.localeCompare(left.createdAt))) {
      if (retained.length < generationLimit && bytes + entry.size <= projectBudget) {
        retained.push(entry);
        bytes += entry.size;
      } else {
        fs.rmSync(entry.recoveryPath, { force: true });
      }
    }
    this.writeIndex(retained);
  }

  private writeIndex(entries: RecoveryGeneration[], projectGuid?: string): void {
    const root =
      !projectGuid || this.project?.descriptor.guid === projectGuid
        ? this.projectRoot()
        : this.knownProjectRoots.get(projectGuid) ?? path.join(this.root, projectGuid);
    writeJsonAtomic(path.join(root, 'index.json'), entries);
  }

  private findGeneration(id: string): RecoveryGeneration | undefined {
    if (this.project) {
      const current = this.snapshot().generations.find((entry) => entry.id === id);
      if (current) return current;
    }
    try {
      for (const entry of fs.readdirSync(this.root, { withFileTypes: true })) {
        if (!entry.isDirectory()) continue;
        const found = this.snapshot(entry.name).generations.find((generation) => generation.id === id);
        if (found) return found;
      }
    } catch {
      // A missing recovery root is the same as no matching generation.
    }
    return undefined;
  }

  private projectRoot(): string {
    if (!this.project) throw new Error('Recovery service has no active project');
    return path.join(this.project.projectRoot, this.project.descriptor.paths.saved, 'Recovery', this.project.descriptor.guid);
  }

  private heartbeatPath(): string {
    return path.join(this.projectRoot(), 'heartbeat');
  }

  private cleanMarkerPath(): string {
    return path.join(this.projectRoot(), 'clean');
  }

  private finiteSetting(
    settings: Record<string, unknown>,
    key: string,
    fallback: number,
    minimum: number,
    maximum: number,
  ): number {
    const value = settings[key];
    return typeof value === 'number' && Number.isFinite(value) && value >= minimum && value <= maximum
      ? value
      : fallback;
  }
}

const readJsonTimestamp = (filePath: string): string => {
  try {
    return fs.readFileSync(filePath, 'utf8').trim();
  } catch {
    return '';
  }
};
