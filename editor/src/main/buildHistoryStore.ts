import fs from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';

import type {
  BuildHistoryChunk,
  BuildHistoryDetail,
  BuildHistoryPage,
  BuildRawLogChunk,
} from '../common/buildHistoryTypes';
import type { BuildMonitorEvent, BuildMonitorRun } from '../common/buildMonitorTypes';
import type { ArcProjectCandidate } from '../common/projectTypes';

const HISTORY_VERSION = 1;
const MAX_RUNS = 100;
const MAX_PAGE_SIZE = 100;
const PAGE_BYTES = 256 * 1024;
const CHUNK_CHARS = 12 * 1024;
const safeId = (value: string): boolean => /^[a-zA-Z0-9_-]{1,100}$/u.test(value);
const terminal = (state: string): boolean => ['succeeded', 'failed', 'cancelled', 'interrupted'].includes(state);

function contained(root: string, candidate: string): boolean {
  const relative = path.relative(root, candidate);
  return relative !== '' && relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
}

/** Never allow an existing saved-path ancestor or history file to be a symlink/junction. */
function safeDirectory(root: string, target: string): void {
  if (!contained(root, target)) throw new Error('Build history directory escapes the project');
  const relative = path.relative(root, target);
  let current = root;
  for (const part of relative.split(path.sep)) {
    current = path.join(current, part);
    if (fs.existsSync(current)) {
      const info = fs.lstatSync(current);
      if (info.isSymbolicLink() || !info.isDirectory()) throw new Error('Unsafe build history directory');
    } else {
      fs.mkdirSync(current);
    }
  }
}

function safeFile(filename: string): void {
  try {
    const info = fs.lstatSync(filename);
    if (info.isSymbolicLink() || !info.isFile()) throw new Error('Unsafe build history file');
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
  }
}

/** Read JSONL in bounded chunks, with byte offsets so a large log never enters renderer memory. */
function readLines<T>(filename: string, requestedCursor: number, limit: number): BuildHistoryChunk<T> {
  const empty = { items: [] as T[], nextCursor: requestedCursor, hasMore: false };
  if (!fs.existsSync(filename)) return empty;
  safeFile(filename);
  const size = fs.statSync(filename).size;
  if (!Number.isSafeInteger(requestedCursor) || requestedCursor < 0 || requestedCursor > size)
    throw new Error('Invalid build history cursor');
  if (requestedCursor === size) return empty;
  const count = Math.min(PAGE_BYTES, size - requestedCursor);
  const buffer = Buffer.allocUnsafe(count);
  const fd = fs.openSync(filename, 'r');
  let bytes = 0;
  try {
    bytes = fs.readSync(fd, buffer, 0, count, requestedCursor);
  } finally {
    fs.closeSync(fd);
  }
  const items: T[] = [];
  let offset = 0;
  while (items.length < limit) {
    const end = buffer.indexOf(10, offset);
    if (end < 0 || end >= bytes) break;
    const line = buffer.subarray(offset, end).toString('utf8').trim();
    offset = end + 1;
    if (line) items.push(JSON.parse(line) as T);
  }
  if (offset === 0 && bytes > 0) throw new Error('Build history record exceeds read limit');
  return {
    items,
    nextCursor: requestedCursor + offset,
    hasMore: requestedCursor + offset < size,
  };
}

/**
 * Project-local, source-control-ignored build history.
 *
 * JSON index + append-only JSONL files avoid native SQLite dependencies in
 * Electron's main process. Build execution must continue if history I/O fails.
 */
export class BuildHistoryStore {
  private root: string | null = null;
  private projectGuid: string | null = null;
  private identity = '';
  private runs: BuildMonitorRun[] = [];
  private storageError?: string;

  /** Returns true when the project identity/path changed. */
  activate(project: ArcProjectCandidate | null): boolean {
    const identity = project
      ? JSON.stringify([project.descriptor.guid, project.projectRoot, project.descriptor.paths.saved])
      : '';
    if (this.identity === identity) return false;
    this.identity = identity;
    this.root = null;
    this.runs = [];
    this.projectGuid = project?.descriptor.guid ?? null;
    this.storageError = undefined;
    if (!project) return true;
    try {
      const projectRoot = fs.realpathSync(project.projectRoot);
      const saved = path.resolve(projectRoot, project.descriptor.paths.saved);
      const root = path.join(saved, 'BuildHistory');
      safeDirectory(projectRoot, root);
      safeDirectory(projectRoot, path.join(root, 'logs'));
      safeDirectory(projectRoot, path.join(root, 'events'));
      safeFile(path.join(root, 'index.json'));
      this.root = root;
      const indexFile = path.join(root, 'index.json');
      if (fs.existsSync(indexFile)) {
        if (fs.statSync(indexFile).size > 4 * 1024 * 1024) throw new Error('Build history index too large');
        const index = JSON.parse(fs.readFileSync(indexFile, 'utf8')) as {
          version?: number;
          projectGuid?: string;
          runs?: BuildMonitorRun[];
        };
        if (index.version !== HISTORY_VERSION || index.projectGuid !== this.projectGuid || !Array.isArray(index.runs))
          throw new Error('Build history index belongs to another project or version');
        this.runs = index.runs
          .filter(
            (run) => run && safeId(run.id) && run.projectGuid === this.projectGuid && typeof run.startedAt === 'string',
          )
          .slice(0, MAX_RUNS);
      }
      let recovered = false;
      for (const run of this.runs) {
        // Upgrade previously recovered histories that incorrectly assigned the
        // next editor startup time as the completion time (e.g. 1401 minutes).
        const legacyInterrupted = run.state === 'failed' && run.error?.startsWith('Interrupted:');
        if (terminal(run.state) && !legacyInterrupted) continue;
        recovered = true;
        run.state = 'interrupted';
        run.error = 'Editor closed before the build completed';
        // We don't know when the process actually stopped. Inventing an end
        // timestamp or elapsed time would mislead users, especially after days.
        run.completedAt = undefined;
        run.durationMs = undefined;
        for (const stage of run.stages ?? []) {
          if (stage.state === 'running' || (legacyInterrupted && stage.state === 'failed')) {
            stage.state = 'interrupted';
            stage.completedAt = undefined;
            stage.durationMs = undefined;
          }
        }
      }
      if (recovered) this.writeIndex();
    } catch (error) {
      this.storageError = String(error);
      this.root = null;
      this.runs = [];
    }
    return true;
  }

  private file(kind: 'logs' | 'events', id: string): string {
    if (!this.root || !safeId(id)) throw new Error('Build history unavailable or invalid run ID');
    return path.join(this.root, kind, `${id}.jsonl`);
  }

  private writeIndex(): void {
    if (!this.root || !this.projectGuid) return;
    const filename = path.join(this.root, 'index.json');
    safeFile(filename);
    const temporary = path.join(this.root, `index-${randomUUID()}.tmp`);
    try {
      fs.writeFileSync(
        temporary,
        JSON.stringify({ version: HISTORY_VERSION, projectGuid: this.projectGuid, runs: this.runs }, null, 2) + '\n',
        { flag: 'wx' },
      );
      fs.renameSync(temporary, filename);
    } finally {
      if (fs.existsSync(temporary)) fs.rmSync(temporary);
    }
  }

  /** Index writes are infrequent (run and stage transitions), never per output chunk. */
  saveRun(run: BuildMonitorRun): void {
    if (!this.root || run.projectGuid !== this.projectGuid || !safeId(run.id)) return;
    try {
      const existing = this.runs.findIndex((item) => item.id === run.id);
      if (existing < 0) this.runs.unshift(structuredClone(run));
      else this.runs[existing] = structuredClone(run);
      while (this.runs.length > MAX_RUNS) {
        const oldest = this.runs.pop()!;
        for (const kind of ['logs', 'events'] as const) {
          const filename = this.file(kind, oldest.id);
          safeFile(filename);
          fs.rmSync(filename, { force: true });
        }
      }
      this.writeIndex();
    } catch (error) {
      this.storageError = String(error);
    }
  }

  private append(kind: 'logs' | 'events', id: string, data: unknown): void {
    if (!this.root || !this.runs.some((run) => run.id === id)) return;
    try {
      const filename = this.file(kind, id);
      safeFile(filename);
      // O_NOFOLLOW prevents a swapped symlink from redirecting a log write.
      const flags =
        fs.constants.O_WRONLY | fs.constants.O_CREAT | fs.constants.O_APPEND | (fs.constants.O_NOFOLLOW ?? 0);
      const fd = fs.openSync(filename, flags, 0o600);
      try {
        fs.writeSync(fd, JSON.stringify(data) + '\n');
      } finally {
        fs.closeSync(fd);
      }
    } catch (error) {
      this.storageError = String(error);
    }
  }

  saveEvent(event: BuildMonitorEvent): void {
    this.append('events', event.runId, event);
  }

  appendRaw(runId: string, stream: 'stdout' | 'stderr', text: string, timestamp: string): void {
    if (!text) return;
    // Bound individual JSONL records, not the total log. Never discard the prefix.
    for (let start = 0; start < text.length;) {
      let end = Math.min(text.length, start + CHUNK_CHARS);
      if (end < text.length && /[\uD800-\uDBFF]/u.test(text[end - 1])) end--;
      this.append('logs', runId, { timestamp, stream, text: text.slice(start, end) } satisfies BuildRawLogChunk);
      start = end;
    }
  }

  list(offset = 0, limit = 50): BuildHistoryPage {
    if (!Number.isSafeInteger(offset) || offset < 0) throw new Error('Invalid build history offset');
    if (!Number.isSafeInteger(limit) || limit < 1 || limit > MAX_PAGE_SIZE)
      throw new Error('Invalid history page size');
    return {
      projectGuid: this.projectGuid,
      total: this.runs.length,
      offset,
      runs: structuredClone(this.runs.slice(offset, offset + limit)),
      ...(this.storageError ? { storageError: this.storageError } : {}),
    };
  }

  details(runId: string, logCursor = 0, eventCursor = 0): BuildHistoryDetail | null {
    if (!safeId(runId)) throw new Error('Invalid build history run ID');
    const run = this.runs.find((candidate) => candidate.id === runId);
    if (!run || !this.root) return null;
    return {
      run: structuredClone(run),
      logs: readLines<BuildRawLogChunk>(this.file('logs', runId), logCursor, 100),
      events: readLines<BuildMonitorEvent>(this.file('events', runId), eventCursor, 100),
    };
  }
}
