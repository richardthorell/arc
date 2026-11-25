import { createHash, randomBytes, timingSafeEqual } from 'node:crypto';
import fs from 'node:fs';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import type { Socket } from 'node:net';
import path from 'node:path';
import type { EditorContentFailureCode, EditorContentServerSnapshot } from '../common/editorContentDiagnosticsTypes';

export type { EditorContentServerSnapshot } from '../common/editorContentDiagnosticsTypes';
import type { LivePlayRemoteRelease } from './livePlayRemoteRelease';

/** C9.4a: main-process-owned read-only serving of already verified L1.3 output.
 * No renderer API, source-tree route, project SQL query, or untrusted filename.
 */
export type EditorContentServerOptions = Readonly<{
  projectRoot: string;
  sessionRoot: string;
  projectGuid: string;
  runtimeBuildId: string;
  sessionId: string;
  instanceId: string;
  port?: number;
  /** Test seam and upper bound for per-target replay history (1..256). */
  journalLimit?: number;
  /** Testable global request rate bound, never higher than 100 requests/second. */
  requestsPerSecondLimit?: number;
}>;

type ObjectRecord = Readonly<{
  file: string;
  size: number;
  sha256: string;
  dev: number;
  ino: number;
  mtimeMs: number;
  ctimeMs: number;
}>;

type Publication = Readonly<{ descriptor: Buffer; hash: string; generation: number; family: string }>;
type FeedTarget = Readonly<{
  platform: string;
  architecture: string;
  renderer: string;
  cookProfile: string;
  textureFamily: string;
}>;
type ImmutableReleaseRef = Readonly<{ path: string; sha256: string; size: number }>;
type FeedEvent = Readonly<{
  sequence: number;
  generation: number;
  release: ImmutableReleaseRef;
  changed: readonly string[];
  removed: readonly string[];
}>;
type FeedJournal = Readonly<{
  engineVersion: string;
  target: FeedTarget;
  oldestSequence: number;
  newestSequence: number;
  currentGeneration: number;
  currentRelease: ImmutableReleaseRef;
  events: readonly FeedEvent[];
}>;
export type PublishedGuidChanges = Readonly<{ changed: readonly string[]; removed: readonly string[] }>;

const guidPattern = /^[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}$/u;
const digestPattern = /^[a-f0-9]{64}$/u;
const profilePattern = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/u;
const familyPattern = /^[a-z0-9_-]{1,32}$/u;
const releasePath =
  /^\/releases\/([a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12})\/([A-Za-z0-9][A-Za-z0-9._-]{0,127})\/current\.json$/u;
const objectPath = /^\/objects\/sha256\/([a-f0-9]{64})$/u;
const feedPath = /^\/sessions\/([A-Za-z0-9_-]{1,128})\/changes$/u;
const feedQuery = /^after=(0|[1-9][0-9]*)(?:&profile=([A-Za-z0-9][A-Za-z0-9._-]{0,127}))?$/u;
const maxJournalEvents = 256;
const maxGuidsPerEvent = 4096;
const maxDescriptor = 8 * 1024 * 1024;
const maxTransfer = 256 * 1024 * 1024;
const noStore = 'no-store, no-cache, must-revalidate';
const maximumConnections = 16;
const defaultRequestsPerSecondLimit = 100;

function within(root: string, candidate: string): boolean {
  const relative = path.relative(root, candidate);
  return relative !== '' && relative !== '..' && !relative.startsWith('..' + path.sep) && !path.isAbsolute(relative);
}

function sealedFile(root: string, candidate: string): string {
  if (!path.isAbsolute(candidate) || !within(root, candidate)) throw new Error('Unapproved private cooked path');
  const stat = fs.lstatSync(candidate);
  if (!stat.isFile() || stat.isSymbolicLink() || fs.realpathSync(candidate) !== candidate)
    throw new Error('Private cooked object has a symlink or invalid file');
  return candidate;
}

async function hashFile(file: string): Promise<string> {
  const digest = createHash('sha256');
  for await (const block of fs.createReadStream(file, { highWaterMark: 1024 * 1024 })) digest.update(block as Buffer);
  return digest.digest('hex');
}

function parseRelease(
  bytes: Buffer,
  release: LivePlayRemoteRelease,
  cookProfile: string,
  family: string,
): {
  objects: Array<{ sha256: string; size: number }>;
  engineVersion: string;
  target: FeedTarget;
} {
  if (!bytes.length || bytes.length > maxDescriptor) throw new Error('Invalid C9 descriptor size');
  const raw: unknown = JSON.parse(bytes.toString('utf8'));
  if (!raw || typeof raw !== 'object') throw new Error('Invalid C9 release descriptor');
  const data = raw as Record<string, unknown>;
  const target = data.target as Record<string, unknown> | undefined;
  if (
    data.format !== 'arc.remote-content' ||
    data.formatVersion !== 1 ||
    data.complete !== true ||
    data.projectGuid !== release.projectGuid ||
    data.baseBuildId !== release.runtimeBuildId ||
    data.generation !== release.generation ||
    typeof data.engineVersion !== 'string' ||
    data.engineVersion.length < 1 ||
    data.engineVersion.length > 128 ||
    !/^[A-Za-z0-9._+-]+$/u.test(data.engineVersion) ||
    !target ||
    !['platform', 'architecture', 'renderer'].every(
      (key) => typeof target[key] === 'string' && profilePattern.test(target[key] as string),
    ) ||
    target.cookProfile !== cookProfile ||
    target.textureFamily !== family ||
    !Array.isArray(data.chunks) ||
    data.chunks.length > 100_000
  )
    throw new Error('C9 descriptor does not match the approved runtime, target or generation');
  const entries = [data.cookManifest, ...data.chunks];
  const objects: Array<{ sha256: string; size: number }> = [];
  const seen = new Set<string>();
  for (const entry of entries) {
    if (!entry || typeof entry !== 'object') throw new Error('Invalid C9 object reference');
    const ref = entry as Record<string, unknown>;
    if (
      typeof ref.sha256 !== 'string' ||
      !digestPattern.test(ref.sha256) ||
      ref.path !== 'objects/sha256/' + ref.sha256 ||
      typeof ref.size !== 'number' ||
      !Number.isSafeInteger(ref.size) ||
      ref.size < 1 ||
      ref.size > 2 * 1024 * 1024 * 1024 ||
      seen.has(ref.sha256)
    )
      throw new Error('Unsafe or duplicate immutable object reference');
    seen.add(ref.sha256);
    objects.push({ sha256: ref.sha256, size: ref.size });
  }
  return {
    objects,
    engineVersion: data.engineVersion as string,
    target: {
      platform: target.platform as string,
      architecture: target.architecture as string,
      renderer: target.renderer as string,
      cookProfile,
      textureFamily: family,
    },
  };
}

/** Only validated authoring GUID summaries appear in journal metadata. */
function guidChanges(
  changes: PublishedGuidChanges | undefined,
  first: boolean,
  sceneAssetGuid: string,
): PublishedGuidChanges {
  if (!changes && !first) throw new Error('New cooked generations require explicit changed/removed GUIDs');
  const changed = changes?.changed ?? [sceneAssetGuid];
  const removed = changes?.removed ?? [];
  if (changed.length > maxGuidsPerEvent || removed.length > maxGuidsPerEvent)
    throw new Error('C9 feed change list exceeds bounded GUID inventory');
  const seen = new Set<string>();
  for (const guid of [...changed, ...removed]) {
    if (!guidPattern.test(guid) || seen.has(guid))
      throw new Error('C9 feed contains an invalid, duplicate or overlapping asset GUID');
    seen.add(guid);
  }
  return { changed: [...changed].sort(), removed: [...removed].sort() };
}

/** Session credentials are editor-main-only and travel to native clients
 * over the already established protected C9.4c pairing path, never through
 * URL, CLI, project settings, logs or renderer IPC.
 */
export class EditorCookedContentServer {
  private server: Server | null = null;
  private readonly sockets = new Set<Socket>();
  private readonly publications = new Map<string, Publication>();
  private readonly feedJournals = new Map<string, FeedJournal>();
  private readonly objects = new Map<string, ObjectRecord>();
  private readonly projectRoot: string;
  private readonly sessionRoot: string;
  private port: number | null = null;
  private authorizationHash: Buffer | null = null;
  private authorizationExpires = 0;
  private requests = 0;
  private bytesSent = 0;
  private rejectedRequests = 0;
  private activeTransfers = 0;
  private windowStart = 0;
  private windowRequests = 0;
  private publishing = false;
  private objectCacheHits = 0;
  private objectResponses = 0;
  private releaseResponses = 0;
  private feedResponses = 0;
  private snapshotResets = 0;
  private publicationFailures = 0;
  private lastFailure: EditorContentServerSnapshot['lastFailure'] = null;

  private noteFailure(code: EditorContentFailureCode, status: number | null = null): void {
    // Intentionally never store a raw exception, request path, header,
    // native process argv, token or authoring filesystem path.
    this.lastFailure = { code, status, atMs: Date.now() };
  }

  constructor(private readonly options: EditorContentServerOptions) {
    if (
      !guidPattern.test(options.projectGuid) ||
      !options.runtimeBuildId ||
      !options.sessionId ||
      !options.instanceId ||
      (options.port !== undefined && (!Number.isInteger(options.port) || options.port < 0 || options.port > 65535)) ||
      (options.journalLimit !== undefined &&
        (!Number.isInteger(options.journalLimit) ||
          options.journalLimit < 1 ||
          options.journalLimit > maxJournalEvents)) ||
      !/^[A-Za-z0-9_-]{1,128}$/u.test(options.sessionId) ||
      (options.requestsPerSecondLimit !== undefined &&
        (!Number.isInteger(options.requestsPerSecondLimit) ||
          options.requestsPerSecondLimit < 1 ||
          options.requestsPerSecondLimit > defaultRequestsPerSecondLimit))
    )
      throw new Error('C9 editor content server requires a trusted session identity');
    this.projectRoot = fs.realpathSync(options.projectRoot);
    this.sessionRoot = fs.realpathSync(options.sessionRoot);
    if (
      !fs.statSync(this.projectRoot).isDirectory() ||
      !fs.statSync(this.sessionRoot).isDirectory() ||
      !path.isAbsolute(options.sessionRoot) ||
      this.projectRoot === this.sessionRoot ||
      within(this.projectRoot, this.sessionRoot)
    )
      throw new Error('C9 content session must be outside the authoring workspace');
  }

  snapshot(): EditorContentServerSnapshot {
    return {
      state: this.server ? 'listening' : 'stopped',
      origin: this.port === null ? null : 'http://127.0.0.1:' + this.port,
      port: this.port,
      bindMode: 'loopback',
      projectGuid: this.options.projectGuid,
      sessionId: this.options.sessionId,
      profiles: [...this.publications].map(([profile, p]) => ({
        profile,
        family: p.family,
        generation: p.generation,
      })),
      changeFeeds: [...this.feedJournals].map(([profile, journal]) => ({
        profile,
        oldestSequence: journal.oldestSequence,
        newestSequence: journal.newestSequence,
      })),
      activeClients: this.sockets.size,
      activeTransfers: this.activeTransfers,
      maximumConnections,
      requestsPerSecondLimit: this.options.requestsPerSecondLimit ?? defaultRequestsPerSecondLimit,
      requests: this.requests,
      bytesSent: this.bytesSent,
      rejectedRequests: this.rejectedRequests,
      objectCacheHits: this.objectCacheHits,
      objectResponses: this.objectResponses,
      releaseResponses: this.releaseResponses,
      feedResponses: this.feedResponses,
      snapshotResets: this.snapshotResets,
      publicationFailures: this.publicationFailures,
      lastFailure: this.lastFailure,
    };
  }

  async start(): Promise<void> {
    if (this.server) throw new Error('C9 content listener is already active');
    const server = createServer((req, res) => this.handle(req, res));
    server.maxConnections = maximumConnections;
    server.headersTimeout = 5000;
    server.requestTimeout = 8000;
    server.keepAliveTimeout = 2000;
    server.on('connection', (socket) => {
      this.sockets.add(socket);
      socket.on('close', () => this.sockets.delete(socket));
    });
    try {
      await new Promise<void>((resolve, reject) => {
        server.once('error', reject);
        server.listen(this.options.port ?? 0, '127.0.0.1', () => {
          server.off('error', reject);
          resolve();
        });
      });
      const address = server.address();
      if (!address || typeof address === 'string') throw new Error('C9 listener has no local port');
      this.server = server;
      this.port = address.port;
      server.on('error', () => {
        this.noteFailure('server-failed');
        if (this.server === server) {
          this.server = null;
          this.port = null;
          this.publications.clear();
          this.feedJournals.clear();
          this.objects.clear();
          this.authorizationHash?.fill(0);
          this.authorizationHash = null;
          this.authorizationExpires = 0;
        }
        for (const socket of this.sockets) socket.destroy();
        try {
          server.close();
        } catch {
          // The socket may have already closed.
        }
      });
    } catch (error) {
      this.noteFailure('bind-failed');
      try {
        server.close();
      } catch {
        /* failed listen has no open port */
      }
      throw new Error(
        'Unable to bind C9 loopback listener: ' + (error instanceof Error ? error.message : 'unknown error'),
      );
    }
  }

  issueClientToken(): string {
    if (!this.server) throw new Error('C9 listener is not active');
    const token = randomBytes(32).toString('hex');
    this.authorizationHash?.fill(0);
    this.authorizationHash = createHash('sha256').update(token).digest();
    this.authorizationExpires = Date.now() + 15 * 60_000;
    return token;
  }

  /** Verify a complete release before atomically advertising its descriptor
   * and object inventory. Failed candidates leave the old generation visible.
   */
  async publish(
    release: LivePlayRemoteRelease,
    cookProfile: string,
    family?: string,
    changes?: PublishedGuidChanges,
  ): Promise<void> {
    const originalServer = this.server;
    if (!originalServer) throw new Error('C9 listener is not active');
    if (this.publishing) throw new Error('C9 publication is already being verified');
    this.publishing = true;
    try {
      if (
        release.projectGuid !== this.options.projectGuid ||
        release.runtimeBuildId !== this.options.runtimeBuildId ||
        release.sessionId !== this.options.sessionId ||
        release.instanceId !== this.options.instanceId ||
        !Number.isSafeInteger(release.generation) ||
        release.generation < 1 ||
        !profilePattern.test(cookProfile)
      )
        throw new Error('C9 release belongs to a different project, target or session');
      if (!release.variants.length || (release.variants.length > 1 && !family))
        throw new Error('Explicit texture family required for multi-variant C9 publication');

      const root = fs.realpathSync(release.root);
      if (
        !within(this.sessionRoot, root) ||
        path.basename(root) !== 'release' ||
        fs.lstatSync(release.root).isSymbolicLink() ||
        !fs.statSync(root).isDirectory()
      )
        throw new Error('Release is outside sealed private session storage');
      const variant = family ? release.variants.find((item) => item.family === family) : release.variants[0];
      if (!variant || !familyPattern.test(variant.family)) throw new Error('Unknown C9 texture family');
      const current = this.publications.get(cookProfile);
      if (current && release.generation <= current.generation)
        throw new Error('C9 generation must increase monotonically for the selected target');

      const descriptorFile = sealedFile(root, variant.descriptorPath);
      if (descriptorFile !== path.join(root, 'targets', variant.family, 'current.json'))
        throw new Error('C9 descriptor is outside its selected target inventory');
      const descriptor = await fs.promises.readFile(descriptorFile);
      const hash = createHash('sha256').update(descriptor).digest('hex');
      if (hash !== variant.descriptorSha256) throw new Error('C9 descriptor SHA-256 changed');
      const parsed = parseRelease(descriptor, release, cookProfile, variant.family);
      const existingJournal = this.feedJournals.get(cookProfile);
      if (
        existingJournal &&
        (existingJournal.engineVersion !== parsed.engineVersion ||
          Object.entries(parsed.target).some(
            ([key, value]) => existingJournal.target[key as keyof FeedTarget] !== value,
          ))
      )
        throw new Error('C9 target or engine identity changed within a feed session');
      const summary = guidChanges(changes, !existingJournal, release.sceneAssetGuid);
      const sequence = (existingJournal?.newestSequence ?? 0) + 1;
      if (!Number.isSafeInteger(sequence)) throw new Error('C9 feed sequence overflow');

      const nextObjects = new Map(this.objects);
      for (const reference of parsed.objects) {
        const file = sealedFile(root, path.join(root, 'objects', 'sha256', reference.sha256));
        const stat = fs.statSync(file);
        if (stat.size !== reference.size || (await hashFile(file)) !== reference.sha256)
          throw new Error('C9 object failed size/hash verification');
        const old = nextObjects.get(reference.sha256);
        if (old && old.size !== reference.size) throw new Error('Incompatible C9 object identity');
        nextObjects.set(reference.sha256, {
          file,
          size: reference.size,
          sha256: reference.sha256,
          dev: stat.dev,
          ino: stat.ino,
          mtimeMs: stat.mtimeMs,
          ctimeMs: stat.ctimeMs,
        });
      }

      // The canonical descriptor itself must be accessible through immutable
      // SHA-addressed storage before its event becomes visible. Returning a
      // mutable current.json URL here would make replay after update unsafe.
      const descriptorStat = fs.statSync(descriptorFile);
      const existingObject = nextObjects.get(hash);
      if (existingObject && existingObject.size !== descriptor.length)
        throw new Error('C9 immutable release descriptor hash collision');
      nextObjects.set(hash, {
        file: descriptorFile,
        size: descriptor.length,
        sha256: hash,
        dev: descriptorStat.dev,
        ino: descriptorStat.ino,
        mtimeMs: descriptorStat.mtimeMs,
        ctimeMs: descriptorStat.ctimeMs,
      });
      const releaseRef: ImmutableReleaseRef = { path: 'objects/sha256/' + hash, sha256: hash, size: descriptor.length };
      const event: FeedEvent = {
        sequence,
        generation: release.generation,
        release: releaseRef,
        changed: summary.changed,
        removed: summary.removed,
      };
      const limit = this.options.journalLimit ?? maxJournalEvents;
      const events = [...(existingJournal?.events ?? []), event].slice(-limit);
      const feed: FeedJournal = {
        engineVersion: parsed.engineVersion,
        target: parsed.target,
        oldestSequence: events[0].sequence,
        newestSequence: sequence,
        currentGeneration: release.generation,
        currentRelease: releaseRef,
        events,
      };
      if (this.server !== originalServer) throw new Error('C9 listener changed during release verification');
      const nextPublications = new Map(this.publications);
      nextPublications.set(cookProfile, { descriptor, hash, generation: release.generation, family: variant.family });
      // No await after this point: a request can observe either old complete
      // generation and journal or new complete generation and journal, never a
      // partially published descriptor or unverified cook object.
      this.objects.clear();
      for (const [name, entry] of nextObjects) this.objects.set(name, entry);
      this.publications.clear();
      for (const [name, entry] of nextPublications) this.publications.set(name, entry);
      this.feedJournals.set(cookProfile, feed);
    } catch (error) {
      this.publicationFailures++;
      this.noteFailure('publish-rejected');
      throw error;
    } finally {
      this.publishing = false;
    }
  }

  async stop(): Promise<void> {
    const server = this.server;
    this.server = null;
    this.port = null;
    this.publications.clear();
    this.feedJournals.clear();
    this.objects.clear();
    this.authorizationHash?.fill(0);
    this.authorizationHash = null;
    this.authorizationExpires = 0;
    for (const socket of this.sockets) socket.destroy();
    this.sockets.clear();
    if (!server) return;
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }

  private authorized(req: IncomingMessage): boolean {
    const header = req.headers.authorization;
    if (!this.authorizationHash || Date.now() >= this.authorizationExpires || typeof header !== 'string') return false;
    const match = /^Bearer ([a-f0-9]{64})$/u.exec(header);
    if (!match) return false;
    return timingSafeEqual(createHash('sha256').update(match[1]).digest(), this.authorizationHash);
  }

  private reject(res: ServerResponse, code: number, message: string, failure?: EditorContentFailureCode): void {
    this.rejectedRequests++;
    this.noteFailure(
      failure ??
        (code === 401
          ? 'unauthorized'
          : code === 403
            ? 'forbidden'
            : code === 429
              ? 'rate-limited'
              : code === 404
                ? 'not-found'
                : code === 416 || code === 412
                  ? 'range-rejected'
                  : code === 503
                    ? 'unavailable'
                    : 'invalid-request'),
      code,
    );
    res.writeHead(code, {
      'Content-Type': 'text/plain; charset=utf-8',
      'Cache-Control': noStore,
      'X-Content-Type-Options': 'nosniff',
      ...(code === 401 ? { 'WWW-Authenticate': 'Bearer' } : {}),
    });
    res.end(message);
  }

  private handle(req: IncomingMessage, res: ServerResponse): void {
    this.requests++;
    const now = Date.now();
    if (now - this.windowStart >= 1000) {
      this.windowStart = now;
      this.windowRequests = 0;
    }
    if (++this.windowRequests > (this.options.requestsPerSecondLimit ?? defaultRequestsPerSecondLimit))
      return this.reject(res, 429, 'Request limit exceeded');
    if (!this.server || this.port === null) return this.reject(res, 503, 'Content server stopping');
    if (req.socket.remoteAddress !== '127.0.0.1') return this.reject(res, 403, 'Loopback only');
    if (req.headers.host !== '127.0.0.1:' + this.port) return this.reject(res, 400, 'Invalid content authority');
    if (req.rawHeaders.length / 2 > 64) return this.reject(res, 431, 'Invalid request headers');
    if (!this.authorized(req)) return this.reject(res, 401, 'Content pairing required');
    if (req.method !== 'GET') return this.reject(res, 405, 'Only read-only GET is supported');
    if (req.headers['content-length'] !== undefined) return this.reject(res, 431, 'Request body is not permitted');
    const requested = req.url ?? '';
    if (requested.length > 512 || requested.includes('%') || requested.includes('#') || requested.includes('\\'))
      return this.reject(res, 400, 'Unsafe content URL');

    // Exact C9.1c polling endpoint. Query strings are forbidden everywhere
    // else; the sole permitted query is a strict ACK cursor plus an optional
    // cook-profile selector for a multi-target editor session.
    const queryStart = requested.indexOf('?');
    const route = queryStart < 0 ? requested : requested.slice(0, queryStart);
    const query = queryStart < 0 ? '' : requested.slice(queryStart + 1);
    const feedRoute = feedPath.exec(route);
    if (feedRoute) {
      if (feedRoute[1] !== this.options.sessionId) return this.reject(res, 404, 'Unknown content session');
      const matched = feedQuery.exec(query);
      if (!matched) return this.reject(res, 400, 'Invalid change-feed cursor or target');
      const afterSequence = Number(matched[1]);
      if (!Number.isSafeInteger(afterSequence)) return this.reject(res, 400, 'Invalid change-feed cursor');
      const profile = matched[2] ?? (this.feedJournals.size === 1 ? this.feedJournals.keys().next().value : undefined);
      if (!profile) return this.reject(res, this.feedJournals.size ? 400 : 503, 'Select a published cook target');
      const journal = this.feedJournals.get(profile);
      if (!journal) return this.reject(res, 404, 'No verified feed for target');
      if (afterSequence > journal.newestSequence) return this.reject(res, 409, 'Change-feed cursor regressed');
      let mode =
        afterSequence === journal.newestSequence
          ? 'up-to-date'
          : afterSequence < journal.oldestSequence - 1 || journal.newestSequence - afterSequence > maxJournalEvents
            ? 'full-snapshot'
            : 'replay';
      let events = mode === 'replay' ? journal.events.filter((entry) => entry.sequence > afterSequence) : [];
      const encode = () =>
        Buffer.from(
          JSON.stringify({
            format: 'arc.remote-content-feed',
            formatVersion: 1,
            projectGuid: this.options.projectGuid,
            engineVersion: journal.engineVersion,
            baseBuildId: this.options.runtimeBuildId,
            target: journal.target,
            afterSequence,
            oldestSequence: journal.oldestSequence,
            newestSequence: journal.newestSequence,
            currentGeneration: journal.currentGeneration,
            currentRelease: journal.currentRelease,
            mode,
            events,
          }),
          'utf8',
        );
      let bytes = encode();
      // Never truncate a replay event list: that would violate native C9.1c
      // contiguous-sequence parsing. Send a full snapshot when bounded
      // journal metadata is too large, even if all sequences are retained.
      if (bytes.length > maxDescriptor && mode === 'replay') {
        mode = 'full-snapshot';
        events = [];
        bytes = encode();
      }
      if (bytes.length > maxDescriptor) return this.reject(res, 503, 'C9 change-feed exceeds metadata bounds');
      res.writeHead(200, {
        'Content-Type': 'application/json',
        'Content-Length': bytes.length,
        'Cache-Control': noStore,
        'X-Content-Type-Options': 'nosniff',
      });
      res.end(bytes);
      this.bytesSent += bytes.length;
      this.feedResponses++;
      if (mode === 'full-snapshot') this.snapshotResets++;
      return;
    }
    if (queryStart >= 0) return this.reject(res, 400, 'Query strings are not permitted on cooked objects');

    const release = releasePath.exec(route);
    if (release) {
      if (release[1] !== this.options.projectGuid) return this.reject(res, 404, 'Unknown project');
      const entry = this.publications.get(release[2]);
      if (!entry) return this.reject(res, 404, 'No verified release for target');
      res.writeHead(200, {
        'Content-Type': 'application/json',
        'Content-Length': entry.descriptor.length,
        'Cache-Control': noStore,
        ETag: '"' + entry.hash + '"',
        'X-Content-Type-Options': 'nosniff',
      });
      res.end(entry.descriptor);
      this.bytesSent += entry.descriptor.length;
      this.releaseResponses++;
      return;
    }

    const object = objectPath.exec(requested);
    if (!object) return this.reject(res, 404, 'Unknown cooked endpoint');
    const entry = this.objects.get(object[1]);
    if (!entry) return this.reject(res, 404, 'Unpublished object');
    const etag = '"' + entry.sha256 + '"';
    if (req.headers['if-match'] && req.headers['if-match'] !== etag) return this.reject(res, 412, 'ETag mismatch');
    let start = 0;
    let end = entry.size - 1;
    const range = req.headers.range;
    if (range !== undefined) {
      if (typeof range !== 'string') return this.reject(res, 400, 'Invalid range header');
      const match = /^bytes=(0|[1-9][0-9]*)-(0|[1-9][0-9]*)$/u.exec(range);
      if (!match) return this.reject(res, 400, 'Only explicit bounded byte ranges are supported');
      start = Number(match[1]);
      end = Number(match[2]);
      if (
        !Number.isSafeInteger(start) ||
        !Number.isSafeInteger(end) ||
        start > end ||
        end >= entry.size ||
        end - start + 1 > maxTransfer
      ) {
        res.setHeader('Content-Range', 'bytes */' + entry.size);
        return this.reject(res, 416, 'Range not satisfiable');
      }
    } else if (entry.size > maxTransfer) {
      return this.reject(res, 413, 'Large cooked objects require bounded ranges');
    }

    let descriptor: number | null = null;
    try {
      // Open once, then fstat the actual descriptor we will stream. Reopening
      // the pathname after lstat creates a symlink/file-replacement race.
      // On platforms without O_NOFOLLOW the inode/realpath checks still
      // reject substituted paths, and the pinned descriptor never follows
      // a later rename while the transfer is in progress.
      descriptor = fs.openSync(entry.file, fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW ?? 0));
      const stat = fs.fstatSync(descriptor);
      const linked = fs.lstatSync(entry.file);
      if (
        !stat.isFile() ||
        !linked.isFile() ||
        linked.isSymbolicLink() ||
        stat.size !== entry.size ||
        stat.dev !== entry.dev ||
        stat.ino !== entry.ino ||
        stat.mtimeMs !== entry.mtimeMs ||
        stat.ctimeMs !== entry.ctimeMs ||
        linked.dev !== stat.dev ||
        linked.ino !== stat.ino ||
        fs.realpathSync(entry.file) !== entry.file
      )
        return this.reject(res, 503, 'Published object changed after verification', 'content-changed');
      const stream = fs.createReadStream(entry.file, {
        fd: descriptor,
        autoClose: true,
        start,
        end,
        highWaterMark: 64 * 1024,
      });
      // createReadStream now owns and closes the exact verified fd.
      descriptor = null;
      this.objectCacheHits++;
      this.objectResponses++;
      this.activeTransfers++;
      res.once('close', () => {
        this.activeTransfers = Math.max(0, this.activeTransfers - 1);
        stream.destroy();
      });
      stream.once('error', () => {
        this.noteFailure('unavailable', 503);
        if (!res.headersSent) this.reject(res, 503, 'Cooked object unavailable');
        else res.destroy();
      });
      res.once('finish', () => {
        this.bytesSent += end - start + 1;
      });
      const headers: Record<string, string | number> = {
        'Content-Type': 'application/octet-stream',
        'Content-Length': end - start + 1,
        'Accept-Ranges': 'bytes',
        'Cache-Control': 'private, max-age=31536000, immutable',
        ETag: etag,
        'X-Content-Type-Options': 'nosniff',
      };
      if (range !== undefined) headers['Content-Range'] = 'bytes ' + start + '-' + end + '/' + entry.size;
      res.writeHead(range !== undefined ? 206 : 200, headers);
      stream.pipe(res);
    } catch {
      if (!res.headersSent) this.reject(res, 503, 'Cooked object unavailable');
      else res.destroy();
    } finally {
      if (descriptor !== null) fs.closeSync(descriptor);
    }
  }
}
